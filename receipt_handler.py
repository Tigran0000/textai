# src/ocr/handlers/receipt_handler.py
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import pytesseract
import re
import time
from PIL import Image
from datetime import datetime

from .base_handler import DocumentHandler

class ReceiptHandler(DocumentHandler):
    """Handler for receipts and invoices with column structure preservation"""
    
    def can_handle(self, image: Image.Image) -> bool:
        """Check if image is likely a receipt based on visual characteristics"""
        try:
            # Check for typical receipt aspect ratio (tall and narrow)
            width, height = image.size
            aspect_ratio = width / height
            
            # Most receipts are narrow (tall rectangles)
            if aspect_ratio > 0.9:  # Not narrow enough
                return False
                
            # Convert to grayscale for analysis
            img_array = np.array(image.convert('L'))
            
            # Check brightness - receipts typically have very white backgrounds
            avg_brightness = np.mean(img_array)
            if avg_brightness < 180:  # Too dark for typical receipt
                return False
                
            # Check for receipt content using OCR
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Create a small version for quick analysis
                small_img = image.copy()
                small_img.thumbnail((600, 800), Image.Resampling.LANCZOS)
                
                # Quick OCR with simple config
                text = pytesseract.image_to_string(small_img, config='--psm 6').lower()
                
                # Look for receipt indicators
                receipt_indicators = [
                    'total', 'subtotal', 'tax', 'cash', 'change',
                    'receipt', 'transaction', 'item', 'qty', 'price',
                    'amount', 'payment', 'store', 'tel:', 'date:',
                    'thank you', '$', '£', '€', 'order', 'customer',
                    'cashier', 'merchant', 'card', 'sale',
                    # Common receipt currencies and symbols
                    'usd', 'eur', 'gbp', 'jpy', 'cad', 'aud'
                ]
                
                # Check for monetary values (like "$10.99")
                money_pattern = r'[$£€]\s*\d+\.\d{2}'
                has_monetary_values = bool(re.search(money_pattern, text))
                
                # Check for multiple receipt indicators
                indicator_count = sum(1 for indicator in receipt_indicators if indicator in text)
                
                # If we found multiple indicators or monetary values, it's likely a receipt
                if indicator_count >= 3 or has_monetary_values:
                    return True
            
            # Check for receipt-like structure (even without OCR)
            # 1. Look for horizontal lines (often used to separate sections in receipts)
            edges = cv2.Canny(img_array, 50, 200)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=width*0.5, maxLineGap=10)
            
            horizontal_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is horizontal
                    if abs(y2 - y1) < 10:
                        horizontal_lines += 1
            
            # 2. Check for centered text pattern (common in receipts)
            # Use horizontal projection to find text line center positions
            h_projection = np.sum(edges, axis=1)
            text_lines = [i for i, val in enumerate(h_projection) if val > width*0.1]
            
            # If we have enough horizontal lines or a narrow image with many text lines,
            # it's likely a receipt
            if horizontal_lines >= 3 or (aspect_ratio < 0.6 and len(text_lines) > 10):
                return True
                
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"Receipt detection error: {e}")
            return False
    
    def _perform_extraction(self, image: Image.Image, preprocess: bool) -> Dict:
        """Extract text from receipt with column structure preservation"""
        try:
            # Use specialized receipt preprocessing if requested
            if preprocess and 'image' in self.processors:
                processed_image = self.processors['image'].preprocess_receipt(image.copy())
            else:
                processed_image = image.copy()
            
            # For receipts, use a hybrid approach with fine-tuned settings
            # First try with Tesseract using a column-aware config
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Specialized config for column data in receipts
                config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
                
                # Get position data for better structure
                data = pytesseract.image_to_data(
                    processed_image, 
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Use data to create better structured receipt text
                text = self._build_receipt_text_structure(data)
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Clean receipt text
                if 'text' in self.processors:
                    text = self.processors['text'].clean_receipt_text(text)
                else:
                    text = self._clean_receipt_text(text)
                
                # Extract structured information
                receipt_info = self._extract_receipt_info(text)
                
                return {
                    'text': text,
                    'confidence': confidence,
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'success': True,
                    'engine': 'receipt_extraction',
                    'receipt_info': receipt_info
                }
            
            else:
                # Fallback to basic extraction if Tesseract isn't available
                return self._extract_with_basic_method(processed_image)
                
        except Exception as e:
            return self._error_result(f"Receipt extraction error: {str(e)}")

    def _build_receipt_text_structure(self, data: dict) -> str:
        """Build structured receipt text from Tesseract data with column alignment"""
        
        lines = {}
        
        # Group words by text line with their positions
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 20 and data['text'][i].strip():  # Lower confidence threshold for receipts
                block_num = data['block_num'][i]
                line_num = data['line_num'][i]
                
                # Create a unique key for this line
                line_key = f"{block_num}_{line_num}"
                
                # Create line if needed
                if line_key not in lines:
                    lines[line_key] = {
                        'words': [],
                        'top': data['top'][i],
                        'left_positions': []
                    }
                
                # Add word with its position
                lines[line_key]['words'].append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'width': data['width'][i],
                    'conf': data['conf'][i]
                })
                
                # Track left position for column analysis
                lines[line_key]['left_positions'].append(data['left'][i])
        
        # Sort lines by vertical position
        sorted_line_keys = sorted(lines.keys(), key=lambda k: lines[k]['top'])
        
        # Analyze receipt structure to find columns
        receipt_columns = self._analyze_receipt_columns(lines)
        
        # Use column information to format receipt text
        result_lines = []
        
        for key in sorted_line_keys:
            line = lines[key]
            
            # Sort words by horizontal position
            sorted_words = sorted(line['words'], key=lambda w: w['left'])
            
            # For single-column entries, simply join all words
            if len(receipt_columns) <= 1:
                line_text = ' '.join(word['text'] for word in sorted_words)
                result_lines.append(line_text)
                continue
            
            # For multi-column receipts, align text to columns
            column_texts = [''] * len(receipt_columns)
            
            for word in sorted_words:
                # Find the column this word belongs to
                column_idx = 0
                for i, col_pos in enumerate(receipt_columns):
                    if abs(word['left'] - col_pos) < 20:
                        column_idx = i
                        break
                
                # Add the word to its column
                if column_texts[column_idx]:
                    column_texts[column_idx] += ' ' + word['text']
                else:
                    column_texts[column_idx] = word['text']
            
            # Format line with proper spacing between columns
            line_parts = []
            for i, col_text in enumerate(column_texts):
                if col_text:
                    if i == 0:  # First column (item descriptions)
                        line_parts.append(col_text)
                    else:  # Numeric columns (price, quantity, etc.)
                        # Right-align numeric columns
                        line_parts.append(col_text.rjust(12))
            
            result_lines.append('  '.join(line_parts))
        
        # Join all lines
        result_text = '\n'.join(result_lines)
        
        # Clean up the text
        result_text = self._clean_receipt_text(result_text)
        
        return result_text

    def _analyze_receipt_columns(self, lines: Dict) -> List[int]:
        """Analyze receipt structure to determine column positions"""
        
        # Collect all left positions
        all_positions = []
        for line_key, line in lines.items():
            all_positions.extend(line['left_positions'])
        
        # If no positions found, return empty list
        if not all_positions:
            return []
            
        # Create position histogram
        position_counts = {}
        for pos in all_positions:
            # Group positions within 10 pixels
            pos_key = pos // 10 * 10
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
        
        # Find position clusters (peaks in the histogram)
        min_count = max(3, len(lines) // 4)  # Minimum frequency to be considered a column
        column_positions = [pos for pos, count in position_counts.items() if count >= min_count]
        
        # Sort positions from left to right
        column_positions.sort()
        
        return column_positions

    def _extract_with_basic_method(self, image: Image.Image) -> Dict:
        """Fallback method for receipt extraction when specialized methods fail"""
        try:
            # Get text using standard OCR settings
            text = pytesseract.image_to_string(image)
            
            # Clean and format as receipt
            cleaned_text = self._clean_receipt_text(text)
            
            # Extract receipt information
            receipt_info = self._extract_receipt_info(cleaned_text)
            
            return {
                'text': cleaned_text,
                'confidence': 70,  # Default confidence for basic method
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text),
                'success': True,
                'engine': 'basic_receipt_extraction',
                'receipt_info': receipt_info
            }
            
        except Exception as e:
            return self._error_result(f"Basic receipt extraction error: {str(e)}")

    def _clean_receipt_text(self, text: str) -> str:
        """Clean receipt text from common OCR errors"""
        if not text:
            return ""
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)         # Multiple spaces to single space
        text = re.sub(r' *\n *', '\n', text)     # Clean up around line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)   # Max two consecutive line breaks
        
        # Fix common receipt OCR errors
        replacements = {
            # Currency symbols
            'S': '$',                            # When preceding a digit
            'S0.': '$0.',                        # Common at start of amounts
            '50.': '$0.',                        # Common at start of amounts
            '5o.': '$0.',                        # Common at start of amounts
            'O.': '0.',                          # Letter O to number 0
            'O,': '0.',                          # Letter O to number 0
            'o.': '0.',                          # Letter o to number 0
            'o,': '0.',                          # Letter o to number 0
            'I.': '1.',                          # Letter I to number 1
            'I,': '1.',                          # Letter I to number 1
            'l.': '1.',                          # Letter l to number 1
            'l,': '1.',                          # Letter l to number 1
            
            # Common receipt terms
            'Tc tal': 'Total',                   # Common misspelling
            'To tal': 'Total',                   # Common misspelling
            'SUB TOTAL': 'SUBTOTAL',             # Remove space
            'SIJBTOTAL': 'SUBTOTAL',             # Fix OCR error
            'ltem': 'Item',                      # Common misspelling
            'ltems': 'Items',                    # Common misspelling
            'Arnount': 'Amount',                 # Common misspelling
            'Thark': 'Thank',                    # Common misspelling
            'Tharks': 'Thanks',                  # Common misspelling
            'Ycu': 'You',                        # Common misspelling
            'Yor': 'You',                        # Common misspelling
            'Retum': 'Return',                   # Common misspelling
            'Excharcge': 'Exchange',             # Common misspelling
            'Exchance': 'Exchange',              # Common misspelling
            'MERGHANT': 'MERCHANT',              # Common misspelling
            'GOPY': 'COPY',                      # Common misspelling
            'GARD': 'CARD',                      # Common misspelling
            'GASH': 'CASH'                       # Common misspelling
        }
        
        # Apply replacements
        for error, correction in replacements.items():
            text = text.replace(error, correction)
        
        # Fix currency symbol at start of string
        text = re.sub(r'^S(\d)', r'$\1', text, flags=re.MULTILINE)
        
        # Fix currency symbol after space
        text = re.sub(r' S(\d)', r' $\1', text)
        
        # Fix decimals in monetary values (ensure two decimal places)
        text = re.sub(r'(\d+)\.(\d)(?!\d)', r'\1.\20', text)  # Add missing zero
        
        return text

    def _extract_receipt_info(self, text: str) -> Dict[str, Any]:
        """Extract structured information from receipt text"""
        
        info = {
            'total': None,
            'subtotal': None,
            'tax': None,
            'date': None,
            'time': None,
            'merchant': None,
            'items': []
        }
        
        # Extract total
        total_patterns = [
            r'(?:total|amount|sum).*?(?:\$|£|€)?\s*([\d,.]+)',
            r'(?:total|amount|sum).*?(\d+\.\d{2})',
            r'(?:\$|£|€)?\s*([\d,.]+).*?(?:total|amount)',
            r'(\d+\.\d{2}).*?(?:total|amount)'
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['total'] = match.group(1).strip()
                break
        
        # Extract subtotal
        subtotal_patterns = [
            r'(?:subtotal|sub-total).*?(?:\$|£|€)?\s*([\d,.]+)',
            r'(?:subtotal|sub-total).*?(\d+\.\d{2})'
        ]
        
        for pattern in subtotal_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['subtotal'] = match.group(1).strip()
                break
        
        # Extract tax
        tax_patterns = [
            r'(?:tax|vat|gst).*?(?:\$|£|€)?\s*([\d,.]+)',
            r'(?:tax|vat|gst).*?(\d+\.\d{2})'
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['tax'] = match.group(1).strip()
                break
        
        # Extract date
        date_patterns = [
            r'(?:date|dt)[^:]*?:?\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(\d{2,4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                info['date'] = match.group(1).strip()
                break
        
        # Extract time
        time_patterns = [
            r'(?:time|tm)[^:]*?:?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)',
            r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                info['time'] = match.group(1).strip()
                break
        
        # Extract merchant (usually in first few lines)
        lines = text.split('\n')
        if lines:
            # First non-empty line often contains merchant name
            for line in lines[:5]:  # Check first 5 lines
                if line.strip() and not re.match(r'^\s*\d', line.strip()):
                    # Skip lines starting with digits (likely dates)
                    if not any(keyword in line.lower() for keyword in ['date', 'time', 'receipt', 'tel', 'phone']):
                        info['merchant'] = line.strip()
                        break
        
        # Extract items with prices (simple pattern)
        item_pattern = r'(.*?)\s+(?:\$|£|€)?\s*(\d+\.\d{2})'
        potential_items = []
        
        # Skip header and total sections
        start_idx = 5  # Skip first few lines (typically header)
        
        for i in range(start_idx, len(lines) - 3):  # Skip last few lines (typically totals)
            line = lines[i].strip()
            if not line:
                continue
                
            match = re.search(item_pattern, line)
            if match and len(match.group(1).strip()) > 1:  # Skip if item name too short
                item = {
                    'name': match.group(1).strip(),
                    'price': match.group(2).strip()
                }
                potential_items.append(item)
        
        # Filter out false positives (items should have similar formats)
        if len(potential_items) >= 2:
            info['items'] = potential_items
        
        return info