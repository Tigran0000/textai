"""
EyeShot AI - Book Document Handler
Specialized handler for book pages and academic documents
Last updated: 2025-06-20 10:18:45 UTC
Author: Tigran0000
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import pytesseract
import re
import time
from PIL import Image

from .base_handler import DocumentHandler

# Try importing BeautifulSoup, but make it optional
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

class BookHandler(DocumentHandler):
    """Handler for book pages with column detection and layout preservation"""
    
    def __init__(self):
        super().__init__()
        self.name = "book"
        self.description = "Handler for book pages and academic documents"
        self.bs4_available = BS4_AVAILABLE
        
    def can_handle(self, image: Image.Image) -> bool:
        """Check if image is likely a book page"""
        try:
            # Check for book-like aspect ratio
            width, height = image.size
            aspect_ratio = width / height
            
            # Most books have aspect ratios within this range
            if 0.6 <= aspect_ratio <= 0.8:
                # Quick check for page number presence
                small_img = image.copy()
                small_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                if hasattr(self.engines, 'tesseract_available') and self.engines.get('tesseract_available', False):
                    # Check for page numbers or chapter markers
                    text = pytesseract.image_to_string(small_img, config='--psm 1').lower()
                    
                    # Book indicators
                    book_indicators = ['chapter', 'page', 'contents', 'preface', 'introduction', 
                                   'rule', 'section', 'appendix']
                    
                    # Check for book indicators or page numbers
                    if any(indicator in text for indicator in book_indicators) or re.search(r'\b\d+\b', text):
                        return True
                
                # Even without OCR, the aspect ratio is a good indicator
                return True
            
            return False
                
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Book page detection error: {e}")
            return False

    def extract_text(self, image: Image.Image, preprocess: bool = True) -> Dict:
        """
        Extract text from image with book-specific optimizations
        
        Args:
            image: PIL Image to process
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary with extraction results
        """
        try:
            # Apply book-specific preprocessing
            if preprocess and 'image' in self.processors:
                processed_img = self.processors['image'].preprocess_for_ocr(image.copy())
                
                # Apply additional book-specific enhancements if available
                if hasattr(self.processors['image'], 'preprocess_book_page'):
                    processed_img = self.processors['image'].preprocess_book_page(processed_img)
            else:
                processed_img = image.copy()
            
            # Detect columns before processing
            columns, column_boxes = self._detect_book_columns(processed_img)
            
            # If multiple columns detected, process each separately
            if columns > 1:
                return self._extract_multi_column_book_text(processed_img, column_boxes)
            else:
                # For single column, use either advanced or basic extraction
                if BS4_AVAILABLE:
                    return self._extract_single_column_book_text(processed_img)
                else:
                    # If BeautifulSoup isn't available, use a simpler approach
                    return self._fallback_extract_text(processed_img)
                
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Book extraction error: {e}")
            return self._fallback_extract_text(image)

    def _fallback_extract_text(self, image: Image.Image) -> Dict:
        """Fallback text extraction when BeautifulSoup isn't available"""
        try:
            # Use standard OCR with settings optimized for books
            config = '--oem 3 --psm 1 -c preserve_interword_spaces=1'
            
            # Extract text
            text = pytesseract.image_to_string(image, config=config)
            
            # Clean up the text for books
            cleaned_text = self._clean_book_text(text)
            
            # Get confidence data
            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': cleaned_text,
                'confidence': avg_confidence,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text),
                'success': True,
                'engine': 'book_standard',
                'has_structure': False
            }
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'char_count': 0,
                'success': False,
                'error': str(e)
            }

    def _detect_book_columns(self, image: Image.Image) -> Tuple[int, List]:
        """
        Detect text columns in book page images
        
        Args:
            image: PIL image of book page
        
        Returns:
            tuple: (number_of_columns, list_of_column_boundaries)
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array.astype(np.uint8)
            
            # Binary threshold to isolate text
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Sum pixels vertically to find text density
            vertical_projection = np.sum(binary, axis=0)
            
            # Smooth projection to reduce noise
            kernel_size = max(5, image.width // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Must be odd
            
            # Apply smoothing if array is large enough
            smoothed = np.zeros_like(vertical_projection, dtype=float)
            
            # Manual smoothing with moving average
            half_window = kernel_size // 2
            for i in range(len(vertical_projection)):
                start = max(0, i - half_window)
                end = min(len(vertical_projection), i + half_window + 1)
                smoothed[i] = np.mean(vertical_projection[start:end])
            
            # Normalize for visualization and analysis
            if np.max(smoothed) > 0:
                normalized = smoothed / np.max(smoothed)
            else:
                return 1, [(0, image.width)]
            
            # Calculate average density
            avg_density = np.mean(normalized)
            
            # Look specifically for a valley in the middle (common in two-column books)
            middle = image.width // 2
            middle_region = normalized[middle - image.width//8:middle + image.width//8]
            
            # If there's a significant drop in the middle, it's likely a two-column layout
            if len(middle_region) > 0 and np.min(middle_region) < avg_density * 0.5:
                # Simple two-column detection - split down the middle
                return 2, [(0, middle), (middle, image.width)]
            
            # Check for more complex multi-column layouts
            valleys = []
            min_valley_width = image.width * 0.02  # Min width of a valley (2% of image width)
            min_valley_drop = avg_density * 0.4  # Valley must be at least 40% lower than average
            
            i = 0
            while i < len(normalized):
                if normalized[i] < avg_density - min_valley_drop:
                    valley_start = i
                    # Find where valley ends
                    while i < len(normalized) and normalized[i] < avg_density - min_valley_drop:
                        i += 1
                    valley_end = i
                    
                    # Check if valley is wide enough to be a column separator
                    if valley_end - valley_start >= min_valley_width:
                        valleys.append((valley_start, valley_end))
                else:
                    i += 1
            
            # If we found valid valleys, define column boundaries
            if valleys:
                column_boundaries = []
                prev_boundary = 0
                
                for valley_start, valley_end in valleys:
                    valley_mid = (valley_start + valley_end) // 2
                    column_boundaries.append((prev_boundary, valley_mid))
                    prev_boundary = valley_mid
                
                # Add final column to right edge
                column_boundaries.append((prev_boundary, image.width))
                
                return len(column_boundaries), column_boundaries
                
            # Default to single column if no clear separations found
            return 1, [(0, image.width)]
            
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Book column detection error: {e}")
            # Default to single column on error
            return 1, [(0, image.width)]

    def _extract_multi_column_book_text(self, image, column_boundaries):
        """Extract text from multi-column book page"""
        try:
            width, height = image.size
            column_texts = []
            total_confidence = 0
            confidence_count = 0
            
            # Process each column separately
            for i, (left, right) in enumerate(column_boundaries):
                # Crop to column boundaries
                column_image = image.crop((left, 0, right, height))
                
                # Process with settings optimized for book columns
                config = '--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1'
                
                # Get text
                column_text = pytesseract.image_to_string(column_image, config=config)
                
                # Get confidence data
                data = pytesseract.image_to_data(
                    column_image, 
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    total_confidence += avg_confidence
                    confidence_count += 1
                
                # Check for drop caps in first paragraph (common in books)
                drop_cap_detected = False
                drop_cap_info = self._detect_drop_cap(column_image)
                
                if drop_cap_info['has_drop_cap']:
                    drop_cap_detected = True
                    drop_cap = drop_cap_info['letter']
                    
                    # Fix the first line with the drop cap
                    lines = column_text.split('\n')
                    if lines and lines[0]:
                        # Check if the first word already contains the drop cap
                        if not lines[0].startswith(drop_cap):
                            # Add drop cap to the beginning of the first line
                            if ' ' in lines[0]:
                                first_word, rest = lines[0].split(' ', 1)
                                lines[0] = f"{drop_cap}{first_word} {rest}"
                            else:
                                lines[0] = f"{drop_cap}{lines[0]}"
                        
                        column_text = '\n'.join(lines)
                
                # Clean up the text for books
                cleaned_text = self._clean_book_text(column_text)
                column_texts.append(cleaned_text)
            
            # Join columns with clear separation
            text = "\n\n".join(column_texts)
            
            # Calculate overall confidence
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'word_count': len(text.split()),
                'char_count': len(text),
                'success': True,
                'engine': 'book_multi_column',
                'has_structure': True,
                'columns': len(column_boundaries),
                'drop_cap_detected': drop_cap_detected
            }
            
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Multi-column book extraction error: {e}")
            # Fallback to standard extraction
            return self._fallback_extract_text(image)

    def _extract_single_column_book_text(self, image):
        """Extract text from single-column book page using BeautifulSoup"""
        if not BS4_AVAILABLE:
            # Fallback if BeautifulSoup isn't available
            return self._fallback_extract_text(image)
            
        try:
            # Use HOCR for better structure preservation with drop caps and paragraphs
            config = '--oem 3 --psm 1 -c preserve_interword_spaces=1 -c textord_tablefind_recognize_tables=0'
            hocr_output = pytesseract.image_to_pdf_or_hocr(image, extension='hocr', config=config)
            
            # Convert HOCR to text with structure preservation
            soup = BeautifulSoup(hocr_output, 'html.parser')
            
            # Extract text blocks with their vertical positions
            blocks = []
            
            # Find paragraphs (ocr_par elements)
            for par in soup.find_all('div', class_='ocr_par'):
                # Get bounding box data
                try:
                    bbox_str = par['title'].split('bbox ')[1].split(';')[0]
                    x1, y1, x2, y2 = map(int, bbox_str.split())
                    
                    # Extract all text from this paragraph
                    text = ' '.join([word.getText() for word in par.find_all('span', class_='ocrx_word')])
                    
                    # Store with position data for ordering
                    blocks.append({
                        'text': text,
                        'y1': y1,
                        'x1': x1,
                        'y2': y2,
                        'x2': x2,
                        'height': y2 - y1
                    })
                except Exception:
                    continue
            
            # Sort blocks by vertical position (top to bottom)
            blocks.sort(key=lambda b: b['y1'])
            
            # Group blocks into header, main content, and footer
            header_blocks = []
            content_blocks = []
            footer_blocks = []
            
            height = image.height
            width = image.width
            
            # Extract page number if present (usually at the top or bottom)
            for i, block in enumerate(blocks):
                # Check if it's a page number (short text with digits)
                if (len(block['text']) <= 5 and
                    re.match(r'^\d+$', block['text'].strip()) and
                    (block['y1'] < height * 0.1 or block['y1'] > height * 0.9)):
                    if block['y1'] < height * 0.1:  # Top of page
                        header_blocks.append(block)
                    else:  # Bottom of page
                        footer_blocks.append(block)
                    blocks[i] = None  # Mark for removal
            
            # Remove processed blocks
            blocks = [b for b in blocks if b is not None]
            
            # Check for title or chapter header
            if blocks and blocks[0]['y1'] < height * 0.2:
                # If first block is a header (all caps or contains "CHAPTER", "SECTION", etc.)
                if (blocks[0]['text'].isupper() or 
                    any(term in blocks[0]['text'].upper() for term in ["CHAPTER", "SECTION", "RULE"])):
                    header_blocks.append(blocks[0])
                    blocks = blocks[1:]
            
            # Check for next section/chapter header at bottom
            if blocks and blocks[-1]['y1'] > height * 0.8:
                if (blocks[-1]['text'].isupper() or 
                    any(term in blocks[-1]['text'].upper() for term in ["CHAPTER", "SECTION", "RULE"])):
                    footer_blocks.append(blocks[-1])
                    blocks = blocks[:-1]
                
            # The rest is content
            content_blocks = blocks
            
            # Build the final text with appropriate section separation
            full_text_parts = []
            
            # Add header content first
            if header_blocks:
                for block in header_blocks:
                    full_text_parts.append(block['text'])
                    
            # Add a separator
            if header_blocks and content_blocks:
                full_text_parts.append("")
                
            # Add main content
            for block in content_blocks:
                full_text_parts.append(block['text'])
                
            # Add a separator before footer
            if content_blocks and footer_blocks:
                full_text_parts.append("")
                
            # Add footer content
            if footer_blocks:
                for block in footer_blocks:
                    full_text_parts.append(block['text'])
                    
            # Join everything with proper paragraph breaks
            full_text = "\n\n".join(full_text_parts)
            
            # Clean up common book text OCR issues
            full_text = self._clean_book_text(full_text)
            
            # Get confidence data from standard OCR for reporting
            data = pytesseract.image_to_data(
                image,
                config='--oem 3 --psm 1',
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'word_count': len(full_text.split()),
                'char_count': len(full_text),
                'success': True,
                'engine': 'book_hocr',
                'has_structure': True
            }
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Book text extraction error: {e}")
            # Fallback to standard extraction
            return self._fallback_extract_text(image)

    def _detect_drop_cap(self, image):
        """
        Advanced drop cap detection for book pages
        """
        try:
            # Convert to numpy for processing
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array.copy()
            
            # Convert to binary for contour detection
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Get image dimensions
            height, width = gray.shape
            
            # Analyze potential drop caps
            drop_cap_info = {
                'has_drop_cap': False,
                'letter': '',
                'coords': (0, 0, 0, 0),
                'size_ratio': 0
            }
            
            # Focus on the top few contours
            for i, contour in enumerate(contours[:10]):
                x, y, w, h = cv2.boundingRect(contour)
                
                # A drop cap has distinctive characteristics:
                # 1. Typically in the first 1/3 of the page
                # 2. Taller than average text height (at least 2.5x)
                # 3. Not too wide (not an illustration or border)
                # 4. Usually in the left side of the page
                
                if (y < height/3 and                     # Near top of page
                    h > 35 and                           # Tall enough
                    h/w > 1 and h/w < 3 and              # Height/width ratio appropriate for a letter
                    x < width/3 and                      # Positioned on left side
                    w < width/4):                        # Not too wide
                    
                    # Extract just this letter for OCR
                    letter_region = gray[y:y+h, x:x+w]
                    
                    # Ensure the region is valid
                    if letter_region.size == 0:
                        continue
                        
                    # Scale up for better OCR
                    try:
                        letter_img = cv2.resize(letter_region, (w*4, h*4))
                    except Exception:
                        continue
                    
                    # Binarize for cleaner OCR
                    _, letter_binary = cv2.threshold(letter_img, 180, 255, cv2.THRESH_BINARY)
                    
                    # OCR just this letter with settings optimized for single characters
                    config = '--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    letter_text = pytesseract.image_to_string(letter_binary, config=config).strip()
                    
                    # If we got a single letter result
                    if len(letter_text) == 1 and letter_text.isalpha():
                        drop_cap_info = {
                            'has_drop_cap': True,
                            'letter': letter_text,
                            'coords': (x, y, w, h),
                            'size_ratio': h / 15  # Approximate ratio to regular text
                        }
                        break
            
            return drop_cap_info
            
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Drop cap detection error: {e}")
            return {'has_drop_cap': False, 'letter': '', 'coords': (0, 0, 0, 0), 'size_ratio': 0}

    def _clean_book_text(self, text):
        """Clean up common OCR issues in book text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)        # Multiple spaces to single
        text = re.sub(r' +\n', '\n', text)      # Remove spaces before line breaks
        text = re.sub(r'\n +', '\n', text)      # Remove spaces after line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive line breaks
        
        # Fix common OCR errors in books
        replacements = {
            "|": "I",               # Vertical bar to I
            "l.": "i.",             # lowercase L with period to i.
            "rnay": "may",          # rn often misrecognized as m
            "rny": "my",            # rn often misrecognized as m
            "tbe": "the",           # t and h sometimes merged
            "tbat": "that",         # t and h sometimes merged
            "arid": "and",          # n sometimes misrecognized as ri
            "aud": "and",           # n sometimes misrecognized as u
            "modem": "modern",      # rn often misrecognized as m
            "lime": "time",         # t sometimes misrecognized as l
            "wc": "we",             # e sometimes misrecognized as c
            "bo": "be",             # e sometimes misrecognized as o
            "thc": "the",           # e sometimes misrecognized as c
            "scction": "section",   # e sometimes misrecognized as c
        }
        
        # Apply replacements
        for error, correction in replacements.items():
            text = text.replace(error, correction)
        
        # Fix spacing around punctuation
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        
        # Fix quotes and special characters
        text = text.replace("''", "\"")
        text = text.replace("``", "\"")
        text = text.replace(",,", "\"")
        
        # Fix spacing around quotes
        text = re.sub(r'"\s+', '"', text)       # No space after opening quote
        text = re.sub(r'\s+"', '"', text)       # No space before closing quote
        
        # Fix broken sentence spacing
        text = text.replace(".\n", ". \n")
        
        # Fix hyphenation at line breaks (common in printed books)
        text = re.sub(r'([a-z])-\s+([a-z])', r'\1\2', text)
        
        return text

    def _error_result(self, error_message: str) -> Dict:
        """Create an error result"""
        return {
            'text': '',
            'confidence': 0,
            'word_count': 0,
            'char_count': 0,
            'success': False,
            'error': error_message
        }