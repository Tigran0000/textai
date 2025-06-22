# src/ocr/handlers/form_handler.py
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import pytesseract
import re
import json
import time
from PIL import Image
from bs4 import BeautifulSoup

from .base_handler import DocumentHandler

class FormHandler(DocumentHandler):
    """Handler for forms with field detection and label-value extraction"""
    
    def can_handle(self, image: Image.Image) -> bool:
        """Check if image contains a form based on visual characteristics"""
        try:
            # Convert image to numpy array for analysis
            img_array = np.array(image.convert('L'))
            height, width = img_array.shape
            
            # Apply threshold for better line detection
            _, binary = cv2.threshold(img_array, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Check for forms using multiple criteria
            form_indicators = 0
            
            # 1. Check for horizontal lines (common in forms)
            edges = cv2.Canny(binary, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 
                1, 
                np.pi/180, 
                threshold=50, 
                minLineLength=width*0.2, 
                maxLineGap=20
            )
            
            horizontal_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is horizontal
                    if abs(y2 - y1) < 10:
                        horizontal_lines += 1
            
            if horizontal_lines >= 3:
                form_indicators += 1
                
            # 2. Check for empty boxes/rectangles (checkbox or input fields)
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours (likely form fields)
            rectangles = 0
            for contour in contours:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if the contour is rectangular
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Typical input field dimensions
                    if 1 < aspect_ratio < 10 and 15 < w < width/2 and 15 < h < height/10:
                        rectangles += 1
            
            if rectangles >= 3:  # Forms typically have multiple fields
                form_indicators += 1
                
            # 3. Check for label-field patterns using OCR
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Create a small version for quick analysis
                small_img = image.copy()
                small_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                # Get text with layout preservation
                text = pytesseract.image_to_string(small_img, config='--psm 6').lower()
                
                # Look for common form patterns
                form_patterns = [
                    r'name\s*:', r'address\s*:', r'date\s*:', r'phone\s*:',
                    r'email\s*:', r'signature\s*:', r'check\s+box',
                    r'please\s+fill', r'required\s+field', r'optional',
                    r'select\s+one', r'form\s*\d*', r'application'
                ]
                
                matches = sum(1 for pattern in form_patterns if re.search(pattern, text))
                if matches >= 2:  # Multiple form indicators found in text
                    form_indicators += 1
                
                # Special case: Form with fields indicated by underscores or dots
                if re.search(r'[_\.]{3,}', text) or re.search(r'□|☐|☑|☒', text):
                    form_indicators += 1
            
            # 4. Check for field-value distribution (label on left, value on right)
            # This is common in many forms
            hocr = pytesseract.image_to_pdf_or_hocr(
                image,
                extension='hocr',
                config='--psm 1'
            )
            
            soup = BeautifulSoup(hocr.decode('utf-8'), 'html.parser')
            
            # Look for lines with text only on the left side (labels) 
            # followed by empty space or underscores (entry fields)
            left_text_only = 0
            
            for line in soup.find_all('span', class_='ocr_line'):
                words = line.find_all('span', class_='ocrx_word')
                if words:
                    # Get the rightmost extent of text in this line
                    text_extent = 0
                    for word in words:
                        bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', word.get('title', ''))
                        if bbox_match:
                            x2 = int(bbox_match.group(3))
                            text_extent = max(text_extent, x2)
                    
                    # If text occupies less than 60% of the width, might be a label with field
                    if text_extent < width * 0.6:
                        left_text_only += 1
            
            if left_text_only >= 3:  # Multiple potential form fields
                form_indicators += 1
            
            # Consider it a form if we have enough indicators
            return form_indicators >= 2
            
        except Exception as e:
            if self.debug_mode:
                print(f"Form detection error: {e}")
            return False
    
    def _perform_extraction(self, image: Image.Image, preprocess: bool) -> Dict:
        """Extract structured data from forms"""
        try:
            # Use specialized form preprocessing if requested
            if preprocess and 'image' in self.processors:
                processed_image = self.processors['image'].preprocess_form(image.copy())
            else:
                processed_image = image.copy()
            
            # Save debug image if enabled
            if hasattr(self, 'debug_mode') and self.debug_mode and hasattr(self, 'save_debug_images') and self.save_debug_images:
                debug_path = os.path.join(self.debug_dir, f"form_processed_{int(time.time())}.png")
                processed_image.save(debug_path)
            
            # Try multiple approaches for form extraction
            
            # 1. Use layout analyzer for structured form extraction if available
            if 'layout_analyzer' in self.processors:
                try:
                    layout_analyzer = self.processors['layout_analyzer']
                    form_data = layout_analyzer.extract_form_structure(np.array(processed_image))
                    
                    if form_data and 'fields' in form_data and len(form_data['fields']) > 0:
                        # Convert form data to text representation
                        form_text = self._convert_form_to_text(form_data['fields'])
                        
                        # Format as JSON as well
                        json_form = json.dumps(form_data.get('fields', {}), indent=2)
                        
                        return {
                            'text': form_text,
                            'json_form': json_form,
                            'confidence': form_data.get('confidence', 70),
                            'field_count': len(form_data.get('fields', {})),
                            'word_count': len(form_text.split()),
                            'char_count': len(form_text),
                            'success': True,
                            'engine': 'layout_analysis_form'
                        }
                except Exception as layout_err:
                    if self.debug_mode:
                        print(f"Layout analysis form extraction error: {layout_err}")
            
            # 2. Use custom field detection algorithm
            fields = self._detect_form_fields(processed_image)
            
            if fields:
                # Convert fields to text format
                form_text = self._convert_form_to_text(fields)
                
                return {
                    'text': form_text,
                    'json_form': json.dumps(fields, indent=2),
                    'fields': fields,
                    'confidence': 75,  # Reasonable default for our custom detection
                    'field_count': len(fields),
                    'word_count': len(form_text.split()),
                    'char_count': len(form_text),
                    'success': True,
                    'engine': 'field_detection_form'
                }
            
            # 3. Fallback to tesseract with form-specific settings
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Use special config for forms
                config = '--oem 3 --psm 4 -c preserve_interword_spaces=1'
                
                # Get HOCR for better structure preservation
                hocr = pytesseract.image_to_pdf_or_hocr(
                    processed_image,
                    extension='hocr',
                    config=config
                )
                
                # Extract text with structure
                text = self._extract_form_text_from_hocr(hocr.decode('utf-8'))
                
                # Extract fields from text
                extracted_fields = self._extract_fields_from_text(text)
                
                # Get confidence data
                data = pytesseract.image_to_data(
                    processed_image, 
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 0
                
                return {
                    'text': text,
                    'fields': extracted_fields,
                    'json_form': json.dumps(extracted_fields, indent=2),
                    'confidence': confidence,
                    'field_count': len(extracted_fields),
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'success': True,
                    'engine': 'tesseract_form'
                }
            else:
                # Fallback to basic extraction if Tesseract isn't available
                return self._extract_with_basic_method(processed_image)
                
        except Exception as e:
            return self._error_result(f"Form extraction error: {str(e)}")
    
    def _detect_form_fields(self, image: Image.Image) -> Dict[str, str]:
        """Detect form fields and their values using visual and textual cues"""
        fields = {}
        
        # Convert to numpy array
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        # Apply threshold to isolate text and form elements
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Get HOCR data for text positioning
        hocr = pytesseract.image_to_pdf_or_hocr(
            image,
            extension='hocr',
            config='--psm 1'
        )
        
        # Parse HOCR to extract text with positions
        soup = BeautifulSoup(hocr.decode('utf-8'), 'html.parser')
        
        # Extract lines of text with their bounding boxes
        lines = []
        for line_elem in soup.find_all('span', class_='ocr_line'):
            # Get line bounding box
            bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line_elem.get('title', ''))
            if not bbox_match:
                continue
                
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            
            # Get line text
            line_text = line_elem.get_text().strip()
            if not line_text:
                continue
                
            lines.append({
                'text': line_text,
                'bbox': (x1, y1, x2, y2)
            })
        
        # Look for typical form field patterns
        field_patterns = [
            r'([A-Za-z][A-Za-z\s]+[A-Za-z]):\s*(.*)',     # Label: Value
            r'([A-Za-z][A-Za-z\s]+[A-Za-z])\s*=\s*(.*)',  # Label = Value
            r'([A-Za-z][A-Za-z\s]+[A-Za-z])\s+-\s*(.*)'   # Label - Value
        ]
        
        # Check lines for field patterns
        for line in lines:
            text = line['text']
            for pattern in field_patterns:
                match = re.match(pattern, text)
                if match:
                    field_name = match.group(1).strip()
                    field_value = match.group(2).strip()
                    
                    if field_name:
                        fields[field_name] = field_value
                    break
        
        # Look for checkbox selections 
        # (common in forms: ☑ Option or ☒ Option)
        for line in lines:
            text = line['text']
            checkbox_match = re.search(r'[☑☒✓✔][ \t]*([A-Za-z][A-Za-z\s]+[A-Za-z])', text)
            if checkbox_match:
                option_text = checkbox_match.group(1).strip()
                fields[f"Selected: {option_text}"] = "Yes"
        
        # Also look for separate label-value pairs
        # (where the label is on one line, value on the next)
        for i in range(len(lines) - 1):
            current_line = lines[i]['text']
            next_line = lines[i+1]['text']
            
            label_match = re.match(r'([A-Za-z][A-Za-z\s]+[A-Za-z]):?\s*$', current_line)
            if label_match and next_line and not re.match(r'([A-Za-z][A-Za-z\s]+[A-Za-z]):?\s*$', next_line):
                field_name = label_match.group(1).strip()
                fields[field_name] = next_line.strip()
        
        return fields
    
    def _extract_form_text_from_hocr(self, hocr_text: str) -> str:
        """Extract form text with structure preservation from HOCR data"""
        try:
            # Parse the HOCR with BeautifulSoup
            soup = BeautifulSoup(hocr_text, 'html.parser')
            
            # Extract paragraphs with their positions
            paragraphs = []
            
            # Process each paragraph in the HOCR
            for par in soup.find_all('p', class_='ocr_par'):
                # Get paragraph bounding box
                bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', par.get('title', ''))
                if not bbox_match:
                    continue
                    
                x1, y1, x2, y2 = map(int, bbox_match.groups())
                
                # Get paragraph text
                par_text = par.get_text().strip()
                
                # Skip empty paragraphs
                if not par_text:
                    continue
                
                # Store paragraph with positioning info
                paragraphs.append({
                    'text': par_text,
                    'bbox': (x1, y1, x2, y2)
                })
            
            # Sort paragraphs top to bottom
            paragraphs.sort(key=lambda p: p['bbox'][1])
            
            # Build form text
            lines = []
            
            # For each paragraph, keep the text
            for par in paragraphs:
                lines.append(par['text'])
            
            # Join paragraphs with blank lines between them
            return '\n\n'.join(lines)
            
        except Exception as e:
            # Return empty string if extraction fails
            print(f"HOCR form extraction error: {e}")
            return ""
    
    def _extract_fields_from_text(self, text: str) -> Dict[str, str]:
        """Extract form fields from structured text"""
        fields = {}
        
        # Look for common field patterns
        field_patterns = [
            r'([A-Za-z][A-Za-z\s]+[A-Za-z]):\s*(.*)',     # Label: Value
            r'([A-Za-z][A-Za-z\s]+[A-Za-z])\s*=\s*(.*)',  # Label = Value
            r'([A-Za-z][A-Za-z\s]+[A-Za-z])\s+-\s*(.*)'   # Label - Value
        ]
        
        # Process each line
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # Check for field patterns in this line
            field_match = None
            for pattern in field_patterns:
                match = re.match(pattern, line)
                if match:
                    field_match = match
                    break
            
            if field_match:
                # Found a field pattern
                field_name = field_match.group(1).strip()
                field_value = field_match.group(2).strip()
                fields[field_name] = field_value
                i += 1
            else:
                # Check if this might be a field label followed by value on next line
                label_match = re.match(r'([A-Za-z][A-Za-z\s]+[A-Za-z]):?\s*$', line)
                if label_match and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    field_name = label_match.group(1).strip()
                    fields[field_name] = next_line
                    i += 2
                else:
                    # Not a field, continue
                    i += 1
        
        # Check for checkbox/radio button patterns
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            checkbox_matches = [
                re.search(r'[☑☒✓✔][ \t]*([A-Za-z][A-Za-z\s]+[A-Za-z])', line),
                re.search(r'\[[xX]\][ \t]*([A-Za-z][A-Za-z\s]+[A-Za-z])', line)
            ]
            
            for match in checkbox_matches:
                if match:
                    option_text = match.group(1).strip()
                    fields[f"Selected: {option_text}"] = "Yes"
        
        return fields
    
    def _extract_with_basic_method(self, image: Image.Image) -> Dict:
        """Fallback method when specialized approaches fail"""
        try:
            # Get raw text
            text = pytesseract.image_to_string(image)
            
            # Try to extract fields
            fields = self._extract_fields_from_text(text)
            
            # Format text as form
            if fields:
                form_text = self._convert_form_to_text(fields)
            else:
                # If no fields detected, return raw text
                form_text = text
            
            return {
                'text': form_text,
                'fields': fields,
                'json_form': json.dumps(fields, indent=2) if fields else "{}",
                'confidence': 60,  # Default confidence for basic method
                'field_count': len(fields),
                'word_count': len(form_text.split()),
                'char_count': len(form_text),
                'success': True,
                'engine': 'basic_form_extraction'
            }
            
        except Exception as e:
            return self._error_result(f"Basic form extraction error: {str(e)}")
    
    def _convert_form_to_text(self, fields: Dict[str, str]) -> str:
        """Convert form fields to text representation"""
        if not fields:
            return ""
        
        lines = []
        
        # Add each field as "Label: Value"
        for field_name, field_value in fields.items():
            lines.append(f"{field_name}: {field_value}")
        
        return '\n'.join(lines)
    
    def _error_result(self, message: str) -> Dict:
        """Create a standardized error result"""
        return {
            'text': '',
            'fields': {},
            'json_form': '{}',
            'confidence': 0,
            'field_count': 0,
            'word_count': 0,
            'char_count': 0,
            'success': False,
            'error': message,
            'engine': 'form_extraction_failed'
        }