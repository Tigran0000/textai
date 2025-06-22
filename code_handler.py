# src/ocr/handlers/code_handler.py
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import pytesseract
import re
import time
from PIL import Image
from bs4 import BeautifulSoup

from .base_handler import DocumentHandler

class CodeHandler(DocumentHandler):
    """Handler for programming code with indentation and syntax preservation"""
    
    def can_handle(self, image: Image.Image) -> bool:
        """Check if image is likely code based on visual characteristics"""
        try:
            # Check aspect ratio (code snippets are usually wider than tall)
            width, height = image.size
            aspect_ratio = width / height
            
            # Most code snippets have wider aspect ratios 
            if aspect_ratio < 1.2:
                return False
                
            # Convert to grayscale for analysis
            img_array = np.array(image.convert('L'))
            
            # Look for indentation patterns
            if TESSERACT_AVAILABLE:
                # Get quick sample with simple config
                text = pytesseract.image_to_string(image, config='--psm 6')
                
                # Check for code indicators
                code_indicators = [
                    # Common code patterns
                    r'def\s+\w+\s*\(',            # Python function def
                    r'function\s+\w+\s*\(',       # JavaScript function
                    r'(public|private|protected)', # Java/C# access modifiers  
                    r'class\s+\w+',               # Class definition
                    r'if\s*\(.+\)\s*[{:]',       # if statement
                    r'for\s*\(.+\)\s*[{:]',      # for loop
                    r'while\s*\(.+\)\s*[{:]',    # while loop
                    r'import\s+[\w.]+',          # import statement
                    r'#include',                  # C/C++ include
                    r'<[/\w]+>',                  # HTML tags
                    r'var\s+\w+\s*=',            # variable declaration
                    r'return\s+.+;',             # return statement
                    
                    # Check for indentation patterns
                    r'^\s{2,}\w+',               # Indented lines
                    r'[{}\[\]();]'               # Common code punctuation
                ]
                
                # Check if text matches any code indicators
                for pattern in code_indicators:
                    if re.search(pattern, text, re.MULTILINE):
                        return True
            
            # Visual analysis for code-like patterns
            # 1. Look for consistent indentation
            edges = cv2.Canny(img_array, 100, 200)
            # Count edge pixels in left margin
            left_margin = edges[:, :int(width * 0.1)]
            left_margin_density = np.sum(left_margin > 0) / left_margin.size
            
            # 2. Look for consistent line heights (code usually has uniform line spacing)
            horizontal_projection = np.sum(edges, axis=1)
            line_positions = []
            in_line = False
            for i, val in enumerate(horizontal_projection):
                if val > width * 0.1 and not in_line:
                    line_positions.append(i)
                    in_line = True
                elif val < width * 0.05 and in_line:
                    in_line = False
            
            # Check line spacing consistency
            if len(line_positions) > 2:
                line_spacings = [line_positions[i+1] - line_positions[i] for i in range(len(line_positions)-1)]
                spacing_std = np.std(line_spacings)
                spacing_mean = np.mean(line_spacings)
                
                # Consistent spacing + low left margin edge density = likely code
                if spacing_std / spacing_mean < 0.2 and left_margin_density < 0.05:
                    return True
            
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"Code detection error: {e}")
            return False
    
    def _perform_extraction(self, image: Image.Image, preprocess: bool) -> Dict:
        """Extract code from image with indentation preservation"""
        try:
            # Use specialized code preprocessing if requested
            if preprocess and 'image' in self.processors:
                processed_image = self.processors['image'].preprocess_code(image.copy())
            else:
                processed_image = image.copy()
            
            # Save debug image if enabled
            if hasattr(self, 'debug_mode') and self.debug_mode and hasattr(self, 'save_debug_images') and self.save_debug_images:
                debug_path = os.path.join(self.debug_dir, f"code_processed_{int(time.time())}.png")
                processed_image.save(debug_path)
            
            # Extract code using HOCR format which preserves spatial information
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Special config for preserving monospaced text and whitespace
                config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
                
                # Extract as HOCR format to better preserve spacing
                hocr = pytesseract.image_to_pdf_or_hocr(
                    processed_image,
                    extension='hocr',
                    config=config
                )
                
                # Convert HOCR to string
                hocr_text = hocr.decode('utf-8')
                
                # Extract lines with proper indentation from HOCR
                code_text = self._extract_code_from_hocr(hocr_text)
                
                # Get confidence data
                data = pytesseract.image_to_data(
                    processed_image, 
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Clean up code text
                if 'text' in self.processors:
                    code_text = self.processors['text'].clean_code_text(code_text)
                
                # Detect programming language
                language = self._detect_programming_language(code_text)
                
                return {
                    'text': code_text,
                    'confidence': confidence,
                    'word_count': len(code_text.split()),
                    'char_count': len(code_text),
                    'success': True,
                    'engine': 'code_extraction',
                    'programming_language': language
                }
            else:
                # Fallback to raw extraction if HOCR doesn't work
                return self._extract_with_raw_lines(processed_image)
                
        except Exception as e:
            return self._error_result(f"Code extraction error: {str(e)}")
    
    def _extract_code_from_hocr(self, hocr_text: str) -> str:
        """Extract code with proper indentation from HOCR data"""
        try:
            # Parse the HOCR with BeautifulSoup
            soup = BeautifulSoup(hocr_text, 'html.parser')
            
            # Extract lines with their positions
            lines = []
            
            # Process each line in the HOCR
            for line_elem in soup.find_all('span', class_='ocr_line'):
                # Get line bounding box
                bbox_str = line_elem.get('title', '')
                bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', bbox_str)
                
                if not bbox_match:
                    continue
                    
                x1, y1, x2, y2 = [int(coord) for coord in bbox_match.groups()]
                
                # Get line text content
                line_text = []
                for word_elem in line_elem.find_all('span', class_='ocrx_word'):
                    word = word_elem.get_text().strip()
                    if word:
                        line_text.append(word)
                
                # Skip empty lines
                if not line_text:
                    continue
                
                # Store line with left position (x1) for indentation calculation
                lines.append((x1, ' '.join(line_text)))
            
            # Find minimum left position (to calculate relative indentation)
            if not lines:
                return ""
                
            min_x = min(line[0] for line in lines)
            
            # Build code with proper indentation
            code_lines = []
            for x, text in lines:
                # Calculate indentation level (rounded to nearest 4 spaces)
                indent_pixels = x - min_x
                indent_level = round(indent_pixels / 8)
                indent_spaces = indent_level * 4
                
                # Add indented line
                code_lines.append(' ' * indent_spaces + text)
            
            # Join lines to form the final code text
            return '\n'.join(code_lines)
            
        except Exception as e:
            # Return empty string if extraction fails
            print(f"HOCR code extraction error: {e}")
            return ""
    
    def _extract_with_raw_lines(self, image: Image.Image) -> Dict:
        """Fallback method using direct line-by-line extraction"""
        try:
            # Use specialized config for code
            config = '--psm 6 -c preserve_interword_spaces=1'
            
            # Get raw text
            raw_text = pytesseract.image_to_string(image, config=config)
            
            # Process and structurize
            lines = raw_text.split('\n')
            processed_lines = []
            
            # Keep track of indentation pattern
            indent_spaces = {}
            
            # Process each line
            for line in lines:
                # Skip empty lines but preserve them in output
                if not line.strip():
                    processed_lines.append('')
                    continue
                
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                
                # Register this indentation level if new
                if leading_spaces not in indent_spaces:
                    indent_spaces[leading_spaces] = len(indent_spaces) * 4
                
                # Normalize indentation to 4-space increments
                normalized_indent = indent_spaces[leading_spaces]
                processed_lines.append(' ' * normalized_indent + line.lstrip())
            
            # Join lines
            code_text = '\n'.join(processed_lines)
            
            # Get confidence
            data = pytesseract.image_to_data(
                image, 
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Detect programming language
            language = self._detect_programming_language(code_text)
            
            return {
                'text': code_text,
                'confidence': confidence,
                'word_count': len(code_text.split()),
                'char_count': len(code_text),
                'success': True,
                'engine': 'raw_line_code_extraction',
                'programming_language': language
            }
            
        except Exception as e:
            return self._error_result(f"Raw code extraction error: {str(e)}")
    
    def _detect_programming_language(self, code_text: str) -> str:
        """Detect the programming language from code text"""
        # Skip if too little text
        if len(code_text.strip()) < 10:
            return "Unknown"
        
        # Dictionary of language indicators (patterns -> language)
        language_patterns = {
            # Python
            r'import\s+[a-zA-Z0-9_.]+': 'Python',
            r'from\s+[a-zA-Z0-9_.]+\s+import': 'Python',
            r'def\s+[a-zA-Z0-9_]+\s*\(': 'Python',
            r'class\s+[A-Z][a-zA-Z0-9_]*\s*:': 'Python',
            r'if\s+.+\s*:': 'Python',
            
            # JavaScript
            r'function\s+[a-zA-Z0-9_]+\s*\(': 'JavaScript',
            r'const\s+[a-zA-Z0-9_]+\s*=': 'JavaScript',
            r'let\s+[a-zA-Z0-9_]+\s*=': 'JavaScript',
            r'var\s+[a-zA-Z0-9_]+\s*=': 'JavaScript',
            r'document\.': 'JavaScript',
            r'window\.': 'JavaScript',
            r'=>\s*{': 'JavaScript',  # Arrow function
            
            # HTML/XML
            r'<\/[a-zA-Z][a-zA-Z0-9]*>': 'HTML/XML',
            r'<[a-zA-Z][a-zA-Z0-9]*\s+[^>]*>': 'HTML/XML',
            r'<!DOCTYPE': 'HTML',
            r'<html': 'HTML',
            
            # CSS
            r'[a-zA-Z0-9_.-]+\s*{\s*[a-zA-Z0-9_-]+:': 'CSS',
            r'@media': 'CSS',
            r'@import': 'CSS',
            
            # C/C++
            r'#include\s*<[a-zA-Z0-9_.]+>': 'C/C++',
            r'int\s+main\s*\(': 'C/C++',
            r'std::': 'C++',
            r'#define': 'C/C++',
            r'void\s+[a-zA-Z0-9_]+\s*\(': 'C/C++',
            
            # Java
            r'public\s+class\s+[A-Z][a-zA-Z0-9_]*': 'Java',
            r'private\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+': 'Java',
            r'protected\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+': 'Java',
            r'import\s+java\.': 'Java',
            r'System\.out\.print': 'Java',
            
            # C#
            r'using\s+System;': 'C#',
            r'namespace\s+[A-Z][a-zA-Z0-9_.]*': 'C#',
            r'Console\.Write': 'C#',
            
            # PHP
            r'<\?php': 'PHP',
            r'\$[a-zA-Z0-9_]+\s*=': 'PHP',
            r'echo\s+': 'PHP',
            
            # Shell script
            r'^#!/bin/(ba)?sh': 'Shell',
            r'^#!/usr/bin/(ba)?sh': 'Shell',
            r'export\s+[A-Z0-9_]+\s*=': 'Shell',
            
            # SQL
            r'SELECT\s+.+\s+FROM': 'SQL',
            r'INSERT\s+INTO': 'SQL',
            r'UPDATE\s+.+\s+SET': 'SQL',
            r'CREATE\s+TABLE': 'SQL',
            
            # Ruby
            r'def\s+[a-z][a-zA-Z0-9_]*\s*(\(.+\))?\s*$': 'Ruby',
            r'require\s+[\'"]': 'Ruby',
            
            # Golang
            r'package\s+[a-zA-Z][a-zA-Z0-9_]*': 'Go',
            r'func\s+[a-zA-Z][a-zA-Z0-9_]*\s*\(': 'Go',
            r'import\s+\(': 'Go',
        }
        
        # Check for patterns in the code text
        for pattern, language in language_patterns.items():
            if re.search(pattern, code_text, re.MULTILINE | re.IGNORECASE):
                return language
        
        # If no specific patterns matched, make a guess based on syntax characteristics
        if '{' in code_text and '}' in code_text:
            if '=>' in code_text:
                return "JavaScript"
            elif ';' in code_text:
                return "C-like"
            else:
                return "JSON/JavaScript"
        elif ':' in code_text and 'def ' in code_text:
            return "Python"
        
        # Default to unknown if no patterns matched
        return "Unknown"
    
    def _fix_code_syntax(self, code_text: str) -> str:
        """Fix common code syntax issues from OCR"""
        # Fix issues with brackets and parentheses
        code_text = re.sub(r'\( ', r'(', code_text)
        code_text = re.sub(r' \)', r')', code_text)
        code_text = re.sub(r'\{ ', r'{', code_text)
        code_text = re.sub(r' \}', r'}', code_text)
        code_text = re.sub(r'\[ ', r'[', code_text)
        code_text = re.sub(r' \]', r']', code_text)
        
        # Fix issues with operators
        code_text = re.sub(r' = ', r'=', code_text)
        code_text = re.sub(r'= ', r'=', code_text)
        code_text = re.sub(r' =', r'=', code_text)
        code_text = re.sub(r' \+ ', r'+', code_text)
        code_text = re.sub(r' - ', r'-', code_text)
        code_text = re.sub(r' \* ', r'*', code_text)
        code_text = re.sub(r' / ', r'/', code_text)
        
        # Fix common OCR errors in code
        code_text = code_text.replace('O', '0')  # Letter O to number 0
        code_text = code_text.replace('l', '1')  # Letter l to number 1
        code_text = code_text.replace('S', '5')  # Letter S to number 5
        
        # Fix indentation
        lines = code_text.split('\n')
        fixed_lines = []
        for line in lines:
            # Count leading spaces
            spaces = len(line) - len(line.lstrip())
            # Round to nearest 4
            spaces = (spaces // 4) * 4
            fixed_lines.append(' ' * spaces + line.lstrip())
        
        return '\n'.join(fixed_lines)