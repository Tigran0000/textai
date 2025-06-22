"""
EyeShot AI - Enhanced Structure-Preserving PDF Handler
Maintains document structure while improving text extraction quality
Author: Tigran0000
Last updated: 2025-06-20 14:33:01 UTC
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image
import pytesseract
import numpy as np

from .base_handler import DocumentHandler

class PDFHandler(DocumentHandler):
    """Enhanced PDF handler with advanced structure preservation and simple bullet normalization"""
    
    def __init__(self):
        super().__init__()
        self.document_type = "pdf"
        
        # Structure detection parameters
        self.line_tolerance = 12  # Pixels tolerance for same line
        self.paragraph_gap_threshold = 20  # Pixels for paragraph separation
        self.section_gap_threshold = 35  # Pixels for section separation
        
        # Simple bullet point mappings - OCR misreads to proper bullets
        self.bullet_fixes = {
            'e ': '• ',
            '« ': '• ',
            '® ': '• ',
            '© ': '• ',
            '¢ ': '• ',
            'o ': '• ',
            '° ': '• ',
            '* ': '• ',
            '- ': '• ',
        }
        
    def extract_text(self, image, preprocess=True) -> Dict:
        """Extract text with enhanced structure preservation"""
        try:
            # Preprocess image for better OCR
            if preprocess and hasattr(self, 'processors') and self.processors.get('image'):
                processed_image = self.processors['image'].preprocess_for_ocr(image.copy())
            else:
                processed_image = image.copy()
            
            # Multi-level OCR extraction
            text = self._enhanced_structure_extraction(processed_image)
            
            # Apply simple bullet normalization
            text = self._normalize_bullets_simple(text)
            
            # Apply structure-aware formatting
            text = self._apply_structure_formatting(text)
            
            # Calculate extraction metrics
            words = [w for w in text.split() if w.strip()]
            
            # Estimate confidence based on text coherence
            confidence = self._estimate_extraction_confidence(text)
            
            return {
                'text': text,
                'confidence': confidence,
                'word_count': len(words),
                'char_count': len(text),
                'success': True,
                'timestamp': '2025-06-20 14:33:01',
                'processor': 'enhanced_structure_with_bullets',
                'user': 'Tigran0000'
            }
            
        except Exception as e:
            return {
                'text': '', 
                'success': False, 
                'error': str(e),
                'timestamp': '2025-06-20 14:33:01',
                'user': 'Tigran0000'
            }
    
    def _enhanced_structure_extraction(self, image):
        """Enhanced extraction with structure analysis"""
        
        # Get detailed OCR data with word-level positioning
        try:
            data = pytesseract.image_to_data(
                image,
                config='--oem 3 --psm 1 -c preserve_interword_spaces=1',
                output_type=pytesseract.Output.DICT
            )
        except Exception:
            # Fallback to simple string extraction
            return pytesseract.image_to_string(image, config='--oem 3 --psm 6')
        
        # Analyze document structure
        structure_info = self._analyze_document_structure(data)
        
        # Build text preserving detected structure
        return self._build_structure_aware_text(data, structure_info)
    
    def _analyze_document_structure(self, data):
        """Analyze document structure from OCR data"""
        
        if not data or 'text' not in data:
            return {'type': 'simple', 'elements': []}
        
        # Extract valid words with positions
        elements = []
        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if conf > 20 and text:  # Lower threshold for structure analysis
                elements.append({
                    'text': text,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'conf': conf,
                    'block': data['block_num'][i],
                    'paragraph': data['par_num'][i],
                    'line': data['line_num'][i]
                })
        
        # Sort elements by reading order
        elements.sort(key=lambda e: (e['y'], e['x']))
        
        # Detect structure patterns
        structure_type = self._detect_structure_type(elements)
        
        return {
            'type': structure_type,
            'elements': elements,
            'has_lists': self._detect_list_patterns(elements),
            'has_headings': self._detect_heading_patterns(elements),
            'has_tables': self._detect_table_patterns(elements),
            'column_count': self._estimate_column_count(elements)
        }
    
    def _detect_structure_type(self, elements):
        """Detect overall document structure type"""
        
        if not elements:
            return 'simple'
        
        # Analyze x-positions for column detection
        x_positions = [e['x'] for e in elements]
        x_variance = np.var(x_positions) if x_positions else 0
        
        # Analyze y-gaps for section detection
        y_positions = [e['y'] for e in elements]
        y_gaps = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        large_gaps = [gap for gap in y_gaps if gap > self.section_gap_threshold]
        
        if len(set([e['x'] for e in elements[:10]])) > 5:  # Multiple start positions
            if x_variance > 5000:
                return 'multi_column'
            else:
                return 'structured_list'
        elif len(large_gaps) > 2:
            return 'sectioned'
        else:
            return 'simple'
    
    def _detect_list_patterns(self, elements):
        """Detect list patterns in the document"""
        
        list_indicators = []
        for element in elements:
            text = element['text'].lower()
            
            # Check for common list markers
            if (text.startswith('•') or text.startswith('*') or 
                re.match(r'^\d+\.', text) or 
                re.match(r'^[a-z]\)', text) or
                text in ['e', '«', '®', '©']):  # OCR misread bullets
                list_indicators.append(element)
        
        return len(list_indicators) > 2
    
    def _detect_heading_patterns(self, elements):
        """Detect heading patterns"""
        
        if not elements:
            return False
        
        # Look for potential headings (isolated lines, different positioning)
        headings = 0
        for i, element in enumerate(elements):
            text = element['text']
            
            # Check if line appears to be a heading
            if (len(text) < 80 and  # Not too long
                ':' in text and len(text.split(':')[0]) < 50):  # Has colon
                headings += 1
            elif (i < len(elements) - 1 and 
                  elements[i+1]['y'] - element['y'] > self.section_gap_threshold):
                headings += 1
        
        return headings > 1
    
    def _detect_table_patterns(self, elements):
        """Detect table-like structures"""
        
        if len(elements) < 6:
            return False
        
        # Look for aligned columns
        x_positions = {}
        for element in elements:
            x = element['x']
            # Group similar x positions
            found_group = False
            for group_x in x_positions:
                if abs(x - group_x) <= 15:  # Tolerance for alignment
                    x_positions[group_x].append(element)
                    found_group = True
                    break
            if not found_group:
                x_positions[x] = [element]
        
        # If we have 3+ aligned columns with multiple rows each
        aligned_columns = [group for group in x_positions.values() if len(group) >= 2]
        return len(aligned_columns) >= 3
    
    def _estimate_column_count(self, elements):
        """Estimate number of columns in document"""
        
        if not elements:
            return 1
        
        # Group elements by approximate x position
        x_groups = {}
        for element in elements:
            x = element['x']
            
            # Find existing group or create new one
            assigned = False
            for group_x in list(x_groups.keys()):
                if abs(x - group_x) <= 30:  # Column tolerance
                    x_groups[group_x].append(element)
                    assigned = True
                    break
            
            if not assigned:
                x_groups[x] = [element]
        
        # Count significant groups (with multiple elements)
        significant_groups = [g for g in x_groups.values() if len(g) >= 3]
        return min(len(significant_groups), 3)  # Cap at 3 columns
    
    def _build_structure_aware_text(self, data, structure_info):
        """Build text output respecting detected structure"""
        
        elements = structure_info['elements']
        if not elements:
            return ""
        
        # Group elements into logical lines
        lines = self._group_elements_into_lines(elements)
        
        # Build output with structure-aware spacing
        return self._format_structured_lines(lines, structure_info)
    
    def _group_elements_into_lines(self, elements):
        """Group elements into logical text lines"""
        
        if not elements:
            return []
        
        lines = []
        current_line = [elements[0]]
        current_baseline = elements[0]['y']
        
        for element in elements[1:]:
            # Calculate if element belongs to current line
            y_diff = abs(element['y'] - current_baseline)
            
            if y_diff <= self.line_tolerance:
                current_line.append(element)
            else:
                # Finalize current line and start new one
                if current_line:
                    current_line.sort(key=lambda e: e['x'])
                    lines.append({
                        'elements': current_line,
                        'baseline': current_baseline,
                        'type': self._classify_line_type(current_line)
                    })
                
                current_line = [element]
                current_baseline = element['y']
        
        # Add final line
        if current_line:
            current_line.sort(key=lambda e: e['x'])
            lines.append({
                'elements': current_line,
                'baseline': current_baseline,
                'type': self._classify_line_type(current_line)
            })
        
        return lines
    
    def _classify_line_type(self, line_elements):
        """Classify the type of a text line"""
        
        if not line_elements:
            return 'empty'
        
        # Combine text from line
        line_text = ' '.join(e['text'] for e in line_elements)
        
        # Classify based on content patterns
        if re.match(r'^\d+\.', line_text.strip()):
            return 'numbered_item'
        elif any(line_text.strip().startswith(marker) for marker in ['•', '*', '-']):
            return 'bullet_item'
        elif line_text.strip().endswith(':') and len(line_text) < 60:
            return 'heading'
        elif len(line_elements) == 1 and len(line_text) < 80:
            return 'title'
        else:
            return 'content'
    
    def _format_structured_lines(self, lines, structure_info):
        """Format lines with structure-aware spacing"""
        
        if not lines:
            return ""
        
        result = []
        prev_baseline = None
        prev_type = None
        
        for line_info in lines:
            elements = line_info['elements']
            baseline = line_info['baseline']
            line_type = line_info['type']
            
            # Build line text with appropriate spacing
            line_text = self._build_line_text(elements, structure_info)
            
            if not line_text.strip():
                continue
            
            # Determine spacing needs
            spacing_needed = self._calculate_spacing_needs(
                baseline, prev_baseline, line_type, prev_type, structure_info
            )
            
            # Add spacing
            if spacing_needed and result and result[-1].strip():
                result.append('')
            
            result.append(line_text)
            
            prev_baseline = baseline
            prev_type = line_type
        
        return '\n'.join(result)
    
    def _build_line_text(self, elements, structure_info):
        """Build text for a line considering structure"""
        
        if not elements:
            return ""
        
        # For single element, return as-is
        if len(elements) == 1:
            return elements[0]['text']
        
        # For multiple elements, consider spacing
        result = [elements[0]['text']]
        
        for i in range(1, len(elements)):
            prev_elem = elements[i-1]
            curr_elem = elements[i]
            
            # Calculate spacing based on gap
            gap = curr_elem['x'] - (prev_elem['x'] + prev_elem['w'])
            
            # Determine appropriate spacing
            if gap > 50:  # Large gap - different sections
                result.append('   ')  # Multiple spaces
            elif gap > 20:  # Medium gap
                result.append('  ')   # Double space
            else:  # Normal gap
                result.append(' ')    # Single space
            
            result.append(curr_elem['text'])
        
        return ''.join(result)
    
    def _calculate_spacing_needs(self, current_baseline, prev_baseline, 
                                current_type, prev_type, structure_info):
        """Calculate if spacing is needed between lines"""
        
        if prev_baseline is None:
            return False
        
        # Calculate vertical gap
        vertical_gap = current_baseline - prev_baseline
        
        # Spacing rules based on gap size and content type
        if vertical_gap > self.section_gap_threshold:
            return True  # Large gap - always space
        elif vertical_gap > self.paragraph_gap_threshold:
            return True  # Medium gap - paragraph break
        elif current_type == 'heading' or current_type == 'title':
            return True  # Before headings
        elif prev_type == 'heading':
            return False  # After headings, no extra space
        else:
            return False  # Normal flow
    
    def _normalize_bullets_simple(self, text):
        """Simple bullet point normalization - most basic approach"""
        
        if not text:
            return text
        
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            if not line.strip():
                normalized_lines.append('')
                continue
            
            # Check if line starts with common OCR bullet misreads
            line_start = line.lstrip()  # Remove leading spaces but preserve indentation level
            leading_spaces = line[:len(line) - len(line_start)]
            
            # Apply simple bullet fixes
            for old_bullet, new_bullet in self.bullet_fixes.items():
                if line_start.startswith(old_bullet):
                    line_start = new_bullet + line_start[len(old_bullet):]
                    break
            
            # Reconstruct line with original indentation
            normalized_lines.append(leading_spaces + line_start)
        
        return '\n'.join(normalized_lines)
    
    def _apply_structure_formatting(self, text):
        """Apply final structure-aware formatting"""
        
        if not text:
            return text
        
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                formatted_lines.append('')
                continue
            
            # Apply universal cleanup only
            line = self._apply_universal_cleanup(line)
            
            formatted_lines.append(line)
        
        # Join and final cleanup
        result = '\n'.join(formatted_lines)
        
        # Limit consecutive blank lines
        result = re.sub(r'\n{4,}', '\n\n\n', result)
        
        return result.strip()
    
    def _apply_universal_cleanup(self, text):
        """Apply only universal text cleanup"""
        
        # Fix multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def _estimate_extraction_confidence(self, text):
        """Estimate confidence based on text coherence"""
        
        if not text:
            return 0
        
        # Basic confidence metrics
        words = text.split()
        if not words:
            return 0
        
        # Check for reasonable word length distribution
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Check for reasonable character distribution
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        
        # Estimate confidence
        confidence = 60  # Base confidence
        
        if 3 <= avg_word_length <= 8:  # Reasonable word length
            confidence += 15
        
        if alpha_ratio > 0.7:  # High alphabetic content
            confidence += 15
        
        if len(words) > 10:  # Sufficient content
            confidence += 10
        
        return min(confidence, 95)  # Cap at 95%