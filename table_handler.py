# src/ocr/handlers/table_handler.py
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

class TableHandler(DocumentHandler):
    """Handler for tables with cell structure preservation"""
    
    def can_handle(self, image: Image.Image) -> bool:
        """Check if image contains a table based on visual characteristics"""
        try:
            # Convert image to numpy array for analysis
            img_array = np.array(image.convert('L'))
            height, width = img_array.shape
            
            # Apply threshold for better line detection
            _, binary = cv2.threshold(img_array, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Find lines using HoughLinesP
            edges = cv2.Canny(binary, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 
                1, 
                np.pi/180, 
                threshold=50, 
                minLineLength=width*0.3, 
                maxLineGap=20
            )
            
            if lines is None or len(lines) < 5:
                return False
            
            # Count horizontal and vertical lines
            horizontal_lines = 0
            vertical_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 20:  # Horizontal line
                    horizontal_lines += 1
                elif abs(x2 - x1) < 20:  # Vertical line
                    vertical_lines += 1
            
            # A table typically has multiple horizontal and vertical lines 
            if horizontal_lines >= 3 and vertical_lines >= 3:
                return True
                
            # Check for grid-like structure using quick OCR
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Create a small version for quick analysis
                small_img = image.copy()
                small_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                # Use HOCR to check for grid-like positioning of words
                hocr = pytesseract.image_to_pdf_or_hocr(
                    small_img,
                    extension='hocr',
                    config='--psm 1'
                )
                
                # Parse HOCR to look for table-like word alignments
                soup = BeautifulSoup(hocr.decode('utf-8'), 'html.parser')
                words = soup.find_all('span', class_='ocrx_word')
                
                # Extract word positions
                word_positions = []
                for word in words:
                    bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', word.get('title', ''))
                    if bbox_match:
                        x1, y1, x2, y2 = map(int, bbox_match.groups())
                        word_positions.append((x1, y1, x2, y2))
                
                # Look for grid alignment patterns in word positions
                if len(word_positions) > 10:  # Need enough words for meaningful analysis
                    # Extract Y-coordinates for line alignment
                    y_coords = [pos[1] for pos in word_positions]
                    
                    # Count words that align horizontally (likely rows)
                    alignment_counts = {}
                    for y in y_coords:
                        y_key = y // 10  # Group within 10 pixels
                        alignment_counts[y_key] = alignment_counts.get(y_key, 0) + 1
                    
                    # Count rows with multiple aligned words
                    aligned_rows = sum(1 for count in alignment_counts.values() if count > 2)
                    
                    # If we have multiple rows with aligned words, likely a table
                    if aligned_rows >= 3:
                        return True
            
            return False
            
        except Exception as e:
            if self.debug_mode:
                print(f"Table detection error: {e}")
            return False
    
    def _perform_extraction(self, image: Image.Image, preprocess: bool) -> Dict:
        """Extract table with cell structure preservation"""
        try:
            # Use specialized table preprocessing if requested
            if preprocess and 'image' in self.processors:
                processed_image = self.processors['image'].preprocess_table(image.copy())
            else:
                processed_image = image.copy()
            
            # Use two approaches for better results:
            # 1. Layout analysis (preferred, extracts the actual grid)
            # 2. Tesseract's built-in table extraction (fallback)
            
            # Try layout analysis first
            if 'layout_analyzer' in self.processors:
                try:
                    layout_analyzer = self.processors['layout_analyzer']
                    table_data = layout_analyzer.extract_table_structure(np.array(processed_image))
                    
                    if table_data and 'cells' in table_data and len(table_data['cells']) > 0:
                        # Convert table to different text formats
                        table_text = self._convert_table_to_text(table_data)
                        markdown_table = table_data.get('markdown_table', '')
                        json_table = table_data.get('json_table', '{}')
                        
                        return {
                            'text': table_text,
                            'markdown_table': markdown_table,
                            'json_table': json_table,
                            'confidence': table_data.get('confidence', 70),
                            'word_count': len(table_text.split()),
                            'char_count': len(table_text),
                            'rows': table_data.get('rows', 0),
                            'columns': table_data.get('columns', 0),
                            'success': True,
                            'engine': 'layout_analysis_table',
                            'has_grid': True
                        }
                except Exception as layout_err:
                    if self.debug_mode:
                        print(f"Layout analysis extraction error: {layout_err}")
            
            # If layout analysis failed, use our own table extraction algorithm
            table_grid = self._extract_table_grid(processed_image)
            
            if table_grid:
                table_data = self._extract_cell_text(processed_image, table_grid)
                
                # Generate output in different formats
                plain_text = self._table_to_plain_text(table_data)
                markdown = self._table_to_markdown(table_data)
                
                return {
                    'text': plain_text,
                    'markdown_table': markdown,
                    'json_table': json.dumps(table_data),
                    'confidence': table_data.get('confidence', 70),
                    'word_count': len(plain_text.split()),
                    'char_count': len(plain_text),
                    'rows': len(table_data['rows']),
                    'columns': len(table_data['columns']) if 'columns' in table_data else 0,
                    'success': True,
                    'engine': 'grid_detection_table',
                    'has_grid': True
                }
            
            # Fallback to standard Tesseract with table optimized settings
            if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                # Special config for tables
                config = '--oem 3 --psm 6 -c tessedit_create_tsv=1'
                
                # Get text
                text = pytesseract.image_to_string(processed_image, config=config)
                
                # Get confidence data
                data = pytesseract.image_to_data(
                    processed_image, 
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Try to structure the text as a table
                structured_text = self._structure_table_text(text, data)
                
                return {
                    'text': structured_text,
                    'confidence': confidence,
                    'word_count': len(structured_text.split()),
                    'char_count': len(structured_text),
                    'success': True,
                    'engine': 'tesseract_table',
                    'has_grid': False  # No grid was detected
                }
            else:
                # Fallback to basic extraction if Tesseract isn't available
                return self._extract_with_basic_method(processed_image)
                
        except Exception as e:
            return self._error_result(f"Table extraction error: {str(e)}")
    
    def _extract_table_grid(self, image: Image.Image) -> Dict[str, Any]:
        """Extract table grid (rows and columns) from the image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            # Apply threshold to create binary image
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            
            # Create kernels for horizontal and vertical line detection
            height, width = binary.shape
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
            
            # Detect horizontal lines
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
            
            # Detect vertical lines
            v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
            
            # Combine horizontal and vertical lines
            table_grid = cv2.add(h_lines, v_lines)
            
            # Find contours of the grid
            contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # No grid found
            if not contours:
                return None
            
            # Extract row and column positions
            h_line_positions = []
            v_line_positions = []
            
            # Get horizontal line positions
            h_projection = np.sum(h_lines, axis=1)
            for i, proj in enumerate(h_projection):
                if proj > width * 0.2:  # Line must span at least 20% of width
                    h_line_positions.append(i)
            
            # Get vertical line positions
            v_projection = np.sum(v_lines, axis=0)
            for i, proj in enumerate(v_projection):
                if proj > height * 0.2:  # Line must span at least 20% of height
                    v_line_positions.append(i)
            
            # Cluster nearby lines (merging lines that are close together)
            h_line_positions = self._cluster_line_positions(h_line_positions, max_gap=5)
            v_line_positions = self._cluster_line_positions(v_line_positions, max_gap=5)
            
            # Need at least 2 horizontal and 2 vertical lines for a table
            if len(h_line_positions) < 2 or len(v_line_positions) < 2:
                return None
            
            # Create grid definition
            table_grid = {
                'rows': h_line_positions,
                'columns': v_line_positions,
                'width': width,
                'height': height
            }
            
            return table_grid
            
        except Exception as e:
            if self.debug_mode:
                print(f"Grid extraction error: {e}")
            return None
    
    def _cluster_line_positions(self, positions: List[int], max_gap: int = 5) -> List[int]:
        """Group nearby line positions and return the center of each cluster"""
        if not positions:
            return []
        
        # Sort positions
        positions = sorted(positions)
        
        # Initialize clusters
        clusters = [[positions[0]]]
        
        # Group positions into clusters
        for pos in positions[1:]:
            if pos - clusters[-1][-1] <= max_gap:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        # Calculate center of each cluster
        return [sum(cluster) // len(cluster) for cluster in clusters]
    
    def _extract_cell_text(self, image: Image.Image, grid: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from each cell in the table grid"""
        # Get rows and columns
        rows = grid['rows']
        columns = grid['columns']
        
        # Initialize table data
        table_data = {
            'cells': [],
            'rows': len(rows) - 1,
            'columns': len(columns) - 1,
            'confidence': 0.0
        }
        
        # Track total confidence
        total_confidence = 0.0
        cell_count = 0
        
        # Process each cell
        for i in range(len(rows) - 1):
            for j in range(len(columns) - 1):
                # Get cell boundaries
                top = rows[i]
                bottom = rows[i + 1]
                left = columns[j]
                right = columns[j + 1]
                
                # Skip tiny cells
                if bottom - top < 5 or right - left < 5:
                    continue
                
                # Add margin inside cell to avoid borders (5% of cell dimensions)
                margin_h = max(1, int((right - left) * 0.05))
                margin_v = max(1, int((bottom - top) * 0.05))
                
                # Extract cell image with internal margin
                cell_img = image.crop((
                    left + margin_h,
                    top + margin_v,
                    right - margin_h,
                    bottom - margin_v
                ))
                
                # Skip empty cells
                if cell_img.width < 5 or cell_img.height < 5:
                    continue
                
                # Extract text from cell
                if 'tesseract_available' in self.engines and self.engines['tesseract_available']:
                    try:
                        # Use settings for small text
                        config = '--psm 7 --oem 3'
                        text = pytesseract.image_to_string(cell_img, config=config).strip()
                        
                        # Get confidence
                        data = pytesseract.image_to_data(
                            cell_img,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        cell_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        # Update confidence stats
                        total_confidence += cell_confidence
                        cell_count += 1
                        
                        # Store cell data
                        table_data['cells'].append({
                            'row': i,
                            'col': j,
                            'text': text,
                            'confidence': cell_confidence,
                            'bbox': (left, top, right, bottom)
                        })
                        
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Cell extraction error: {e}")
        
        # Calculate average confidence
        if cell_count > 0:
            table_data['confidence'] = total_confidence / cell_count
        
        return table_data
    
    def _extract_with_basic_method(self, image: Image.Image) -> Dict:
        """Fallback method using direct extraction with text alignment"""
        try:
            # Get raw text
            text = pytesseract.image_to_string(image)
            
            # Try to detect and format as a table
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Look for separator lines (like +----+----+)
            separator_pattern = r'[+\-|=]+'
            rows = []
            current_row = []
            
            for line in lines:
                if re.match(separator_pattern, line):
                    # This is a separator line, complete current row
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                else:
                    # This is data, add to current row
                    current_row.append(line)
            
            # Add final row if needed
            if current_row:
                rows.append(current_row)
            
            # Format as a table
            if rows:
                plain_text = '\n'.join([' | '.join(row) for row in rows])
            else:
                plain_text = text
            
            return {
                'text': plain_text,
                'confidence': 60,  # Default confidence for basic method
                'word_count': len(plain_text.split()),
                'char_count': len(plain_text),
                'success': True,
                'engine': 'basic_table_extraction'
            }
            
        except Exception as e:
            return self._error_result(f"Basic table extraction error: {str(e)}")
    
    def _table_to_plain_text(self, table_data: Dict[str, Any]) -> str:
        """Convert table data to plain text format"""
        if not table_data or 'cells' not in table_data:
            return ""
        
        # Get dimensions
        rows = max([cell['row'] for cell in table_data['cells']]) + 1
        cols = max([cell['col'] for cell in table_data['cells']]) + 1
        
        # Create empty grid
        grid = [['' for _ in range(cols)] for _ in range(rows)]
        
        # Fill grid with cell values
        for cell in table_data['cells']:
            r, c = cell['row'], cell['col']
            grid[r][c] = cell.get('text', '').strip()
        
        # Create plain text representation with fixed column widths
        col_widths = [max(len(grid[r][c]) for r in range(rows)) + 2 for c in range(cols)]
        
        result = []
        for row in grid:
            line = []
            for i, cell in enumerate(row):
                line.append(cell.ljust(col_widths[i]))
            result.append(''.join(line))
        
        return '\n'.join(result)
    
    def _table_to_markdown(self, table_data: Dict[str, Any]) -> str:
        """Convert table data to markdown format"""
        if not table_data or 'cells' not in table_data:
            return ""
        
        # Get dimensions
        rows = max([cell['row'] for cell in table_data['cells']]) + 1
        cols = max([cell['col'] for cell in table_data['cells']]) + 1
        
        # Create empty grid
        grid = [['' for _ in range(cols)] for _ in range(rows)]
        
        # Fill grid with cell values
        for cell in table_data['cells']:
            r, c = cell['row'], cell['col']
            grid[r][c] = cell.get('text', '').strip()
        
        # Create markdown table
        markdown = []
        
        # Header row
        markdown.append('| ' + ' | '.join(grid[0]) + ' |')
        
        # Separator row
        markdown.append('| ' + ' | '.join(['---' for _ in range(cols)]) + ' |')
        
        # Data rows
        for row in grid[1:]:
            markdown.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(markdown)
    
    def _structure_table_text(self, text: str, data: Dict[str, Any]) -> str:
        """Try to structure raw text as a table based on word positioning"""
        # Initialize structured text
        lines = []
        line_data = {}
        
        # Group words by lines with their positions
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30 and data['text'][i].strip():
                block_num = data['block_num'][i]
                line_num = data['line_num'][i]
                
                # Create unique key for this line
                line_key = f"{block_num}_{line_num}"
                
                # Create line if needed
                if line_key not in line_data:
                    line_data[line_key] = {
                        'words': [],
                        'top': data['top'][i],
                        'left_positions': []
                    }
                
                # Add word with its position
                line_data[line_key]['words'].append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'width': data['width'][i],
                    'conf': data['conf'][i]
                })
                
                # Track left position for column analysis
                line_data[line_key]['left_positions'].append(data['left'][i])
        
        # Sort lines by vertical position
        sorted_line_keys = sorted(line_data.keys(), key=lambda k: line_data[k]['top'])
        
        # Find potential column boundaries
        all_left_positions = []
        for key, line in line_data.items():
            all_left_positions.extend(line['left_positions'])
        
        # Create position histogram
        position_counts = {}
        for pos in all_left_positions:
            # Group positions within 20 pixels
            pos_key = pos // 20 * 20
            position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
        
        # Find position clusters (potential column starts)
        min_count = max(2, len(line_data) // 4)  # Minimum frequency to be considered a column
        column_positions = [pos for pos, count in position_counts.items() 
                          if count >= min_count]
        
        # Sort positions from left to right
        column_positions.sort()
        
        # Process each line with column awareness
        for key in sorted_line_keys:
            line = line_data[key]
            
            # Sort words by horizontal position
            sorted_words = sorted(line['words'], key=lambda w: w['left'])
            
            if len(column_positions) <= 1:
                # No columns detected, use simple text
                line_text = ' '.join(word['text'] for word in sorted_words)
                lines.append(line_text)
            else:
                # Create array for each column
                columns = [''] * (len(column_positions) + 1)
                
                for word in sorted_words:
                    # Find which column this word belongs to
                    col_idx = 0
                    for i, pos in enumerate(column_positions):
                        if word['left'] >= pos - 10:
                            col_idx = i
                    
                    # Add word to its column
                    if columns[col_idx]:
                        columns[col_idx] += ' ' + word['text']
                    else:
                        columns[col_idx] = word['text']
                
                # Create line with proper column structure
                line_parts = []
                
                for col_text in columns:
                    if col_text:
                        line_parts.append(col_text)
                
                lines.append(' | '.join(line_parts))
        
        # Join all lines
        result = '\n'.join(lines)
        
        return result
    
    def _convert_table_to_text(self, table_data: Dict[str, Any]) -> str:
        """Convert table data to text representation"""
        if 'cells' not in table_data:
            return ""
        
        rows = []
        
        # Get dimensions
        max_row = max([cell.get('row', 0) for cell in table_data['cells']])
        max_col = max([cell.get('col', 0) for cell in table_data['cells']])
        
        # Create empty grid
        grid = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        # Fill in cell values
        for cell in table_data['cells']:
            r, c = cell.get('row', 0), cell.get('col', 0)
            if 0 <= r <= max_row and 0 <= c <= max_col:
                grid[r][c] = cell.get('text', '')
        
        # Convert to text
        for row in grid:
            rows.append(' | '.join(row))
        
        return '\n'.join(rows)