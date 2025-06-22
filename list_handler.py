"""
EyeShot AI - List Document Handler
Specialized handler for documents with bullet points and lists
Last updated: 2025-06-20 10:02:38 UTC
Author: Tigran0000
"""

import re
from typing import Dict, List, Any, Optional
from PIL import Image

from .base_handler import DocumentHandler

class ListHandler(DocumentHandler):
    """
    Specialized handler for documents with bullet points and lists.
    Optimizes extraction and formatting of bulleted and numbered lists.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "list"
        self.description = "Handler for documents with bullet points and lists"
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply specialized preprocessing for list documents"""
        # Lists often benefit from enhanced contrast to make bullet points clearer
        image = self.processors['image'].enhance_contrast(image, factor=1.2)
        image = self.processors['image'].sharpen(image, factor=1.3)
        return image
        
    def extract_text(self, image: Image.Image, preprocess: bool = True) -> Dict[str, Any]:
        """
        Extract text with special handling for bullet points and lists
        
        Args:
            image: PIL Image to process
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary with extraction results
        """
        # Extract text using base handler
        result = super().extract_text(image, preprocess)
        
        # Apply specialized post-processing for lists
        if result['success'] and result.get('text'):
            # Process lists specifically
            result['text'] = self._process_lists(result['text'])
            
            # Update word count after processing
            result['word_count'] = len(result['text'].split())
            result['char_count'] = len(result['text'])
        
        return result
    
    def _process_lists(self, text: str) -> str:
        """
        Process and format lists in extracted text
        
        Args:
            text: Extracted text
            
        Returns:
            Text with improved list formatting
        """
        if not text:
            return text
            
        processed_text = text
        
        # Step 1: Identify and fix bullet points
        # Various bullet characters that might be recognized incorrectly
        bullet_chars = [
            '•', '⁃', '⦁', '▪', '▫', '◦', '⦿', '⚫', '⚬',  # Common bullet chars
            'o', '*', '-', '>', '+',  # ASCII alternatives often used
        ]
        
        # Create regex pattern to match bullet chars at line start (possibly with spaces)
        bullet_pattern = '|'.join([re.escape(c) for c in bullet_chars])
        bullet_regex = rf'^\s*({bullet_pattern})\s*'
        
        # Process line by line
        lines = processed_text.split('\n')
        for i, line in enumerate(lines):
            # Check for bullet points at start of line
            match = re.match(bullet_regex, line)
            if match:
                # Standardize bullet format: "• " followed by text
                bullet_char = match.group(1)
                remaining_text = line[match.end():].lstrip()
                
                # Format to ensure space after bullet
                lines[i] = f"• {remaining_text}"
                
                # Check for common OCR errors in bullet points
                if i > 0 and lines[i-1]:
                    # Fix cases where phase becomes "phas•" (merges with bullet)
                    prev_line = lines[i-1]
                    if prev_line.endswith(('Phas', 'phas')):
                        lines[i-1] = prev_line + 'e'
        
        # Step 2: Identify and fix numbered lists
        numbered_regex = r'^\s*(\d+)[\.:\)]\s*'
        for i, line in enumerate(lines):
            match = re.match(numbered_regex, line)
            if match:
                # Standardize numbered list format: "1. " followed by text
                number = match.group(1)
                remaining_text = line[match.end():].lstrip()
                
                # Format to ensure space after number
                lines[i] = f"{number}. {remaining_text}"
        
        # Step 3: Identify and fix lettered lists
        lettered_regex = r'^\s*([a-zA-Z])[\.:\)]\s*'
        for i, line in enumerate(lines):
            match = re.match(lettered_regex, line)
            if match:
                # Standardize lettered list format: "a. " or "a) " followed by text
                letter = match.group(1)
                remaining_text = line[match.end():].lstrip()
                
                # Keep original separator (. or ))
                separator = "." if "." in line[:match.end()] else ")"
                
                # Format to ensure space after letter
                lines[i] = f"{letter}{separator} {remaining_text}"
        
        # Ensure proper line breaks between list items
        formatted_lines = []
        in_list = False
        
        for i, line in enumerate(lines):
            is_list_item = (
                re.match(bullet_regex, line) or 
                re.match(numbered_regex, line) or 
                re.match(lettered_regex, line)
            )
            
            if is_list_item:
                # If this is a new list (not continuing), add extra line break
                if not in_list and formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                    
                in_list = True
            elif line.strip() and in_list:
                # Non-empty line after list - end list
                if i > 0:
                    formatted_lines.append('')
                in_list = False
                
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)