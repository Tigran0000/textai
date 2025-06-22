"""
EyeShot AI - Enhanced Document Handler
Base class for all document type handlers with aggressive text detection
Last updated: 2025-06-20 10:49:47 UTC
Author: Tigran0000
"""

import os
import sys
import time
import math
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io

class DocumentHandler:
    """
    Base class for document-specific OCR handling with enhanced detection capabilities.
    Provides common functionality for all document types.
    """
    
    def __init__(self):
        """Initialize document handler"""
        self.name = "standard"
        self.description = "Standard document handler"
        self.processors = {}
        self.engines = {}
        self.languages = ['en']
        self.quality_level = 'balanced'
        self.debug_mode = False
        self.save_debug_images = False
        self.debug_dir = 'ocr_debug'
        
    def set_processors(self, processors: Dict):
        """Set processor instances for this handler"""
        self.processors = processors
    
    def register_engines(self, engines: Dict):
        """Register available OCR engines with the handler"""
        self.engines = engines
    
    def set_languages(self, languages: List[str]):
        """Set supported languages for OCR"""
        self.languages = languages
    
    def set_quality_level(self, level: str):
        """Set quality level for OCR (speed, balanced, quality)"""
        self.quality_level = level
    
    def set_debug_mode(self, debug_mode: bool):
        """Enable/disable debug mode"""
        self.debug_mode = debug_mode
    
    def set_save_debug_images(self, save_images: bool, debug_dir: str = 'ocr_debug'):
        """Configure debug image saving"""
        self.save_debug_images = save_images
        self.debug_dir = debug_dir
        
        # Create directory if enabled
        if self.save_debug_images and not os.path.exists(self.debug_dir):
            try:
                os.makedirs(self.debug_dir)
            except Exception as e:
                print(f"Failed to create debug directory: {e}")
                self.save_debug_images = False
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply document-specific preprocessing to optimize OCR results
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image
        """
        # Basic preprocessing using the image processor
        if 'image' in self.processors:
            # Apply standard preprocessing
            enhanced_image = self.processors['image'].preprocess_for_ocr(image)
            return enhanced_image
        return image
    
    def extract_text(self, image: Image.Image, preprocess: bool = True) -> Dict:
        """
        Extract text from image with document-specific optimizations
        
        Args:
            image: PIL Image to process
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary with extraction results
        """
        try:
            # Make a copy to avoid modifying original
            original_image = image.copy()
            
            # Initial preprocessing if enabled
            if preprocess:
                processed_image = self.preprocess_image(original_image)
            else:
                processed_image = original_image.copy()
            
            # Save debug image if enabled
            if self.debug_mode and self.save_debug_images:
                self._save_debug_image(processed_image, "initial_preprocess")
            
            # Try standard extraction first
            results = self._try_multiple_approaches(processed_image)
            
            # Check if we got good results
            best_result = self._get_best_result(results)
            
            # If no good results, try aggressive preprocessing
            if not best_result.get('text', '').strip() or best_result.get('confidence', 0) < 30:
                if self.debug_mode:
                    print("Standard approaches failed, trying aggressive preprocessing...")
                
                # Try multiple preprocessing variations
                enhanced_images = self._create_enhanced_variations(original_image)
                
                # Process each enhanced image
                for idx, (method, enhanced_img) in enumerate(enhanced_images):
                    if self.debug_mode and self.save_debug_images:
                        self._save_debug_image(enhanced_img, f"enhanced_{method}")
                    
                    # Try OCR on this variation
                    variation_results = self._try_multiple_approaches(enhanced_img)
                    
                    # Get best result for this variation
                    variation_best = self._get_best_result(variation_results)
                    
                    # Add method info
                    variation_best['preprocessing_method'] = method
                    
                    # Add to overall results
                    results.extend(variation_results)
                
                # Get best result across all attempts
                best_result = self._get_best_result(results)
            
            # Post-process the extracted text
            if 'text' in self.processors and best_result.get('text'):
                best_result['text'] = self.processors['text'].clean_extracted_text(
                    best_result['text'], 
                    document_type=self.name
                )
            
            # If still no text, use most aggressive OCR approach as last resort
            if not best_result.get('text', '').strip():
                if self.debug_mode:
                    print("All standard methods failed, using last-resort approach...")
                
                # Last resort: try pure binarization with multiple thresholds
                binary_results = self._try_binary_thresholds(original_image)
                
                if binary_results and any(r.get('text', '').strip() for r in binary_results):
                    best_binary = self._get_best_result(binary_results)
                    
                    # Use binary result if it has text
                    if best_binary.get('text', '').strip():
                        best_result = best_binary
            
            # If we have text but confidence is low, try to fix it with language model
            # This would integrate with a language model for text correction, but is optional
            
            # Add metadata
            best_result['success'] = bool(best_result.get('text', '').strip())
            best_result['extraction_engine'] = best_result.get('extraction_engine', 'unknown')
            best_result['document_type'] = self.name
            best_result['word_count'] = len(best_result.get('text', '').split()) if best_result.get('text') else 0
            best_result['char_count'] = len(best_result.get('text', '')) if best_result.get('text') else 0
            
            return best_result
            
        except Exception as e:
            if self.debug_mode:
                print(f"Extraction error in {self.name} handler: {e}")
                traceback.print_exc()
                
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'char_count': 0,
                'extraction_mode': self.name,
                'success': False,
                'error': str(e)
            }
    
    def _create_enhanced_variations(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """
        Create multiple enhanced variations of the image for OCR
        
        Args:
            image: Original PIL Image
            
        Returns:
            List of (method_name, enhanced_image) tuples
        """
        variations = []
        
        try:
            # 1. High contrast enhancement
            contrast_img = ImageEnhance.Contrast(image).enhance(2.0)
            variations.append(("high_contrast", contrast_img))
            
            # 2. Sharpened image
            sharp_img = image.filter(ImageFilter.SHARPEN)
            sharp_img = sharp_img.filter(ImageFilter.SHARPEN)  # Double sharpening
            variations.append(("sharpened", sharp_img))
            
            # 3. Grayscale conversion
            gray_img = image.convert('L')
            variations.append(("grayscale", gray_img))
            
            # 4. Binarization (black and white)
            binary_img = gray_img.point(lambda x: 0 if x < 128 else 255, '1')
            variations.append(("binary", binary_img))
            
            # 5. Inverted image (sometimes works better for light text on dark backgrounds)
            inverted_img = ImageOps.invert(gray_img)
            variations.append(("inverted", inverted_img))
            
            # 6. Combination: contrast + sharpen
            combo1_img = ImageEnhance.Contrast(sharp_img).enhance(1.8)
            variations.append(("contrast_sharp", combo1_img))
            
            # 7. Combination: contrast + binary
            combo2_gray = ImageEnhance.Contrast(gray_img).enhance(2.0)
            combo2_img = combo2_gray.point(lambda x: 0 if x < 140 else 255, '1')
            variations.append(("contrast_binary", combo2_img))
            
            # 8. Brightness increased
            bright_img = ImageEnhance.Brightness(image).enhance(1.3)
            variations.append(("bright", bright_img))
            
            # 9. Edge enhancement
            edge_img = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            variations.append(("edge_enhance", edge_img))
            
            # 10. Denoised image
            denoise_img = image.filter(ImageFilter.MedianFilter(size=3))
            variations.append(("denoised", denoise_img))
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error creating image variations: {e}")
        
        return variations
    
    def _try_binary_thresholds(self, image: Image.Image) -> List[Dict]:
        """
        Try multiple binary thresholds as last resort
        
        Args:
            image: Original PIL Image
            
        Returns:
            List of extraction results
        """
        results = []
        
        try:
            # Convert to grayscale
            gray_img = image.convert('L')
            
            # Try multiple threshold levels
            thresholds = [80, 100, 120, 140, 160, 180, 200]
            
            for threshold in thresholds:
                # Create binary image with this threshold
                binary_img = gray_img.point(lambda x: 0 if x < threshold else 255, '1')
                
                if self.debug_mode and self.save_debug_images:
                    self._save_debug_image(binary_img, f"binary_threshold_{threshold}")
                
                # Try OCR with Tesseract
                if self.engines.get('tesseract_available', False):
                    try:
                        import pytesseract
                        
                        # Use raw text extraction with minimal processing
                        config = '--oem 0 --psm 6'
                        text = pytesseract.image_to_string(binary_img, config=config)
                        
                        if text.strip():
                            results.append({
                                'text': text,
                                'confidence': 50,  # Arbitrary confidence for binary threshold
                                'extraction_engine': f'tesseract_binary_{threshold}',
                                'preprocessing': f'binary_threshold_{threshold}'
                            })
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Binary threshold OCR error at {threshold}: {e}")
        except Exception as e:
            if self.debug_mode:
                print(f"Binary threshold processing error: {e}")
        
        return results
    
    def _try_multiple_approaches(self, image: Image.Image) -> List[Dict]:
        """
        Try multiple extraction approaches and return all results
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of extraction results
        """
        results = []
        
        # Try tesseract with different settings
        if self.engines.get('tesseract_available', False):
            try:
                import pytesseract
                
                # Check if tesseract is properly installed and working
                try:
                    pytesseract.get_tesseract_version()
                except Exception as e:
                    if self.debug_mode:
                        print(f"Tesseract not properly installed: {e}")
                    return results
                
                # Prepare language string
                lang_str = '+'.join(self.languages[:3]) if self.languages else 'eng'
                
                # 1. Try default mode (automatic page segmentation)
                result_default = self._extract_with_tesseract(
                    image, 
                    config='--oem 3 --psm 3',
                    languages=lang_str
                )
                results.append(result_default)
                
                # 2. Try single text block mode (assumes a single block of text)
                result_block = self._extract_with_tesseract(
                    image, 
                    config='--oem 3 --psm 6',
                    languages=lang_str
                )
                results.append(result_block)
                
                # 3. Try sparse text mode (no specific text orientation or ordering)
                result_sparse = self._extract_with_tesseract(
                    image, 
                    config='--oem 3 --psm 11',
                    languages=lang_str
                )
                results.append(result_sparse)
                
                # 4. Try page segmentation with orientation and script detection
                result_orient = self._extract_with_tesseract(
                    image, 
                    config='--oem 3 --psm 1',
                    languages=lang_str
                )
                results.append(result_orient)
                
                # 5. Legacy engine with single column detection
                result_legacy = self._extract_with_tesseract(
                    image, 
                    config='--oem 0 --psm 4',
                    languages=lang_str
                )
                results.append(result_legacy)
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Tesseract extraction error: {e}")
        
        # Try EasyOCR if available
        if self.engines.get('easyocr_available', False) and 'easyocr_reader' in self.engines:
            try:
                reader = self.engines['easyocr_reader']
                if reader:
                    # Extract using EasyOCR
                    ocr_result = reader.readtext(image)
                    
                    # Convert to standardized format
                    combined_text = " ".join([res[1] for res in ocr_result])
                    avg_confidence = sum([res[2] for res in ocr_result]) / len(ocr_result) if ocr_result else 0
                    
                    results.append({
                        'text': combined_text,
                        'confidence': avg_confidence * 100,  # Convert to 0-100 scale
                        'extraction_engine': 'easyocr'
                    })
            except Exception as e:
                if self.debug_mode:
                    print(f"EasyOCR extraction error: {e}")
        
        # Try PaddleOCR if available
        if self.engines.get('paddleocr_available', False) and 'paddleocr_engine' in self.engines:
            try:
                engine = self.engines['paddleocr_engine']
                if engine:
                    # Convert PIL to OpenCV format if needed
                    import numpy as np
                    img_arr = np.array(image)
                    
                    # Extract using PaddleOCR
                    paddle_result = engine.ocr(img_arr, cls=True)
                    
                    # Process result
                    text_parts = []
                    confidence_sum = 0
                    count = 0
                    
                    # Extract text from result structure
                    for line in paddle_result:
                        for item in line:
                            if isinstance(item, list) and len(item) > 1:
                                text = item[1][0]  # Text content
                                conf = item[1][1]  # Confidence score
                                text_parts.append(text)
                                confidence_sum += conf
                                count += 1
                    
                    # Combine results
                    combined_text = " ".join(text_parts)
                    avg_confidence = (confidence_sum / count) * 100 if count > 0 else 0
                    
                    results.append({
                        'text': combined_text,
                        'confidence': avg_confidence,
                        'extraction_engine': 'paddleocr'
                    })
            except Exception as e:
                if self.debug_mode:
                    print(f"PaddleOCR extraction error: {e}")
                    
        return results
    
    def _extract_with_tesseract(self, image: Image.Image, config: str, languages: str) -> Dict:
        """
        Extract text using Tesseract with specific configuration
        
        Args:
            image: PIL Image to process
            config: Tesseract configuration string
            languages: Language string (e.g., 'eng+fra')
            
        Returns:
            Extraction result dictionary
        """
        try:
            import pytesseract
            
            # Get text
            text = pytesseract.image_to_string(image, lang=languages, config=config)
            
            # Get confidence and other data
            try:
                data = pytesseract.image_to_data(
                    image, 
                    lang=languages,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except:
                # If confidence calculation fails, use a default value
                avg_confidence = 50 if text.strip() else 0
            
            return {
                'text': text,
                'confidence': avg_confidence,
                'extraction_engine': f'tesseract_{config.replace(" ", "_")}',
                'word_count': len(text.split()) if text else 0,
                'char_count': len(text) if text else 0
            }
        except Exception as e:
            if self.debug_mode:
                print(f"Tesseract extraction error with config {config}: {e}")
                
            return {
                'text': '',
                'confidence': 0,
                'extraction_engine': f'tesseract_{config.replace(" ", "_")}_failed',
                'word_count': 0,
                'char_count': 0,
                'error': str(e)
            }
    
    def _get_best_result(self, results: List[Dict]) -> Dict:
        """
        Select the best result from multiple extraction attempts
        
        Args:
            results: List of extraction results
            
        Returns:
            Best extraction result
        """
        if not results:
            return {
                'text': '',
                'confidence': 0,
                'extraction_engine': 'none'
            }
            
        # First, filter for results that actually have text
        text_results = [r for r in results if r.get('text', '').strip()]
        
        # If no result has text, return the first one
        if not text_results:
            return results[0]
            
        # Calculate text quality score for each result
        scored_results = []
        for result in text_results:
            text = result.get('text', '')
            
            # Basic quality checks
            word_count = len(text.split())
            avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
            has_punctuation = any(p in text for p in '.,:;?!')
            confidence = result.get('confidence', 0)
            
            # Calculate quality score (higher is better)
            # Factors: confidence, word count, reasonable word length, presence of punctuation
            quality_score = (
                confidence * 0.5 +  # 50% weight on confidence
                min(100, word_count * 2) * 0.3 +  # 30% weight on word count (up to 50 words)
                (50 if 3 <= avg_word_length <= 10 else 0) * 0.1 +  # 10% weight on reasonable word length
                (20 if has_punctuation else 0) * 0.1  # 10% weight on punctuation
            )
            
            scored_results.append((result, quality_score))
        
        # Sort by quality score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Get best result
        best_result = scored_results[0][0] if scored_results else text_results[0]
        
        # If all confidences are very low, try combining results
        low_confidence = all(r.get('confidence', 0) < 40 for r in results)
        
        if low_confidence and len(text_results) > 1:
            # Filter out duplicate or nearly identical results
            unique_texts = {}
            for r in text_results:
                text = r.get('text', '').strip()
                if text and not any(self._text_similarity(text, t) > 0.8 for t in unique_texts):
                    unique_texts[text] = r
            
            # If we have multiple unique texts, combine them
            if len(unique_texts) > 1:
                combined_text = "\n\n".join(unique_texts.keys())
                
                # Take the average confidence
                avg_confidence = sum([r.get('confidence', 0) for r in unique_texts.values()]) / len(unique_texts)
                
                # Return combined result if it's better than individual ones
                if len(combined_text) > len(best_result.get('text', '')):
                    return {
                        'text': combined_text,
                        'confidence': avg_confidence,
                        'extraction_engine': 'combined',
                        'combined_from': [r.get('extraction_engine', 'unknown') for r in unique_texts.values()]
                    }
        
        return best_result

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two text strings
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score (0-1, higher means more similar)
        """
        # Simple similarity calculation based on shared words
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        shared = words1.intersection(words2)
        
        return len(shared) / max(1, (len(words1) + len(words2)) / 2)

    def _save_debug_image(self, image: Image.Image, stage: str):
        """Save an image for debugging purposes"""
        try:
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
                
            # Generate timestamp string
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{stage}_{timestamp}.png"
            filepath = os.path.join(self.debug_dir, filename)
            
            image.save(filepath)
            
            if self.debug_mode:
                print(f"Saved debug image: {filepath}")
        except Exception as e:
            if self.debug_mode:
                print(f"Failed to save debug image: {e}")