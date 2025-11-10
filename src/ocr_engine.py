from __future__ import annotations

import gc
import inspect
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from config.settings import Settings
from utils.memory_monitor import MemoryMonitor


@dataclass(frozen=True)
class OCRResult:
    """Result information returned by the OCR engine."""
    
    output_path: Path
    duration_seconds: float
    total_texts_found: int
    average_confidence: float
    low_confidence_count: int


@dataclass(frozen=True)
class TextDetection:
    """Individual text detection with metadata."""
    
    text: str
    confidence: float
    bbox: List[List[int]]  # 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    center_x: int
    center_y: int


class OCREngine:
    """PaddleOCR-based text recognition engine for QC form processing.
    
    Can be used as a context manager for automatic resource cleanup:
        with OCREngine(settings, logger) as engine:
            result = engine.process(image_path)
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger
        self._memory_monitor = MemoryMonitor(logger)
        # Initialize OCR engine immediately (eager initialization)
        self._ocr_engine = self._initialize_ocr()

    def __enter__(self) -> 'OCREngine':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    def close(self) -> None:
        """Clean up OCR engine resources."""
        if self._ocr_engine is not None:
            self._logger.debug("Releasing OCR engine resources")
            self._ocr_engine = None

    def _initialize_ocr(self) -> PaddleOCR:
        """Initialize PaddleOCR engine with configured settings."""
        self._logger.info("Initializing PaddleOCR engine...")
        
        # Build desired parameters
        ocr_params = {
            'use_angle_cls': True,  # Enable text angle classification
            'lang': 'ru',  # Russian language support
            'show_log': False,  # Suppress PaddleOCR internal logging
            'det_limit_side_len': self._settings.ocr_det_limit_side_len,  # Limit detection size
            'det_limit_type': 'max',  # Maximum dimension limit
        }
        
        # Add GPU parameter only if enabled
        if self._settings.ocr_use_gpu:
            ocr_params['use_gpu'] = True
        
        # Filter out unsupported parameters by checking PaddleOCR.__init__ signature
        try:
            sig = inspect.signature(PaddleOCR.__init__)
            supported_params = set(sig.parameters.keys())
            
            # Filter params to only include supported ones
            filtered_params = {}
            skipped_params = []
            
            for key, value in ocr_params.items():
                if key in supported_params:
                    filtered_params[key] = value
                else:
                    skipped_params.append(key)
            
            # Log skipped parameters for debugging
            if skipped_params:
                self._logger.debug(
                    "Skipping unsupported PaddleOCR parameters: %s", 
                    ", ".join(skipped_params)
                )
            
            ocr_params = filtered_params
            
        except Exception as e:
            self._logger.debug("Could not inspect PaddleOCR signature: %s. Using all parameters.", e)
            
        try:
            ocr_engine = PaddleOCR(**ocr_params)
        except Exception as e:
            # Try with minimal parameters if advanced ones fail
            self._logger.warning("Failed to initialize with advanced parameters: %s", e)
            self._logger.info("Falling back to minimal PaddleOCR initialization...")
            try:
                ocr_engine = PaddleOCR(lang='ru')
            except Exception as e2:
                self._logger.error("Failed to initialize PaddleOCR even with minimal parameters: %s", e2)
                # Try with no parameters at all
                ocr_engine = PaddleOCR()
        self._logger.info("PaddleOCR engine initialized successfully")
        return ocr_engine

    def _prepare_image_for_ocr(self, image_path: Path) -> Tuple[np.ndarray, bool]:
        """Load image and ensure it's not too large for OCR processing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (processed_image, was_downscaled)
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image '{image_path}'")
        
        height, width = image.shape[:2]
        max_dimension = self._settings.ocr_max_image_dimension
        
        self._logger.debug(
            "Image dimensions: %dx%d pixels (%.2f MP)",
            width, height, (width * height) / 1_000_000
        )
        
        # Дополнительная диагностика изображения
        self._logger.debug("Image color space: %s", "COLOR" if len(image.shape) == 3 else "GRAYSCALE")
        if len(image.shape) == 3:
            self._logger.debug("Image channels: %d", image.shape[2])
        self._logger.debug("Image dtype: %s", image.dtype)
        self._logger.debug("Image value range: min=%d, max=%d", image.min(), image.max())
        
        # Check if downscaling is needed
        if max(width, height) > max_dimension:
            # Calculate scale to fit within max_dimension
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            self._logger.warning(
                "Image too large for OCR (%dx%d). Downscaling to %dx%d (scale=%.3f) to prevent OOM/crash",
                width, height, new_width, new_height, scale
            )
            
            # Downscale using high-quality interpolation
            image = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA  # Best for downscaling
            )
            
            self._logger.info(
                "Image downscaled from %.2f MP to %.2f MP for OCR stability",
                (width * height) / 1_000_000,
                (new_width * new_height) / 1_000_000
            )
            
            return image, True
        
        return image, False

    def process(self, input_path: Path, output_path: Optional[Path] = None) -> OCRResult:
        """Run OCR processing on image and save results to JSON."""
        start_time = time.perf_counter()
        
        # Prepare image for OCR (with safety downscaling if needed)
        image, was_downscaled = self._prepare_image_for_ocr(input_path)
        self._logger.debug("Loaded image '%s' with shape %s", input_path, image.shape)
        
        if was_downscaled:
            # Save downscaled image temporarily for OCR
            temp_path = input_path.parent / f"{input_path.stem}_ocr_temp{input_path.suffix}"
            cv2.imwrite(str(temp_path), image)
            ocr_input_path = temp_path
        else:
            ocr_input_path = input_path
        
        # Perform OCR using pre-initialized engine
        self._logger.info("Starting OCR text recognition on image %dx%d...", 
                         image.shape[1], image.shape[0])
        
        try:
            self._logger.debug("Calling PaddleOCR.ocr()...")
            mem_before_ocr = self._memory_monitor.log_memory("before OCR", level="INFO")
            
            try:
                ocr_results = self._ocr_engine.ocr(str(ocr_input_path), cls=True)
            except TypeError as e:
                if 'cls' in str(e):
                    # Fallback to OCR without cls parameter
                    self._logger.debug("OCR cls parameter not supported, trying without it")
                    ocr_results = self._ocr_engine.ocr(str(ocr_input_path))
                else:
                    raise
            
            self._logger.debug("PaddleOCR.ocr() completed successfully")
            mem_after_ocr = self._memory_monitor.log_memory_delta(
                mem_before_ocr, 
                "after OCR"
            )
            
            # Диагностика структуры результатов OCR
            self._logger.debug("=" * 60)
            self._logger.debug("OCR RESULTS STRUCTURE ANALYSIS")
            self._logger.debug("=" * 60)
            self._logger.debug("ocr_results type: %s", type(ocr_results).__name__)
            self._logger.debug("ocr_results is None: %s", ocr_results is None)
            
            if ocr_results:
                self._logger.debug("ocr_results length: %d", len(ocr_results))
                
                for idx, page_result in enumerate(ocr_results):
                    self._logger.debug("  Page %d type: %s", idx, type(page_result).__name__)
                    self._logger.debug("  Page %d is None: %s", idx, page_result is None)
                    
                    if page_result:
                        if isinstance(page_result, list):
                            self._logger.debug("  Page %d length: %d items", idx, len(page_result))
                            if len(page_result) > 0:
                                self._logger.debug("    First item type: %s", type(page_result[0]).__name__)
                                self._logger.debug("    First item preview: %s", str(page_result[0])[:200])
                        elif hasattr(page_result, 'keys'):
                            self._logger.debug("  Page %d keys: %s", idx, list(page_result.keys())[:10])
                        elif hasattr(page_result, '__dict__'):
                            self._logger.debug("  Page %d attributes: %s", idx, list(vars(page_result).keys())[:10])
                    else:
                        self._logger.warning("  Page %d is empty/None", idx)
            
            self._logger.debug("=" * 60)
            
        except Exception as e:
            self._logger.error(
                "PaddleOCR.ocr() failed with exception: %s", 
                type(e).__name__, 
                exc_info=True
            )
            raise
        finally:
            # Clean up temporary file if created
            if was_downscaled and ocr_input_path.exists():
                try:
                    ocr_input_path.unlink()
                    self._logger.debug("Removed temporary OCR file: %s", ocr_input_path)
                except Exception as e:
                    self._logger.warning("Failed to remove temporary file %s: %s", 
                                        ocr_input_path, e)
        
        # Log OCR results summary
        self._logger.debug("OCR detected %d pages", len(ocr_results) if ocr_results else 0)
        
        # Process results
        text_detections = self._process_ocr_results(ocr_results)
        
        # Create structured output
        output_data = self._create_output_structure(input_path, text_detections, start_time)
        
        # Save to JSON
        destination = output_path or self._build_output_path(input_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        duration = time.perf_counter() - start_time
        
        # Calculate metrics
        avg_confidence = self._calculate_average_confidence(text_detections)
        low_confidence_count = sum(
            1 for detection in text_detections 
            if detection.confidence < self._settings.ocr_confidence_threshold * 1.2
        )
        
        self._logger.info(
            "OCR processing completed in %.3f seconds. Found %d text regions (avg confidence: %.3f)",
            duration, len(text_detections), avg_confidence
        )
        
        if low_confidence_count > 0:
            self._logger.warning(
                "%d text regions have low confidence (< %.2f)",
                low_confidence_count, self._settings.ocr_confidence_threshold * 1.2
            )
        
        # Force garbage collection after OCR
        gc.collect()
        self._memory_monitor.log_memory("after cleanup", level="DEBUG")
        
        return OCRResult(
            output_path=destination,
            duration_seconds=duration,
            total_texts_found=len(text_detections),
            average_confidence=avg_confidence,
            low_confidence_count=low_confidence_count
        )

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and validate image file."""
        if not path.exists():
            raise FileNotFoundError(f"Input image '{path}' does not exist")
        
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image '{path}'. Ensure it is a valid image file.")
        
        return image

    def _process_ocr_results(self, ocr_results: List[Any]) -> List[TextDetection]:
        """Convert PaddleOCR results to structured TextDetection objects."""
        text_detections = []
        
        self._logger.debug("=" * 60)
        self._logger.debug("PROCESSING OCR RESULTS")
        self._logger.debug("=" * 60)
        
        # Валидация входных данных
        if ocr_results is None:
            self._logger.warning("OCR results is None")
            return text_detections
        
        if not ocr_results:
            self._logger.warning("OCR results is empty list")
            return text_detections
        
        if len(ocr_results) == 0:
            self._logger.warning("OCR results has length 0")
            return text_detections
        
        self._logger.debug("OCR results has %d pages", len(ocr_results))
        
        # Проверка первой страницы
        ocr_result_obj = ocr_results[0]
        
        if ocr_result_obj is None:
            self._logger.warning("First page (ocr_results[0]) is None")
            return text_detections
        
        if not ocr_result_obj:
            self._logger.warning("First page (ocr_results[0]) is empty/falsy")
            return text_detections
        
        # Детальная диагностика первой страницы
        self._logger.debug("First page type: %s", type(ocr_result_obj).__name__)
        self._logger.debug("First page has 'json': %s", hasattr(ocr_result_obj, 'json'))
        self._logger.debug("First page has 'keys': %s", hasattr(ocr_result_obj, 'keys'))
        self._logger.debug("First page is list: %s", isinstance(ocr_result_obj, list))
        self._logger.debug("First page is dict: %s", isinstance(ocr_result_obj, dict))
        
        if isinstance(ocr_result_obj, list):
            self._logger.debug("First page list length: %d", len(ocr_result_obj))
            if len(ocr_result_obj) > 0:
                self._logger.debug("First item in list type: %s", type(ocr_result_obj[0]).__name__)
                self._logger.debug("First item preview: %s", str(ocr_result_obj[0])[:300])
        
        if hasattr(ocr_result_obj, 'keys'):
            available_keys = list(ocr_result_obj.keys())
            self._logger.debug("Available keys: %s", available_keys)
        
        # Check if this is a PaddleX OCRResult object (dict-like)
        if hasattr(ocr_result_obj, 'json') and hasattr(ocr_result_obj, 'keys'):
            self._logger.info("Processing PaddleX OCRResult object (dict-like)")
            try:
                # Extract OCR data from PaddleX result object
                
                # Extract the main OCR data
                rec_texts = ocr_result_obj.get('rec_texts', [])
                rec_scores = ocr_result_obj.get('rec_scores', [])
                rec_polys = ocr_result_obj.get('rec_polys', [])
                
                self._logger.info("Found %d texts, %d scores, %d polygons", 
                                len(rec_texts), len(rec_scores), len(rec_polys))
                
                # Process each text detection by combining the three lists
                min_length = min(len(rec_texts), len(rec_scores), len(rec_polys))
                for i in range(min_length):
                    try:
                        text = rec_texts[i] if i < len(rec_texts) else ""
                        confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                        poly = rec_polys[i] if i < len(rec_polys) else None
                        
                        # Filter by confidence threshold
                        if confidence < self._settings.ocr_confidence_threshold:
                            self._logger.debug("Skipping text with low confidence: '%s' (%.3f)", text, confidence)
                            continue
                        
                        # Convert numpy polygon to bbox format
                        bbox = [[0, 0], [100, 0], [100, 50], [0, 50]]  # Default
                        center_x = center_y = 0
                        
                        if poly is not None:
                            try:
                                # Convert numpy array to list of [x, y] points
                                if hasattr(poly, 'tolist'):
                                    points = poly.tolist()
                                else:
                                    points = list(poly)
                                
                                if len(points) >= 4:
                                    bbox = [[int(p[0]), int(p[1])] for p in points[:4]]
                                    # Calculate center
                                    center_x = int(sum(p[0] for p in points[:4]) / 4)
                                    center_y = int(sum(p[1] for p in points[:4]) / 4)
                                    
                            except Exception as e:
                                self._logger.debug("Failed to process polygon %d: %s", i, e)
                        
                        detection = TextDetection(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                            center_x=center_x,
                            center_y=center_y
                        )
                        
                        text_detections.append(detection)
                        
                        self._logger.debug(
                            "Detected text: '%s' (confidence: %.3f, center: %d,%d)",
                            text, confidence, center_x, center_y
                        )
                        
                    except Exception as e:
                        self._logger.warning("Failed to process text detection %d: %s", i, e)
                        continue
                        
            except Exception as e:
                self._logger.error("Failed to process PaddleX OCRResult: %s", e)
                
        else:
            # Fallback to original processing for standard PaddleOCR format
            self._logger.info("Processing standard PaddleOCR format")
            self._logger.debug("Items to process: %d", len(ocr_result_obj))
            
            processed_count = 0
            skipped_low_confidence = 0
            skipped_errors = 0
            
            for idx, line_result in enumerate(ocr_result_obj):
                try:
                    self._logger.debug("  Processing line %d/%d", idx + 1, len(ocr_result_obj))
                    self._logger.debug("    Line type: %s", type(line_result).__name__)
                    self._logger.debug("    Line length: %d", len(line_result) if hasattr(line_result, '__len__') else 0)
                    
                    if len(line_result) < 2:
                        self._logger.warning("    Line %d has insufficient data (length=%d)", idx, len(line_result))
                        skipped_errors += 1
                        continue
                        
                    bbox = line_result[0] 
                    text_info = line_result[1]
                    
                    self._logger.debug("    bbox type: %s, length: %d", type(bbox).__name__, len(bbox) if hasattr(bbox, '__len__') else 0)
                    self._logger.debug("    text_info type: %s, length: %d", type(text_info).__name__, len(text_info) if hasattr(text_info, '__len__') else 0)
                    
                    text = text_info[0] if text_info[0] else ""
                    confidence = float(text_info[1]) if text_info[1] else 0.0
                    
                    self._logger.debug("    text: '%s', confidence: %.3f", text[:50], confidence)
                    
                    if confidence < self._settings.ocr_confidence_threshold:
                        self._logger.debug("    SKIPPED: confidence %.3f < threshold %.3f", 
                                         confidence, self._settings.ocr_confidence_threshold)
                        skipped_low_confidence += 1
                        continue
                    
                    center_x = int(sum(point[0] for point in bbox) / 4)
                    center_y = int(sum(point[1] for point in bbox) / 4)
                    bbox_int = [[int(point[0]), int(point[1])] for point in bbox]
                    
                    detection = TextDetection(
                        text=text,
                        confidence=confidence,
                        bbox=bbox_int,
                        center_x=center_x,
                        center_y=center_y
                    )
                    
                    text_detections.append(detection)
                    processed_count += 1
                    
                    self._logger.debug("    ✓ ADDED to detections (total: %d)", processed_count)
                    
                except (IndexError, ValueError, TypeError) as e:
                    self._logger.error("    ✗ ERROR processing line %d: %s", idx, e, exc_info=True)
                    skipped_errors += 1
                    continue
            
            # Итоговая статистика
            self._logger.info("Standard format processing summary:")
            self._logger.info("  Total items: %d", len(ocr_result_obj))
            self._logger.info("  Successfully processed: %d", processed_count)
            self._logger.info("  Skipped (low confidence): %d", skipped_low_confidence)
            self._logger.info("  Skipped (errors): %d", skipped_errors)
            self._logger.info("  Final detections: %d", len(text_detections))
        
        return text_detections

    def _create_output_structure(self, input_path: Path, detections: List[TextDetection], start_time: float) -> Dict[str, Any]:
        """Create structured JSON output with metadata and text detections."""
        duration = time.perf_counter() - start_time
        
        return {
            "document_info": {
                "source_file": input_path.name,
                "processing_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ocr_engine": "PaddleOCR",
                "language": "ru"
            },
            "processing_metrics": {
                "total_time_ms": int(duration * 1000),
                "texts_detected": len(detections),
                "average_confidence": self._calculate_average_confidence(detections),
                "low_confidence_threshold": self._settings.ocr_confidence_threshold,
                "low_confidence_count": sum(
                    1 for d in detections 
                    if d.confidence < self._settings.ocr_confidence_threshold * 1.2
                )
            },
            "text_regions": [
                {
                    "text": detection.text,
                    "confidence": round(detection.confidence, 3),
                    "bbox": detection.bbox,
                    "center": [detection.center_x, detection.center_y],
                    "char_count": len(detection.text)
                }
                for detection in detections
            ]
        }

    def _calculate_average_confidence(self, detections: List[TextDetection]) -> float:
        """Calculate average confidence score across all detections."""
        if not detections:
            return 0.0
        return sum(detection.confidence for detection in detections) / len(detections)

    def _build_output_path(self, input_path: Path) -> Path:
        """Generate output path with -texts suffix."""
        output_dir = self._settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem = input_path.stem
        # Remove existing -cor suffix if present from preprocessing
        if stem.endswith('-cor'):
            stem = stem[:-4]
        
        filename = f"{stem}-texts.json"
        return output_dir / filename
