from __future__ import annotations

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
        # Initialize PaddleOCR with basic parameters - start simple
        ocr_params = {
            'use_angle_cls': True,  # Enable text angle classification
            'lang': 'ru',  # Russian language support
            'show_log': False,  # Suppress PaddleOCR internal logging
        }
        
        # Add GPU parameter only if enabled
        if self._settings.ocr_use_gpu:
            ocr_params['use_gpu'] = True
            
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

    def process(self, input_path: Path, output_path: Optional[Path] = None) -> OCRResult:
        """Run OCR processing on image and save results to JSON."""
        start_time = time.perf_counter()
        
        # Load and validate input image
        image = self._load_image(input_path)
        self._logger.debug("Loaded image '%s' with shape %s", input_path, image.shape)
        
        # Perform OCR using pre-initialized engine
        self._logger.info("Starting OCR text recognition...")
        try:
            ocr_results = self._ocr_engine.ocr(str(input_path), cls=True)
        except TypeError as e:
            if 'cls' in str(e):
                # Fallback to OCR without cls parameter
                self._logger.debug("OCR cls parameter not supported, trying without it")
                ocr_results = self._ocr_engine.ocr(str(input_path))
            else:
                raise
        
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
        
        if not ocr_results or not ocr_results[0]:
            self._logger.warning("No text detected in image")
            return text_detections
        
        # Handle PaddleX OCRResult object
        ocr_result_obj = ocr_results[0]
        
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
            for line_result in ocr_result_obj:
                try:
                    if len(line_result) < 2:
                        continue
                        
                    bbox = line_result[0] 
                    text_info = line_result[1]
                    text = text_info[0] if text_info[0] else ""
                    confidence = float(text_info[1]) if text_info[1] else 0.0
                    
                    if confidence < self._settings.ocr_confidence_threshold:
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
                    
                except (IndexError, ValueError, TypeError) as e:
                    self._logger.warning("Failed to process OCR result: %s", e)
                    continue
        
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
