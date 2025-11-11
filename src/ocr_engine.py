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
from region_detector import DocumentRegion, RegionDetector
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
    region_id: Optional[str] = None
    local_bbox: Optional[List[List[int]]] = None
    local_center: Optional[Tuple[int, int]] = None


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

    def _log_image_properties(self, image: np.ndarray) -> None:
        """Log diagnostic information about the image."""
        height, width = image.shape[:2]
        self._logger.debug(
            "Image dimensions: %dx%d pixels (%.2f MP)",
            width,
            height,
            (width * height) / 1_000_000,
        )
        self._logger.debug(
            "Image color space: %s", "COLOR" if image.ndim == 3 else "GRAYSCALE"
        )
        if image.ndim == 3:
            self._logger.debug("Image channels: %d", image.shape[2])
        self._logger.debug("Image dtype: %s", image.dtype)
        self._logger.debug("Image value range: min=%d, max=%d", image.min(), image.max())

    def _resize_for_ocr(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize image if it exceeds OCR safety limits."""
        height, width = image.shape[:2]
        max_dimension = self._settings.ocr_max_image_dimension
        if max(width, height) <= max_dimension:
            return image, 1.0

        scale = max_dimension / float(max(width, height))
        new_width = int(width * scale)
        new_height = int(height * scale)

        self._logger.warning(
            "Image too large for OCR (%dx%d). Downscaling to %dx%d (scale=%.3f) to prevent OOM/crash",
            width,
            height,
            new_width,
            new_height,
            scale,
        )

        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        self._logger.info(
            "Image downscaled from %.2f MP to %.2f MP for OCR stability",
            (width * height) / 1_000_000,
            (new_width * new_height) / 1_000_000,
        )
        return resized, scale

    def _run_paddle_ocr(self, image: np.ndarray) -> List[Any]:
        """Execute PaddleOCR on the provided image array."""
        self._logger.info(
            "Starting OCR text recognition on image %dx%d...",
            image.shape[1],
            image.shape[0],
        )
        mem_before_ocr = self._memory_monitor.log_memory("before OCR", level="INFO")

        try:
            self._logger.debug("Calling PaddleOCR.ocr()...")
            try:
                ocr_results = self._ocr_engine.ocr(image, cls=True)
            except TypeError as err:
                if "cls" in str(err):
                    self._logger.debug(
                        "OCR cls parameter not supported, trying without it"
                    )
                    ocr_results = self._ocr_engine.ocr(image)
                else:
                    raise
            self._logger.debug("PaddleOCR.ocr() completed successfully")
        except Exception as exc:  # pragma: no cover - passthrough for logging
            self._logger.error(
                "PaddleOCR.ocr() failed with exception: %s",
                type(exc).__name__,
                exc_info=True,
            )
            raise
        finally:
            self._memory_monitor.log_memory_delta(mem_before_ocr, "after OCR")

        return ocr_results

    def _log_ocr_structure(self, ocr_results: List[Any]) -> None:
        """Log diagnostic information about PaddleOCR output."""
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
                        self._logger.debug(
                            "  Page %d length: %d items", idx, len(page_result)
                        )
                        if len(page_result) > 0:
                            self._logger.debug(
                                "    First item type: %s",
                                type(page_result[0]).__name__,
                            )
                            self._logger.debug(
                                "    First item preview: %s",
                                str(page_result[0])[:200],
                            )
                    elif hasattr(page_result, "keys"):
                        self._logger.debug(
                            "  Page %d keys: %s",
                            list(page_result.keys())[:10],
                        )
                    elif hasattr(page_result, "__dict__"):
                        self._logger.debug(
                            "  Page %d attributes: %s",
                            list(vars(page_result).keys())[:10],
                        )
                else:
                    self._logger.warning("  Page %d is empty/None", idx)

        self._logger.debug("=" * 60)

    def process(self, input_path: Path, output_path: Optional[Path] = None) -> OCRResult:
        """Run OCR processing on image and save results to JSON."""
        start_time = time.perf_counter()
        
        original_image = self._load_image(input_path)
        self._log_image_properties(original_image)
        prepared_image, scale_factor = self._resize_for_ocr(original_image)

        ocr_results = self._run_paddle_ocr(prepared_image)
        self._log_ocr_structure(ocr_results)

        self._logger.debug(
            "OCR detected %d pages", len(ocr_results) if ocr_results else 0
        )

        text_detections = self._process_ocr_results(
            ocr_results, scale=scale_factor if scale_factor > 0 else 1.0
        )

        output_data = self._create_output_structure(
            input_path, text_detections, start_time
        )

        destination = output_path or self._build_output_path(input_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        duration = time.perf_counter() - start_time

        avg_confidence = self._calculate_average_confidence(text_detections)
        low_confidence_count = sum(
            1
            for detection in text_detections
            if detection.confidence < self._settings.ocr_confidence_threshold * 1.2
        )

        self._logger.info(
            "OCR processing completed in %.3f seconds. Found %d text regions (avg confidence: %.3f)",
            duration,
            len(text_detections),
            avg_confidence,
        )

        if low_confidence_count > 0:
            self._logger.warning(
                "%d text regions have low confidence (< %.2f)",
                low_confidence_count,
                self._settings.ocr_confidence_threshold * 1.2,
            )

        gc.collect()
        self._memory_monitor.log_memory("after cleanup", level="DEBUG")

        return OCRResult(
            output_path=destination,
            duration_seconds=duration,
            total_texts_found=len(text_detections),
            average_confidence=avg_confidence,
            low_confidence_count=low_confidence_count,
        )

    def process_regions(
        self,
        image_path: Path,
        regions: List[DocumentRegion],
        output_path: Optional[Path] = None,
    ) -> OCRResult:
        """Run OCR for each detected region and aggregate results."""
        if not regions:
            raise ValueError("Region list cannot be empty for process_regions()")

        start_time = time.perf_counter()
        base_image = self._load_image(image_path)
        self._log_image_properties(base_image)

        region_detector = RegionDetector(settings=self._settings, logger=self._logger)

        aggregated_detections: List[TextDetection] = []
        detections_by_region: Dict[str, List[TextDetection]] = {}

        for region in regions:
            self._logger.info(
                "Processing region '%s' (y: %d-%d) detected via %s",
                region.region_id,
                region.y_start,
                region.y_end,
                region.detection_method,
            )
            region_image, region_scale = region_detector.extract_region(
                base_image, region
            )
            prepared_region, ocr_scale = self._resize_for_ocr(region_image)
            combined_scale = region_scale * ocr_scale

            ocr_results = self._run_paddle_ocr(prepared_region)
            self._log_ocr_structure(ocr_results)

            region_detections = self._process_ocr_results(
                ocr_results,
                scale=combined_scale if combined_scale > 0 else 1.0,
                offset_x=0,
                offset_y=region.y_start,
                region_id=region.region_id,
            )

            aggregated_detections.extend(region_detections)
            detections_by_region[region.region_id] = region_detections
            self._logger.info(
                "Region '%s' yielded %d detections (avg confidence: %.3f)",
                region.region_id,
                len(region_detections),
                self._calculate_average_confidence(region_detections),
            )

        output_data = self._create_output_structure(
            image_path,
            aggregated_detections,
            start_time,
            regions=regions,
            grouped_detections=detections_by_region,
        )

        destination = output_path or self._build_output_path(image_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        duration = time.perf_counter() - start_time
        avg_confidence = self._calculate_average_confidence(aggregated_detections)
        low_confidence_count = sum(
            1
            for detection in aggregated_detections
            if detection.confidence < self._settings.ocr_confidence_threshold * 1.2
        )

        self._logger.info(
            "Regional OCR processing completed in %.3f seconds. Total detections: %d (avg confidence: %.3f)",
            duration,
            len(aggregated_detections),
            avg_confidence,
        )

        if low_confidence_count > 0:
            self._logger.warning(
                "%d detections across regions have low confidence (< %.2f)",
                low_confidence_count,
                self._settings.ocr_confidence_threshold * 1.2,
            )

        gc.collect()
        self._memory_monitor.log_memory("after cleanup", level="DEBUG")

        return OCRResult(
            output_path=destination,
            duration_seconds=duration,
            total_texts_found=len(aggregated_detections),
            average_confidence=avg_confidence,
            low_confidence_count=low_confidence_count,
        )

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and validate image file."""
        if not path.exists():
            raise FileNotFoundError(f"Input image '{path}' does not exist")
        
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image '{path}'. Ensure it is a valid image file.")
        
        return image

    def _process_ocr_results(
        self,
        ocr_results: List[Any],
        *,
        scale: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
        region_id: Optional[str] = None,
    ) -> List[TextDetection]:
        """Convert PaddleOCR results to structured TextDetection objects."""
        text_detections = []
        inv_scale = 1.0 / scale if scale not in (0.0, 0) else 1.0
        
        self._logger.debug("=" * 60)
        self._logger.debug("PROCESSING OCR RESULTS")
        self._logger.debug("=" * 60)
        self._logger.debug(
            "Processing with scale=%.3f (inv_scale=%.3f), offsets x=%d, y=%d, region=%s",
            scale,
            inv_scale,
            offset_x,
            offset_y,
            region_id,
        )
        
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
                        local_bbox = [[0, 0], [100, 0], [100, 50], [0, 50]]
                        center_local_x = center_local_y = 0
                        
                        if poly is not None:
                            try:
                                # Convert numpy array to list of [x, y] points
                                if hasattr(poly, 'tolist'):
                                    points = poly.tolist()
                                else:
                                    points = list(poly)
                                
                                if len(points) >= 4:
                                    local_bbox = [
                                        [
                                            int(p[0] * inv_scale),
                                            int(p[1] * inv_scale),
                                        ]
                                        for p in points[:4]
                                    ]
                                    center_local_x = int(
                                        sum(point[0] for point in local_bbox) / 4
                                    )
                                    center_local_y = int(
                                        sum(point[1] for point in local_bbox) / 4
                                    )
                                    
                            except Exception as e:
                                self._logger.debug("Failed to process polygon %d: %s", i, e)
                        
                        global_bbox = [
                            [
                                local_point[0] + offset_x,
                                local_point[1] + offset_y,
                            ]
                            for local_point in local_bbox
                        ]
                        detection = TextDetection(
                            text=text,
                            confidence=confidence,
                            bbox=global_bbox,
                            center_x=center_local_x + offset_x,
                            center_y=center_local_y + offset_y,
                            region_id=region_id,
                            local_bbox=local_bbox,
                            local_center=(center_local_x, center_local_y),
                        )
                        
                        text_detections.append(detection)
                        
                        self._logger.debug(
                            "Detected text: '%s' (confidence: %.3f, center: %d,%d)",
                            text,
                            confidence,
                            detection.center_x,
                            detection.center_y,
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
                    
                    local_points = [
                        [
                            int(point[0] * inv_scale),
                            int(point[1] * inv_scale),
                        ]
                        for point in bbox
                    ]
                    global_points = [
                        [
                            local_point[0] + offset_x,
                            local_point[1] + offset_y,
                        ]
                        for local_point in local_points
                    ]
                    center_local_x = int(sum(point[0] for point in local_points) / 4)
                    center_local_y = int(sum(point[1] for point in local_points) / 4)
                    center_x = center_local_x + offset_x
                    center_y = center_local_y + offset_y
                    
                    detection = TextDetection(
                        text=text,
                        confidence=confidence,
                        bbox=global_points,
                        center_x=center_x,
                        center_y=center_y,
                        region_id=region_id,
                        local_bbox=local_points,
                        local_center=(center_local_x, center_local_y),
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

    def _create_output_structure(
        self,
        input_path: Path,
        detections: List[TextDetection],
        start_time: float,
        regions: Optional[List[DocumentRegion]] = None,
        grouped_detections: Optional[Dict[str, List[TextDetection]]] = None,
    ) -> Dict[str, Any]:
        """Create structured JSON output with metadata and text detections."""
        duration = time.perf_counter() - start_time

        def _detection_to_dict(detection: TextDetection) -> Dict[str, Any]:
            data: Dict[str, Any] = {
                "text": detection.text,
                "confidence": round(detection.confidence, 3),
                "bbox": detection.bbox,
                "center": [detection.center_x, detection.center_y],
                "char_count": len(detection.text),
            }
            if detection.region_id:
                data["region_id"] = detection.region_id
            if detection.local_bbox:
                data["bbox_region"] = detection.local_bbox
            if detection.local_center:
                data["center_region"] = list(detection.local_center)
            return data

        output: Dict[str, Any] = {
            "document_info": {
                "source_file": input_path.name,
                "processing_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ocr_engine": "PaddleOCR",
                "language": "ru",
            },
            "processing_metrics": {
                "total_time_ms": int(duration * 1000),
                "texts_detected": len(detections),
                "average_confidence": self._calculate_average_confidence(detections),
                "low_confidence_threshold": self._settings.ocr_confidence_threshold,
                "low_confidence_count": sum(
                    1
                    for d in detections
                    if d.confidence < self._settings.ocr_confidence_threshold * 1.2
                ),
            },
            "text_regions": [_detection_to_dict(detection) for detection in detections],
        }

        if regions:
            output["regions_detected"] = [
                {
                    "region_id": region.region_id,
                    "y_start_norm": region.y_start_norm,
                    "y_end_norm": region.y_end_norm,
                    "y_start": region.y_start,
                    "y_end": region.y_end,
                    "detection_method": region.detection_method,
                    "confidence": round(region.confidence, 3),
                }
                for region in regions
            ]

        if grouped_detections:
            grouped_output: Dict[str, List[Dict[str, Any]]] = {}
            for region_key, region_detections in grouped_detections.items():
                grouped_output[region_key] = [
                    _detection_to_dict(detection) for detection in region_detections
                ]
            output["ocr_results_by_region"] = grouped_output

        return output

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
