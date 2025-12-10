from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
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
from ocr_engine_factory import OCREngineFactory
from parallel_ocr_worker import create_engine_params
from region_detector import DocumentRegion, RegionDetector
from table_detector import TableDetector
from table_processor import TableProcessor
from utils.json_utils import convert_numpy_types
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

    def __init__(
        self, settings: Settings, logger: logging.Logger, engine_mode: str = "full"
    ) -> None:
        """Initialize OCR engine.

        Args:
            settings: Application settings
            logger: Logger instance
            engine_mode: Engine type - 'full', 'detection', or 'recognition'
        """
        self._settings = settings
        self._logger = logger
        self._memory_monitor = MemoryMonitor(logger)
        self._engine_mode = engine_mode
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
        if self._engine_mode == "detection":
            return OCREngineFactory.create_detection_engine(self._settings, self._logger)
        elif self._engine_mode == "recognition":
            return OCREngineFactory.create_recognition_engine(self._settings, self._logger)
        else:
            return OCREngineFactory.create_full_engine(self._settings, self._logger)

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

        height, width = original_image.shape[:2]
        output_data = self._create_output_structure(
            input_path, text_detections, start_time, image_width=width, image_height=height
        )

        destination = output_path or self._build_output_path(input_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)

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

        # Choose parallel or sequential processing
        if self._should_use_parallel_regions_processing(regions):
            return self._process_regions_parallel(image_path, regions, output_path)
        else:
            return self._process_regions_sequential(image_path, regions, output_path)

    def _should_use_parallel_regions_processing(self, regions: List[DocumentRegion]) -> bool:
        """Check if parallel processing should be used for regions.

        Args:
            regions: List of document regions

        Returns:
            True if parallel processing should be used
        """
        if not self._settings.enable_parallel_processing:
            return False
        return len(regions) >= self._settings.parallel_min_regions_for_parallelization

    def _process_regions_sequential(
        self,
        image_path: Path,
        regions: List[DocumentRegion],
        output_path: Optional[Path] = None,
    ) -> OCRResult:
        """Run OCR for each detected region sequentially (original implementation)."""
        if not regions:
            raise ValueError("Region list cannot be empty for process_regions()")

        start_time = time.perf_counter()
        base_image = self._load_image(image_path)
        self._log_image_properties(base_image)

        region_detector = RegionDetector(settings=self._settings, logger=self._logger)

        aggregated_detections: List[TextDetection] = []
        detections_by_region: Dict[str, List[TextDetection]] = {}
        table_data_by_region: Dict[str, Dict[str, Any]] = {}

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

            # Check if table detection is enabled for defects zone
            if (
                region.region_id == "defects"
                and self._settings.enable_table_detection
            ):
                # Try table detection
                table_detector = TableDetector(self._settings, self._logger)
                grid = table_detector.detect_structure(region_image)

                if grid and self._validate_table_grid(grid):
                    # Process as table with cell-by-cell OCR
                    self._logger.info(
                        "Table structure detected in defects zone: %d rows × %d columns",
                        grid.num_rows,
                        grid.num_cols,
                    )
                    table_processor = TableProcessor(
                        self._settings, self._logger, self
                    )
                    cells = table_processor.extract_cells(region_image, grid)

                    if cells:
                        # Store structured table data
                        table_data_by_region[region.region_id] = {
                            "type": "table",
                            "cells": [
                                {
                                    "row_idx": c.row_idx,
                                    "col_idx": c.col_idx,
                                    "x_start": c.x_start,
                                    "y_start": c.y_start + region.y_start,  # Adjust to full image coordinates
                                    "x_end": c.x_end,
                                    "y_end": c.y_end + region.y_start,
                                    "text": c.text,
                                    "confidence": c.confidence,
                                    "field_name": c.field_name,
                                }
                                for c in cells
                            ],
                            "grid": {
                                "rows": grid.rows,
                                "cols": grid.cols,
                                "num_rows": grid.num_rows,
                                "num_cols": grid.num_cols,
                                "confidence": grid.confidence,
                            },
                        }

                        # Also create flat detections for backward compatibility
                        # Convert cells to TextDetection format
                        region_detections = []
                        for cell in cells:
                            # Create bbox from cell coordinates
                            bbox = [
                                [cell.x_start, cell.y_start],
                                [cell.x_end, cell.y_start],
                                [cell.x_end, cell.y_end],
                                [cell.x_start, cell.y_end],
                            ]
                            # Adjust to full image coordinates
                            global_bbox = [
                                [p[0], p[1] + region.y_start] for p in bbox
                            ]
                            detection = TextDetection(
                                text=cell.text,
                                confidence=cell.confidence,
                                bbox=global_bbox,
                                center_x=(cell.x_start + cell.x_end) // 2,
                                center_y=(cell.y_start + cell.y_end) // 2
                                + region.y_start,
                                region_id=region.region_id,
                                local_bbox=bbox,
                                local_center=(
                                    (cell.x_start + cell.x_end) // 2,
                                    (cell.y_start + cell.y_end) // 2,
                                ),
                            )
                            region_detections.append(detection)

                        aggregated_detections.extend(region_detections)
                        detections_by_region[region.region_id] = region_detections
                        self._logger.info(
                            "Table processing for region '%s' yielded %d cells (avg confidence: %.3f)",
                            region.region_id,
                            len(cells),
                            self._calculate_average_confidence(region_detections),
                        )
                        continue

            # Standard OCR for non-table regions or fallback
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

        height, width = base_image.shape[:2]
        output_data = self._create_output_structure(
            image_path,
            aggregated_detections,
            start_time,
            image_width=width,
            image_height=height,
            regions=regions,
            grouped_detections=detections_by_region,
            table_data=table_data_by_region,
        )

        destination = output_path or self._build_output_path(image_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)

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

    def _process_regions_parallel(
        self,
        image_path: Path,
        regions: List[DocumentRegion],
        output_path: Optional[Path] = None,
    ) -> OCRResult:
        """Run OCR for regions in parallel using ProcessPoolExecutor."""
        if not regions:
            raise ValueError("Region list cannot be empty for process_regions()")

        start_time = time.perf_counter()
        base_image = self._load_image(image_path)
        self._log_image_properties(base_image)

        region_detector = RegionDetector(settings=self._settings, logger=self._logger)

        aggregated_detections: List[TextDetection] = []
        detections_by_region: Dict[str, List[TextDetection]] = {}
        table_data_by_region: Dict[str, Dict[str, Any]] = {}

        # Separate table and non-table regions
        table_regions: List[DocumentRegion] = []
        non_table_regions: List[DocumentRegion] = []

        for region in regions:
            if (
                region.region_id == "defects"
                and self._settings.enable_table_detection
            ):
                table_regions.append(region)
            else:
                non_table_regions.append(region)

        # Process table regions sequentially (they use parallel cell processing internally)
        for region in table_regions:
            self._logger.info(
                "Processing table region '%s' (y: %d-%d) detected via %s",
                region.region_id,
                region.y_start,
                region.y_end,
                region.detection_method,
            )
            region_image, region_scale = region_detector.extract_region(
                base_image, region
            )

            # Try table detection
            table_detector = TableDetector(self._settings, self._logger)
            grid = table_detector.detect_structure(region_image)

            if grid and self._validate_table_grid(grid):
                # Process as table with cell-by-cell OCR (uses parallel processing internally)
                self._logger.info(
                    "Table structure detected in defects zone: %d rows × %d columns",
                    grid.num_rows,
                    grid.num_cols,
                )
                table_processor = TableProcessor(
                    self._settings, self._logger, self
                )
                cells = table_processor.extract_cells(region_image, grid)

                if cells:
                    # Store structured table data
                    table_data_by_region[region.region_id] = {
                        "type": "table",
                        "cells": [
                            {
                                "row_idx": c.row_idx,
                                "col_idx": c.col_idx,
                                "x_start": c.x_start,
                                "y_start": c.y_start + region.y_start,
                                "x_end": c.x_end,
                                "y_end": c.y_end + region.y_start,
                                "text": c.text,
                                "confidence": c.confidence,
                                "field_name": c.field_name,
                            }
                            for c in cells
                        ],
                        "grid": {
                            "rows": grid.rows,
                            "cols": grid.cols,
                            "num_rows": grid.num_rows,
                            "num_cols": grid.num_cols,
                            "confidence": grid.confidence,
                        },
                    }

                    # Convert cells to TextDetection format
                    region_detections = []
                    for cell in cells:
                        bbox = [
                            [cell.x_start, cell.y_start],
                            [cell.x_end, cell.y_start],
                            [cell.x_end, cell.y_end],
                            [cell.x_start, cell.y_end],
                        ]
                        global_bbox = [
                            [p[0], p[1] + region.y_start] for p in bbox
                        ]
                        detection = TextDetection(
                            text=cell.text,
                            confidence=cell.confidence,
                            bbox=global_bbox,
                            center_x=(cell.x_start + cell.x_end) // 2,
                            center_y=(cell.y_start + cell.y_end) // 2
                            + region.y_start,
                            region_id=region.region_id,
                            local_bbox=bbox,
                            local_center=(
                                (cell.x_start + cell.x_end) // 2,
                                (cell.y_start + cell.y_end) // 2,
                            ),
                        )
                        region_detections.append(detection)

                    aggregated_detections.extend(region_detections)
                    detections_by_region[region.region_id] = region_detections
                    self._logger.info(
                        "Table processing for region '%s' yielded %d cells (avg confidence: %.3f)",
                        region.region_id,
                        len(cells),
                        self._calculate_average_confidence(region_detections),
                    )

        # Process non-table regions in parallel
        if non_table_regions:
            if self._settings.parallel_use_separate_engines:
                # Use detection + recognition split approach
                self._logger.info(
                    "Processing %d non-table regions in parallel (det+rec split, %d workers)",
                    len(non_table_regions),
                    self._settings.parallel_regions_workers,
                )
                self._process_regions_parallel_det_rec(
                    base_image,
                    non_table_regions,
                    region_detector,
                    aggregated_detections,
                    detections_by_region,
                )
            else:
                # Use simple parallel approach (full engines in workers)
                self._logger.info(
                    "Processing %d non-table regions in parallel (full engines, %d workers)",
                    len(non_table_regions),
                    self._settings.parallel_regions_workers,
                )
                self._process_regions_parallel_simple(
                    base_image,
                    non_table_regions,
                    region_detector,
                    aggregated_detections,
                    detections_by_region,
                )

        height, width = base_image.shape[:2]
        output_data = self._create_output_structure(
            image_path,
            aggregated_detections,
            start_time,
            image_width=width,
            image_height=height,
            regions=regions,
            grouped_detections=detections_by_region,
            table_data=table_data_by_region,
        )

        destination = output_path or self._build_output_path(image_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)

        duration = time.perf_counter() - start_time
        avg_confidence = self._calculate_average_confidence(aggregated_detections)
        low_confidence_count = sum(
            1
            for detection in aggregated_detections
            if detection.confidence < self._settings.ocr_confidence_threshold * 1.2
        )

        self._logger.info(
            "Regional OCR processing completed in %.3f seconds (parallel). Total detections: %d (avg confidence: %.3f)",
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

    def _process_regions_parallel_simple(
        self,
        base_image: np.ndarray,
        regions: List[DocumentRegion],
        region_detector: RegionDetector,
        aggregated_detections: List[TextDetection],
        detections_by_region: Dict[str, List[TextDetection]],
    ) -> None:
        """Process regions in parallel using full OCR engines in workers."""
        from parallel_ocr_worker import _image_to_bytes

        # Prepare region tasks
        region_tasks = []
        for region in regions:
            region_image, region_scale = region_detector.extract_region(
                base_image, region
            )
            prepared_region, ocr_scale = self._resize_for_ocr(region_image)
            combined_scale = region_scale * ocr_scale

            # Convert to bytes for pickling
            region_image_bytes = _image_to_bytes(prepared_region)

            region_tasks.append({
                "region": region,
                "image_bytes": region_image_bytes,
                "scale": combined_scale if combined_scale > 0 else 1.0,
                "offset_y": region.y_start,
            })

        # Process in parallel
        try:
            from parallel_ocr_worker import process_region_worker_standalone

            # Prepare task data with all necessary settings (for pickling)
            prepared_tasks = []
            for task in region_tasks:
                prepared_task = {
                    "region": task["region"],
                    "image_bytes": task["image_bytes"],
                    "scale": task["scale"],
                    "offset_y": task["offset_y"],
                    "ocr_language": self._settings.ocr_language,
                    "ocr_use_gpu": self._settings.ocr_use_gpu,
                    "ocr_det_limit_side_len": self._settings.ocr_det_limit_side_len,
                    "ocr_confidence_threshold": self._settings.ocr_confidence_threshold,
                }
                prepared_tasks.append(prepared_task)

            with ProcessPoolExecutor(
                max_workers=self._settings.parallel_regions_workers
            ) as executor:
                futures = {
                    executor.submit(process_region_worker_standalone, task): task
                    for task in prepared_tasks
                }

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        region_detections = []
                        for det_dict in result["detections"]:
                            detection = TextDetection(
                                text=det_dict["text"],
                                confidence=det_dict["confidence"],
                                bbox=det_dict["bbox"],
                                center_x=det_dict["center_x"],
                                center_y=det_dict["center_y"],
                                region_id=det_dict["region_id"],
                                local_bbox=det_dict["local_bbox"],
                                local_center=det_dict["local_center"],
                            )
                            region_detections.append(detection)

                        aggregated_detections.extend(region_detections)
                        detections_by_region[result["region_id"]] = region_detections
                        self._logger.info(
                            "Region '%s' yielded %d detections (avg confidence: %.3f)",
                            result["region_id"],
                            len(region_detections),
                            self._calculate_average_confidence(region_detections),
                        )
                    except Exception as e:
                        self._logger.error(
                            "Failed to process region '%s': %s",
                            task["region"].region_id,
                            e,
                        )

        except Exception as e:
            self._logger.error(
                "Parallel region processing failed, falling back to sequential: %s", e
            )
            # Fallback to sequential for remaining regions
            for region in regions:
                region_image, region_scale = region_detector.extract_region(
                    base_image, region
                )
                prepared_region, ocr_scale = self._resize_for_ocr(region_image)
                combined_scale = region_scale * ocr_scale

                ocr_results = self._run_paddle_ocr(prepared_region)
                region_detections = self._process_ocr_results(
                    ocr_results,
                    scale=combined_scale if combined_scale > 0 else 1.0,
                    offset_x=0,
                    offset_y=region.y_start,
                    region_id=region.region_id,
                )

                aggregated_detections.extend(region_detections)
                detections_by_region[region.region_id] = region_detections

    def _process_regions_parallel_det_rec(
        self,
        base_image: np.ndarray,
        regions: List[DocumentRegion],
        region_detector: RegionDetector,
        aggregated_detections: List[TextDetection],
        detections_by_region: Dict[str, List[TextDetection]],
    ) -> None:
        """Process regions using detection+recognition split (recommended approach).

        First, detect text blocks in each region using detection engine.
        Then, extract crops and recognize them in parallel using recognition-only engines.
        """
        # For now, fall back to simple parallel (det+rec split is more complex)
        # This can be enhanced in the future
        self._logger.info(
            "Det+rec split approach not yet fully implemented, using simple parallel"
        )
        self._process_regions_parallel_simple(
            base_image,
            regions,
            region_detector,
            aggregated_detections,
            detections_by_region,
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
        image_width: int,
        image_height: int,
        regions: Optional[List[DocumentRegion]] = None,
        grouped_detections: Optional[Dict[str, List[TextDetection]]] = None,
        table_data: Optional[Dict[str, Dict[str, Any]]] = None,
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
                "image_width": image_width,
                "image_height": image_height,
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

        # Add table data if available
        if table_data:
            output["table_data_by_region"] = table_data

        return output

    def _calculate_average_confidence(self, detections: List[TextDetection]) -> float:
        """Calculate average confidence score across all detections."""
        if not detections:
            return 0.0
        return sum(detection.confidence for detection in detections) / len(detections)

    def _validate_table_grid(self, grid) -> bool:
        """Validate table grid before processing.

        Args:
            grid: TableGrid object

        Returns:
            True if grid is valid for processing
        """
        if grid is None:
            return False
        if grid.num_rows < 2 or grid.num_cols < 2:
            self._logger.debug(
                "Table grid validation failed: insufficient rows/cols (%d rows, %d cols)",
                grid.num_rows,
                grid.num_cols,
            )
            return False
        if grid.confidence < 0.3:
            self._logger.debug(
                "Table grid validation failed: low confidence (%.2f)", grid.confidence
            )
            return False
        return True

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
