"""Worker functions for parallel OCR processing using ProcessPoolExecutor."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR

from config.settings import Settings
from ocr_engine_factory import OCREngineFactory


@dataclass
class EngineParams:
    """Parameters for OCR engine initialization (serializable)."""

    ocr_language: str
    ocr_use_gpu: bool
    ocr_det_limit_side_len: int
    ocr_confidence_threshold: float
    ocr_max_image_dimension: int


@dataclass
class CellOCRTask:
    """Task data for cell OCR processing."""

    cell_image_bytes: bytes
    row_idx: int
    col_idx: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    field_name: Optional[str]


@dataclass
class CellOCRResult:
    """Result from cell OCR processing."""

    row_idx: int
    col_idx: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    text: str
    confidence: float
    field_name: Optional[str]


@dataclass
class RegionOCRTask:
    """Task data for region OCR processing."""

    region_image_bytes: bytes
    region_id: Optional[str]
    offset_x: int
    offset_y: int


@dataclass
class RegionOCRResult:
    """Result from region OCR processing."""

    detections: List[Dict[str, Any]]
    region_id: Optional[str]


# Global engine cache per worker process (initialized once per worker)
_worker_recognition_engine: Optional[PaddleOCR] = None
_worker_engine_params: Optional[EngineParams] = None
_worker_cells_processed: int = 0
_worker_memory_baseline_mb: float = 0.0
_worker_last_memory_check_mb: float = 0.0


def _image_to_bytes(image: np.ndarray) -> bytes:
    """Convert numpy image array to bytes for pickling.

    Args:
        image: Image array

    Returns:
        Serialized image bytes
    """
    is_success, buffer = cv2.imencode('.jpg', image)
    if not is_success:
        raise ValueError("Failed to encode image to bytes")
    return buffer.tobytes()


def _bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert bytes back to numpy image array.

    Args:
        image_bytes: Serialized image bytes

    Returns:
        Image array
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    return image


def _get_or_create_recognition_engine(engine_params: EngineParams) -> PaddleOCR:
    """Get or create recognition engine for worker process.

    Engines are cached per worker process to avoid re-initialization.
    Performs self-monitoring and restarts if memory exceeds threshold.

    Args:
        engine_params: Engine configuration parameters

    Returns:
        Recognition-only PaddleOCR engine
    """
    global _worker_recognition_engine, _worker_engine_params
    global _worker_cells_processed, _worker_memory_baseline_mb, _worker_last_memory_check_mb

    # Check memory and restart if needed (every 10 cells processed)
    if _worker_recognition_engine is not None and _worker_cells_processed > 0:
        if _worker_cells_processed % 10 == 0:  # Check every 10 cells
            try:
                import psutil
                import gc
                
                process = psutil.Process()
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                _worker_last_memory_check_mb = current_memory_mb
                
                # Use 80% of typical reload threshold (700 MB default) = 560 MB
                # Workers are lighter than full engines, so use lower threshold
                restart_threshold_mb = 560.0
                
                if current_memory_mb > restart_threshold_mb:
                    worker_logger = logging.getLogger(__name__)
                    worker_logger.warning(
                        "Worker memory (%.1f MB) exceeds threshold (%.1f MB). "
                        "Performing self-restart...",
                        current_memory_mb,
                        restart_threshold_mb,
                    )
                    
                    # Clear engine and run GC
                    _worker_recognition_engine = None
                    gc.collect()
                    
                    # Reset metrics
                    _worker_cells_processed = 0
                    _worker_memory_baseline_mb = process.memory_info().rss / 1024 / 1024
                    
                    worker_logger.info(
                        "Worker engine restarted. Memory after restart: %.1f MB",
                        _worker_memory_baseline_mb,
                    )
            except Exception as e:
                # If memory monitoring fails, log but continue
                logging.getLogger(__name__).debug(
                    "Worker memory check failed: %s", e
                )

    # Check if we need to create a new engine
    if (
        _worker_recognition_engine is None
        or _worker_engine_params != engine_params
    ):
        # Create temporary settings object for factory
        class TempSettings:
            ocr_language = engine_params.ocr_language
            ocr_use_gpu = engine_params.ocr_use_gpu
            ocr_det_limit_side_len = engine_params.ocr_det_limit_side_len

        temp_settings = TempSettings()
        temp_logger = logging.getLogger(__name__)

        # Suppress warnings during engine initialization in worker processes
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # Also suppress stderr output from PaddleOCR initialization
            import sys
            import os
            from io import StringIO
            
            # Temporarily redirect stderr to suppress PaddleOCR initialization messages
            old_stderr = sys.stderr
            try:
                sys.stderr = StringIO()
                _worker_recognition_engine = OCREngineFactory.create_recognition_engine(
                    temp_settings, temp_logger
                )
            finally:
                sys.stderr = old_stderr
        _worker_engine_params = engine_params
        
        # Track baseline memory after initialization
        try:
            import psutil
            process = psutil.Process()
            _worker_memory_baseline_mb = process.memory_info().rss / 1024 / 1024
            _worker_last_memory_check_mb = _worker_memory_baseline_mb
        except Exception:
            pass

    return _worker_recognition_engine


def recognize_cell_worker(
    task: CellOCRTask, engine_params: EngineParams
) -> CellOCRResult:
    """Worker function to recognize text in a single table cell.

    This function is called by ProcessPoolExecutor workers.
    Each worker initializes its own OCR engine.

    Args:
        task: Cell OCR task with image and metadata
        engine_params: Engine configuration parameters

    Returns:
        Cell OCR result with text and confidence
    """
    try:
        # Decode image from bytes
        cell_image = _bytes_to_image(task.cell_image_bytes)

        # Resize if needed
        height, width = cell_image.shape[:2]
        max_dimension = engine_params.ocr_max_image_dimension
        scale = 1.0

        if max(width, height) > max_dimension:
            scale = max_dimension / float(max(width, height))
            new_width = int(width * scale)
            new_height = int(height * scale)
            cell_image = cv2.resize(
                cell_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        # Get or create recognition engine
        rec_engine = _get_or_create_recognition_engine(engine_params)
        
        # Increment cells processed counter (for memory monitoring)
        global _worker_cells_processed
        _worker_cells_processed += 1

        # Run recognition
        # Note: Angle classification is controlled via use_angle_cls during initialization,
        # not via cls parameter in .ocr() method calls
        try:
            # Try new API with det=False
            try:
                ocr_results = rec_engine.ocr(cell_image, det=False)
            except TypeError:
                # Fallback if det parameter not supported - use full OCR
                ocr_results = rec_engine.ocr(cell_image)
        except Exception as err:
            # Log error and re-raise
            logging.getLogger(__name__).error(
                "OCR failed for cell: %s", str(err)[:200]
            )
            raise

        # Process results
        texts = []
        confidences = []

        if ocr_results and ocr_results[0]:
            for line_result in ocr_results[0]:
                if line_result and len(line_result) >= 2:
                    # Handle both formats:
                    # - New format (recognition only): [text, confidence]
                    # - Old format (full OCR): [[bbox], [text, confidence]]
                    if isinstance(line_result[0], str):
                        # Recognition-only format: [text, confidence]
                        text = line_result[0] if line_result[0] else ""
                        confidence = 0.0
                        if len(line_result) > 1 and line_result[1] is not None:
                            try:
                                confidence = float(line_result[1])
                            except (ValueError, TypeError):
                                # If confidence is not a number, try to parse it
                                logging.getLogger(__name__).debug(
                                    "Invalid confidence value: %s (type: %s), using 0.0",
                                    line_result[1],
                                    type(line_result[1]).__name__
                                )
                                confidence = 0.0
                    else:
                        # Full OCR format: [[bbox], [text, confidence]]
                        text_info = line_result[1] if len(line_result) > 1 else None
                        if text_info and len(text_info) >= 2:
                            text = text_info[0] if text_info[0] else ""
                            confidence = 0.0
                            if text_info[1] is not None:
                                try:
                                    confidence = float(text_info[1])
                                except (ValueError, TypeError):
                                    # If confidence is not a number, try to parse it
                                    logging.getLogger(__name__).debug(
                                        "Invalid confidence value: %s (type: %s), using 0.0",
                                        text_info[1],
                                        type(text_info[1]).__name__
                                    )
                                    confidence = 0.0
                        else:
                            continue

                    # Filter by confidence threshold
                    if confidence >= engine_params.ocr_confidence_threshold:
                        texts.append(text)
                        confidences.append(confidence)

        # Combine texts
        combined_text = " ".join(texts).strip()
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return CellOCRResult(
            row_idx=task.row_idx,
            col_idx=task.col_idx,
            x_start=task.x_start,
            y_start=task.y_start,
            x_end=task.x_end,
            y_end=task.y_end,
            text=combined_text,
            confidence=avg_confidence,
            field_name=task.field_name,
        )

    except Exception as e:
        # Return empty result on error
        logging.getLogger(__name__).warning(
            "Cell OCR worker failed for cell [%d,%d]: %s",
            task.row_idx,
            task.col_idx,
            e,
        )
        return CellOCRResult(
            row_idx=task.row_idx,
            col_idx=task.col_idx,
            x_start=task.x_start,
            y_start=task.y_start,
            x_end=task.x_end,
            y_end=task.y_end,
            text="",
            confidence=0.0,
            field_name=task.field_name,
        )


def process_region_worker_standalone(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone worker function for region processing (can be pickled).

    Args:
        task_data: Dictionary with region data, image bytes, scale, and offset

    Returns:
        Dictionary with region_id and detections list
    """
    from ocr_engine_factory import OCREngineFactory

    region = task_data["region"]
    image_bytes = task_data["image_bytes"]
    scale = task_data["scale"]
    offset_y = task_data["offset_y"]
    ocr_language = task_data["ocr_language"]
    ocr_use_gpu = task_data["ocr_use_gpu"]
    ocr_det_limit_side_len = task_data["ocr_det_limit_side_len"]
    ocr_confidence_threshold = task_data["ocr_confidence_threshold"]

    # Decode image
    region_image = _bytes_to_image(image_bytes)

    # Create full OCR engine in worker
    class TempSettings:
        ocr_language = ocr_language
        ocr_use_gpu = ocr_use_gpu
        ocr_det_limit_side_len = ocr_det_limit_side_len

    temp_settings = TempSettings()
    temp_logger = logging.getLogger(__name__)
    
    # Suppress warnings during engine initialization in worker processes
    import warnings
    import sys
    from io import StringIO
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Temporarily redirect stderr to suppress PaddleOCR initialization messages
        old_stderr = sys.stderr
        try:
            sys.stderr = StringIO()
            worker_engine = OCREngineFactory.create_full_engine(temp_settings, temp_logger)
        finally:
            sys.stderr = old_stderr

    # Run OCR
    # Note: Angle classification is controlled via use_angle_cls during initialization,
    # not via cls parameter in .ocr() method calls
    ocr_results = worker_engine.ocr(region_image)

    # Process results
    detections = []
    inv_scale = 1.0 / scale if scale > 0 else 1.0

    if ocr_results and ocr_results[0]:
        for line_result in ocr_results[0]:
            if len(line_result) >= 2:
                bbox = line_result[0]
                text_info = line_result[1]
                text = text_info[0] if text_info[0] else ""
                
                # Safe confidence parsing
                confidence = 0.0
                if text_info[1] is not None:
                    try:
                        confidence = float(text_info[1])
                    except (ValueError, TypeError):
                        logging.getLogger(__name__).debug(
                            "Invalid confidence value: %s (type: %s), using 0.0",
                            text_info[1],
                            type(text_info[1]).__name__
                        )
                        confidence = 0.0

                if confidence >= ocr_confidence_threshold:
                    local_points = [
                        [int(point[0] * inv_scale), int(point[1] * inv_scale)]
                        for point in bbox
                    ]
                    global_points = [
                        [p[0], p[1] + offset_y] for p in local_points
                    ]
                    center_local_x = int(sum(p[0] for p in local_points) / 4)
                    center_local_y = int(sum(p[1] for p in local_points) / 4)

                    detection = {
                        "text": text,
                        "confidence": confidence,
                        "bbox": global_points,
                        "center_x": center_local_x,
                        "center_y": center_local_y + offset_y,
                        "local_bbox": local_points,
                        "local_center": (center_local_x, center_local_y),
                        "region_id": region.region_id,
                    }
                    detections.append(detection)

    return {
        "region_id": region.region_id,
        "detections": detections,
    }


def recognize_region_worker(
    task: RegionOCRTask, engine_params: EngineParams
) -> RegionOCRResult:
    """Worker function to recognize text in a document region.

    This function is called by ProcessPoolExecutor workers.
    Each worker initializes its own OCR engine.

    Args:
        task: Region OCR task with image and metadata
        engine_params: Engine configuration parameters

    Returns:
        Region OCR result with text detections
    """
    try:
        # Decode image from bytes
        region_image = _bytes_to_image(task.region_image_bytes)

        # Resize if needed
        height, width = region_image.shape[:2]
        max_dimension = engine_params.ocr_max_image_dimension
        scale = 1.0

        if max(width, height) > max_dimension:
            scale = max_dimension / float(max(width, height))
            new_width = int(width * scale)
            new_height = int(height * scale)
            region_image = cv2.resize(
                region_image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        # Get or create recognition engine
        rec_engine = _get_or_create_recognition_engine(engine_params)
        
        # Increment cells processed counter (for memory monitoring)
        global _worker_cells_processed
        _worker_cells_processed += 1

        # Run recognition
        # Note: Angle classification is controlled via use_angle_cls during initialization,
        # not via cls parameter in .ocr() method calls
        try:
            # Try new API with det=False
            try:
                ocr_results = rec_engine.ocr(region_image, det=False)
            except TypeError:
                # Fallback if det parameter not supported - use full OCR
                ocr_results = rec_engine.ocr(region_image)
        except Exception as err:
            # Log error and re-raise
            logging.getLogger(__name__).error(
                "OCR failed for region: %s", str(err)[:200]
            )
            raise

        # Process results into detection format
        detections = []
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        if ocr_results and ocr_results[0]:
            for line_result in ocr_results[0]:
                if line_result and len(line_result) >= 2:
                    text = line_result[0] if line_result[0] else ""
                    confidence = float(line_result[1]) if line_result[1] else 0.0

                    # Filter by confidence threshold
                    if confidence >= engine_params.ocr_confidence_threshold:
                        # Create detection dict (simplified, as we don't have bbox from rec-only)
                        detection = {
                            "text": text,
                            "confidence": confidence,
                            "region_id": task.region_id,
                        }
                        detections.append(detection)

        return RegionOCRResult(
            detections=detections,
            region_id=task.region_id,
        )

    except Exception as e:
        # Return empty result on error
        logging.getLogger(__name__).warning(
            "Region OCR worker failed for region '%s': %s",
            task.region_id,
            e,
        )
        return RegionOCRResult(
            detections=[],
            region_id=task.region_id,
        )


def create_engine_params(settings: Settings) -> EngineParams:
    """Create EngineParams from Settings object.

    Args:
        settings: Application settings

    Returns:
        Serializable engine parameters
    """
    return EngineParams(
        ocr_language=settings.ocr_language,
        ocr_use_gpu=settings.ocr_use_gpu,
        ocr_det_limit_side_len=settings.ocr_det_limit_side_len,
        ocr_confidence_threshold=settings.ocr_confidence_threshold,
        ocr_max_image_dimension=settings.ocr_max_image_dimension,
    )

