"""Test fixtures for header field extraction."""

from typing import Any, Dict, List


def create_clean_ocr_detections() -> List[Dict[str, Any]]:
    """Create clean OCR detections (baseline - good recognition).

    Returns:
        List of detection dictionaries with clean recognition
    """
    return [
        {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],  # 80% X, 2.8% Y
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        },
        {
            "text": "057/25",
            "confidence": 0.98,
            "center": [2000, 150],
            "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
        },
        {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2200, 100],
            "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
        },
        {
            "text": "15",
            "confidence": 0.97,
            "center": [2180, 150],
            "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
        },
        {
            "text": "10",
            "confidence": 0.97,
            "center": [2220, 150],
            "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
        },
        {
            "text": "2025",
            "confidence": 0.98,
            "center": [2260, 150],
            "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
        },
    ]


def create_slash_misrecognition_detections() -> List[Dict[str, Any]]:
    """Create OCR detections with slash misrecognition (I, l, |).

    Returns:
        List of detection dictionaries with slash errors
    """
    return [
        {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        },
        {
            "text": "057I25",  # I instead of /
            "confidence": 0.85,
            "center": [2000, 150],
            "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
        },
        {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2200, 100],
            "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
        },
        {
            "text": "01",
            "confidence": 0.97,
            "center": [2180, 150],
            "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
        },
        {
            "text": "12",
            "confidence": 0.97,
            "center": [2220, 150],
            "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
        },
        {
            "text": "2025",
            "confidence": 0.98,
            "center": [2260, 150],
            "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
        },
    ]


def create_low_confidence_detections() -> List[Dict[str, Any]]:
    """Create OCR detections with low confidence values.

    Returns:
        List of detection dictionaries with low confidence
    """
    return [
        {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        },
        {
            "text": "057/25",
            "confidence": 0.35,  # Below typical threshold
            "center": [2000, 150],
            "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
        },
        {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2200, 100],
            "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
        },
        {
            "text": "15",
            "confidence": 0.35,  # Low confidence
            "center": [2180, 150],
            "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
        },
        {
            "text": "10",
            "confidence": 0.35,
            "center": [2220, 150],
            "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
        },
        {
            "text": "2025",
            "confidence": 0.35,
            "center": [2260, 150],
            "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
        },
    ]


def create_multiple_errors_detections() -> List[Dict[str, Any]]:
    """Create OCR detections with multiple errors combined.

    Returns:
        List of detection dictionaries with multiple errors
    """
    return [
        {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        },
        {
            "text": "057l25",  # l instead of /, low confidence
            "confidence": 0.38,
            "center": [2000, 150],
            "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
        },
        {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2200, 100],
            "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
        },
        {
            "text": "O1",  # O instead of 0
            "confidence": 0.42,
            "center": [2180, 150],
            "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
        },
        {
            "text": "12",
            "confidence": 0.45,
            "center": [2220, 150],
            "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
        },
        {
            "text": "25",  # 2-digit year
            "confidence": 0.40,
            "center": [2260, 150],
            "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
        },
    ]


def create_split_cell_detections() -> List[Dict[str, Any]]:
    """Create OCR detections with blank number split across cells.

    Returns:
        List of detection dictionaries with split number
    """
    return [
        {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        },
        {
            "text": "057",
            "confidence": 0.98,
            "center": [2000, 150],
            "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
        },
        {
            "text": "/",
            "confidence": 0.90,
            "center": [2020, 150],
            "bbox": [[2015, 130], [2025, 130], [2025, 170], [2015, 170]],
        },
        {
            "text": "25",
            "confidence": 0.97,
            "center": [2040, 150],
            "bbox": [[2030, 130], [2050, 130], [2050, 170], [2030, 170]],
        },
        {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2200, 100],
            "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
        },
        {
            "text": "15",
            "confidence": 0.97,
            "center": [2180, 150],
            "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
        },
        {
            "text": "10",
            "confidence": 0.97,
            "center": [2220, 150],
            "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
        },
        {
            "text": "2025",
            "confidence": 0.98,
            "center": [2260, 150],
            "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
        },
    ]


def create_outside_region_detections() -> List[Dict[str, Any]]:
    """Create OCR detections outside the target region.

    Returns:
        List of detection dictionaries outside top-right region
    """
    return [
        {
            "text": "Номер",
            "confidence": 0.95,
            "center": [200, 100],  # 8% X - outside region (70-100%)
            "bbox": [[150, 80], [250, 80], [250, 120], [150, 120]],
        },
        {
            "text": "057/25",
            "confidence": 0.98,
            "center": [200, 150],  # Outside region
            "bbox": [[180, 130], [220, 130], [220, 170], [180, 170]],
        },
        {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2000, 500],  # 80% X but 14% Y - outside region (0-8%)
            "bbox": [[1950, 480], [2050, 480], [2050, 520], [1950, 520]],
        },
    ]


def get_image_dimensions() -> tuple[int, int]:
    """Get standard test image dimensions.

    Returns:
        Tuple of (width, height) in pixels
    """
    return 2480, 3508  # Typical A4 scan dimensions

