"""Image generation utilities for testing."""
import cv2
import numpy as np
from pathlib import Path


def create_test_document_image(width: int = 800, height: int = 1000) -> np.ndarray:
    """Create synthetic document image for testing.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Synthetic document image with header, defects, and analysis sections
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Add header section
    cv2.rectangle(image, (50, 50), (750, 200), (0, 0, 0), 2)
    # Add defects section
    cv2.rectangle(image, (50, 250), (750, 600), (0, 0, 0), 2)
    # Add analysis section
    cv2.rectangle(image, (50, 650), (750, 950), (0, 0, 0), 2)
    return image


def create_rotated_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by specified angle.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        Rotated image with same dimensions
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


