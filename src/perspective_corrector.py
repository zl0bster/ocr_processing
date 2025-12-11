from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings


class PerspectiveCorrector:
    """Handles perspective distortion correction for document images."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        """Initialize perspective corrector."""
        self._settings = settings
        self._logger = logger

    def correct(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Correct perspective distortion in image.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (corrected_image, was_corrected)
                - corrected_image: Perspective-corrected image (or original if correction failed)
                - was_corrected: True if correction was applied
        """
        # Validate input
        if image is None or image.size == 0:
            self._logger.warning("Input image is None or empty, skipping perspective correction")
            return image, False

        height, width = image.shape[:2]
        if height < 50 or width < 50:
            self._logger.debug("Image too small (%dx%d), skipping perspective correction", width, height)
            return image, False

        self._logger.debug("Attempting perspective correction on image with shape %s", image.shape)

        try:
            # Step 1: Prepare image (grayscale + binarization)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Optional: morphological opening for cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            # Step 2: Find document contour
            contour = self._find_document_contour(binary)
            if contour is None:
                self._logger.debug("No suitable document contour found, skipping perspective correction")
                return image, False

            # Step 3: Get document corners
            corners = self._get_document_corners(contour)
            if corners is None:
                self._logger.debug("Could not extract 4 corners from contour, skipping perspective correction")
                return image, False

            # Step 4: Validate corners
            if not self._validate_corners(corners, width, height):
                self._logger.debug("Corner validation failed, skipping perspective correction")
                return image, False

            # Step 5: Calculate target size
            target_width, target_height = self._calculate_target_size(corners)

            # Step 6: Apply perspective transform
            corrected = self._apply_perspective_transform(image, corners, (target_width, target_height))

            self._logger.debug(
                "Perspective correction applied: (%dx%d) -> (%dx%d)",
                width, height, target_width, target_height
            )
            return corrected, True

        except Exception as e:
            self._logger.warning(
                "Perspective correction failed: %s, returning original image",
                e,
                exc_info=self._logger.level == logging.DEBUG
            )
            return image, False

    def _find_document_contour(self, binary: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the largest rectangular contour (document boundary).

        Args:
            binary: Binary image

        Returns:
            Contour points or None if not found
        """
        image_area = binary.shape[0] * binary.shape[1]
        min_area = image_area * self._settings.perspective_min_area_ratio

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Sort by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find first contour that meets area requirement
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                return contour

        return None

    def _get_document_corners(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 4 corner points from contour.

        Args:
            contour: Input contour

        Returns:
            4 corner points in order (TL, TR, BR, BL) or None if not quadrilateral
        """
        # Calculate perimeter
        peri = cv2.arcLength(contour, True)
        epsilon = self._settings.perspective_corner_epsilon * peri

        # Approximate polygon
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if we have exactly 4 points
        if len(approx) != 4:
            self._logger.debug("Approximated polygon has %d points, expected 4", len(approx))
            return None

        # Reshape to (4, 2) and order points
        pts = approx.reshape(4, 2).astype(np.float32)
        ordered_pts = self._order_points(pts)

        return ordered_pts

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points in consistent sequence: top-left, top-right, bottom-right, bottom-left.

        Algorithm:
        1. Sum of x+y (TL has smallest, BR has largest)
        2. Difference of x-y (TR has smallest, BL has largest)

        Args:
            pts: 4 unordered points

        Returns:
            Ordered points (TL, TR, BR, BL)
        """
        # pts shape: (4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum of x+y (TL has smallest, BR has largest)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR

        # Difference of x-y (TR has smallest, BL has largest)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL

        return rect

    def _validate_corners(self, corners: np.ndarray, image_width: int, image_height: int) -> bool:
        """
        Validate corner points.

        Checks:
        - Minimum distance between corners
        - Convex quadrilateral

        Args:
            corners: Ordered corner points (TL, TR, BR, BL)
            image_width: Image width
            image_height: Image height

        Returns:
            True if corners are valid
        """
        min_distance = self._settings.perspective_min_corner_distance

        # Check distances between adjacent corners
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            distance = np.linalg.norm(p2 - p1)

            if distance < min_distance:
                self._logger.debug(
                    "Corner distance too small: %.1f < %d (between corners %d and %d)",
                    distance, min_distance, i, (i + 1) % 4
                )
                return False

        # Check if quadrilateral is convex (all interior angles < 180)
        # For a convex quadrilateral, cross products of adjacent edges should have same sign
        cross_products = []
        for i in range(4):
            v1 = corners[(i + 1) % 4] - corners[i]
            v2 = corners[(i + 2) % 4] - corners[(i + 1) % 4]
            cross = np.cross(v1, v2)
            cross_products.append(cross)

        # All cross products should have the same sign (or zero)
        signs = [np.sign(cp) for cp in cross_products if abs(cp) > 1e-6]
        if len(signs) > 0 and not all(s == signs[0] for s in signs):
            self._logger.debug("Quadrilateral is not convex")
            return False

        return True

    def _calculate_target_size(self, pts: np.ndarray) -> Tuple[int, int]:
        """
        Calculate target image dimensions based on document corners.

        Args:
            pts: Ordered corner points (TL, TR, BR, BL)

        Returns:
            (width, height) tuple
        """
        # pts: [TL, TR, BR, BL]
        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[2] - pts[1])
        max_height = int(max(height_left, height_right))

        # Clamp to limits
        max_width = min(max_width, self._settings.perspective_target_width_limit)
        max_height = min(max_height, self._settings.perspective_target_height_limit)

        return (max_width, max_height)

    def _apply_perspective_transform(
        self, image: np.ndarray, src_pts: np.ndarray, dst_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply perspective transformation to image.

        Args:
            image: Input image
            src_pts: 4 source points (TL, TR, BR, BL)
            dst_size: Target (width, height)

        Returns:
            Transformed image
        """
        target_width, target_height = dst_size

        # Destination points: rectangle
        dst_pts = np.array([
            [0, 0],  # TL
            [target_width, 0],  # TR
            [target_width, target_height],  # BR
            [0, target_height]  # BL
        ], dtype=np.float32)

        # Calculate transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply transformation
        warped = cv2.warpPerspective(
            image,
            M,
            (target_width, target_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return warped




