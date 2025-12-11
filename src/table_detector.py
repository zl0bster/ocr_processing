"""Table structure detection for defects zone processing."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional

import cv2
import numpy as np

from config.settings import Settings


@dataclass(frozen=True)
class TableGrid:
    """Detected table structure with row and column boundaries."""

    rows: List[int]  # Y-coordinates of horizontal lines (sorted)
    cols: List[int]  # X-coordinates of vertical lines (sorted)
    confidence: float  # Overall detection confidence (0-1)
    num_rows: int  # Number of data rows (excluding header)
    num_cols: int  # Number of columns
    method: str  # Detection method used ('morphology', 'template')


class TableDetector:
    """Detect table structure (rows and columns) in document regions."""

    def __init__(self, settings: Settings, logger: logging.Logger):
        """Initialize table detector."""
        self._settings = settings
        self._logger = logger

    def detect_structure(
        self,
        image: np.ndarray,
        strategy: Optional[str] = None,
    ) -> Optional[TableGrid]:
        """Detect table grid (rows and columns).

        Args:
            image: Preprocessed (binarized) region containing table
            strategy: 'morphology' | 'template' | 'auto' (default from settings)

        Returns:
            TableGrid with detected boundaries, or None if detection failed
        """
        if image is None or image.size == 0:
            self._logger.warning("Table detection: input image is None or empty")
            return None

        height, width = image.shape[:2]
        if height < 50 or width < 50:
            self._logger.debug(
                "Table detection: image too small (%dx%d), skipping", width, height
            )
            return None

        selected_strategy = strategy or self._settings.table_detection_strategy
        self._logger.debug("Table detection strategy: %s", selected_strategy)

        if selected_strategy == "auto":
            # Try morphology first, fallback to template
            grid = self._detect_morphology(image)
            if grid:
                return grid
            # Template fallback would go here if implemented
            return None
        elif selected_strategy == "morphology":
            return self._detect_morphology(image)
        elif selected_strategy == "template":
            # Template-based detection (future implementation)
            self._logger.warning("Template-based table detection not yet implemented")
            return None
        else:
            self._logger.warning("Unknown table detection strategy: %s", selected_strategy)
            return None

    def _detect_morphology(self, image: np.ndarray) -> Optional[TableGrid]:
        """Detect table structure using morphological operations.

        Algorithm:
        1. Binarize image (inverted for white-on-black)
        2. Create horizontal/vertical kernels
        3. Apply morphological opening to extract lines
        4. Find contours and extract coordinates
        5. Merge close positions
        6. Validate grid

        Returns:
            TableGrid if successful, None otherwise
        """
        height, width = image.shape[:2]

        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Binarize (inverted for white text on black background)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Detect horizontal lines
        h_lines = self._detect_horizontal_lines(binary, width, height)
        if not h_lines or len(h_lines) < 2:
            self._logger.debug(
                "Table detection: insufficient horizontal lines (%d)", len(h_lines) if h_lines else 0
            )
            return None

        # Detect vertical lines
        v_lines = self._detect_vertical_lines(binary, width, height)
        if not v_lines or len(v_lines) < 2:
            self._logger.debug(
                "Table detection: insufficient vertical lines (%d)", len(v_lines) if v_lines else 0
            )
            return None

        # Validate grid
        if not self._validate_grid(h_lines, v_lines, width, height):
            self._logger.debug("Table detection: grid validation failed")
            return None

        # Calculate number of rows/columns (subtract 1 for boundaries)
        num_rows = len(h_lines) - 1
        num_cols = len(v_lines) - 1

        # Calculate confidence based on line detection quality
        confidence = self._calculate_confidence(h_lines, v_lines, width, height)

        self._logger.info(
            "Table structure detected: %d rows, %d columns (confidence: %.2f)",
            num_rows,
            num_cols,
            confidence,
        )

        return TableGrid(
            rows=h_lines,
            cols=v_lines,
            confidence=confidence,
            num_rows=num_rows,
            num_cols=num_cols,
            method="morphology",
        )

    def _detect_horizontal_lines(
        self, binary: np.ndarray, width: int, height: int
    ) -> List[int]:
        """Detect horizontal lines using morphological operations.

        Args:
            binary: Binary image (inverted)
            width: Image width
            height: Image height

        Returns:
            List of Y-coordinates (sorted, ascending)
        """
        # Create horizontal kernel
        kernel_width = max(10, int(width * self._settings.table_h_kernel_ratio))
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, 1)
        )

        # Apply morphological opening
        h_lines_img = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
        )

        # Find contours
        contours, _ = cv2.findContours(
            h_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        line_positions: List[int] = []
        min_length = int(width * self._settings.table_line_min_length_ratio)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_length:
                # Store center Y coordinate
                line_positions.append(y + h // 2)

        # Merge close positions
        merged = self._merge_close_positions(
            line_positions, self._settings.table_line_merge_threshold
        )

        # Sort ascending
        merged.sort()

        self._logger.debug(
            "Detected %d horizontal lines: %s",
            len(merged),
            [f"y={y} ({y/height:.1%})" for y in merged],
        )

        return merged

    def _detect_vertical_lines(
        self, binary: np.ndarray, width: int, height: int
    ) -> List[int]:
        """Detect vertical lines using morphological operations.

        Args:
            binary: Binary image (inverted)
            width: Image width
            height: Image height

        Returns:
            List of X-coordinates (sorted, ascending)
        """
        # Create vertical kernel
        kernel_height = max(10, int(height * self._settings.table_v_kernel_ratio))
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, kernel_height)
        )

        # Apply morphological opening
        v_lines_img = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1
        )

        # Find contours
        contours, _ = cv2.findContours(
            v_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        line_positions: List[int] = []
        min_length = int(height * self._settings.table_line_min_length_ratio)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= min_length:
                # Store center X coordinate
                line_positions.append(x + w // 2)

        # Merge close positions
        merged = self._merge_close_positions(
            line_positions, self._settings.table_line_merge_threshold
        )

        # Sort ascending
        merged.sort()

        self._logger.debug(
            "Detected %d vertical lines: %s",
            len(merged),
            [f"x={x} ({x/width:.1%})" for x in merged],
        )

        return merged

    def _merge_close_positions(
        self, positions: List[int], threshold: int
    ) -> List[int]:
        """Merge positions that are very close (within threshold).

        Args:
            positions: List of position values (should be sorted)
            threshold: Merge if distance < threshold

        Returns:
            Merged list (positions moved to midpoint of clusters)
        """
        if not positions:
            return []

        sorted_positions = sorted(positions)
        merged = [sorted_positions[0]]

        for pos in sorted_positions[1:]:
            if abs(pos - merged[-1]) <= threshold:
                # Merge: move to midpoint
                merged[-1] = int((merged[-1] + pos) / 2)
            else:
                merged.append(pos)

        return merged

    def _validate_grid(
        self, rows: List[int], cols: List[int], width: int, height: int
    ) -> bool:
        """Validate detected grid.

        Checks:
        - At least 2 rows and 2 columns
        - Reasonable spacing (not too close, not too far apart)
        - Lines are within image bounds

        Args:
            rows: Y-coordinates of horizontal lines
            cols: X-coordinates of vertical lines
            width: Image width
            height: Image height

        Returns:
            True if grid is valid
        """
        if len(rows) < 2 or len(cols) < 2:
            self._logger.debug(
                "Grid validation failed: insufficient lines (rows=%d, cols=%d)",
                len(rows),
                len(cols),
            )
            return False

        # Check all lines are within bounds
        if rows[0] < 0 or rows[-1] > height or cols[0] < 0 or cols[-1] > width:
            self._logger.debug("Grid validation failed: lines outside image bounds")
            return False

        # Check minimum spacing (at least 10 pixels between lines)
        min_spacing = 10
        for i in range(len(rows) - 1):
            if rows[i + 1] - rows[i] < min_spacing:
                self._logger.debug(
                    "Grid validation failed: rows too close (y=%d to y=%d)",
                    rows[i],
                    rows[i + 1],
                )
                return False

        for i in range(len(cols) - 1):
            if cols[i + 1] - cols[i] < min_spacing:
                self._logger.debug(
                    "Grid validation failed: cols too close (x=%d to x=%d)",
                    cols[i],
                    cols[i + 1],
                )
                return False

        return True

    def _calculate_confidence(
        self, rows: List[int], cols: List[int], width: int, height: int
    ) -> float:
        """Calculate detection confidence based on grid quality.

        Args:
            rows: Y-coordinates of horizontal lines
            cols: X-coordinates of vertical lines
            width: Image width
            height: Image height

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from number of detected lines
        row_score = min(len(rows) / 10.0, 1.0)  # More lines = better (up to 10)
        col_score = min(len(cols) / 10.0, 1.0)

        # Spacing regularity (more regular = higher confidence)
        row_spacing_regularity = self._calculate_spacing_regularity(rows)
        col_spacing_regularity = self._calculate_spacing_regularity(cols)

        # Combine scores
        confidence = (
            (row_score + col_score) * 0.3
            + (row_spacing_regularity + col_spacing_regularity) * 0.2
        )

        return min(max(confidence, 0.0), 1.0)

    def _calculate_spacing_regularity(self, positions: List[int]) -> float:
        """Calculate how regular spacing is between positions.

        Args:
            positions: List of sorted positions

        Returns:
            Regularity score (0.0-1.0), higher = more regular
        """
        if len(positions) < 2:
            return 0.0

        spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        if not spacings:
            return 0.0

        mean_spacing = sum(spacings) / len(spacings)
        if mean_spacing == 0:
            return 0.0

        # Calculate coefficient of variation (lower = more regular)
        variance = sum((s - mean_spacing) ** 2 for s in spacings) / len(spacings)
        std_dev = variance ** 0.5
        cv = std_dev / mean_spacing if mean_spacing > 0 else 1.0

        # Convert to regularity score (inverse of CV)
        regularity = 1.0 / (1.0 + cv)
        return regularity








