"""Detection and extraction of logical document regions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config.region_templates import load_region_templates
from config.settings import Settings


@dataclass(frozen=True)
class DocumentRegion:
    """Normalized and absolute coordinates of a document region."""

    region_id: str
    y_start_norm: float
    y_end_norm: float
    y_start: int
    y_end: int
    detection_method: str
    confidence: float


class RegionDetector:
    """Detect logical regions in QC forms and extract their contents."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger
        self._templates = load_region_templates(
            settings.region_template_file, logger
        )

    def detect_zones(
        self,
        image: np.ndarray,
        strategy: Optional[str] = None,
        template_name: Optional[str] = None,
    ) -> List[DocumentRegion]:
        """Detect logical zones on the provided image.

        Args:
            image: Source image (H, W, C).
            strategy: Optional override for detection strategy.
            template_name: Optional template to use for template strategy.

        Returns:
            List of detected document regions.
        """
        if image is None or image.size == 0:
            raise ValueError("Image for zone detection cannot be empty")

        if not self._settings.enable_region_detection:
            self._logger.debug(
                "Region detection disabled in settings. Falling back to template split."
            )
            return self._detect_template_based(
                image,
                template_name=template_name,
                detection_method="template-disabled",
            )

        height = image.shape[0]
        selected_strategy = (strategy or self._settings.region_detection_strategy).lower()
        self._logger.debug("Selected region detection strategy: %s", selected_strategy)

        detection_order: Sequence[str]
        if selected_strategy == "auto":
            detection_order = ("adaptive", "text_based", "template")
        else:
            detection_order = (selected_strategy,)

        for method in detection_order:
            if method == "adaptive":
                regions = self._detect_adaptive_lines(image, template_name=template_name)
            elif method == "text_based":
                regions = self._detect_text_projection(image, template_name=template_name)
            elif method == "template":
                regions = self._detect_template_based(image, template_name=template_name)
            else:
                self._logger.warning("Unknown region detection strategy: %s", method)
                continue

            if self._validate_regions(regions, height):
                self._logger.info(
                    "Region detection succeeded using strategy '%s' with %d regions",
                    method,
                    len(regions),
                )
                return regions

            self._logger.debug(
                "Strategy '%s' produced invalid regions. Trying next fallback.", method
            )

        # Final fallback to template-based segmentation
        self._logger.warning("Falling back to template-based region detection.")
        return self._detect_template_based(image, template_name=template_name)

    def extract_region(
        self,
        image: np.ndarray,
        region: DocumentRegion,
        margin: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """Extract ROI for the specified region.

        Args:
            image: Source image (H, W, C).
            region: Region definition.
            margin: Optional margin in pixels (applied to top/bottom).

        Returns:
            Tuple with cropped image and scale factor applied (1.0 means no scaling).
        """
        height, width = image.shape[:2]
        top = max(0, region.y_start - margin)
        bottom = min(height, region.y_end + margin)
        if bottom <= top:
            raise ValueError(
                f"Invalid crop boundaries for region '{region.region_id}': "
                f"{top}..{bottom}"
            )

        cropped = image[top:bottom, 0:width].copy()
        region_height = cropped.shape[0]
        region_width = cropped.shape[1]
        scale_factor = 1.0

        if self._should_downscale_region(region_width, region_height):
            target_width, target_height = self._calculate_region_resize(
                region_width, region_height
            )
            if target_width != region_width or target_height != region_height:
                self._logger.debug(
                    "Downscaling region '%s' from %dx%d to %dx%d",
                    region.region_id,
                    region_width,
                    region_height,
                    target_width,
                    target_height,
                )
                cropped = cv2.resize(
                    cropped, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
                scale_factor = target_height / float(region_height)

        return cropped, scale_factor

    def _detect_template_based(
        self,
        image: np.ndarray,
        template_name: Optional[str] = None,
        detection_method: str = "template",
    ) -> List[DocumentRegion]:
        """Use predefined templates with normalized coordinates."""
        height = image.shape[0]
        template_key = template_name or self._settings.template_name
        template = self._templates.get(template_key)

        if not template:
            self._logger.warning(
                "Template '%s' not found. Falling back to default template.", template_key
            )
            template = next(iter(self._templates.values()), [])

        regions: List[DocumentRegion] = []
        for idx, region_def in enumerate(template):
            y_start_norm = float(region_def.get("y_start_norm", 0.0))
            y_end_norm = float(region_def.get("y_end_norm", 1.0))
            y_start, y_end = self._denormalize_coordinates(
                height, y_start_norm, y_end_norm
            )
            region_id = region_def.get("region_id", f"region_{idx}")
            confidence = float(region_def.get("confidence", 1.0))

            regions.append(
                DocumentRegion(
                    region_id=region_id,
                    y_start_norm=y_start_norm,
                    y_end_norm=y_end_norm,
                    y_start=y_start,
                    y_end=y_end,
                    detection_method=detection_method,
                    confidence=confidence,
                )
            )

        return regions

    def _detect_adaptive_lines(
        self, image: np.ndarray, template_name: Optional[str] = None
    ) -> List[DocumentRegion]:
        """Detect regions using horizontal line morphology with constraint validation."""
        gray = self._to_gray(image)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        width = gray.shape[1]
        height = gray.shape[0]
        kernel_width = max(10, width // 2)
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, 1)
        )
        detected_lines = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
        )

        contours, _ = cv2.findContours(
            detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        line_positions: List[int] = []
        min_width_ratio = self._settings.region_horizontal_line_min_width_ratio

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            width_ratio = w / float(width)
            if width_ratio < min_width_ratio:
                continue
            line_positions.append(y + h // 2)

        line_positions = self._merge_close_positions(line_positions, threshold=10)
        self._logger.debug(
            "Detected %d horizontal line candidates: %s",
            len(line_positions),
            [f"y={y} ({y/height:.1%})" for y in sorted(line_positions)],
        )

        # Filter lines for valid header boundaries
        valid_header_lines = [
            y for y in line_positions if self._validate_line_as_header_boundary(y, height)
        ]

        if not valid_header_lines:
            self._logger.debug(
                "No valid header boundaries found in detected lines. "
                "Searching fallback range [%d, %d]",
                int(height * self._settings.region_min_header_ratio),
                int(height * self._settings.region_max_header_ratio),
            )
            # Try fallback search in expected header range
            min_header_y = int(height * self._settings.region_min_header_ratio)
            max_header_y = int(height * self._settings.region_max_header_ratio)
            fallback_header = self._find_fallback_line_in_range(
                gray, min_header_y, max_header_y
            )
            if fallback_header:
                valid_header_lines = [fallback_header]
                self._logger.debug(
                    "Found fallback header boundary at y=%d (%.1f%%)",
                    fallback_header,
                    fallback_header / height * 100,
                )
            else:
                self._logger.debug("Fallback header search failed. Returning empty.")
                return []

        # Select best header boundary (closest to middle of valid range)
        header_mid = height * (
            self._settings.region_min_header_ratio
            + self._settings.region_max_header_ratio
        ) / 2.0
        header_boundary = min(
            valid_header_lines, key=lambda y: abs(y - header_mid)
        )
        self._logger.debug(
            "Selected header boundary at y=%d (%.1f%%)",
            header_boundary,
            header_boundary / height * 100,
        )

        # Find valid defects boundary below header
        min_defects_y = header_boundary + int(
            height * self._settings.region_min_defects_ratio
        )
        max_defects_y = header_boundary + int(
            height * self._settings.region_max_defects_ratio
        )
        max_defects_y = min(max_defects_y, height - 1)

        valid_defects_lines = [
            y
            for y in line_positions
            if y > header_boundary
            and self._validate_line_as_defects_boundary(y, header_boundary, height)
        ]

        if not valid_defects_lines:
            self._logger.debug(
                "No valid defects boundaries found. Searching fallback range [%d, %d]",
                min_defects_y,
                max_defects_y,
            )
            fallback_defects = self._find_fallback_line_in_range(
                gray, min_defects_y, max_defects_y
            )
            if fallback_defects:
                valid_defects_lines = [fallback_defects]
                self._logger.debug(
                    "Found fallback defects boundary at y=%d (%.1f%%)",
                    fallback_defects,
                    fallback_defects / height * 100,
                )
            else:
                self._logger.debug("Fallback defects search failed. Returning empty.")
                return []

        # Select best defects boundary
        defects_mid = (min_defects_y + max_defects_y) / 2.0
        defects_boundary = min(
            valid_defects_lines, key=lambda y: abs(y - defects_mid)
        )
        self._logger.debug(
            "Selected defects boundary at y=%d (%.1f%%)",
            defects_boundary,
            defects_boundary / height * 100,
        )

        # Build boundaries and create regions
        boundaries = [0, header_boundary, defects_boundary, height]
        regions = self._regions_from_boundaries(
            boundaries, height, detection_method="adaptive", confidence=0.85
        )

        # Cross-validate with template
        if not self._cross_validate_with_template(regions, template_name):
            self._logger.debug(
                "Adaptive regions failed template cross-validation. Returning empty."
            )
            return []

        self._logger.debug(
            "Adaptive detection succeeded with regions: %s",
            [
                f"{r.region_id} y={r.y_start}-{r.y_end} ({r.y_start_norm:.1%}-{r.y_end_norm:.1%})"
                for r in regions
            ],
        )

        return regions

    def _validate_line_as_header_boundary(
        self, y_pos: int, height: int
    ) -> bool:
        """Check if line position is valid for header boundary."""
        ratio = y_pos / float(height)
        return (
            self._settings.region_min_header_ratio
            <= ratio
            <= self._settings.region_max_header_ratio
        )

    def _validate_line_as_defects_boundary(
        self, y_pos: int, header_end: int, height: int
    ) -> bool:
        """Check if line position creates valid defects region."""
        defects_height = y_pos - header_end
        defects_ratio = defects_height / float(height)
        return (
            self._settings.region_min_defects_ratio
            <= defects_ratio
            <= self._settings.region_max_defects_ratio
        )

    def _find_fallback_line_in_range(
        self, gray: np.ndarray, min_y: int, max_y: int
    ) -> Optional[int]:
        """Search for horizontal line within specified Y range using projection."""
        if min_y >= max_y or min_y < 0 or max_y > gray.shape[0]:
            return None

        # Extract region of interest
        roi = gray[min_y:max_y, :]
        if roi.size == 0:
            return None

        # Binarize the ROI
        _, binary = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Horizontal projection: sum of black pixels per row
        projection = np.sum(binary == 0, axis=1).astype(np.float32)

        if projection.size == 0:
            return None

        # Find row with minimum projection (likely a horizontal line/separator)
        # Use a small window to smooth and find local minima
        window = max(3, projection.size // 20)
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / float(window)
            smoothed = np.convolve(projection, kernel, mode="same")
        else:
            smoothed = projection

        # Find local minima (potential line positions)
        # Look for rows with projection below median
        threshold = np.percentile(smoothed, 30)
        candidate_indices = np.where(smoothed <= threshold)[0]

        if candidate_indices.size == 0:
            return None

        # Return the candidate closest to the middle of the range
        mid_range = (max_y - min_y) // 2
        best_idx = int(np.argmin(np.abs(candidate_indices - mid_range)))
        return min_y + candidate_indices[best_idx]

    def _cross_validate_with_template(
        self,
        regions: List[DocumentRegion],
        template_name: Optional[str] = None,
    ) -> bool:
        """Validate adaptive regions against template expectations."""
        template_key = template_name or self._settings.template_name
        template = self._templates.get(template_key)
        if not template:
            return True  # No template to validate against

        tolerance = self._settings.region_adaptive_template_tolerance

        for adaptive_region in regions:
            # Find matching template region
            template_region = next(
                (t for t in template if t["region_id"] == adaptive_region.region_id),
                None,
            )
            if not template_region:
                continue

            # Check if normalized coords are within tolerance
            expected_start = template_region["y_start_norm"]
            expected_end = template_region["y_end_norm"]

            start_deviation = abs(adaptive_region.y_start_norm - expected_start)
            end_deviation = abs(adaptive_region.y_end_norm - expected_end)

            if start_deviation > tolerance or end_deviation > tolerance:
                self._logger.debug(
                    "Region '%s' deviates from template: start=%.3f (expected %.3f), end=%.3f (expected %.3f)",
                    adaptive_region.region_id,
                    adaptive_region.y_start_norm,
                    expected_start,
                    adaptive_region.y_end_norm,
                    expected_end,
                )
                return False

        return True

    def _detect_text_projection(
        self, image: np.ndarray, template_name: Optional[str] = None
    ) -> List[DocumentRegion]:
        """Detect regions using horizontal projection profile with constraint validation."""
        gray = self._to_gray(image)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        height = binary.shape[0]
        projection = np.sum(binary == 0, axis=1).astype(np.float32)
        window = max(5, binary.shape[0] // 50)
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.convolve(projection, kernel, mode="same")

        threshold = np.percentile(smoothed, 20)
        candidate_indices = np.where(smoothed <= threshold)[0]
        segments = self._group_indices(candidate_indices)

        if len(segments) < 2:
            return []

        segments = sorted(segments, key=lambda seg: seg[1] - seg[0], reverse=True)[:2]
        line_positions = sorted(int((start + end) / 2) for start, end in segments)

        self._logger.debug(
            "Text projection detected %d line candidates: %s",
            len(line_positions),
            [f"y={y} ({y/height:.1%})" for y in line_positions],
        )

        # Filter lines for valid header boundaries
        valid_header_lines = [
            y for y in line_positions if self._validate_line_as_header_boundary(y, height)
        ]

        if not valid_header_lines:
            self._logger.debug(
                "No valid header boundaries found in text projection. "
                "Searching fallback range [%d, %d]",
                int(height * self._settings.region_min_header_ratio),
                int(height * self._settings.region_max_header_ratio),
            )
            # Try fallback search in expected header range
            min_header_y = int(height * self._settings.region_min_header_ratio)
            max_header_y = int(height * self._settings.region_max_header_ratio)
            fallback_header = self._find_fallback_line_in_range(
                gray, min_header_y, max_header_y
            )
            if fallback_header:
                valid_header_lines = [fallback_header]
                self._logger.debug(
                    "Found fallback header boundary at y=%d (%.1f%%)",
                    fallback_header,
                    fallback_header / height * 100,
                )
            else:
                self._logger.debug("Fallback header search failed. Returning empty.")
                return []

        # Select best header boundary (closest to middle of valid range)
        header_mid = height * (
            self._settings.region_min_header_ratio
            + self._settings.region_max_header_ratio
        ) / 2.0
        header_boundary = min(
            valid_header_lines, key=lambda y: abs(y - header_mid)
        )
        self._logger.debug(
            "Selected header boundary at y=%d (%.1f%%)",
            header_boundary,
            header_boundary / height * 100,
        )

        # Find valid defects boundary below header
        min_defects_y = header_boundary + int(
            height * self._settings.region_min_defects_ratio
        )
        max_defects_y = header_boundary + int(
            height * self._settings.region_max_defects_ratio
        )
        max_defects_y = min(max_defects_y, height - 1)

        valid_defects_lines = [
            y
            for y in line_positions
            if y > header_boundary
            and self._validate_line_as_defects_boundary(y, header_boundary, height)
        ]

        if not valid_defects_lines:
            self._logger.debug(
                "No valid defects boundaries found. Searching fallback range [%d, %d]",
                min_defects_y,
                max_defects_y,
            )
            fallback_defects = self._find_fallback_line_in_range(
                gray, min_defects_y, max_defects_y
            )
            if fallback_defects:
                valid_defects_lines = [fallback_defects]
                self._logger.debug(
                    "Found fallback defects boundary at y=%d (%.1f%%)",
                    fallback_defects,
                    fallback_defects / height * 100,
                )
            else:
                self._logger.debug("Fallback defects search failed. Returning empty.")
                return []

        # Select best defects boundary
        defects_mid = (min_defects_y + max_defects_y) / 2.0
        defects_boundary = min(
            valid_defects_lines, key=lambda y: abs(y - defects_mid)
        )
        self._logger.debug(
            "Selected defects boundary at y=%d (%.1f%%)",
            defects_boundary,
            defects_boundary / height * 100,
        )

        # Build boundaries and create regions
        boundaries = [0, header_boundary, defects_boundary, height]
        regions = self._regions_from_boundaries(
            boundaries, height, detection_method="text_based", confidence=0.75
        )

        # Cross-validate with template
        if not self._cross_validate_with_template(regions, template_name):
            self._logger.debug(
                "Text projection regions failed template cross-validation. Returning empty."
            )
            return []

        self._logger.debug(
            "Text projection detection succeeded with regions: %s",
            [
                f"{r.region_id} y={r.y_start}-{r.y_end} ({r.y_start_norm:.1%}-{r.y_end_norm:.1%})"
                for r in regions
            ],
        )

        return regions

    def _regions_from_boundaries(
        self,
        boundaries: Sequence[int],
        image_height: int,
        detection_method: str,
        confidence: float,
    ) -> List[DocumentRegion]:
        """Convert boundary positions to DocumentRegion objects."""
        if len(boundaries) < 2:
            return []

        regions: List[DocumentRegion] = []
        for idx in range(len(boundaries) - 1):
            y_start = max(0, min(image_height, boundaries[idx]))
            y_end = max(0, min(image_height, boundaries[idx + 1]))
            if y_end <= y_start:
                continue
            y_start_norm, y_end_norm = self._normalize_coordinates(
                image_height, y_start, y_end
            )
            region_id = ("header", "defects", "analysis")[idx] if idx < 3 else f"region_{idx}"
            regions.append(
                DocumentRegion(
                    region_id=region_id,
                    y_start_norm=y_start_norm,
                    y_end_norm=y_end_norm,
                    y_start=y_start,
                    y_end=y_end,
                    detection_method=detection_method,
                    confidence=confidence,
                )
            )
        return regions

    @staticmethod
    def _normalize_coordinates(
        image_height: int, y_start: int, y_end: int
    ) -> Tuple[float, float]:
        """Normalize absolute coordinates to 0.0-1.0 range."""
        if image_height <= 0:
            raise ValueError("Image height must be positive")
        return y_start / float(image_height), y_end / float(image_height)

    @staticmethod
    def _denormalize_coordinates(
        image_height: int, y_start_norm: float, y_end_norm: float
    ) -> Tuple[int, int]:
        """Convert normalized coordinates to absolute pixel values."""
        y_start = int(round(image_height * y_start_norm))
        y_end = int(round(image_height * y_end_norm))
        y_start = max(0, min(image_height, y_start))
        y_end = max(0, min(image_height, y_end))
        if y_end < y_start:
            y_end = y_start
        return y_start, y_end

    @staticmethod
    def _merge_close_positions(
        positions: Iterable[int], threshold: int
    ) -> List[int]:
        """Merge line positions that are closer than threshold."""
        sorted_positions = sorted(positions)
        if not sorted_positions:
            return []

        merged = [sorted_positions[0]]
        for pos in sorted_positions[1:]:
            if abs(pos - merged[-1]) <= threshold:
                merged[-1] = int((merged[-1] + pos) / 2)
            else:
                merged.append(pos)
        return merged

    @staticmethod
    def _group_indices(indices: np.ndarray) -> List[Tuple[int, int]]:
        """Group contiguous indices into segments."""
        if indices.size == 0:
            return []
        segments: List[Tuple[int, int]] = []
        start = int(indices[0])
        prev = int(indices[0])
        for idx in indices[1:]:
            current = int(idx)
            if current == prev + 1:
                prev = current
                continue
            segments.append((start, prev))
            start = current
            prev = current
        segments.append((start, prev))
        return segments

    def _validate_regions(
        self,
        regions: Sequence[DocumentRegion],
        image_height: int,
    ) -> bool:
        """Validate that regions cover the full image height."""
        if not regions:
            return False

        coverage_start = regions[0].y_start
        coverage_end = regions[-1].y_end
        if coverage_start > 0 or coverage_end < image_height:
            return False

        total_coverage = sum(region.y_end - region.y_start for region in regions)
        if total_coverage < image_height * 0.95:
            return False

        min_confidence = min(region.confidence for region in regions)
        if min_confidence < self._settings.region_min_confidence:
            self._logger.debug(
                "Detected regions rejected due to low confidence: %.2f", min_confidence
            )
            return False

        return True

    def _should_downscale_region(self, width: int, height: int) -> bool:
        """Determine whether the region should be downscaled before OCR."""
        return (
            width > self._settings.region_max_width
            or height > self._settings.region_max_height
        )

    def _calculate_region_resize(
        self, width: int, height: int
    ) -> Tuple[int, int]:
        """Calculate safe resize dimensions for a region."""
        width_scale = self._settings.region_max_width / float(width)
        height_scale = self._settings.region_max_height / float(height)
        scale = min(width_scale, height_scale, 1.0)
        if scale >= 1.0:
            return width, height
        return int(width * scale), int(height * scale)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

