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
                regions = self._detect_adaptive_lines(image)
            elif method == "text_based":
                regions = self._detect_text_projection(image)
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

    def _detect_adaptive_lines(self, image: np.ndarray) -> List[DocumentRegion]:
        """Detect regions using horizontal line morphology."""
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
        if len(line_positions) < 2:
            return []

        line_positions = sorted(line_positions)[:2]
        boundaries = [0, *line_positions, height]
        return self._regions_from_boundaries(
            boundaries, height, detection_method="adaptive", confidence=0.85
        )

    def _detect_text_projection(self, image: np.ndarray) -> List[DocumentRegion]:
        """Detect regions using horizontal projection profile."""
        gray = self._to_gray(image)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

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

        boundaries = [0, *line_positions, binary.shape[0]]
        return self._regions_from_boundaries(
            boundaries, binary.shape[0], detection_method="text_based", confidence=0.75
        )

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

