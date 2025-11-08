from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings


@dataclass(frozen=True)
class PreprocessorResult:
    """Result information returned by the preprocessing pipeline."""

    output_path: Path
    duration_seconds: float
    deskew_angle: Optional[float]


class ImagePreprocessor:
    """Image preprocessing pipeline for deskewing and quality enhancement."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger

    def process(self, input_path: Path, output_path: Optional[Path] = None) -> PreprocessorResult:
        """Run preprocessing pipeline and persist the result."""
        import time

        start_time = time.perf_counter()
        image = self._load_image(input_path)
        self._logger.debug("Loaded image '%s' with shape %s", input_path, image.shape)

        processed_image, angle = self._apply_pipeline(image)
        destination = output_path or self._build_output_path(input_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(destination), processed_image)
        if not success:
            raise RuntimeError(f"Failed to store processed image at '{destination}'")

        duration = time.perf_counter() - start_time
        self._logger.info("Preprocessing completed in %.3f seconds", duration)
        self._logger.info("Processed image stored at '%s'", destination)

        return PreprocessorResult(
            output_path=destination,
            duration_seconds=duration,
            deskew_angle=angle,
        )

    def _apply_pipeline(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Apply deskewing and enhancement steps sequentially."""
        deskew_angle: Optional[float] = None
        if self._settings.enable_deskew:
            image, deskew_angle = self._deskew(image)
            if deskew_angle is not None:
                self._logger.debug("Deskew applied with angle %.3f degrees", deskew_angle)
            else:
                self._logger.debug("Deskew skipped (no text contours detected)")

        enhanced_image = self._enhance(image)
        return enhanced_image, deskew_angle

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from disk and validate its existence."""
        if not path.exists():
            raise FileNotFoundError(f"Input image '{path}' does not exist")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image '{path}'. Ensure it is a valid image file.")
        return image

    def _build_output_path(self, input_path: Path) -> Path:
        """Generate output path in configured output directory with suffix."""
        output_dir = self._settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = self._settings.processed_suffix
        stem = input_path.stem
        extension = input_path.suffix or ".jpg"
        filename = f"{stem}{suffix}{extension}"
        return output_dir / filename

    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Correct image rotation based on detected text angle."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        coords = np.column_stack(np.where(binary > 0))
        if coords.size == 0:
            return image, None

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45.0:
            angle = -(90.0 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.1:
            return image, 0.0

        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, angle

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """Improve image clarity for OCR processing."""
        scaled = self._scale_image(image)
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        channel_l, channel_a, channel_b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(channel_l)
        lab_enhanced = cv2.merge((enhanced_l, channel_a, channel_b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        if self._settings.enable_denoising:
            self._logger.debug("Applying denoising filter")
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            )

        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return binary

    def _scale_image(self, image: np.ndarray) -> np.ndarray:
        """Scale image according to configuration."""
        scale_factor = self._settings.image_scale_factor
        if scale_factor == 1.0:
            return image

        interpolation = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        self._logger.debug(
            "Scaling image from (%d, %d) to (%d, %d)",
            width,
            height,
            new_width,
            new_height,
        )
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

