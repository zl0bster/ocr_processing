from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
        """Correct image rotation based on detected text lines using Hough transform."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect text lines using Hough transform
        lines = self._detect_text_lines(gray)
        if len(lines) == 0:
            return image, 0.0
        
        # Calculate skew angle from detected lines
        angle = self._calculate_skew_angle(lines)
        
        # Only apply rotation if confident and angle is significant
        if not self._should_apply_rotation(angle):
            return image, 0.0

        # Apply rotation
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

    def _detect_text_lines(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect text lines using Hough line transform."""
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        return lines if lines is not None else []

    def _calculate_skew_angle(self, lines: List[np.ndarray]) -> float:
        """Calculate skew angle from detected lines."""
        if len(lines) == 0:
            return 0.0
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle of the line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            
            # Normalize angle to [-45, 45] range for text lines
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            
            # Only consider lines that could be text (not too steep)
            if abs(angle) < 45:
                angles.append(angle)
        
        if len(angles) == 0:
            return 0.0
        
        # Use median angle to avoid outliers
        return float(np.median(angles))

    def _should_apply_rotation(self, angle: float) -> bool:
        """Determine if rotation should be applied based on angle and confidence."""
        # Only rotate if angle is significant (> 2 degrees)
        min_rotation_threshold = 2.0
        
        # Don't rotate if angle is too small
        if abs(angle) < min_rotation_threshold:
            return False
        
        # Don't rotate if angle seems unrealistic for text skew
        max_rotation_threshold = 45.0
        if abs(angle) > max_rotation_threshold:
            return False
        
        return True

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

        if self._settings.enable_illumination_correction:
            kernel_size = self._settings.illumination_kernel
            self._logger.debug(
                "Applying illumination correction with kernel size %d", kernel_size
            )
            gray_float = gray.astype(np.float32)
            background = cv2.GaussianBlur(gray_float, (kernel_size, kernel_size), 0)
            # Avoid division by zero by adding a small epsilon
            background = np.maximum(background, 1e-6)
            normalized = cv2.divide(gray_float, background, scale=255.0)
            gray = np.clip(normalized, 0, 255).astype(np.uint8)

        if self._settings.binarization_mode == "otsu":
            kernel_size = self._settings.gaussian_blur_kernel
            if kernel_size > 1:
                self._logger.debug(
                    "Applying Gaussian blur with kernel size %d before Otsu thresholding",
                    kernel_size,
                )
                gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            _, binary = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:
            self._logger.debug("Using adaptive thresholding mode")
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
        """Scale image adaptively based on megapixels to optimize OCR."""
        height, width = image.shape[:2]
        original_mp = (width * height) / 1_000_000
        
        # Skip scaling if already high resolution
        if original_mp >= self._settings.adaptive_scaling_min_mp:
            self._logger.debug(
                "Skipping scaling: image has %.2f MP (≥%.1f MP threshold)",
                original_mp,
                self._settings.adaptive_scaling_min_mp
            )
            return image
        
        # Target ~5 MP with max dimensions 3000×2000
        target_mp = self._settings.adaptive_scaling_target_mp
        max_width = self._settings.adaptive_scaling_max_width
        max_height = self._settings.adaptive_scaling_max_height
        
        # Calculate scale factor based on megapixels
        scale_factor = (target_mp / original_mp) ** 0.5
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure we don't exceed max dimensions (maintain aspect ratio)
        width_scale = max_width / new_width if new_width > max_width else 1.0
        height_scale = max_height / new_height if new_height > max_height else 1.0

        final_scale = min(width_scale, height_scale)
        if final_scale < 1.0:
            new_width = int(new_width * final_scale)
            new_height = int(new_height * final_scale)
        
        final_mp = (new_width * new_height) / 1_000_000
        
        self._logger.debug(
            "Adaptive scaling: %.2f MP → %.2f MP, (%d×%d) → (%d×%d), factor=%.2f",
            original_mp, final_mp, width, height, new_width, new_height, scale_factor
        )
        
        interpolation = cv2.INTER_CUBIC
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

