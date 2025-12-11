"""Synthetic document image generators for region detection testing."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def add_realistic_text(
    image: np.ndarray,
    region: Tuple[int, int, int, int],  # (x, y, width, height)
    density: str = "medium",
    language: str = "ru",
) -> np.ndarray:
    """
    Add realistic text-like elements using cv2.putText.
    Simulates Russian OCR text with various fonts and sizes.

    Args:
        image: Image to add text to
        region: Region coordinates (x, y, width, height)
        density: Text density ("low", "medium", "high")
        language: Language for text simulation ("ru" for Russian)

    Returns:
        Image with added text elements
    """
    x, y, width, height = region
    image_copy = image.copy()

    # Russian text samples for realistic simulation
    text_samples_ru = [
        "АКТ № 034/25",
        "Дата: 05.11.2025",
        "Контролер: Денисова Л.В.",
        "ГЕОМЕТРИЯ",
        "ОТВЕРСТИЯ",
        "ПОВЕРХНОСТЬ",
        "Проверено: 10",
        "С дефектами: 2",
        "Годно: 8",
        "Отклонение: 0.5 мм",
        "Причина: несоответствие",
        "Решение: доработать",
    ]

    # Determine spacing based on density
    if density == "low":
        line_spacing = 50
        font_scale = 0.6
    elif density == "medium":
        line_spacing = 35
        font_scale = 0.7
    else:  # high
        line_spacing = 25
        font_scale = 0.8

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    color = (0, 0, 0)  # Black text

    current_y = y + 30
    text_index = 0

    while current_y < y + height - 20 and text_index < len(text_samples_ru):
        text = text_samples_ru[text_index % len(text_samples_ru)]
        text_x = x + 20

        # Get text size to ensure it fits
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        if text_x + text_width > x + width:
            text_x = x + 10

        cv2.putText(
            image_copy,
            text,
            (text_x, current_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        current_y += line_spacing
        text_index += 1

    return image_copy


def create_synthetic_document(
    width: int = 1500,
    height: int = 2000,
    header_ratio: float = 0.15,
    defects_ratio: float = 0.40,
    with_horizontal_lines: bool = True,
    line_thickness: int = 3,
    text_density: str = "medium",  # "low", "medium", "high"
    add_noise: bool = False,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Generate synthetic document with configurable parameters.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        header_ratio: Header section height as fraction of total (0.0-1.0)
        defects_ratio: Defects section height as fraction of total (0.0-1.0)
        with_horizontal_lines: Add clear boundary lines between sections
        line_thickness: Thickness of boundary lines in pixels
        text_density: Amount of text content ("low", "medium", "high")
        add_noise: Add realistic noise/artifacts to image
        save_path: Optional path to save image for debugging

    Returns:
        Generated image as numpy array (BGR format)
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Calculate section boundaries
    header_end = int(height * header_ratio)
    defects_end = int(height * (header_ratio + defects_ratio))
    analysis_start = defects_end

    # Add text content to each section
    # Header section (0 to header_end)
    image = add_realistic_text(
        image,
        (50, 50, width - 100, header_end - 100),
        density=text_density,
        language="ru",
    )

    # Defects section (header_end to defects_end)
    image = add_realistic_text(
        image,
        (50, header_end + 20, width - 100, defects_end - header_end - 40),
        density=text_density,
        language="ru",
    )

    # Analysis section (defects_end to height)
    image = add_realistic_text(
        image,
        (50, analysis_start + 20, width - 100, height - analysis_start - 40),
        density=text_density,
        language="ru",
    )

    # Add horizontal separator lines if requested
    if with_horizontal_lines:
        # Header boundary line
        cv2.line(
            image,
            (0, header_end),
            (width, header_end),
            (0, 0, 0),
            line_thickness,
        )

        # Defects boundary line
        cv2.line(
            image,
            (0, defects_end),
            (width, defects_end),
            (0, 0, 0),
            line_thickness,
        )

    # Add noise if requested (simulates scanning artifacts)
    if add_noise:
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        image = cv2.add(image, noise)

    # Save image if path provided (for debugging)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)

    return image


def create_document_with_lines(
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Create document with clear horizontal separators (adaptive strategy test).

    Args:
        save_path: Optional path to save image for debugging

    Returns:
        Synthetic document image with clear boundary lines
    """
    return create_synthetic_document(
        with_horizontal_lines=True,
        line_thickness=3,
        text_density="medium",
        save_path=save_path,
    )


def create_document_with_text_blocks(
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Create document with dense text blocks and gaps (text projection test).

    Args:
        save_path: Optional path to save image for debugging

    Returns:
        Synthetic document image with dense text and clear gaps
    """
    return create_synthetic_document(
        with_horizontal_lines=False,
        text_density="high",
        save_path=save_path,
    )


def create_document_no_boundaries(
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Create document without clear boundaries (template fallback test).

    Args:
        save_path: Optional path to save image for debugging

    Returns:
        Synthetic document image without clear section boundaries
    """
    return create_synthetic_document(
        with_horizontal_lines=False,
        text_density="low",
        add_noise=True,
        save_path=save_path,
    )





