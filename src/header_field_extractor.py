"""Specialized extractor for blank number and date fields in header region.

This module provides region-based extraction with format validation and OCR
error correction for blank numbers (XXX/YYY format) and dates (DD/MM/YYYY).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings
from models.form_data import FieldValue


@dataclass(frozen=True)
class FieldRegion:
    """Region boundaries for field extraction (in percentage of image dimensions)."""

    x_min: float  # Minimum X percentage (0-100)
    x_max: float  # Maximum X percentage (0-100)
    y_min: float  # Minimum Y percentage (0-100)
    y_max: float  # Maximum Y percentage (0-100)


class HeaderFieldExtractor:
    """Specialized extractor for blank number and date from header region.

    This extractor uses region-based filtering to locate fields in the top-right
    corner of the document, validates formats, and corrects common OCR errors.
    """

    def __init__(self, settings: Settings, logger: logging.Logger):
        """Initialize header field extractor.

        Args:
            settings: Application settings
            logger: Logger instance
        """
        self._settings = settings
        self._logger = logger

        # Define top-right region for blank number and date
        # Default: 70-100% width, 0-8% height
        self._top_right_region = FieldRegion(
            x_min=getattr(settings, "header_blank_area_x_min", 70.0),
            x_max=100.0,
            y_min=0.0,
            y_max=getattr(settings, "header_blank_area_y_max", 8.0),
        )

        # Minimum confidence threshold for small text
        self._min_confidence = getattr(settings, "header_blank_min_confidence", 0.4)

        # Enable/disable corrections
        self._enable_slash_correction = getattr(
            settings, "header_enable_slash_correction", True
        )
        self._enable_digit_correction = getattr(
            settings, "header_enable_digit_correction", True
        )

    def extract_blank_number_and_date(
        self,
        header_detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Tuple[Optional[FieldValue], Optional[FieldValue]]:
        """Extract blank number and date from header region.

        Args:
            header_detections: List of text detections from header region
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Tuple of (blank_number, blank_date) FieldValue objects or (None, None)
        """
        if not header_detections:
            return None, None

        # Step 1: Filter detections by top-right region
        region_dets = self._filter_by_region(
            header_detections, self._top_right_region, image_width, image_height
        )

        if not region_dets:
            self._logger.debug(
                "No detections found in top-right region (%.1f-%.1f%% X, %.1f-%.1f%% Y)",
                self._top_right_region.x_min,
                self._top_right_region.x_max,
                self._top_right_region.y_min,
                self._top_right_region.y_max,
            )
            return None, None

        # Step 2: Find header keywords
        nomer_header = self._find_header(region_dets, "номер")
        data_header = self._find_header(region_dets, "дата")

        # Step 3: Extract blank number
        blank_number = None
        if nomer_header:
            blank_number = self._extract_blank_number(
                nomer_header, region_dets, image_width, image_height
            )

        # Step 4: Extract date
        blank_date = None
        if data_header:
            blank_date = self._extract_date(
                data_header, region_dets, image_width, image_height
            )

        return blank_number, blank_date

    def _filter_by_region(
        self,
        detections: List[Dict[str, Any]],
        region: FieldRegion,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Filter detections by coordinate region.

        Args:
            detections: List of detection dictionaries
            region: FieldRegion with percentage boundaries
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Filtered list of detections within the region
        """
        filtered = []
        for det in detections:
            center = det.get("center", [0, 0])
            if len(center) < 2:
                continue

            center_x, center_y = center[0], center[1]

            # Convert to percentages
            x_pct = (center_x / width) * 100 if width > 0 else 0
            y_pct = (center_y / height) * 100 if height > 0 else 0

            # Check if within region
            if (
                region.x_min <= x_pct <= region.x_max
                and region.y_min <= y_pct <= region.y_max
            ):
                filtered.append(det)

        return filtered

    def _find_header(
        self, detections: List[Dict[str, Any]], keyword: str
    ) -> Optional[Dict[str, Any]]:
        """Find header detection by keyword.

        Args:
            detections: List of detection dictionaries
            keyword: Keyword to search for (e.g., "номер", "дата")

        Returns:
            Detection dictionary or None
        """
        keyword_lower = keyword.lower()
        for det in detections:
            text = det.get("text", "").strip()
            if keyword_lower in text.lower():
                return det
        return None

    def _extract_blank_number(
        self,
        nomer_header: Dict[str, Any],
        region_dets: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Optional[FieldValue]:
        """Extract blank number from below "Номер" header.

        Args:
            nomer_header: Detection dictionary for "Номер" header
            region_dets: All detections in the region
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            FieldValue with blank number or None
        """
        # Find texts below header
        header_x = nomer_header["center"][0]
        header_y = nomer_header["center"][1]
        header_y_bottom = max([p[1] for p in nomer_header["bbox"]])

        # Search for values below header
        # Y tolerance: 2.9% of image height (≈100px at 3500px)
        # X tolerance: 2% of image width (≈50px at 2500px)
        max_distance_y = int((2.9 / 100) * image_height)
        max_distance_x = int((2.0 / 100) * image_width)

        candidates = []
        for det in region_dets:
            if det == nomer_header:
                continue

            det_center = det.get("center", [0, 0])
            if len(det_center) < 2:
                continue

            det_x, det_y = det_center[0], det_center[1]
            det_y_top = min([p[1] for p in det["bbox"]])

            y_distance = det_y_top - header_y_bottom
            x_distance = abs(det_x - header_x)

            # Allow small overlap tolerance
            overlap_tolerance = int((0.3 / 100) * image_height)
            if (
                -overlap_tolerance <= y_distance <= max_distance_y
                and x_distance <= max_distance_x
            ):
                text = det.get("text", "").strip()
                confidence = det.get("confidence", 0.0)

                # Check if it contains numbers
                if re.search(r"\d+", text) and confidence >= self._min_confidence:
                    candidates.append((det, abs(y_distance)))

        if not candidates:
            return None

        # Sort by Y distance (closest first)
        candidates.sort(key=lambda x: x[1])
        best_det, _ = candidates[0]

        # Extract and combine text from all candidates in same cell area
        # (blank number might be split: "057" and "25" in separate detections)
        cell_texts = []
        cell_confidences = []
        best_y = best_det["center"][1]

        for det, _ in candidates:
            det_y = det["center"][1]
            # Group detections on same line (within 5% of image height)
            if abs(det_y - best_y) <= int((5.0 / 100) * image_height):
                text = det.get("text", "").strip()
                if text:
                    cell_texts.append(text)
                    cell_confidences.append(det.get("confidence", 0.0))

        # Combine texts (could be "057" + "/" + "25" or "057/25" as one)
        combined_text = "".join(cell_texts) if cell_texts else best_det.get("text", "").strip()

        # If combined text has no slash and looks like two numbers, try to insert slash
        # Pattern: 3 digits followed by 1-2 digits (e.g., "05725" → "057/25")
        if "/" not in combined_text and re.match(r"^\d{3}\d{1,2}$", combined_text):
            # Insert slash after first 3 digits
            combined_text = combined_text[:3] + "/" + combined_text[3:]

        # Validate and correct
        return self._validate_and_correct_blank_number(
            combined_text, sum(cell_confidences) / len(cell_confidences) if cell_confidences else best_det.get("confidence", 0.0)
        )

    def _extract_date(
        self,
        data_header: Dict[str, Any],
        region_dets: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Optional[FieldValue]:
        """Extract date from below "Дата" header.

        Args:
            data_header: Detection dictionary for "Дата" header
            region_dets: All detections in the region
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            FieldValue with date in DD/MM/YYYY format or None
        """
        # Find texts below header
        header_x = data_header["center"][0]
        header_y = data_header["center"][1]
        header_y_bottom = max([p[1] for p in data_header["bbox"]])

        # Search for date parts (usually 3 cells: DD, MM, YYYY)
        max_distance_y = int((2.9 / 100) * image_height)
        max_distance_x = int((2.0 / 100) * image_width)

        date_parts = []
        for det in region_dets:
            if det == data_header:
                continue

            det_center = det.get("center", [0, 0])
            if len(det_center) < 2:
                continue

            det_x, det_y = det_center[0], det_center[1]
            det_y_top = min([p[1] for p in det["bbox"]])

            y_distance = det_y_top - header_y_bottom
            x_distance = abs(det_x - header_x)

            overlap_tolerance = int((0.3 / 100) * image_height)
            if (
                -overlap_tolerance <= y_distance <= max_distance_y
                and x_distance <= max_distance_x
            ):
                text = det.get("text", "").strip()
                confidence = det.get("confidence", 0.0)

                # Check for numeric content
                if re.search(r"\d+", text) and confidence >= self._min_confidence:
                    date_parts.append((det, det_x))  # Store with X coordinate for sorting

        if not date_parts:
            return None

        # Sort by X coordinate (left to right: DD, MM, YYYY)
        date_parts.sort(key=lambda x: x[1])
        sorted_dets = [det for det, _ in date_parts]

        # Extract numeric parts
        numeric_parts = []
        confidences = []
        for det in sorted_dets:
            text = det.get("text", "").strip()
            # Extract all numbers from text
            numbers = re.findall(r"\d+", text)
            for num in numbers:
                numeric_parts.append(num)
                confidences.append(det.get("confidence", 0.0))

        if len(numeric_parts) < 2:
            return None

        # Assemble date: expect DD, MM, YYYY or DD, MM, YY
        day = numeric_parts[0].zfill(2) if len(numeric_parts) > 0 else "01"
        month = numeric_parts[1].zfill(2) if len(numeric_parts) > 1 else "01"

        # Handle year
        if len(numeric_parts) >= 3:
            year = numeric_parts[2]
            # If year is 2 digits, assume 20XX
            if len(year) == 2:
                year = "20" + year
            elif len(year) == 4:
                pass  # Already 4 digits
            else:
                # Try to combine with next part if available
                if len(numeric_parts) >= 4:
                    year = numeric_parts[2] + numeric_parts[3]
                else:
                    year = "20" + year[:2] if len(year) >= 2 else "2025"
        else:
            year = "2025"  # Default fallback

        assembled_date = f"{day}/{month}/{year}"

        # Validate date format
        if self._validate_date_format(assembled_date):
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return FieldValue(
                value=assembled_date,
                confidence=avg_confidence,
                source="header",
                validated=True,
                corrected=self._enable_slash_correction or self._enable_digit_correction,
            )

        return None

    def _validate_and_correct_blank_number(
        self, text: str, confidence: float
    ) -> Optional[FieldValue]:
        """Validate and correct blank number format.

        Args:
            text: Raw text from OCR
            confidence: OCR confidence value

        Returns:
            FieldValue with validated blank number or None
        """
        if not text:
            return None

        # Apply OCR corrections
        corrected = text
        if self._enable_slash_correction:
            corrected = self._correct_slash_chars(corrected)
        if self._enable_digit_correction:
            corrected = self._correct_digit_chars(corrected)

        # Validate format: XXX/YYY (1-3 digits / 1-2 digits)
        pattern = r"^(\d{1,3})/(\d{1,2})$"
        match = re.match(pattern, corrected)

        if match:
            # Ensure proper formatting
            part1 = match.group(1).zfill(3)  # Pad to 3 digits
            part2 = match.group(2).zfill(2)  # Pad to 2 digits
            formatted = f"{part1}/{part2}"

            return FieldValue(
                value=formatted,
                confidence=confidence,
                source="header",
                validated=True,
                corrected=corrected != text,
            )
        else:
            self._logger.warning(
                "Invalid blank number format: '%s' (corrected: '%s')", text, corrected
            )
            return None

    def _correct_slash_chars(self, text: str) -> str:
        """Correct slash character misrecognition.

        Replaces I, l, | with / in numeric context.

        Args:
            text: Text to correct

        Returns:
            Corrected text
        """
        # Pattern: digit + (I|l|∣||) + digit → digit + / + digit
        corrected = re.sub(r"([0-9])([Il|∣])([0-9])", r"\1/\3", text)

        # Also handle cases where slash is at start/end of number sequence
        # e.g., "057I" or "I25" (less common but possible)
        corrected = re.sub(r"([0-9]+)([Il|∣])([0-9]+)", r"\1/\3", corrected)

        return corrected

    def _correct_digit_chars(self, text: str) -> str:
        """Correct digit character misrecognition.

        Common OCR errors: O→0, S→5, Z→2, g→9

        Args:
            text: Text to correct

        Returns:
            Corrected text
        """
        # Only apply in context of blank numbers (mostly digits)
        if not re.search(r"\d", text):
            return text

        corrections = {
            "O": "0",  # Letter O to zero
            "S": "5",  # Letter S to five
            "Z": "2",  # Letter Z to two
            "g": "9",  # Letter g to nine
            "З": "3",  # Cyrillic З to 3
            "О": "0",  # Cyrillic О to 0
            "Б": "8",  # Cyrillic Б to 8
        }

        corrected = text
        for wrong, correct in corrections.items():
            # Only replace if surrounded by digits or at boundary
            pattern = rf"([0-9\s]*){re.escape(wrong)}([0-9\s]*)"
            # Use lambda to avoid issues with backreferences when correct contains digits
            corrected = re.sub(pattern, lambda m: m.group(1) + correct + m.group(2), corrected)

        return corrected

    def _validate_date_format(self, date_str: str) -> bool:
        """Validate date format DD/MM/YYYY.

        Args:
            date_str: Date string to validate

        Returns:
            True if format is valid
        """
        pattern = r"^(\d{2})/(\d{2})/(\d{4})$"
        match = re.match(pattern, date_str)
        if not match:
            return False

        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))

        # Basic validation
        if not (1 <= day <= 31):
            return False
        if not (1 <= month <= 12):
            return False
        if not (2000 <= year <= 2100):  # Reasonable year range
            return False

        return True

