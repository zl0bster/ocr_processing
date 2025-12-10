"""Form extraction module for structured data extraction from OCR results.

This module extracts structured business data from regional OCR results,
including header fields, defects tables, and analysis sections.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings
from models.form_data import (
    AnalysisData,
    AnalysisRow,
    DefectBlock,
    DefectRow,
    FieldValue,
    FinalDecision,
    HeaderData,
    StickerData,
    dataclass_to_dict,
)
from utils.json_utils import convert_numpy_types


@dataclass(frozen=True)
class ExtractionResult:
    """Result information returned by the form extractor."""

    output_path: Path
    duration_seconds: float
    header_fields_extracted: int
    defect_blocks_found: int
    analysis_rows_found: int
    mandatory_fields_missing: int


class FormExtractor:
    """Extract structured data from regional OCR results."""

    def __init__(self, settings: Settings, logger: logging.Logger):
        self._settings = settings
        self._logger = logger
        self._mandatory_fields = {
            "act_number",
            "act_date",
            "template_revision",
            "part_line_number",
            "quantity_checked",
            "control_type",
            "inspector_name",
        }
        self._image_width: int = 2480  # Default fallback to typical A4 scan
        self._image_height: int = 3508  # Default fallback to typical A4 scan

    def extract(
        self, ocr_json_path: Path, output_path: Optional[Path] = None
    ) -> ExtractionResult:
        """Main extraction method.

        Args:
            ocr_json_path: Path to corrected OCR JSON with regional data
            output_path: Optional custom output path

        Returns:
            ExtractionResult with processing metrics
        """
        start_time = time.perf_counter()

        # Load OCR results
        with open(ocr_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)

        # Check for regional data
        if "ocr_results_by_region" not in ocr_data:
            raise ValueError(
                "OCR results must contain 'ocr_results_by_region'. "
                "Use OCREngine.process_regions() first."
            )

        # Extract and store image dimensions
        self._image_width, self._image_height = self._get_image_dimensions(ocr_data)

        regions_data = ocr_data["ocr_results_by_region"]
        table_data = ocr_data.get("table_data_by_region", {})

        # Extract each section
        header_data = self._extract_header(regions_data.get("header", []))
        # Check if defects zone has table data
        defects_table_data = table_data.get("defects")
        if defects_table_data and defects_table_data.get("type") == "table":
            defects_data = self._extract_defects_from_table(defects_table_data)
        else:
            defects_data = self._extract_defects(regions_data.get("defects", []))
        analysis_data = self._extract_analysis(regions_data.get("analysis", []))

        # Validate
        validation_results = self._validate_extracted_data(
            header_data, defects_data, analysis_data
        )

        # Build final structure
        output_data = {
            "document_info": ocr_data.get("document_info", {}),
            "header": dataclass_to_dict(header_data),
            "defects": [dataclass_to_dict(block) for block in defects_data],
            "analysis": dataclass_to_dict(analysis_data),
            "validation_results": validation_results,
            "corrections_applied": ocr_data.get("corrections_applied", {}),
            "processing_metrics": {
                **ocr_data.get("processing_metrics", {}),
                "extraction_time_ms": int((time.perf_counter() - start_time) * 1000),
                "defect_blocks_detected": len(defects_data),
                "defect_rows_extracted": sum(len(block.rows) for block in defects_data),
                "analysis_rows_extracted": len(analysis_data.deviations),
            },
        }

        # Save
        destination = output_path or self._build_output_path(ocr_json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)

        duration = time.perf_counter() - start_time

        self._logger.info(
            "Data extraction completed in %.3f seconds. Output: %s",
            duration,
            destination,
        )

        return ExtractionResult(
            output_path=destination,
            duration_seconds=duration,
            header_fields_extracted=self._count_extracted_fields(header_data),
            defect_blocks_found=len(defects_data),
            analysis_rows_found=len(analysis_data.deviations),
            mandatory_fields_missing=validation_results.get(
                "mandatory_fields_missing_count", 0
            ),
        )

    def _build_output_path(self, input_path: Path) -> Path:
        """Generate output path with -data suffix.

        Args:
            input_path: Input OCR results file path

        Returns:
            Output path for extracted data
        """
        output_dir = self._settings.output_dir
        stem = input_path.stem
        # Remove existing -corrected suffix if present
        if stem.endswith("-corrected"):
            stem = stem[:-10]
        filename = f"{stem}-data.json"
        return output_dir / filename

    def _count_extracted_fields(self, header: HeaderData) -> int:
        """Count extracted header fields.

        Args:
            header: Header data structure

        Returns:
            Number of extracted fields
        """
        count = 0
        for field_name in self._mandatory_fields:
            field_value = getattr(header, field_name, None)
            if field_value is not None:
                count += 1
        return count

    def _get_image_dimensions(self, ocr_data: Dict[str, Any]) -> Tuple[int, int]:
        """Extract image dimensions from OCR data.

        Args:
            ocr_data: OCR results data dictionary

        Returns:
            Tuple of (width, height) in pixels
        """
        doc_info = ocr_data.get("document_info", {})
        width = doc_info.get("image_width", 2480)  # Fallback to typical A4 scan
        height = doc_info.get("image_height", 3508)  # Fallback to typical A4 scan
        return width, height

    def _pct_x(self, pixels: float) -> float:
        """Convert X pixels to percentage of image width.

        Args:
            pixels: Pixel value

        Returns:
            Percentage value
        """
        return (pixels / self._image_width) * 100

    def _pct_y(self, pixels: float) -> float:
        """Convert Y pixels to percentage of image height.

        Args:
            pixels: Pixel value

        Returns:
            Percentage value
        """
        return (pixels / self._image_height) * 100

    def _x_pixels(self, percent: float) -> int:
        """Convert X percentage to pixels.

        Args:
            percent: Percentage value (e.g., 12.0 for 12%)

        Returns:
            Pixel value
        """
        return int((percent / 100) * self._image_width)

    def _y_pixels(self, percent: float) -> int:
        """Convert Y percentage to pixels.

        Args:
            percent: Percentage value (e.g., 2.0 for 2%)

        Returns:
            Pixel value
        """
        return int((percent / 100) * self._image_height)

    def _extract_header(self, header_texts: List[Dict[str, Any]]) -> HeaderData:
        """Extract header section data.

        Args:
            header_texts: List of text detections from header region

        Returns:
            HeaderData with extracted fields
        """
        header = HeaderData()

        # Detect sticker (PRIORITY SOURCE)
        sticker_data = self._detect_sticker_data(header_texts)

        if sticker_data:
            header.sticker_data = sticker_data
            header.part_line_number = sticker_data.part_line_number
            header.quantity_ordered = sticker_data.quantity_ordered

        # Extract document metadata
        header.act_number = self._extract_act_number(header_texts)
        header.act_date = self._extract_act_date(header_texts)
        header.template_revision = self._extract_template_revision(header_texts)

        # Extract quantities
        header.quantity_checked = self._extract_quantity_field(
            header_texts, "проверено"
        )
        header.quantity_defective = self._extract_quantity_field(
            header_texts, "дефектами"
        )
        header.quantity_passed = self._extract_quantity_field(
            header_texts, "годно"
        )

        # Extract control type and inspector name
        header.control_type = self._extract_control_type(header_texts)
        header.inspector_name = self._extract_inspector_name(header_texts)

        # If no sticker, extract handwritten part details
        if not sticker_data:
            header.part_line_number = self._extract_part_line_number(header_texts)
            header.part_designation = self._extract_part_designation(header_texts)
            header.part_name = self._extract_part_name(header_texts)

        return header

    def _find_column_header(
        self, keyword: str, detections: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Locate column header by keyword.

        Args:
            keyword: Header keyword to search for (e.g., "Номер", "Дата", "Рев")
            detections: List of text detections from header region

        Returns:
            Detection dict with header text or None
        """
        keyword_lower = keyword.lower()
        for det in detections:
            text = det.get("text", "").strip()
            if keyword_lower in text.lower():
                return det
        return None

    def _find_texts_below_header(
        self,
        header_keyword: str,
        detections: List[Dict[str, Any]],
        max_distance_y: int = 100,
        max_distance_x: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find texts vertically below a header keyword.

        Args:
            header_keyword: Header keyword to search for
            detections: List of text detections from header region
            max_distance_y: Maximum vertical distance below header (pixels)
            max_distance_x: Maximum horizontal offset from header center (pixels)

        Returns:
            List of detection dicts found below the header
        """
        header_det = self._find_column_header(header_keyword, detections)
        if not header_det:
            return []

        header_x = header_det["center"][0]
        header_y = header_det["center"][1]
        header_y_bottom = max([p[1] for p in header_det["bbox"]])

        candidates = []
        for det in detections:
            if det == header_det:
                continue

            det_y = det["center"][1]
            det_x = det["center"][0]
            det_y_top = min([p[1] for p in det["bbox"]])

            # Check if text is below header (Y distance) and aligned (X distance)
            y_distance = det_y_top - header_y_bottom
            x_distance = abs(det_x - header_x)

            # Allow small negative overlap (0.3% of image height ≈ 10px at 3500px)
            overlap_tolerance = self._y_pixels(0.3)
            if -overlap_tolerance <= y_distance <= max_distance_y and x_distance <= max_distance_x:
                candidates.append((det, abs(y_distance)))

        # Sort by Y distance (closest first)
        candidates.sort(key=lambda x: x[1])
        return [det for det, _ in candidates]

    def _find_texts_to_right(
        self,
        anchor_keyword: str,
        detections: List[Dict[str, Any]],
        y_tolerance: int = 30,
        max_distance_x: int = 200,
    ) -> List[Dict[str, Any]]:
        """Find texts horizontally to the right of an anchor keyword.

        Args:
            anchor_keyword: Keyword to search for (e.g., "ПРОВЕРЕНО", "ГОДНО")
            detections: List of text detections from header region
            y_tolerance: Maximum vertical offset (pixels)
            max_distance_x: Maximum horizontal distance to the right (pixels)

        Returns:
            List of detection dicts found to the right
        """
        anchor_lower = anchor_keyword.lower()
        anchor_detections = [
            d for d in detections if anchor_lower in d.get("text", "").lower()
        ]

        if not anchor_detections:
            return []

        candidates = []
        for anchor_det in anchor_detections:
            anchor_x = anchor_det["center"][0]
            anchor_x_right = max([p[0] for p in anchor_det["bbox"]])
            anchor_y = anchor_det["center"][1]

            for det in detections:
                if det == anchor_det:
                    continue

                det_x = det["center"][0]
                det_x_left = min([p[0] for p in det["bbox"]])
                det_y = det["center"][1]

                # Check if text is to the right and on same line
                x_distance = det_x_left - anchor_x_right
                y_distance = abs(det_y - anchor_y)

                if 0 < x_distance <= max_distance_x and y_distance <= y_tolerance:
                    candidates.append((det, x_distance))

        # Sort by X distance (closest first)
        candidates.sort(key=lambda x: x[1])
        return [det for det, _ in candidates]

    def _assemble_date_from_parts(
        self, date_texts: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Assemble date from multiple text elements.

        Args:
            date_texts: List of detection dicts containing date parts

        Returns:
            Assembled date string in DD/MM/YYYY format or None
        """
        if not date_texts:
            return None

        # Extract numeric parts and sort by X coordinate (left to right)
        parts = []
        for det in date_texts:
            text = det.get("text", "").strip()
            # Extract numbers from text
            numbers = re.findall(r"\d+", text)
            if numbers:
                for num in numbers:
                    parts.append((num, det["center"][0]))

        if not parts:
            return None

        # Sort by X coordinate
        parts.sort(key=lambda x: x[1])
        numeric_parts = [part[0] for part in parts]

        # Try to assemble date: expect DD, MM, YYYY or DD, MM, YY, YY
        if len(numeric_parts) >= 3:
            day = numeric_parts[0].zfill(2)
            month = numeric_parts[1].zfill(2)

            # Handle year: could be 4 digits or 2+2 digits
            if len(numeric_parts) >= 4 and len(numeric_parts[2]) == 2:
                year = numeric_parts[2] + numeric_parts[3]
            elif len(numeric_parts[2]) == 4:
                year = numeric_parts[2]
            elif len(numeric_parts) >= 4:
                year = numeric_parts[2] + numeric_parts[3]
            else:
                year = numeric_parts[2]

            # Validate year length
            if len(year) == 2:
                year = "20" + year  # Assume 20XX for 2-digit years

            return f"{day}/{month}/{year}"

        return None

    def _detect_sticker_data(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[StickerData]:
        """Detect sticker with part details (PRIORITY SOURCE).

        Sticker characteristics:
        - Compact group (x_range < 400px, y_range < 300px)
        - Keywords: "строка:", "кол-во:", "план", "дата:", "заказа", "клиент"
        - Usually in upper-left quadrant

        Args:
            detections: List of text detections from header region

        Returns:
            StickerData if sticker detected, None otherwise
        """
        sticker_keywords = [
            "строка:",
            "кол-во:",
            "количество:",
            "план",
            "дата:",
            "заказа",
            "клиент",
        ]

        # Filter candidates
        candidates = [
            d
            for d in detections
            if any(kw in d.get("text", "").lower() for kw in sticker_keywords)
        ]

        if len(candidates) < 2:
            return None

        # Check spatial compactness
        x_coords = [d["center"][0] for d in candidates]
        y_coords = [d["center"][1] for d in candidates]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # X range: 16% of image width (≈400px at 2500px)
        # Y range: 8.5% of image height (≈300px at 3500px)
        if x_range > self._x_pixels(16.0) or y_range > self._y_pixels(8.5):
            return None

        self._logger.info("Sticker detected with %d fields", len(candidates))

        # Extract fields
        sticker = StickerData()

        for det in candidates:
            text = det.get("text", "")
            text_lower = text.lower()
            confidence = det.get("confidence", 0.0)

            # Extract part_line_number
            if "строка" in text_lower:
                match = re.search(r"строка[:\s]*(\d+)", text_lower)
                if match:
                    sticker.part_line_number = FieldValue(
                        value=match.group(1),
                        confidence=confidence,
                        source="sticker",
                    )

            # Extract quantity_ordered
            if "кол-во" in text_lower or "количество" in text_lower:
                match = re.search(r"(\d+)", text)
                if match:
                    sticker.quantity_ordered = FieldValue(
                        value=match.group(1),
                        confidence=confidence,
                        source="sticker",
                    )

            # Extract order_number
            if "заказа" in text_lower or "№ заказа" in text_lower:
                match = re.search(r"(\d+)", text)
                if match:
                    sticker.order_number = FieldValue(
                        value=match.group(1),
                        confidence=confidence,
                        source="sticker",
                    )

            # Extract order_date
            if "дата" in text_lower:
                match = re.search(
                    r"(\d{1,2})[.\s](\d{1,2})[.\s](\d{4})", text
                )
                if match:
                    sticker.order_date = FieldValue(
                        value=f"{match.group(1)}.{match.group(2)}.{match.group(3)}",
                        confidence=confidence,
                        source="sticker",
                    )

            # Extract client
            if "клиент" in text_lower:
                # Extract text after "клиент:" or "клиент"
                match = re.search(r"клиент[:\s]+(.+)", text_lower)
                if match:
                    sticker.client = FieldValue(
                        value=match.group(1).strip(),
                        confidence=confidence,
                        source="sticker",
                    )

        # Return sticker only if we found at least part_line_number
        if sticker.part_line_number is not None:
            return sticker

        return None

    def _extract_act_number(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract act number from below "Номер" header.

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with act number or None
        """
        # Find texts below "Номер" header
        # Y tolerance: 2.9% of image height (≈100px at 3500px)
        # X tolerance: 2% of image width (≈50px at 2500px)
        texts_below = self._find_texts_below_header(
            "Номер", detections,
            max_distance_y=self._y_pixels(2.9),
            max_distance_x=self._x_pixels(2.0)
        )

        if not texts_below:
            return None

        # Look for first numeric text
        for det in texts_below:
            text = det.get("text", "").strip()
            # Check if it's a number (could be "030" or "030/25" format)
            if re.match(r"^\d+", text):
                # Extract the number part
                match = re.search(r"^(\d+)", text)
                if match:
                    return FieldValue(
                        value=match.group(1),
                        confidence=det.get("confidence", 0.0),
                        source="header",
                        validated=True,
                    )

        return None

    def _extract_act_date(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract date by assembling parts between 'Номер' and 'Рев' headers.

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with date in DD/MM/YYYY format or None
        """
        # Find all three headers
        nomer_header = self._find_column_header("Номер", detections)
        data_header = self._find_column_header("Дата", detections)
        rev_header = self._find_column_header("Рев", detections)

        if not data_header:
            return None

        header_y = data_header["center"][1]

        # Define X-zone: BETWEEN Номер and Рев (or use full range if headers missing)
        x_min = nomer_header["center"][0] if nomer_header else 0
        x_max = rev_header["center"][0] if rev_header else self._image_width

        # Search for numeric texts in the zone
        # Y tolerance: 2% of image height (≈70px at 3500px)
        y_tolerance = self._y_pixels(2.0)

        date_parts = []
        for det in detections:
            if det in [nomer_header, data_header, rev_header]:
                continue

            det_x = det["center"][0]
            det_y = det["center"][1]
            text = det.get("text", "").strip()

            # Check for numeric content
            if re.search(r"\d+", text):
                y_distance = abs(det_y - header_y)
                in_x_range = x_min < det_x < x_max

                if in_x_range and y_distance < y_tolerance:
                    date_parts.append(det)

        if not date_parts:
            return None

        # Assemble date from parts
        assembled_date = self._assemble_date_from_parts(date_parts)
        if assembled_date:
            avg_confidence = sum(d.get("confidence", 0.0) for d in date_parts) / len(date_parts)
            return FieldValue(
                value=assembled_date,
                confidence=avg_confidence,
                source="header",
                validated=True,
            )

        return None

    def _extract_template_revision(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract template revision from below "Рев" header in upper right corner.

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with template revision (e.g., A3, A4) or None
        """
        # Find texts below "Рев" header
        # Y tolerance: 2.9% of image height (≈100px at 3500px)
        # X tolerance: 2% of image width (≈50px at 2500px)
        texts_below = self._find_texts_below_header(
            "Рев", detections,
            max_distance_y=self._y_pixels(2.9),
            max_distance_x=self._x_pixels(2.0)
        )

        if not texts_below:
            return None

        # Look for first alphanumeric text (format: A3, A4, etc.)
        for det in texts_below:
            text = det.get("text", "").strip()
            # Match pattern like "A3", "A4", "B1", etc.
            match = re.search(r"([A-ZА-Я])\s*(\d+)", text)
            if match:
                return FieldValue(
                    value=f"{match.group(1)}{match.group(2)}",
                    confidence=det.get("confidence", 0.0),
                    source="header",
                    validated=True,
                )
            # Also try if text is already in correct format
            if re.match(r"^[A-ZА-Я]\d+$", text):
                return FieldValue(
                    value=text,
                    confidence=det.get("confidence", 0.0),
                    source="header",
                    validated=True,
                )

        return None

    def _extract_quantity_field(
        self, detections: List[Dict[str, Any]], keyword: str
    ) -> Optional[FieldValue]:
        """Extract numeric value from "КОЛ-ВО" column to the right of status keyword.

        Args:
            detections: List of text detections from header region
            keyword: Keyword to search for (e.g., "проверено", "годно", "дефектами")

        Returns:
            FieldValue with quantity or None
        """
        # Map input keywords to status keywords
        keyword_mapping = {
            "проверено": "ПРОВЕРЕНО",
            "годно": "ГОДНО",
            "дефектами": "С ДЕФЕКТАМИ",
        }

        keyword_lower = keyword.lower()
        status_keyword = keyword_mapping.get(keyword_lower)

        if not status_keyword:
            # Fallback: use original keyword
            status_keyword = keyword.upper()

        # First, find "КОЛ-ВО" header to identify the column
        kolvo_header = self._find_column_header("КОЛ-ВО", detections)
        if not kolvo_header:
            # Try alternative spelling
            kolvo_header = self._find_column_header("кол-во", detections)

        # Find status keyword detections
        status_detections = [
            d
            for d in detections
            if status_keyword.lower() in d.get("text", "").lower()
        ]

        if not status_detections:
            return None

        # For each status keyword, find number to its right
        for status_det in status_detections:
            status_x = status_det["center"][0]
            status_x_right = max([p[0] for p in status_det["bbox"]])
            status_y = status_det["center"][1]

            # Find texts to the right of this specific status detection
            texts_to_right = []
            for det in detections:
                if det == status_det:
                    continue

                det_x = det["center"][0]
                det_x_left = min([p[0] for p in det["bbox"]])
                det_y = det["center"][1]

                # Check if text is to the right and on same line
                x_distance = det_x_left - status_x_right
                y_distance = abs(det_y - status_y)

                # X tolerance: 12% of image width (≈300px at 2500px)
                # Y tolerance: 1.5% of image height (≈50px at 3500px)
                x_tolerance = self._x_pixels(12.0)
                y_tolerance = self._y_pixels(1.5)

                if 0 < x_distance <= x_tolerance and y_distance <= y_tolerance:
                    texts_to_right.append((det, x_distance))

            # Sort by X distance (closest first)
            texts_to_right.sort(key=lambda x: x[1])

            # Look for numeric text
            for det, _ in texts_to_right:
                text = det.get("text", "").strip()
                # Check if it's a pure number
                match = re.search(r"^(\d+)$", text)
                if match:
                    # Verify it's in the "КОЛ-ВО" column area if we found the header
                    if kolvo_header:
                        kolvo_x = kolvo_header["center"][0]
                        det_x = det["center"][0]
                        # Check if number is roughly aligned with КОЛ-ВО column
                        # X alignment tolerance: 4% of image width (≈100px at 2500px)
                        if abs(det_x - kolvo_x) < self._x_pixels(4.0):
                            return FieldValue(
                                value=match.group(1),
                                confidence=det.get("confidence", 0.0),
                                source="header",
                            )
                    else:
                        # If no КОЛ-ВО header found, just return the number
                        return FieldValue(
                            value=match.group(1),
                            confidence=det.get("confidence", 0.0),
                            source="header",
                        )

        return None

    def _extract_control_type(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract control type (операционный, входной, выходной).

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with control type or None
        """
        control_types = ["операционный", "входной", "выходной"]

        # Look for checkmark or control type text
        for det in detections:
            text_lower = det.get("text", "").lower()

            # Check for checkmark symbols
            if any(
                symbol in text_lower
                for symbol in ["✓", "☑", "☐", "v", "v", "x", "х"]
            ):
                # Look for control type nearby
                det_y = det["center"][1]
                nearby = [
                    d
                    for d in detections
                    if abs(d["center"][1] - det_y) < 50
                ]

                for nearby_det in nearby:
                    text = nearby_det.get("text", "").lower()
                    for control_type in control_types:
                        if control_type in text:
                            return FieldValue(
                                value=control_type,
                                confidence=nearby_det.get("confidence", 0.0),
                                source="header",
                            )

            # Direct match
            for control_type in control_types:
                if control_type in text_lower:
                    return FieldValue(
                        value=control_type,
                        confidence=det.get("confidence", 0.0),
                        source="header",
                    )

        return None

    def _extract_inspector_name(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract inspector name near "КОНТРОЛЕР ОТК" or "ФИО" label.

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with inspector name or None
        """
        # Find "КОНТРОЛЕР ОТК", "ФИО" or similar
        controller_keywords = ["контролер", "контролёр", "отк", "фио"]

        controller_detections = [
            d
            for d in detections
            if any(kw in d.get("text", "").lower() for kw in controller_keywords)
        ]

        if not controller_detections:
            return None

        # Look for name below or to the right of controller label
        candidates = []
        for controller_det in controller_detections:
            controller_x = controller_det["center"][0]
            controller_y = controller_det["center"][1]
            controller_x_right = max([p[0] for p in controller_det["bbox"]])

            # Expanded search area: Y + 0.6% to Y + 4.3%, X ± 8%, X + 2% to X + 16%
            y_offset_min = self._y_pixels(0.6)
            y_offset_max = self._y_pixels(4.3)
            x_tolerance = self._x_pixels(8.0)
            x_right_min = self._x_pixels(2.0)
            x_right_max = self._x_pixels(16.0)

            nearby = [
                d
                for d in detections
                if controller_y + y_offset_min < d["center"][1] < controller_y + y_offset_max
                and (
                    abs(d["center"][0] - controller_x) < x_tolerance  # Below
                    or (controller_x_right + x_right_min < d["center"][0] < controller_x_right + x_right_max)  # To the right
                )
            ]

            for nearby_det in nearby:
                text = nearby_det.get("text", "").strip()
                confidence = nearby_det.get("confidence", 0.0)

                # Skip very short texts or pure numbers
                if len(text) < 2 or text.isdigit():  # 3 → 2
                    continue

                # Accept texts with confidence > 0.4 (handwritten signatures may have lower confidence)
                if confidence < 0.4:  # 0.5 → 0.4
                    continue

                # Name typically has 1-4 words (accept single-word surnames)
                words = text.split()
                if 1 <= len(words) <= 4:  # 2 → 1 (accept single-word names)
                    # Check if it looks like a name (has alphabetic characters)
                    if any(word and word.isalpha() for word in words):
                        # Bonus for uppercase (better confidence)
                        has_uppercase = any(word and word[0].isupper() for word in words)
                        adjusted_conf = confidence * 1.1 if has_uppercase else confidence
                        candidates.append((nearby_det, adjusted_conf))

        # Fallback: accept any alphabetic text with confidence >= 0.3
        if not candidates:
            for controller_det in controller_detections:
                controller_x = controller_det["center"][0]
                controller_y = controller_det["center"][1]
                controller_x_right = max([p[0] for p in controller_det["bbox"]])

                y_offset_min = self._y_pixels(0.6)
                y_offset_max = self._y_pixels(4.3)
                x_tolerance = self._x_pixels(8.0)
                x_right_min = self._x_pixels(2.0)
                x_right_max = self._x_pixels(16.0)

                nearby = [
                    d
                    for d in detections
                    if controller_y + y_offset_min < d["center"][1] < controller_y + y_offset_max
                    and (
                        abs(d["center"][0] - controller_x) < x_tolerance
                        or (controller_x_right + x_right_min < d["center"][0] < controller_x_right + x_right_max)
                    )
                ]

                for nearby_det in nearby:
                    text = nearby_det.get("text", "").strip()
                    confidence = nearby_det.get("confidence", 0.0)
                    if len(text) >= 2 and confidence >= 0.3:
                        words = text.split()
                        if any(word.isalpha() for word in words):
                            candidates.append((nearby_det, confidence * 0.9))  # Lower weight

        if not candidates:
            return None

        # Sort by confidence (highest first) and return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_det, best_confidence = candidates[0]

        return FieldValue(
            value=best_det.get("text", "").strip(),
            confidence=best_confidence,
            source="header",
        )

    def _extract_part_line_number(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract part line number (handwritten, not from sticker).

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with part line number or None
        """
        # Look for "Строка:" label
        for det in detections:
            text_lower = det.get("text", "").lower()
            if "строка" in text_lower:
                # Extract number from same or nearby detection
                match = re.search(r"(\d+)", det.get("text", ""))
                if match:
                    return FieldValue(
                        value=match.group(1),
                        confidence=det.get("confidence", 0.0),
                        source="header",
                    )

                # Look nearby
                det_x = det["center"][0]
                det_y = det["center"][1]
                nearby = [
                    d
                    for d in detections
                    if abs(d["center"][1] - det_y) < 30
                    and det_x + 20 < d["center"][0] < det_x + 200
                ]

                for nearby_det in nearby:
                    match = re.search(r"^(\d+)$", nearby_det.get("text", "").strip())
                    if match:
                        return FieldValue(
                            value=match.group(1),
                            confidence=nearby_det.get("confidence", 0.0),
                            source="header",
                        )

        return None

    def _extract_part_designation(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract part designation (handwritten).

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with part designation or None
        """
        # Look for "Обозначение:" or "Изделие:" label
        for det in detections:
            text_lower = det.get("text", "").lower()
            if "обозначение" in text_lower or "изделие" in text_lower:
                # Look for text to the right
                det_x = det["center"][0]
                det_y = det["center"][1]
                nearby = [
                    d
                    for d in detections
                    if abs(d["center"][1] - det_y) < 30
                    and det_x + 20 < d["center"][0] < det_x + 400
                ]

                if nearby:
                    # Take first nearby text as designation
                    text = nearby[0].get("text", "").strip()
                    if text and len(text) > 2:
                        return FieldValue(
                            value=text,
                            confidence=nearby[0].get("confidence", 0.0),
                            source="header",
                        )

        return None

    def _extract_part_name(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[FieldValue]:
        """Extract part name (handwritten).

        Args:
            detections: List of text detections from header region

        Returns:
            FieldValue with part name or None
        """
        # Look for "Наименование:" label
        for det in detections:
            text_lower = det.get("text", "").lower()
            if "наименование" in text_lower:
                # Look for text to the right
                det_x = det["center"][0]
                det_y = det["center"][1]
                nearby = [
                    d
                    for d in detections
                    if abs(d["center"][1] - det_y) < 30
                    and det_x + 20 < d["center"][0] < det_x + 400
                ]

                if nearby:
                    # Take first nearby text as name
                    text = nearby[0].get("text", "").strip()
                    if text and len(text) > 1:
                        return FieldValue(
                            value=text,
                            confidence=nearby[0].get("confidence", 0.0),
                            source="header",
                        )

        return None

    def _extract_defects(
        self, defects_texts: List[Dict[str, Any]]
    ) -> List[DefectBlock]:
        """Extract defects table data.

        Args:
            defects_texts: List of text detections from defects region

        Returns:
            List of DefectBlock structures
        """
        if not defects_texts:
            return []

        # Split into horizontal blocks
        blocks = self._detect_horizontal_blocks(defects_texts)

        result = []
        for block_detections in blocks:
            block_data = self._parse_defect_block(block_detections)
            if block_data:
                result.append(block_data)

        self._logger.info(
            "Extracted %d defect blocks from defects zone", len(result)
        )

        return result

    def _extract_defects_from_table(
        self, table_data: Dict[str, Any]
    ) -> List[DefectBlock]:
        """Extract defects from structured table data.

        Args:
            table_data: Table data dictionary with 'cells' and 'grid' keys

        Returns:
            List of DefectBlock structures
        """
        cells = table_data.get("cells", [])
        if not cells:
            self._logger.warning("Table data has no cells")
            return []

        self._logger.info(
            "Extracting defects from structured table data (%d cells)", len(cells)
        )

        # Group cells by row
        rows_dict: Dict[int, List[Dict[str, Any]]] = {}
        for cell in cells:
            row_idx = cell.get("row_idx", 0)
            if row_idx not in rows_dict:
                rows_dict[row_idx] = []
            rows_dict[row_idx].append(cell)

        # Sort rows by index
        sorted_rows = sorted(rows_dict.keys())

        # First row is typically header - skip it
        data_rows = sorted_rows[1:] if len(sorted_rows) > 1 else sorted_rows

        # For now, create a single block with all rows
        # In the future, we could split into multiple blocks based on cell content
        rows_list = []
        for row_idx in data_rows:
            row_cells = rows_dict[row_idx]
            # Sort cells by column index
            row_cells.sort(key=lambda c: c.get("col_idx", 0))

            # Extract row data
            row_data = DefectRow()
            row_data.source_texts = [c.get("text", "") for c in row_cells]
            row_data.confidence_avg = (
                sum(c.get("confidence", 0.0) for c in row_cells) / len(row_cells)
                if row_cells
                else 0.0
            )

            # Map cells to row fields based on position
            if len(row_cells) > 0:
                row_data.row_number = row_cells[0].get("text", "").strip()
            if len(row_cells) > 1:
                row_data.parameter = row_cells[1].get("text", "").strip()
            if len(row_cells) > 2:
                row_data.fact_value = row_cells[2].get("text", "").strip()
            if len(row_cells) > 3:
                row_data.deviation = row_cells[3].get("text", "").strip()
            if len(row_cells) > 4:
                row_data.quantity = row_cells[4].get("text", "").strip()

            rows_list.append(row_data)

        # Create a single defect block
        # In the future, we could classify blocks based on cell content
        block = DefectBlock(
            block_type="other",  # Will be classified later if needed
            block_title="",  # No title in structured data
            column_headers=[],  # Headers not extracted from table data yet
            rows=rows_list,
        )

        self._logger.info(
            "Extracted %d rows from structured table data", len(rows_list)
        )

        return [block]

    def _detect_horizontal_blocks(
        self, detections: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Split defects zone into 2-3 horizontal blocks by X gaps.

        Args:
            detections: List of text detections from defects region

        Returns:
            List of detection groups (blocks)
        """
        if not detections:
            return []

        # Sort by X coordinate
        sorted_dets = sorted(detections, key=lambda d: d["center"][0])

        blocks = []
        current_block = [sorted_dets[0]]

        for i in range(1, len(sorted_dets)):
            gap = sorted_dets[i]["center"][0] - sorted_dets[i - 1]["center"][0]

            # Horizontal separator threshold: 12% of image width (≈300px at 2500px)
            if gap > self._x_pixels(12.0):
                blocks.append(current_block)
                current_block = [sorted_dets[i]]
            else:
                current_block.append(sorted_dets[i])

        blocks.append(current_block)

        self._logger.debug(
            "Detected %d horizontal blocks in defects zone", len(blocks)
        )
        return blocks

    def _parse_defect_block(
        self, block_detections: List[Dict[str, Any]]
    ) -> Optional[DefectBlock]:
        """Parse one defects table block.

        Args:
            block_detections: List of text detections for one block

        Returns:
            DefectBlock structure or None
        """
        if not block_detections:
            return None

        # 1. Extract block title (top texts)
        sorted_by_y = sorted(block_detections, key=lambda d: d["center"][1])

        if not sorted_by_y:
            return None

        title_y_threshold = sorted_by_y[0]["center"][1] + 50
        title_texts = [
            d for d in sorted_by_y if d["center"][1] < title_y_threshold
        ]
        block_title = " ".join([d.get("text", "") for d in title_texts])

        # 2. Classify block type
        block_type = self._classify_defect_block(block_title)

        # 3. Get table data (below title)
        table_texts = [
            d for d in sorted_by_y if d["center"][1] >= title_y_threshold
        ]

        # 4. Group into rows
        rows = self._group_texts_into_rows(table_texts)

        if not rows:
            return None

        # 5. First row = column headers
        column_headers = [d.get("text", "") for d in rows[0]]

        # 6. Parse data rows
        data_rows = []
        for row_texts in rows[1:]:
            row_data = self._parse_defect_row(
                row_texts, column_headers, block_type
            )
            if row_data:
                data_rows.append(row_data)

        return DefectBlock(
            block_type=block_type,
            block_title=block_title,
            column_headers=column_headers,
            rows=data_rows,
        )

    def _classify_defect_block(self, title: str) -> str:
        """Classify block type by title keywords.

        Args:
            title: Block title text

        Returns:
            Block type: "geometry", "holes", "surface", or "other"
        """
        title_lower = title.lower()

        if "геометрия" in title_lower:
            return "geometry"
        elif "отверстия" in title_lower or "вал" in title_lower:
            return "holes"
        elif "поверхность" in title_lower:
            return "surface"
        else:
            return "other"

    def _group_texts_into_rows(
        self, texts: List[Dict[str, Any]], y_tolerance: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """Group texts into table rows by Y-coordinate proximity.

        Args:
            texts: List of text detections
            y_tolerance: Y-coordinate tolerance for grouping (pixels). If None, uses 0.6% of image height.

        Returns:
            List of rows, each row is a list of text detections
        """
        if not texts:
            return []

        # Default tolerance: 0.6% of image height (≈20px at 3500px)
        if y_tolerance is None:
            y_tolerance = self._y_pixels(0.6)

        sorted_texts = sorted(texts, key=lambda t: (t["center"][1], t["center"][0]))

        rows = []
        current_row = [sorted_texts[0]]
        current_y = sorted_texts[0]["center"][1]

        for text in sorted_texts[1:]:
            if abs(text["center"][1] - current_y) <= y_tolerance:
                current_row.append(text)
            else:
                rows.append(current_row)
                current_row = [text]
                current_y = text["center"][1]

        rows.append(current_row)

        self._logger.debug(
            "Grouped %d texts into %d table rows", len(texts), len(rows)
        )
        return rows

    def _parse_defect_row(
        self,
        row_texts: List[Dict[str, Any]],
        headers: List[str],
        block_type: str,
    ) -> DefectRow:
        """Parse single defects table row.

        Args:
            row_texts: List of text detections for one row
            headers: Column headers
            block_type: Block type (geometry/holes/surface)

        Returns:
            DefectRow structure
        """
        # Sort texts left to right
        sorted_texts = sorted(row_texts, key=lambda t: t["center"][0])

        # Create row
        row = DefectRow()
        row.source_texts = [t.get("text", "") for t in sorted_texts]
        row.confidence_avg = (
            sum(t.get("confidence", 0) for t in sorted_texts) / len(sorted_texts)
            if sorted_texts
            else 0.0
        )

        # Simple positional mapping based on column count
        if len(sorted_texts) > 0:
            row.row_number = sorted_texts[0].get("text", "").strip()

        if block_type == "surface":
            # Surface block: №, ДЕФЕКТ, КОЛ-ВО
            if len(sorted_texts) > 1:
                row.defect = sorted_texts[1].get("text", "").strip()
            if len(sorted_texts) > 2:
                row.quantity = sorted_texts[2].get("text", "").strip()
        else:
            # Geometry/holes block: №, ПАРАМЕТР, ФАКТ/СТАТУС, ОТКЛ, КОЛ-ВО
            if len(sorted_texts) > 1:
                row.parameter = sorted_texts[1].get("text", "").strip()
            if len(sorted_texts) > 2:
                row.fact_value = sorted_texts[2].get("text", "").strip()
            if len(sorted_texts) > 3:
                row.deviation = sorted_texts[3].get("text", "").strip()
            if len(sorted_texts) > 4:
                row.quantity = sorted_texts[4].get("text", "").strip()

        return row

    def _extract_analysis(
        self, analysis_texts: List[Dict[str, Any]]
    ) -> AnalysisData:
        """Extract analysis section data.

        Args:
            analysis_texts: List of text detections from analysis region

        Returns:
            AnalysisData structure
        """
        if not analysis_texts:
            return AnalysisData()

        # Parse deviations table
        deviations = self._parse_analysis_deviations(analysis_texts)

        # Parse final decision
        final_decision = self._parse_final_decision(analysis_texts)

        return AnalysisData(deviations=deviations, final_decision=final_decision)

    def _parse_analysis_deviations(
        self, texts: List[Dict[str, Any]]
    ) -> List[AnalysisRow]:
        """Parse deviation analysis table.

        Args:
            texts: List of text detections from analysis region

        Returns:
            List of AnalysisRow structures
        """
        # Group into rows
        rows = self._group_texts_into_rows(texts)

        # Skip header row, parse data rows
        data_rows = []
        for row_texts in rows[1:]:  # Skip header
            sorted_texts = sorted(row_texts, key=lambda t: t["center"][0])

            if len(sorted_texts) < 2:
                continue

            row = AnalysisRow(
                row_number=(
                    sorted_texts[0].get("text", "").strip()
                    if len(sorted_texts) > 0
                    else None
                ),
                operation=(
                    sorted_texts[1].get("text", "").strip()
                    if len(sorted_texts) > 1
                    else None
                ),
                cause=(
                    sorted_texts[2].get("text", "").strip()
                    if len(sorted_texts) > 2
                    else None
                ),
                responsible=(
                    sorted_texts[3].get("text", "").strip()
                    if len(sorted_texts) > 3
                    else None
                ),
                decision=(
                    sorted_texts[4].get("text", "").strip()
                    if len(sorted_texts) > 4
                    else None
                ),
                confidence_avg=(
                    sum(t.get("confidence", 0) for t in sorted_texts)
                    / len(sorted_texts)
                    if sorted_texts
                    else 0.0
                ),
            )
            data_rows.append(row)

        return data_rows

    def _parse_final_decision(
        self, texts: List[Dict[str, Any]]
    ) -> Optional[FinalDecision]:
        """Parse final decision block.

        Args:
            texts: List of text detections from analysis region

        Returns:
            FinalDecision structure or None
        """
        decision = FinalDecision()

        # Action keywords
        action_keywords = {
            "использовать": "использовать",
            "доработать": "доработать",
            "не использовать": "не использовать",
        }

        # Look for action keywords
        for det in texts:
            text = det.get("text", "")
            text_lower = text.lower()

            for keyword, action in action_keywords.items():
                if keyword in text_lower:
                    decision.action = action

                    # Try to find quantity near this text
                    match = re.search(r"(\d+)", text)
                    if match:
                        try:
                            decision.quantity = int(match.group(1))
                        except ValueError:
                            pass

        # Extract names (look for text near "Руководитель" and "Представитель ОТК")
        for det in texts:
            text_lower = det.get("text", "").lower()

            if "руководитель" in text_lower:
                # Look for name nearby
                det_x = det["center"][0]
                det_y = det["center"][1]
                nearby = [
                    d
                    for d in texts
                    if abs(d["center"][1] - det_y) < 50
                    and det_x + 50 < d["center"][0] < det_x + 400
                ]

                for nearby_det in nearby:
                    text = nearby_det.get("text", "").strip()
                    words = text.split()
                    if 2 <= len(words) <= 4:
                        if any(word and word[0].isupper() for word in words):
                            decision.manager_name = text
                            break

            if "представитель" in text_lower and "отк" in text_lower:
                # Look for name nearby
                det_x = det["center"][0]
                det_y = det["center"][1]
                nearby = [
                    d
                    for d in texts
                    if abs(d["center"][1] - det_y) < 50
                    and det_x + 50 < d["center"][0] < det_x + 400
                ]

                for nearby_det in nearby:
                    text = nearby_det.get("text", "").strip()
                    words = text.split()
                    if 2 <= len(words) <= 4:
                        if any(word and word[0].isupper() for word in words):
                            decision.otk_representative = text
                            break

        # Return decision only if action is found
        if decision.action:
            return decision

        return None

    def _validate_extracted_data(
        self,
        header: HeaderData,
        defects: List[DefectBlock],
        analysis: AnalysisData,
    ) -> Dict[str, Any]:
        """Validate mandatory fields and mark suspicious values.

        Args:
            header: Header data structure
            defects: List of defect blocks
            analysis: Analysis data structure

        Returns:
            Validation results dictionary
        """
        mandatory_missing = []
        suspicious_fields = []
        errors = []

        # Check mandatory header fields
        for field_name in self._mandatory_fields:
            field_value = getattr(header, field_name, None)

            if field_value is None:
                mandatory_missing.append(field_name)
                errors.append(
                    {
                        "field": field_name,
                        "message": "Field is missing",
                        "severity": "error",
                    }
                )
                self._logger.warning("Mandatory field '%s' is missing", field_name)
            elif isinstance(field_value, FieldValue):
                # Check confidence
                if field_value.confidence < 0.7:
                    field_value.suspicious = True
                    suspicious_fields.append(field_name)
                    errors.append(
                        {
                            "field": field_name,
                            "message": f"Low confidence: {field_value.confidence:.3f}",
                            "severity": "warning",
                        }
                    )
                    self._logger.warning(
                        "Mandatory field '%s' has low confidence: %.3f",
                        field_name,
                        field_value.confidence,
                    )

        return {
            "total_fields_validated": len(self._mandatory_fields),
            "mandatory_fields_missing": mandatory_missing,
            "mandatory_fields_missing_count": len(mandatory_missing),
            "suspicious_fields": suspicious_fields,
            "errors": errors,
        }

