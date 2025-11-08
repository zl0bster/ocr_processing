"""Field validation module for verifying OCR-extracted data quality.

This module validates corrected OCR results against defined field rules,
checking formats, data types, and confidence scores.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings
from config.validation_rules import (
    get_rule,
    infer_field_type,
    validate_confidence,
    REQUIRED_FIELDS,
    CONFIDENCE_THRESHOLDS
)


@dataclass(frozen=True)
class ValidationResult:
    """Result information returned by the field validator."""
    
    output_path: Path
    duration_seconds: float
    total_fields: int
    validated_fields: int
    failed_validations: int
    validation_rate: float


class FieldValidator:
    """Validator for checking OCR field data against defined rules."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger

    def process(self, input_path: Path, output_path: Optional[Path] = None) -> ValidationResult:
        """Validate fields in corrected OCR results.
        
        Args:
            input_path: Path to corrected OCR results JSON file (-corrected.json)
            output_path: Optional custom output path for validated results
            
        Returns:
            ValidationResult with validation metrics
        """
        start_time = time.perf_counter()
        
        # Load corrected OCR results
        corrected_data = self._load_corrected_results(input_path)
        
        # Perform validation on text regions
        validation_results = self._validate_text_regions(corrected_data)
        
        # Create output structure with validation metadata
        output_data = self._create_output_structure(
            corrected_data, validation_results, start_time
        )
        
        # Save to JSON
        destination = output_path or self._build_output_path(input_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        duration = time.perf_counter() - start_time
        
        # Calculate metrics
        total_fields = validation_results["total_fields"]
        validated_fields = validation_results["validated_fields"]
        failed_validations = validation_results["failed_validations"]
        validation_rate = validated_fields / total_fields if total_fields > 0 else 0.0
        
        self._logger.info(
            "Field validation completed in %.3f seconds. Validated %d/%d fields (%.1f%%), %d failures",
            duration, validated_fields, total_fields, validation_rate * 100, failed_validations
        )
        
        if failed_validations > 0:
            self._logger.warning(
                "%d field validations failed - review validation_results in output",
                failed_validations
            )
        
        return ValidationResult(
            output_path=destination,
            duration_seconds=duration,
            total_fields=total_fields,
            validated_fields=validated_fields,
            failed_validations=failed_validations,
            validation_rate=validation_rate
        )

    def _load_corrected_results(self, path: Path) -> Dict[str, Any]:
        """Load corrected OCR results from JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Corrected results file '{path}' does not exist")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._logger.debug("Loaded corrected results from '%s'", path)
        return data

    def _validate_text_regions(self, corrected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all text regions in the corrected data.
        
        Args:
            corrected_data: Corrected OCR results dictionary
            
        Returns:
            Dictionary containing validation results and errors
        """
        text_regions = corrected_data.get("text_regions", [])
        
        validation_errors = []
        low_confidence_warnings = []
        validated_count = 0
        failed_count = 0
        
        for idx, region in enumerate(text_regions):
            text = region.get("text", "")
            confidence = region.get("confidence", 0.0)
            was_corrected = region.get("corrected", False)
            
            # Skip empty text
            if not text or text.strip() == "":
                continue
            
            # Infer field type based on text content
            field_type = infer_field_type(text)
            
            # Get validation rule for this field type
            rule = get_rule(field_type)
            
            if rule:
                # Validate the text against the rule
                is_valid, error_message = rule.validate(text)
                
                if is_valid:
                    validated_count += 1
                else:
                    failed_count += 1
                    validation_errors.append({
                        "index": idx,
                        "field": f"text_region_{idx}",
                        "field_type": field_type,
                        "value": text,
                        "error": error_message,
                        "confidence": confidence,
                        "corrected": was_corrected
                    })
                    
                    self._logger.debug(
                        "Validation failed at index %d (%s): %s - '%s'",
                        idx, field_type, error_message, text
                    )
            
            # Check confidence levels
            is_acceptable, warning = validate_confidence(confidence, "normal")
            if not is_acceptable or warning:
                low_confidence_warnings.append({
                    "index": idx,
                    "field": f"text_region_{idx}",
                    "value": text,
                    "confidence": confidence,
                    "warning": warning or f"Confidence below threshold",
                    "corrected": was_corrected
                })
                
                if not is_acceptable:
                    self._logger.warning(
                        "Low confidence at index %d: %.3f - '%s'",
                        idx, confidence, text[:50]
                    )
        
        return {
            "total_fields": len(text_regions),
            "validated_fields": validated_count,
            "failed_validations": failed_count,
            "errors": validation_errors,
            "low_confidence_warnings": low_confidence_warnings
        }

    def _create_output_structure(
        self,
        corrected_data: Dict[str, Any],
        validation_results: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Create output JSON structure with validation metadata.
        
        Args:
            corrected_data: Corrected OCR results
            validation_results: Validation results and errors
            start_time: Processing start time
            
        Returns:
            Complete output structure with all metadata
        """
        duration = time.perf_counter() - start_time
        
        # Get previous metrics
        previous_metrics = corrected_data.get("processing_metrics", {})
        
        # Build output structure
        output = {
            "document_info": corrected_data.get("document_info", {}),
            "text_regions": corrected_data.get("text_regions", []),
            "corrections_applied": corrected_data.get("corrections_applied", {}),
            "validation_results": {
                "total_fields": validation_results["total_fields"],
                "validated_fields": validation_results["validated_fields"],
                "failed_validations": validation_results["failed_validations"],
                "validation_rate": (
                    validation_results["validated_fields"] / validation_results["total_fields"]
                    if validation_results["total_fields"] > 0 else 0.0
                ),
                "errors": validation_results["errors"],
                "low_confidence_warnings": validation_results["low_confidence_warnings"]
            },
            "processing_metrics": {
                # Previous metrics
                "ocr_time_ms": previous_metrics.get("ocr_time_ms", 0),
                "correction_time_ms": previous_metrics.get("correction_time_ms", 0),
                
                # Validation metrics
                "validation_time_ms": int(duration * 1000),
                
                # Counts
                "texts_detected": previous_metrics.get("texts_detected", 0),
                "corrections_applied": previous_metrics.get("corrections_applied", 0),
                "fields_validated": validation_results["validated_fields"],
                "validation_failures": validation_results["failed_validations"],
                
                # Combined totals
                "total_time_ms": previous_metrics.get("total_time_ms", 0) + int(duration * 1000),
                "average_confidence": previous_metrics.get("average_confidence", 0.0)
            }
        }
        
        return output

    def _build_output_path(self, input_path: Path) -> Path:
        """Generate output path, keeping -corrected suffix for validated results.
        
        Args:
            input_path: Input corrected results file path
            
        Returns:
            Output path for validated results (same as input, overwrite)
        """
        # For validation, we typically update the corrected file in place
        # or keep the same name since validation adds metadata
        return input_path


def validate_date_format(date_str: str) -> tuple[bool, Optional[str]]:
    """Validate date string format (DD.MM.YYYY).
    
    Args:
        date_str: Date string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import re
    from datetime import datetime
    
    # Check format with regex
    pattern = r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$"
    match = re.match(pattern, date_str)
    
    if not match:
        return False, "Date must be in DD.MM.YYYY format"
    
    # Extract components
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    
    # Validate ranges
    if month < 1 or month > 12:
        return False, f"Invalid month: {month}"
    
    if day < 1 or day > 31:
        return False, f"Invalid day: {day}"
    
    # Try to create actual date to check validity (e.g., 31.02.2025 is invalid)
    try:
        datetime(year, month, day)
    except ValueError as e:
        return False, f"Invalid date: {e}"
    
    return True, None


def validate_numeric_field(value_str: str, allow_decimal: bool = True) -> tuple[bool, Optional[str]]:
    """Validate numeric field value.
    
    Args:
        value_str: String value to validate
        allow_decimal: Whether to allow decimal points
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import re
    
    if allow_decimal:
        pattern = r"^\d+\.?\d*$"
        error_msg = "Must be a valid number"
    else:
        pattern = r"^\d+$"
        error_msg = "Must be a valid integer"
    
    if not re.match(pattern, value_str.strip()):
        return False, error_msg
    
    return True, None

