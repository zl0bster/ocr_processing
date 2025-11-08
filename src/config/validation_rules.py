"""Validation rules for QC form fields.

This module defines validation patterns and rules for different field types
found in quality control forms, including dates, numbers, and text patterns.
"""

import re
from typing import Any, Dict, List, Optional

# Validation rules for different field types
VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    # Act number format: digits/digits (e.g., "001/2025", "123/24")
    "act_number": {
        "pattern": r"^\d+/\d+$",
        "required": True,
        "example": "001/2025",
        "description": "Act number in format: number/year"
    },
    
    # Date format: DD.MM.YYYY or D.M.YYYY
    "date": {
        "pattern": r"^\d{1,2}\.\d{1,2}\.\d{4}$",
        "required": True,
        "example": "15.10.2025",
        "description": "Date in DD.MM.YYYY format"
    },
    
    # Revision number: typically 01, 02, etc.
    "revision": {
        "pattern": r"^\d{2}$",
        "required": False,
        "example": "01",
        "description": "Two-digit revision number"
    },
    
    # Quantities: positive integers
    "quantity": {
        "pattern": r"^\d+$",
        "type": "integer",
        "min": 0,
        "required": False,
        "example": "10",
        "description": "Positive integer quantity"
    },
    
    # Decimal measurements: numbers with optional decimal point
    "measurement": {
        "pattern": r"^\d+\.?\d*$",
        "type": "float",
        "required": False,
        "example": "100.05",
        "description": "Numeric measurement value"
    },
    
    # Measurement with tolerance: number ± tolerance (e.g., "100±0.1")
    "measurement_with_tolerance": {
        "pattern": r"^\d+\.?\d*[±]\d+\.?\d*$",
        "required": False,
        "example": "100±0.1",
        "description": "Measurement with tolerance (e.g., 100±0.1)"
    },
    
    # Status/decision: one of predefined values
    "status": {
        "allowed_values": ["годен", "брак", "доработать", "утилизировать"],
        "required": False,
        "example": "годен",
        "description": "Status or decision value"
    },
    
    # Defect code: typically letter+digits
    "defect_code": {
        "pattern": r"^[А-ЯA-Z]\d{3,}$",
        "required": False,
        "example": "Д001",
        "description": "Defect code (letter + digits)"
    },
    
    # Inspector name: Cyrillic text with initials
    "inspector": {
        "pattern": r"^[А-Яа-я\s]+\s+[А-Я]\.[А-Я]\.$",
        "required": False,
        "example": "Иванов И.И.",
        "description": "Inspector name with initials"
    },
    
    # General text field: non-empty text
    "text": {
        "min_length": 1,
        "required": False,
        "example": "Описание дефекта",
        "description": "General text field"
    }
}

# Confidence thresholds for validation warnings
CONFIDENCE_THRESHOLDS = {
    "high": 0.9,      # High confidence - reliable data
    "medium": 0.7,    # Medium confidence - acceptable
    "low": 0.5,       # Low confidence - warning threshold
    "critical": 0.3   # Critical - likely incorrect
}

# Required fields for complete document
REQUIRED_FIELDS = [
    "act_number",
    "date"
]


class ValidationRule:
    """Represents a validation rule for a field type."""
    
    def __init__(self, rule_name: str, rule_config: Dict[str, Any]):
        self.name = rule_name
        self.pattern = rule_config.get("pattern")
        self.required = rule_config.get("required", False)
        self.type = rule_config.get("type", "string")
        self.min_value = rule_config.get("min")
        self.max_value = rule_config.get("max")
        self.min_length = rule_config.get("min_length")
        self.max_length = rule_config.get("max_length")
        self.allowed_values = rule_config.get("allowed_values", [])
        self.example = rule_config.get("example", "")
        self.description = rule_config.get("description", "")
        
        # Compile regex pattern if provided
        self._regex = re.compile(self.pattern) if self.pattern else None
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a value against this rule.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passed
            - error_message: None if valid, error description if invalid
        """
        # Check if required and missing
        if self.required and (value is None or value == ""):
            return False, f"Required field is missing"
        
        # If not required and empty, skip validation
        if not self.required and (value is None or value == ""):
            return True, None
        
        # Convert value to string for pattern matching
        value_str = str(value).strip()
        
        # Check allowed values
        if self.allowed_values and value_str.lower() not in [v.lower() for v in self.allowed_values]:
            return False, f"Value must be one of: {', '.join(self.allowed_values)}"
        
        # Check pattern
        if self._regex and not self._regex.match(value_str):
            return False, f"Value does not match expected pattern. Example: {self.example}"
        
        # Check length constraints
        if self.min_length and len(value_str) < self.min_length:
            return False, f"Value too short (min length: {self.min_length})"
        
        if self.max_length and len(value_str) > self.max_length:
            return False, f"Value too long (max length: {self.max_length})"
        
        # Type-specific validation
        if self.type == "integer":
            try:
                int_value = int(value_str)
                if self.min_value is not None and int_value < self.min_value:
                    return False, f"Value below minimum ({self.min_value})"
                if self.max_value is not None and int_value > self.max_value:
                    return False, f"Value above maximum ({self.max_value})"
            except ValueError:
                return False, "Value must be an integer"
        
        elif self.type == "float":
            try:
                float_value = float(value_str)
                if self.min_value is not None and float_value < self.min_value:
                    return False, f"Value below minimum ({self.min_value})"
                if self.max_value is not None and float_value > self.max_value:
                    return False, f"Value above maximum ({self.max_value})"
            except ValueError:
                return False, "Value must be a number"
        
        return True, None


def get_rule(rule_name: str) -> Optional[ValidationRule]:
    """Get a validation rule by name.
    
    Args:
        rule_name: Name of the rule to retrieve
        
    Returns:
        ValidationRule instance or None if not found
    """
    rule_config = VALIDATION_RULES.get(rule_name)
    if rule_config:
        return ValidationRule(rule_name, rule_config)
    return None


def get_all_rules() -> Dict[str, ValidationRule]:
    """Get all available validation rules.
    
    Returns:
        Dictionary mapping rule names to ValidationRule instances
    """
    return {
        name: ValidationRule(name, config)
        for name, config in VALIDATION_RULES.items()
    }


def infer_field_type(text: str) -> str:
    """Infer the most likely field type based on text content.
    
    Args:
        text: Text to analyze
        
    Returns:
        Inferred field type name
    """
    text = text.strip()
    
    # Try to match against patterns
    for rule_name, rule_config in VALIDATION_RULES.items():
        if "pattern" in rule_config:
            pattern = rule_config["pattern"]
            if re.match(pattern, text):
                return rule_name
    
    # Default to text type
    return "text"


def validate_confidence(confidence: float, field_importance: str = "normal") -> tuple[bool, str]:
    """Validate if confidence score meets threshold requirements.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        field_importance: Importance level ("critical", "high", "normal", "low")
        
    Returns:
        Tuple of (is_acceptable, warning_message)
    """
    # Adjust threshold based on field importance
    if field_importance == "critical":
        threshold = CONFIDENCE_THRESHOLDS["medium"]
        label = "critical field"
    elif field_importance == "high":
        threshold = CONFIDENCE_THRESHOLDS["low"]
        label = "important field"
    else:
        threshold = CONFIDENCE_THRESHOLDS["critical"]
        label = "field"
    
    if confidence < threshold:
        return False, f"Low confidence for {label}: {confidence:.2f} (threshold: {threshold:.2f})"
    
    if confidence < CONFIDENCE_THRESHOLDS["low"]:
        return True, f"Below recommended confidence for {label}: {confidence:.2f}"
    
    return True, ""

