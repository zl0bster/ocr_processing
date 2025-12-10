"""Unit tests for validation rules module."""
import pytest

from src.config.validation_rules import (
    VALIDATION_RULES,
    ValidationRule,
    get_rule,
    get_all_rules,
    infer_field_type,
    validate_confidence,
    REQUIRED_FIELDS,
    CONFIDENCE_THRESHOLDS,
)


@pytest.mark.unit
class TestValidationRuleActNumber:
    """Test ValidationRule for act_number field type."""

    def test_valid_act_number_formats(self):
        """Test valid act_number formats."""
        # Arrange
        rule = ValidationRule("act_number", VALIDATION_RULES["act_number"])

        # Act & Assert
        valid_cases = ["001/2025", "123/24", "1/2025", "999/99"]
        for value in valid_cases:
            is_valid, error = rule.validate(value)
            assert is_valid is True, f"'{value}' should be valid: {error}"
            assert error is None

    @pytest.mark.parametrize(
        "invalid_value",
        [
            "abc/2025",  # Non-numeric prefix
            "001/abc",  # Non-numeric suffix
            "001-2025",  # Wrong separator
            "001 2025",  # Space separator
            "001/2025/extra",  # Extra parts
            "/2025",  # Missing prefix
            "001/",  # Missing suffix
            "",  # Empty
        ],
    )
    def test_invalid_act_number_formats(self, invalid_value):
        """Test invalid act_number formats."""
        # Arrange
        rule = ValidationRule("act_number", VALIDATION_RULES["act_number"])

        # Act
        is_valid, error = rule.validate(invalid_value)

        # Assert
        assert is_valid is False
        assert error is not None

    def test_required_act_number_missing(self):
        """Test that missing required act_number returns error."""
        # Arrange
        rule = ValidationRule("act_number", VALIDATION_RULES["act_number"])

        # Act
        is_valid, error = rule.validate(None)

        # Assert
        assert is_valid is False
        assert "Required field is missing" in error


@pytest.mark.unit
class TestValidationRuleDate:
    """Test ValidationRule for date field type."""

    @pytest.mark.parametrize(
        "valid_date",
        [
            "15.10.2025",
            "1.1.2025",
            "31.12.2024",
            "5.5.2023",
        ],
    )
    def test_valid_date_formats(self, valid_date):
        """Test valid date formats."""
        # Arrange
        rule = ValidationRule("date", VALIDATION_RULES["date"])

        # Act
        is_valid, error = rule.validate(valid_date)

        # Assert
        assert is_valid is True
        assert error is None

    @pytest.mark.parametrize(
        "invalid_date",
        [
            "15/10/2025",  # Wrong separator
            "15-10-2025",  # Wrong separator
            "15.10.25",  # Short year
            "2025.10.15",  # Wrong order
            "",  # Empty
        ],
    )
    def test_invalid_date_formats(self, invalid_date):
        """Test invalid date formats."""
        # Arrange
        rule = ValidationRule("date", VALIDATION_RULES["date"])

        # Act
        is_valid, error = rule.validate(invalid_date)

        # Assert
        assert is_valid is False
        assert error is not None

    def test_date_pattern_only_checks_format(self):
        """Test that date pattern only checks format, not calendar validity."""
        # Note: The regex pattern r"^\d{1,2}\.\d{1,2}\.\d{4}$" only validates format,
        # not actual date validity. So "32.10.2025" and "15.13.2025" will pass pattern check.
        # Arrange
        rule = ValidationRule("date", VALIDATION_RULES["date"])

        # Act
        is_valid_32, _ = rule.validate("32.10.2025")
        is_valid_13, _ = rule.validate("15.13.2025")

        # Assert - pattern matches, but these are invalid calendar dates
        # The pattern validation passes, but these would fail in real date parsing
        assert is_valid_32 is True  # Pattern matches
        assert is_valid_13 is True  # Pattern matches

    def test_required_date_missing(self):
        """Test that missing required date returns error."""
        # Arrange
        rule = ValidationRule("date", VALIDATION_RULES["date"])

        # Act
        is_valid, error = rule.validate("")

        # Assert
        assert is_valid is False
        assert "Required field is missing" in error


@pytest.mark.unit
class TestValidationRuleQuantity:
    """Test ValidationRule for quantity field type."""

    @pytest.mark.parametrize("valid_quantity", ["0", "1", "10", "100", "9999"])
    def test_valid_quantities(self, valid_quantity):
        """Test valid quantity values."""
        # Arrange
        rule = ValidationRule("quantity", VALIDATION_RULES["quantity"])

        # Act
        is_valid, error = rule.validate(valid_quantity)

        # Assert
        assert is_valid is True
        assert error is None

    @pytest.mark.parametrize(
        "invalid_quantity",
        [
            "-1",  # Negative
            "1.5",  # Decimal
            "abc",  # Non-numeric
            "1.0",  # Float format
        ],
    )
    def test_invalid_quantities(self, invalid_quantity):
        """Test invalid quantity values."""
        # Arrange
        rule = ValidationRule("quantity", VALIDATION_RULES["quantity"])

        # Act
        is_valid, error = rule.validate(invalid_quantity)

        # Assert
        assert is_valid is False
        assert error is not None

    def test_optional_quantity_empty(self):
        """Test that empty optional quantity is valid."""
        # Arrange
        rule = ValidationRule("quantity", VALIDATION_RULES["quantity"])

        # Act
        is_valid, error = rule.validate("")

        # Assert
        assert is_valid is True
        assert error is None


@pytest.mark.unit
class TestValidationRuleMeasurement:
    """Test ValidationRule for measurement field type."""

    @pytest.mark.parametrize(
        "valid_measurement",
        [
            "100",
            "100.05",
            "0.5",
            "123.456",
        ],
    )
    def test_valid_measurements(self, valid_measurement):
        """Test valid measurement values."""
        # Arrange
        rule = ValidationRule("measurement", VALIDATION_RULES["measurement"])

        # Act
        is_valid, error = rule.validate(valid_measurement)

        # Assert
        assert is_valid is True
        assert error is None

    @pytest.mark.parametrize(
        "invalid_measurement",
        [
            "abc",
            "100mm",  # With unit
            "-100",  # Negative
        ],
    )
    def test_invalid_measurements(self, invalid_measurement):
        """Test invalid measurement values."""
        # Arrange
        rule = ValidationRule("measurement", VALIDATION_RULES["measurement"])

        # Act
        is_valid, error = rule.validate(invalid_measurement)

        # Assert
        assert is_valid is False
        assert error is not None


@pytest.mark.unit
class TestValidationRuleStatus:
    """Test ValidationRule for status field type."""

    @pytest.mark.parametrize(
        "valid_status",
        [
            "годен",
            "брак",
            "доработать",
            "утилизировать",
        ],
    )
    def test_valid_status_values(self, valid_status):
        """Test valid status values."""
        # Arrange
        rule = ValidationRule("status", VALIDATION_RULES["status"])

        # Act
        is_valid, error = rule.validate(valid_status)

        # Assert
        assert is_valid is True
        assert error is None

    @pytest.mark.parametrize(
        "invalid_status",
        [
            "invalid",
            "принят",
        ],
    )
    def test_invalid_status_values(self, invalid_status):
        """Test invalid status values."""
        # Arrange
        rule = ValidationRule("status", VALIDATION_RULES["status"])

        # Act
        is_valid, error = rule.validate(invalid_status)

        # Assert
        assert is_valid is False
        assert error is not None

    def test_status_case_insensitive(self):
        """Test that status validation is case-insensitive."""
        # Arrange
        rule = ValidationRule("status", VALIDATION_RULES["status"])

        # Act
        is_valid, error = rule.validate("Годен")  # Capitalized

        # Assert - case-insensitive check should pass
        assert is_valid is True
        assert error is None

    def test_status_empty_optional_field(self):
        """Test that empty status is valid for optional field."""
        # Arrange
        rule = ValidationRule("status", VALIDATION_RULES["status"])

        # Act
        is_valid, error = rule.validate("")

        # Assert - optional field, empty is valid
        assert is_valid is True
        assert error is None


@pytest.mark.unit
class TestValidationRuleGeneral:
    """Test general ValidationRule functionality."""

    def test_required_field_missing(self):
        """Test that required fields return error when missing."""
        # Arrange
        rule = ValidationRule("act_number", VALIDATION_RULES["act_number"])

        # Act
        is_valid, error = rule.validate(None)

        # Assert
        assert is_valid is False
        assert "Required field is missing" in error

    def test_optional_field_empty(self):
        """Test that optional fields accept empty values."""
        # Arrange
        rule = ValidationRule("quantity", VALIDATION_RULES["quantity"])

        # Act
        is_valid, error = rule.validate(None)

        # Assert
        assert is_valid is True
        assert error is None

    def test_validation_rule_initialization(self):
        """Test ValidationRule initialization."""
        # Arrange & Act
        rule = ValidationRule("act_number", VALIDATION_RULES["act_number"])

        # Assert
        assert rule.name == "act_number"
        assert rule.pattern is not None
        assert rule.required is True
        assert rule._regex is not None


@pytest.mark.unit
class TestGetRule:
    """Test get_rule function."""

    def test_get_existing_rule(self):
        """Test getting an existing rule."""
        # Act
        rule = get_rule("act_number")

        # Assert
        assert rule is not None
        assert isinstance(rule, ValidationRule)
        assert rule.name == "act_number"

    def test_get_nonexistent_rule(self):
        """Test getting a non-existent rule returns None."""
        # Act
        rule = get_rule("nonexistent_rule")

        # Assert
        assert rule is None

    @pytest.mark.parametrize(
        "rule_name",
        [
            "act_number",
            "date",
            "quantity",
            "measurement",
            "status",
        ],
    )
    def test_get_all_field_types(self, rule_name):
        """Test getting all field type rules."""
        # Act
        rule = get_rule(rule_name)

        # Assert
        assert rule is not None
        assert isinstance(rule, ValidationRule)


@pytest.mark.unit
class TestGetAllRules:
    """Test get_all_rules function."""

    def test_get_all_rules_returns_dict(self):
        """Test that get_all_rules returns a dictionary."""
        # Act
        rules = get_all_rules()

        # Assert
        assert isinstance(rules, dict)
        assert len(rules) > 0

    def test_all_rules_are_validation_rules(self):
        """Test that all returned rules are ValidationRule instances."""
        # Act
        rules = get_all_rules()

        # Assert
        for rule_name, rule in rules.items():
            assert isinstance(rule, ValidationRule)
            assert rule.name == rule_name

    def test_all_validation_rules_included(self):
        """Test that all VALIDATION_RULES are included."""
        # Act
        rules = get_all_rules()

        # Assert
        assert set(rules.keys()) == set(VALIDATION_RULES.keys())


@pytest.mark.unit
class TestInferFieldType:
    """Test infer_field_type function."""

    def test_infer_act_number(self):
        """Test inferring act_number type."""
        # Act
        field_type = infer_field_type("001/2025")

        # Assert
        assert field_type == "act_number"

    def test_infer_date(self):
        """Test inferring date type."""
        # Act
        field_type = infer_field_type("15.10.2025")

        # Assert
        assert field_type == "date"

    def test_infer_quantity(self):
        """Test inferring quantity type."""
        # Note: "10" matches revision pattern (^\d{2}$) first, so use a different number
        # Act
        field_type = infer_field_type("100")

        # Assert
        assert field_type == "quantity"

    def test_infer_measurement(self):
        """Test inferring measurement type."""
        # Act
        field_type = infer_field_type("100.05")

        # Assert
        assert field_type == "measurement"

    def test_infer_text_default(self):
        """Test that unknown text defaults to text type."""
        # Act
        field_type = infer_field_type("Some random text")

        # Assert
        assert field_type == "text"

    def test_infer_with_whitespace(self):
        """Test that whitespace is stripped before inference."""
        # Act
        field_type = infer_field_type("  001/2025  ")

        # Assert
        assert field_type == "act_number"


@pytest.mark.unit
class TestValidateConfidence:
    """Test validate_confidence function."""

    def test_high_confidence_critical_field(self):
        """Test high confidence for critical field."""
        # Act
        is_acceptable, warning = validate_confidence(0.95, "critical")

        # Assert
        assert is_acceptable is True
        assert warning == ""

    def test_low_confidence_critical_field(self):
        """Test low confidence for critical field."""
        # Act
        is_acceptable, warning = validate_confidence(0.5, "critical")

        # Assert
        assert is_acceptable is False
        assert "critical field" in warning.lower()

    def test_high_confidence_normal_field(self):
        """Test high confidence for normal field."""
        # Act
        is_acceptable, warning = validate_confidence(0.9, "normal")

        # Assert
        assert is_acceptable is True

    def test_low_confidence_normal_field(self):
        """Test low confidence for normal field."""
        # Note: Normal fields use CONFIDENCE_THRESHOLDS["critical"] = 0.3 as threshold
        # Act
        is_acceptable, warning = validate_confidence(0.2, "normal")

        # Assert
        assert is_acceptable is False  # 0.2 < 0.3 threshold
        assert "Low confidence" in warning

    @pytest.mark.parametrize(
        "confidence,importance,expected_acceptable",
        [
            (0.95, "critical", True),
            (0.7, "critical", True),  # At threshold (medium = 0.7)
            (0.5, "critical", False),  # Below threshold
            (0.9, "high", True),
            (0.6, "high", True),  # At threshold (low = 0.5, so 0.6 > 0.5)
            (0.4, "high", False),  # Below threshold (low = 0.5)
            (0.9, "normal", True),
            (0.4, "normal", True),  # Above critical threshold (0.3)
            (0.2, "normal", False),  # Below critical threshold (0.3)
        ],
    )
    def test_confidence_thresholds(self, confidence, importance, expected_acceptable):
        """Test confidence thresholds for different importance levels."""
        # Act
        is_acceptable, warning = validate_confidence(confidence, importance)

        # Assert
        assert is_acceptable == expected_acceptable


@pytest.mark.unit
class TestValidationRulesConstants:
    """Test validation rules constants."""

    def test_required_fields_defined(self):
        """Test that REQUIRED_FIELDS is defined."""
        # Assert
        assert isinstance(REQUIRED_FIELDS, list)
        assert len(REQUIRED_FIELDS) > 0
        assert "act_number" in REQUIRED_FIELDS
        assert "date" in REQUIRED_FIELDS

    def test_confidence_thresholds_defined(self):
        """Test that CONFIDENCE_THRESHOLDS is defined."""
        # Assert
        assert isinstance(CONFIDENCE_THRESHOLDS, dict)
        assert "high" in CONFIDENCE_THRESHOLDS
        assert "medium" in CONFIDENCE_THRESHOLDS
        assert "low" in CONFIDENCE_THRESHOLDS
        assert "critical" in CONFIDENCE_THRESHOLDS
        assert all(0.0 <= v <= 1.0 for v in CONFIDENCE_THRESHOLDS.values())

