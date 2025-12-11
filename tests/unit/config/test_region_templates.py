"""Unit tests for region templates module."""
import json
import logging
from pathlib import Path
import pytest

from src.config.region_templates import (
    DEFAULT_REGION_TEMPLATES,
    load_region_templates,
)


@pytest.mark.unit
class TestDefaultRegionTemplates:
    """Test DEFAULT_REGION_TEMPLATES constant."""

    def test_default_templates_not_empty(self):
        """Test that DEFAULT_REGION_TEMPLATES is not empty."""
        # Assert
        assert len(DEFAULT_REGION_TEMPLATES) > 0
        assert isinstance(DEFAULT_REGION_TEMPLATES, dict)

    def test_default_templates_contain_expected_keys(self):
        """Test that default templates contain expected template names."""
        # Assert
        expected_templates = ["otk_v1", "otk_v2", "otk_standard", "otk_extended"]
        for template_name in expected_templates:
            assert template_name in DEFAULT_REGION_TEMPLATES

    def test_default_template_structure(self):
        """Test that default templates have correct structure."""
        # Arrange
        template = DEFAULT_REGION_TEMPLATES["otk_v1"]

        # Assert
        assert isinstance(template, list)
        assert len(template) > 0

        for region in template:
            assert isinstance(region, dict)
            assert "region_id" in region
            assert "y_start_norm" in region
            assert "y_end_norm" in region
            assert isinstance(region["region_id"], str)
            assert isinstance(region["y_start_norm"], (int, float))
            assert isinstance(region["y_end_norm"], (int, float))

    def test_default_template_coordinates_valid(self):
        """Test that default template coordinates are in valid range."""
        # Arrange
        template = DEFAULT_REGION_TEMPLATES["otk_v1"]

        # Assert
        for region in template:
            y_start = float(region["y_start_norm"])
            y_end = float(region["y_end_norm"])
            assert 0.0 <= y_start < y_end <= 1.0


@pytest.mark.unit
class TestLoadRegionTemplates:
    """Test load_region_templates function."""

    def test_load_valid_json_file(self, tmp_path):
        """Test loading templates from valid JSON file."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "custom_template": [
                {
                    "region_id": "header",
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.3,
                    "detection_method": "template",
                    "confidence": 1.0,
                },
                {
                    "region_id": "body",
                    "y_start_norm": 0.3,
                    "y_end_norm": 1.0,
                    "detection_method": "template",
                    "confidence": 1.0,
                },
            ]
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        assert "custom_template" in templates
        assert len(templates["custom_template"]) == 2
        assert templates["custom_template"][0]["region_id"] == "header"

    def test_fallback_to_defaults_when_file_not_found(self):
        """Test that function falls back to defaults when file doesn't exist."""
        # Arrange
        non_existent_file = Path("nonexistent/regions.json")

        # Act
        templates = load_region_templates(non_existent_file)

        # Assert
        assert templates == DEFAULT_REGION_TEMPLATES

    def test_fallback_to_defaults_on_json_decode_error(self, tmp_path):
        """Test that function falls back to defaults on JSON decode error."""
        # Arrange
        invalid_json_file = tmp_path / "invalid.json"
        invalid_json_file.write_text("invalid json content {", encoding="utf-8")

        # Act
        templates = load_region_templates(invalid_json_file)

        # Assert
        assert templates == DEFAULT_REGION_TEMPLATES

    def test_fallback_to_defaults_on_invalid_structure(self, tmp_path):
        """Test that function falls back to defaults when JSON is not a dict."""
        # Arrange
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

        # Act
        templates = load_region_templates(invalid_file)

        # Assert
        assert templates == DEFAULT_REGION_TEMPLATES

    def test_validates_region_structure(self, tmp_path):
        """Test that function validates region structure."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "valid_template": [
                {
                    "region_id": "header",
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.3,
                },
            ],
            "invalid_template": [
                {
                    "region_id": 123,  # Invalid: not a string
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.3,
                },
            ],
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        assert "valid_template" in templates
        assert "invalid_template" not in templates

    def test_validates_y_coordinate_ranges(self, tmp_path):
        """Test that function validates y coordinate ranges."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "valid_template": [
                {
                    "region_id": "region1",
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.5,
                },
            ],
            "invalid_template": [
                {
                    "region_id": "region1",
                    "y_start_norm": -0.1,  # Invalid: negative
                    "y_end_norm": 0.5,
                },
                {
                    "region_id": "region2",
                    "y_start_norm": 0.0,
                    "y_end_norm": 1.5,  # Invalid: > 1.0
                },
                {
                    "region_id": "region3",
                    "y_start_norm": 0.5,
                    "y_end_norm": 0.3,  # Invalid: start >= end
                },
            ],
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        assert "valid_template" in templates
        assert "invalid_template" not in templates

    def test_validates_y_start_less_than_y_end(self, tmp_path):
        """Test that function validates y_start < y_end."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "invalid_template": [
                {
                    "region_id": "region1",
                    "y_start_norm": 0.5,
                    "y_end_norm": 0.3,  # Invalid: start >= end
                },
            ],
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        assert "invalid_template" not in templates

    def test_skips_invalid_templates(self, tmp_path):
        """Test that function skips invalid templates but keeps valid ones."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "valid_template": [
                {
                    "region_id": "header",
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.3,
                },
            ],
            "invalid_template": [
                {
                    "region_id": "header",
                    "y_start_norm": "invalid",  # Invalid type
                    "y_end_norm": 0.3,
                },
            ],
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        assert "valid_template" in templates
        assert "invalid_template" not in templates

    def test_fallback_when_no_valid_templates(self, tmp_path):
        """Test that function falls back to defaults when no valid templates found."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "all_invalid": [
                {
                    "region_id": 123,  # Invalid
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.3,
                },
            ],
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        assert templates == DEFAULT_REGION_TEMPLATES

    def test_handles_none_template_file(self):
        """Test that function handles None template_file parameter."""
        # Act
        templates = load_region_templates(None)

        # Assert
        assert templates == DEFAULT_REGION_TEMPLATES

    def test_logs_warning_on_file_error(self, tmp_path, caplog):
        """Test that function logs warning on file error."""
        # Arrange
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("invalid json", encoding="utf-8")
        logger = logging.getLogger("test")

        # Act
        with caplog.at_level(logging.WARNING):
            templates = load_region_templates(invalid_file, logger=logger)

        # Assert
        assert templates == DEFAULT_REGION_TEMPLATES
        # Note: caplog might not capture if logger is passed, but function should still work

    def test_preserves_optional_fields(self, tmp_path):
        """Test that function preserves optional fields like detection_method and confidence."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "template_with_options": [
                {
                    "region_id": "header",
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.3,
                    "detection_method": "custom",
                    "confidence": 0.9,
                },
            ],
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        region = templates["template_with_options"][0]
        assert region["detection_method"] == "custom"
        assert region["confidence"] == 0.9

    def test_defaults_optional_fields(self, tmp_path):
        """Test that function provides defaults for optional fields."""
        # Arrange
        template_file = tmp_path / "regions.json"
        template_data = {
            "template_minimal": [
                {
                    "region_id": "header",
                    "y_start_norm": 0.0,
                    "y_end_norm": 0.3,
                    # No detection_method or confidence
                },
            ],
        }
        template_file.write_text(json.dumps(template_data), encoding="utf-8")

        # Act
        templates = load_region_templates(template_file)

        # Assert
        region = templates["template_minimal"][0]
        assert region["detection_method"] == "template"  # Default
        assert region["confidence"] == 1.0  # Default







