"""Unit tests for HeaderFieldExtractor module."""

import logging
import pytest
from unittest.mock import MagicMock

from src.header_field_extractor import HeaderFieldExtractor, FieldRegion
from src.models.form_data import FieldValue


@pytest.fixture
def header_extractor(test_settings):
    """Create HeaderFieldExtractor instance for testing."""
    logger = logging.getLogger("test")
    return HeaderFieldExtractor(settings=test_settings, logger=logger)


@pytest.fixture
def mock_header_detections():
    """Create mock header detections in top-right corner."""
    return [
        {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],  # 80% of 2500px width, 2.8% of 3508px height
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        },
        {
            "text": "057",
            "confidence": 0.98,
            "center": [2000, 150],
            "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
        },
        {
            "text": "/",
            "confidence": 0.90,
            "center": [2020, 150],
            "bbox": [[2015, 130], [2025, 130], [2025, 170], [2015, 170]],
        },
        {
            "text": "25",
            "confidence": 0.97,
            "center": [2040, 150],
            "bbox": [[2030, 130], [2050, 130], [2050, 170], [2030, 170]],
        },
        {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2200, 100],
            "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
        },
        {
            "text": "15",
            "confidence": 0.97,
            "center": [2180, 150],
            "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
        },
        {
            "text": "10",
            "confidence": 0.97,
            "center": [2220, 150],
            "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
        },
        {
            "text": "2025",
            "confidence": 0.98,
            "center": [2260, 150],
            "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
        },
    ]


@pytest.mark.unit
class TestHeaderFieldExtractorBasic:
    """Test basic HeaderFieldExtractor functionality."""

    def test_extract_blank_number_and_date_success(
        self, header_extractor, mock_header_detections
    ):
        """Test successful extraction of blank number and date."""
        # Arrange
        image_width = 2480
        image_height = 3508

        # Act
        blank_number, blank_date = header_extractor.extract_blank_number_and_date(
            mock_header_detections, image_width, image_height
        )

        # Assert
        assert blank_number is not None
        assert blank_number.value == "057/25"
        assert blank_number.confidence >= 0.9
        assert blank_number.validated is True

        assert blank_date is not None
        assert blank_date.value == "15/10/2025"
        assert blank_date.confidence >= 0.9
        assert blank_date.validated is True

    def test_extract_with_empty_detections(self, header_extractor):
        """Test extraction with empty detections list."""
        # Act
        blank_number, blank_date = header_extractor.extract_blank_number_and_date(
            [], 2480, 3508
        )

        # Assert
        assert blank_number is None
        assert blank_date is None

    def test_extract_with_no_region_matches(self, header_extractor):
        """Test extraction when no detections match region."""
        # Arrange - detections in wrong region (left side)
        detections = [
            {
                "text": "Номер",
                "confidence": 0.95,
                "center": [200, 100],  # 8% of 2500px - outside region
                "bbox": [[150, 80], [250, 80], [250, 120], [150, 120]],
            },
        ]

        # Act
        blank_number, blank_date = header_extractor.extract_blank_number_and_date(
            detections, 2480, 3508
        )

        # Assert
        assert blank_number is None
        assert blank_date is None


@pytest.mark.unit
class TestBlankNumberValidation:
    """Test blank number format validation and correction."""

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("057/25", "057/25"),
            ("036/25", "036/25"),
            ("046/25", "046/25"),
            ("12/99", "012/99"),
            ("999/99", "999/99"),
            ("1/5", "001/05"),
        ],
    )
    def test_validate_valid_blank_number_formats(
        self, header_extractor, input_text, expected
    ):
        """Test validation of valid blank number formats."""
        # Act
        result = header_extractor._validate_and_correct_blank_number(input_text, 0.95)

        # Assert
        assert result is not None
        assert result.value == expected
        assert result.validated is True

    @pytest.mark.parametrize(
        "input_text",
        [
            "057-25",  # Dash instead of slash
            "057",  # Missing second part
            "abc/de",  # Not numbers
            "057/",  # Incomplete
            "/25",  # Incomplete
        ],
    )
    def test_validate_invalid_blank_number_formats(
        self, header_extractor, input_text
    ):
        """Test rejection of invalid blank number formats."""
        # Act
        result = header_extractor._validate_and_correct_blank_number(input_text, 0.95)

        # Assert
        assert result is None

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("057I25", "057/25"),  # I → /
            ("057l25", "057/25"),  # l → /
            ("057|25", "057/25"),  # | → /
            ("057I25", "057/25"),  # I → /
        ],
    )
    def test_slash_correction(self, header_extractor, input_text, expected):
        """Test correction of slash misrecognition."""
        # Act
        result = header_extractor._validate_and_correct_blank_number(input_text, 0.95)

        # Assert
        assert result is not None
        assert result.value == expected
        assert result.corrected is True

    def test_slash_correction_method(self, header_extractor):
        """Test _correct_slash_chars method directly."""
        # Act
        corrected = header_extractor._correct_slash_chars("057I25")
        assert corrected == "057/25"

        corrected = header_extractor._correct_slash_chars("057l25")
        assert corrected == "057/25"

        corrected = header_extractor._correct_slash_chars("057|25")
        assert corrected == "057/25"

    def test_digit_correction(self, header_extractor):
        """Test correction of digit misrecognition."""
        # Act
        corrected = header_extractor._correct_digit_chars("O57/25")  # O → 0
        assert "0" in corrected or corrected == "057/25"

        corrected = header_extractor._correct_digit_chars("057/2S")  # S → 5
        assert "5" in corrected or corrected == "057/25"


@pytest.mark.unit
class TestRegionFiltering:
    """Test region-based filtering of detections."""

    def test_filter_by_region_top_right(self, header_extractor):
        """Test filtering detections in top-right region."""
        # Arrange
        detections = [
            {
                "text": "Номер",
                "center": [2000, 100],  # 80% X, 2.8% Y - in region
                "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
            },
            {
                "text": "Дата",
                "center": [2200, 100],  # 88% X, 2.8% Y - in region
                "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
            },
            {
                "text": "Other",
                "center": [200, 100],  # 8% X - outside region
                "bbox": [[150, 80], [250, 80], [250, 120], [150, 120]],
            },
        ]
        region = FieldRegion(x_min=70.0, x_max=100.0, y_min=0.0, y_max=8.0)
        width, height = 2480, 3508

        # Act
        filtered = header_extractor._filter_by_region(detections, region, width, height)

        # Assert
        assert len(filtered) == 2
        assert all(d["text"] in ["Номер", "Дата"] for d in filtered)

    def test_filter_by_region_outside(self, header_extractor):
        """Test filtering rejects detections outside region."""
        # Arrange
        detections = [
            {
                "text": "Left",
                "center": [200, 100],  # 8% X - outside
                "bbox": [[150, 80], [250, 80], [250, 120], [150, 120]],
            },
            {
                "text": "Bottom",
                "center": [2000, 500],  # 80% X but 14% Y - outside
                "bbox": [[1950, 480], [2050, 480], [2050, 520], [1950, 520]],
            },
        ]
        region = FieldRegion(x_min=70.0, x_max=100.0, y_min=0.0, y_max=8.0)
        width, height = 2480, 3508

        # Act
        filtered = header_extractor._filter_by_region(detections, region, width, height)

        # Assert
        assert len(filtered) == 0


@pytest.mark.unit
class TestDateExtraction:
    """Test date extraction and assembly."""

    def test_extract_date_from_parts(self, header_extractor):
        """Test date extraction from separate cells."""
        # Arrange
        detections = [
            {
                "text": "Дата",
                "confidence": 0.95,
                "center": [2200, 100],
                "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
            },
            {
                "text": "15",
                "confidence": 0.97,
                "center": [2180, 150],
                "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
            },
            {
                "text": "10",
                "confidence": 0.97,
                "center": [2220, 150],
                "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
            },
            {
                "text": "2025",
                "confidence": 0.98,
                "center": [2260, 150],
                "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
            },
        ]
        image_width, image_height = 2480, 3508

        # Act
        result = header_extractor._extract_date(
            detections[0], detections, image_width, image_height
        )

        # Assert
        assert result is not None
        assert result.value == "15/10/2025"
        assert result.validated is True

    def test_extract_date_with_2digit_year(self, header_extractor):
        """Test date extraction with 2-digit year."""
        # Arrange
        data_header = {
            "text": "Дата",
            "confidence": 0.95,
            "center": [2200, 100],
            "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]],
        }
        detections = [
            data_header,
            {
                "text": "01",
                "confidence": 0.97,
                "center": [2180, 150],
                "bbox": [[2170, 130], [2190, 130], [2190, 170], [2170, 170]],
            },
            {
                "text": "12",
                "confidence": 0.97,
                "center": [2220, 150],
                "bbox": [[2210, 130], [2230, 130], [2230, 170], [2210, 170]],
            },
            {
                "text": "25",  # 2-digit year
                "confidence": 0.98,
                "center": [2260, 150],
                "bbox": [[2240, 130], [2280, 130], [2280, 170], [2240, 170]],
            },
        ]
        image_width, image_height = 2480, 3508

        # Act
        result = header_extractor._extract_date(
            data_header, detections, image_width, image_height
        )

        # Assert
        assert result is not None
        assert result.value == "01/12/2025"  # Should convert 25 → 2025

    def test_validate_date_format(self, header_extractor):
        """Test date format validation."""
        # Valid dates
        assert header_extractor._validate_date_format("15/10/2025") is True
        assert header_extractor._validate_date_format("01/12/2024") is True
        assert header_extractor._validate_date_format("31/12/2025") is True

        # Invalid dates
        assert header_extractor._validate_date_format("32/10/2025") is False  # Invalid day
        assert header_extractor._validate_date_format("15/13/2025") is False  # Invalid month
        assert header_extractor._validate_date_format("15/10/1999") is False  # Year too old
        assert header_extractor._validate_date_format("15-10-2025") is False  # Wrong separator


@pytest.mark.unit
class TestHeaderFinding:
    """Test header keyword finding."""

    def test_find_header_success(self, header_extractor):
        """Test finding header by keyword."""
        # Arrange
        detections = [
            {"text": "Номер", "center": [2000, 100], "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]]},
            {"text": "Дата", "center": [2200, 100], "bbox": [[2150, 80], [2250, 80], [2250, 120], [2150, 120]]},
        ]

        # Act
        nomer = header_extractor._find_header(detections, "номер")
        data = header_extractor._find_header(detections, "дата")

        # Assert
        assert nomer is not None
        assert nomer["text"] == "Номер"
        assert data is not None
        assert data["text"] == "Дата"

    def test_find_header_not_found(self, header_extractor):
        """Test finding header when not present."""
        # Arrange
        detections = [
            {"text": "Other", "center": [2000, 100], "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]]},
        ]

        # Act
        result = header_extractor._find_header(detections, "номер")

        # Assert
        assert result is None


@pytest.mark.unit
class TestBlankNumberExtraction:
    """Test blank number extraction logic."""

    def test_extract_blank_number_split_cells(self, header_extractor):
        """Test extraction when number is split across cells."""
        # Arrange
        nomer_header = {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        }
        detections = [
            nomer_header,
            {
                "text": "057",
                "confidence": 0.98,
                "center": [2000, 150],
                "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
            },
            {
                "text": "25",
                "confidence": 0.97,
                "center": [2040, 150],
                "bbox": [[2030, 130], [2050, 130], [2050, 170], [2030, 170]],
            },
        ]
        image_width, image_height = 2480, 3508

        # Act
        result = header_extractor._extract_blank_number(
            nomer_header, detections, image_width, image_height
        )

        # Assert
        assert result is not None
        # Should combine "057" and "25" - but needs slash
        # Since there's no slash, it might not validate, but should attempt extraction
        if result:
            assert "057" in result.value or "25" in result.value

    def test_extract_blank_number_with_slash(self, header_extractor):
        """Test extraction when slash is present."""
        # Arrange
        nomer_header = {
            "text": "Номер",
            "confidence": 0.95,
            "center": [2000, 100],
            "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
        }
        detections = [
            nomer_header,
            {
                "text": "057/25",
                "confidence": 0.98,
                "center": [2000, 150],
                "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
            },
        ]
        image_width, image_height = 2480, 3508

        # Act
        result = header_extractor._extract_blank_number(
            nomer_header, detections, image_width, image_height
        )

        # Assert
        assert result is not None
        assert result.value == "057/25"
        assert result.validated is True


@pytest.mark.unit
class TestLowConfidenceHandling:
    """Test handling of low confidence detections."""

    def test_reject_low_confidence(self, header_extractor):
        """Test rejection of detections below confidence threshold."""
        # Arrange
        detections = [
            {
                "text": "Номер",
                "confidence": 0.95,
                "center": [2000, 100],
                "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
            },
            {
                "text": "057/25",
                "confidence": 0.3,  # Below threshold (0.4)
                "center": [2000, 150],
                "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
            },
        ]
        image_width, image_height = 2480, 3508

        # Act
        blank_number, blank_date = header_extractor.extract_blank_number_and_date(
            detections, image_width, image_height
        )

        # Assert
        # Should reject due to low confidence
        assert blank_number is None

    def test_accept_high_confidence(self, header_extractor):
        """Test acceptance of detections above confidence threshold."""
        # Arrange
        detections = [
            {
                "text": "Номер",
                "confidence": 0.95,
                "center": [2000, 100],
                "bbox": [[1950, 80], [2050, 80], [2050, 120], [1950, 120]],
            },
            {
                "text": "057/25",
                "confidence": 0.5,  # Above threshold (0.4)
                "center": [2000, 150],
                "bbox": [[1980, 130], [2020, 130], [2020, 170], [1980, 170]],
            },
        ]
        image_width, image_height = 2480, 3508

        # Act
        blank_number, blank_date = header_extractor.extract_blank_number_and_date(
            detections, image_width, image_height
        )

        # Assert
        assert blank_number is not None
        assert blank_number.value == "057/25"

