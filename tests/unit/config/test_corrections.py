"""Unit tests for OCR error correction dictionary."""
import pytest

from src.config.corrections import (
    OCR_CORRECTIONS,
    FUZZY_CORRECTIONS,
    get_correction,
    apply_corrections_to_text_list,
)


@pytest.mark.unit
class TestGetCorrection:
    """Test get_correction function."""

    @pytest.mark.parametrize(
        "original,expected",
        [
            ("Homep", "Номер"),
            ("Номео", "Номер"),
            ("НомеD", "Номер"),
            ("PeB", "Рев"),
            ("Pев", "Рев"),
            ("РeB", "Рев"),
            ("годeн", "годен"),
            ("браk", "брак"),
            ("ГЕОМЕТРUЯ", "ГЕОМЕТРИЯ"),
            ("ОТВЕРСТUЯ", "ОТВЕРСТИЯ"),
            ("РЕЗЬБA", "РЕЗЬБА"),
        ],
    )
    def test_exact_match_corrections(self, original, expected):
        """Test that exact matches from OCR_CORRECTIONS are corrected."""
        # Arrange & Act
        corrected, was_corrected = get_correction(original)

        # Assert
        assert corrected == expected
        assert was_corrected is True

    @pytest.mark.parametrize(
        "original,expected",
        [
            ("номер", "Номер"),
            ("дата", "Дата"),
            ("рев", "Рев"),
            ("изделие", "Изделие"),
            ("годен", "годен"),
            ("брак", "брак"),
        ],
    )
    def test_fuzzy_match_case_insensitive(self, original, expected):
        """Test that fuzzy matches (case-insensitive) are corrected."""
        # Arrange & Act
        corrected, was_corrected = get_correction(original)

        # Assert
        assert corrected == expected
        assert was_corrected is True

    @pytest.mark.parametrize(
        "text",
        [
            "unknown_text",  # Not in dictionary
            "12345",  # Numbers
            "",  # Empty string
        ],
    )
    def test_no_correction_for_valid_text(self, text):
        """Test that valid text or unknown text returns unchanged."""
        # Arrange & Act
        corrected, was_corrected = get_correction(text)

        # Assert
        assert corrected == text
        assert was_corrected is False

    def test_correction_for_fuzzy_matches(self):
        """Test that fuzzy matches (case variations) are corrected."""
        # Note: "Номер" and "Дата" are in FUZZY_CORRECTIONS, so they get corrected
        # Arrange & Act
        corrected_nomer, was_corrected_nomer = get_correction("Номер")
        corrected_data, was_corrected_data = get_correction("Дата")

        # Assert
        assert was_corrected_nomer is True
        assert was_corrected_data is True
        assert corrected_nomer == "Номер"
        assert corrected_data == "Дата"

    def test_return_tuple_structure(self):
        """Test that get_correction returns correct tuple structure."""
        # Arrange & Act
        result = get_correction("Homep")

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        corrected_text, was_corrected = result
        assert isinstance(corrected_text, str)
        assert isinstance(was_corrected, bool)


@pytest.mark.unit
class TestApplyCorrectionsToTextList:
    """Test apply_corrections_to_text_list function."""

    def test_processes_list_correctly(self):
        """Test that corrections are applied to all items in list."""
        # Arrange
        texts = ["Homep", "Дaта", "годeн", "unknown"]

        # Act
        corrected_texts, correction_records = apply_corrections_to_text_list(texts)

        # Assert
        assert len(corrected_texts) == 4
        assert corrected_texts[0] == "Номер"
        assert corrected_texts[1] == "Дата"
        assert corrected_texts[2] == "годен"
        assert corrected_texts[3] == "unknown"

    def test_returns_correction_records(self):
        """Test that correction records contain correct metadata."""
        # Arrange
        texts = ["Homep", "Дaта", "correct_text"]

        # Act
        corrected_texts, correction_records = apply_corrections_to_text_list(texts)

        # Assert
        assert len(correction_records) == 2
        assert correction_records[0]["index"] == 0
        assert correction_records[0]["original"] == "Homep"
        assert correction_records[0]["corrected"] == "Номер"
        assert correction_records[1]["index"] == 1
        assert correction_records[1]["original"] == "Дaта"
        assert correction_records[1]["corrected"] == "Дата"

    def test_empty_list(self):
        """Test that empty list returns empty results."""
        # Arrange
        texts = []

        # Act
        corrected_texts, correction_records = apply_corrections_to_text_list(texts)

        # Assert
        assert corrected_texts == []
        assert correction_records == []

    def test_no_corrections_needed(self):
        """Test that texts not in correction dictionaries return no corrections."""
        # Arrange - use texts that are not in OCR_CORRECTIONS or FUZZY_CORRECTIONS
        texts = ["unknown_text", "12345", "some_other_text"]

        # Act
        corrected_texts, correction_records = apply_corrections_to_text_list(texts)

        # Assert
        assert corrected_texts == texts
        assert correction_records == []

    def test_fuzzy_corrections_applied(self):
        """Test that fuzzy corrections are applied correctly."""
        # Arrange - texts that match FUZZY_CORRECTIONS
        texts = ["Номер", "Дата", "годен"]

        # Act
        corrected_texts, correction_records = apply_corrections_to_text_list(texts)

        # Assert
        assert len(correction_records) == 3  # All three are in FUZZY_CORRECTIONS
        assert all(record["original"] == record["corrected"] for record in correction_records)

    def test_multiple_corrections_in_list(self):
        """Test that multiple corrections are applied correctly."""
        # Arrange
        texts = ["Homep", "Дaта", "годeн", "браk", "ГЕОМЕТРUЯ"]

        # Act
        corrected_texts, correction_records = apply_corrections_to_text_list(texts)

        # Assert
        assert len(correction_records) == 5
        assert all(record["index"] == idx for idx, record in enumerate(correction_records))
        assert all(record["original"] != record["corrected"] for record in correction_records)

    def test_correction_record_structure(self):
        """Test that correction records have required fields."""
        # Arrange
        texts = ["Homep"]

        # Act
        corrected_texts, correction_records = apply_corrections_to_text_list(texts)

        # Assert
        assert len(correction_records) == 1
        record = correction_records[0]
        assert "index" in record
        assert "original" in record
        assert "corrected" in record
        assert isinstance(record["index"], int)
        assert isinstance(record["original"], str)
        assert isinstance(record["corrected"], str)


@pytest.mark.unit
class TestCorrectionsDictionary:
    """Test corrections dictionary structure."""

    def test_ocr_corrections_not_empty(self):
        """Test that OCR_CORRECTIONS dictionary is not empty."""
        # Assert
        assert len(OCR_CORRECTIONS) > 0
        assert isinstance(OCR_CORRECTIONS, dict)

    def test_fuzzy_corrections_not_empty(self):
        """Test that FUZZY_CORRECTIONS dictionary is not empty."""
        # Assert
        assert len(FUZZY_CORRECTIONS) > 0
        assert isinstance(FUZZY_CORRECTIONS, dict)

    def test_all_corrections_are_strings(self):
        """Test that all correction values are strings."""
        # Assert
        for key, value in OCR_CORRECTIONS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

        for key, value in FUZZY_CORRECTIONS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

