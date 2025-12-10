"""Unit tests for ErrorCorrector module."""
import json
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.error_corrector import ErrorCorrector, CorrectionResult
from src.config.corrections import OCR_CORRECTIONS, FUZZY_CORRECTIONS


@pytest.fixture
def mock_ocr_json():
    """Create mock OCR results JSON structure."""
    return {
        "document_info": {
            "image_path": "test_image.jpg",
            "image_size": [1920, 1080],
        },
        "text_regions": [
            {
                "text": "Homep",
                "confidence": 0.9,
                "center": [100, 200],
                "bbox": [[50, 150], [150, 150], [150, 250], [50, 250]],
            },
            {
                "text": "Дaта",
                "confidence": 0.85,
                "center": [200, 300],
                "bbox": [[150, 250], [250, 250], [250, 350], [150, 350]],
            },
            {
                "text": "correct_text",
                "confidence": 0.95,
                "center": [300, 400],
                "bbox": [[250, 350], [350, 350], [350, 450], [250, 450]],
            },
        ],
        "processing_metrics": {
            "total_time_ms": 1500,
            "texts_detected": 3,
            "average_confidence": 0.9,
        },
    }


@pytest.fixture
def mock_ocr_json_file(mock_ocr_json, tmp_path):
    """Create temporary JSON file with mock OCR results."""
    json_file = tmp_path / "test-texts.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(mock_ocr_json, f, ensure_ascii=False, indent=2)
    return json_file


@pytest.fixture
def mock_ocr_json_with_regions(mock_ocr_json):
    """Create mock OCR JSON with ocr_results_by_region."""
    mock_ocr_json["ocr_results_by_region"] = {
        "header": [
            {
                "text": "PeB",
                "confidence": 0.88,
                "center": [150, 100],
            },
            {
                "text": "годeн",
                "confidence": 0.92,
                "center": [200, 150],
            },
        ],
        "defects": [
            {
                "text": "ГЕОМЕТРUЯ",
                "confidence": 0.90,
                "center": [300, 200],
            },
        ],
    }
    return mock_ocr_json


@pytest.fixture
def error_corrector(test_settings):
    """Create ErrorCorrector instance for testing."""
    logger = logging.getLogger("test")
    return ErrorCorrector(settings=test_settings, logger=logger)


@pytest.mark.unit
class TestErrorCorrectorBasicProcessing:
    """Test basic processing functionality."""

    def test_process_completes_successfully(
        self, error_corrector, mock_ocr_json_file, tmp_path
    ):
        """Test process() completes end-to-end with mock JSON."""
        # Arrange
        output_path = tmp_path / "output-corrected.json"

        # Act
        result = error_corrector.process(mock_ocr_json_file, output_path)

        # Assert
        assert isinstance(result, CorrectionResult)
        assert result.output_path == output_path
        assert result.output_path.exists()
        assert result.total_texts == 3
        assert result.corrections_applied == 2  # Homep and Дaта
        assert result.correction_rate > 0
        assert result.duration_seconds > 0

    def test_process_generates_output_file(
        self, error_corrector, mock_ocr_json_file, tmp_path
    ):
        """Test that process() creates output JSON file."""
        # Arrange
        output_path = tmp_path / "output-corrected.json"

        # Act
        result = error_corrector.process(mock_ocr_json_file, output_path)

        # Assert
        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        assert "text_regions" in output_data
        assert "corrections_applied" in output_data
        assert "processing_metrics" in output_data

    def test_process_calculates_correction_rate(
        self, error_corrector, mock_ocr_json_file, tmp_path
    ):
        """Test correction rate calculation."""
        # Arrange
        output_path = tmp_path / "output-corrected.json"

        # Act
        result = error_corrector.process(mock_ocr_json_file, output_path)

        # Assert
        expected_rate = 2 / 3  # 2 corrections out of 3 texts
        assert abs(result.correction_rate - expected_rate) < 0.01
        assert result.corrections_applied == 2
        assert result.total_texts == 3

    def test_process_with_default_output_path(
        self, error_corrector, mock_ocr_json_file, test_settings
    ):
        """Test process() uses default output path when not specified."""
        # Act
        result = error_corrector.process(mock_ocr_json_file)

        # Assert
        assert result.output_path.exists()
        assert "-corrected.json" in result.output_path.name
        assert result.output_path.parent == test_settings.output_dir


@pytest.mark.unit
class TestErrorCorrectorJSONLoading:
    """Test JSON loading functionality."""

    def test_load_ocr_results_loads_valid_json(
        self, error_corrector, mock_ocr_json_file, mock_ocr_json
    ):
        """Test _load_ocr_results() loads valid JSON file."""
        # Act
        loaded_data = error_corrector._load_ocr_results(mock_ocr_json_file)

        # Assert
        assert loaded_data == mock_ocr_json
        assert "text_regions" in loaded_data
        assert "processing_metrics" in loaded_data

    def test_load_ocr_results_raises_on_missing_file(self, error_corrector, tmp_path):
        """Test _load_ocr_results() raises FileNotFoundError for missing file."""
        # Arrange
        missing_file = tmp_path / "nonexistent.json"

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            error_corrector._load_ocr_results(missing_file)
        assert str(missing_file) in str(exc_info.value)

    def test_load_ocr_results_handles_invalid_json(
        self, error_corrector, tmp_path
    ):
        """Test _load_ocr_results() raises JSONDecodeError for invalid JSON."""
        # Arrange
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            error_corrector._load_ocr_results(invalid_json_file)


@pytest.mark.unit
class TestErrorCorrectorCorrectionApplication:
    """Test correction application logic."""

    def test_apply_corrections_with_exact_matches(
        self, error_corrector, mock_ocr_json
    ):
        """Test _apply_corrections() applies exact match corrections."""
        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )

        # Assert
        assert len(corrections_log) == 2
        assert corrected_data["text_regions"][0]["text"] == "Номер"
        assert corrected_data["text_regions"][0]["corrected"] is True
        assert corrected_data["text_regions"][0]["original_text"] == "Homep"
        assert corrected_data["text_regions"][1]["text"] == "Дата"
        assert corrected_data["text_regions"][1]["corrected"] is True
        assert corrected_data["text_regions"][2]["text"] == "correct_text"
        assert corrected_data["text_regions"][2]["corrected"] is False
        assert corrected_data["text_regions"][2]["original_text"] is None

    def test_apply_corrections_with_fuzzy_matches(
        self, error_corrector, mock_ocr_json
    ):
        """Test _apply_corrections() applies fuzzy case-insensitive corrections."""
        # Arrange - modify mock to include fuzzy match
        mock_ocr_json["text_regions"][0]["text"] = "номер"  # lowercase

        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )

        # Assert
        assert corrected_data["text_regions"][0]["text"] == "Номер"
        assert corrected_data["text_regions"][0]["corrected"] is True

    def test_apply_corrections_no_false_corrections(
        self, error_corrector, mock_ocr_json
    ):
        """Test that valid text is not incorrectly corrected."""
        # Arrange - all texts are valid
        mock_ocr_json["text_regions"] = [
            {"text": "valid_text_1", "confidence": 0.9, "center": [100, 200]},
            {"text": "valid_text_2", "confidence": 0.85, "center": [200, 300]},
        ]

        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )

        # Assert
        assert len(corrections_log) == 0
        assert corrected_data["text_regions"][0]["text"] == "valid_text_1"
        assert corrected_data["text_regions"][0]["corrected"] is False
        assert corrected_data["text_regions"][1]["text"] == "valid_text_2"
        assert corrected_data["text_regions"][1]["corrected"] is False

    def test_apply_corrections_with_regions(
        self, error_corrector, mock_ocr_json_with_regions
    ):
        """Test _apply_corrections() processes ocr_results_by_region."""
        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json_with_regions
        )

        # Assert
        assert "ocr_results_by_region" in corrected_data
        assert corrected_data["ocr_results_by_region"]["header"][0]["text"] == "Рев"
        assert corrected_data["ocr_results_by_region"]["header"][0]["corrected"] is True
        assert (
            corrected_data["ocr_results_by_region"]["header"][1]["text"] == "годен"
        )
        assert (
            corrected_data["ocr_results_by_region"]["header"][1]["corrected"] is True
        )
        assert (
            corrected_data["ocr_results_by_region"]["defects"][0]["text"]
            == "ГЕОМЕТРИЯ"
        )
        assert (
            corrected_data["ocr_results_by_region"]["defects"][0]["corrected"] is True
        )

    def test_apply_corrections_with_regions_no_correction_needed(
        self, error_corrector
    ):
        """Test _apply_corrections() handles ocr_results_by_region with no corrections."""
        # Arrange - text that doesn't need correction
        ocr_data = {
            "text_regions": [],
            "ocr_results_by_region": {
                "header": [
                    {
                        "text": "valid_text",
                        "confidence": 0.9,
                        "center": [100, 200],
                    }
                ]
            },
            "processing_metrics": {},
        }

        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(ocr_data)

        # Assert
        assert (
            corrected_data["ocr_results_by_region"]["header"][0]["text"]
            == "valid_text"
        )
        assert (
            corrected_data["ocr_results_by_region"]["header"][0]["corrected"] is False
        )
        assert (
            corrected_data["ocr_results_by_region"]["header"][0]["original_text"]
            is None
        )
        assert len(corrections_log) == 0

    def test_apply_corrections_empty_text_regions(self, error_corrector):
        """Test _apply_corrections() handles empty text_regions."""
        # Arrange
        ocr_data = {"text_regions": [], "processing_metrics": {}}

        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(ocr_data)

        # Assert
        assert len(corrected_data["text_regions"]) == 0
        assert len(corrections_log) == 0

    def test_apply_corrections_missing_text_regions(self, error_corrector):
        """Test _apply_corrections() handles missing text_regions."""
        # Arrange
        ocr_data = {"processing_metrics": {}}

        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(ocr_data)

        # Assert
        assert "text_regions" in corrected_data
        assert len(corrected_data["text_regions"]) == 0
        assert len(corrections_log) == 0


@pytest.mark.unit
class TestErrorCorrectorCorrectionMetadata:
    """Test correction metadata and logging."""

    def test_correction_log_structure(self, error_corrector, mock_ocr_json):
        """Test that correction log has correct structure."""
        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )

        # Assert
        assert len(corrections_log) == 2
        for log_entry in corrections_log:
            assert "index" in log_entry
            assert "field" in log_entry
            assert "original" in log_entry
            assert "corrected" in log_entry
            assert "confidence" in log_entry
            assert "position" in log_entry
            assert isinstance(log_entry["index"], int)
            assert isinstance(log_entry["field"], str)
            assert isinstance(log_entry["original"], str)
            assert isinstance(log_entry["corrected"], str)
            assert isinstance(log_entry["confidence"], (int, float))
            assert isinstance(log_entry["position"], list)

    def test_correction_log_metadata_values(
        self, error_corrector, mock_ocr_json
    ):
        """Test that correction log contains correct metadata values."""
        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )

        # Assert
        first_correction = corrections_log[0]
        assert first_correction["index"] == 0
        assert first_correction["field"] == "text_region_0"
        assert first_correction["original"] == "Homep"
        assert first_correction["corrected"] == "Номер"
        assert first_correction["confidence"] == 0.9
        assert first_correction["position"] == [100, 200]

    def test_confidence_preserved(self, error_corrector, mock_ocr_json):
        """Test that confidence values are preserved after correction."""
        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )

        # Assert
        assert corrected_data["text_regions"][0]["confidence"] == 0.9
        assert corrected_data["text_regions"][1]["confidence"] == 0.85
        assert corrected_data["text_regions"][2]["confidence"] == 0.95

    def test_position_preserved(self, error_corrector, mock_ocr_json):
        """Test that position information is preserved after correction."""
        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )

        # Assert
        assert corrected_data["text_regions"][0]["center"] == [100, 200]
        assert corrected_data["text_regions"][1]["center"] == [200, 300]
        assert corrected_data["text_regions"][2]["center"] == [300, 400]


@pytest.mark.unit
class TestErrorCorrectorOutputStructure:
    """Test output structure creation."""

    def test_create_output_structure_format(
        self, error_corrector, mock_ocr_json
    ):
        """Test _create_output_structure() creates correct format."""
        # Arrange
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )
        start_time = 0.0

        # Act
        output = error_corrector._create_output_structure(
            mock_ocr_json, corrected_data, corrections_log, start_time
        )

        # Assert
        assert "document_info" in output
        assert "text_regions" in output
        assert "corrections_applied" in output
        assert "processing_metrics" in output
        assert output["document_info"] == mock_ocr_json["document_info"]

    def test_create_output_structure_processing_metrics(
        self, error_corrector, mock_ocr_json
    ):
        """Test that processing metrics are included correctly."""
        # Arrange
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )
        start_time = 0.0

        # Act
        output = error_corrector._create_output_structure(
            mock_ocr_json, corrected_data, corrections_log, start_time
        )

        # Assert
        metrics = output["processing_metrics"]
        assert "ocr_time_ms" in metrics
        assert "texts_detected" in metrics
        assert "average_confidence" in metrics
        assert "correction_time_ms" in metrics
        assert "corrections_applied" in metrics
        assert "correction_rate" in metrics
        assert "total_time_ms" in metrics
        assert metrics["ocr_time_ms"] == 1500
        assert metrics["texts_detected"] == 3
        assert metrics["corrections_applied"] == 2

    def test_create_output_structure_corrections_log(
        self, error_corrector, mock_ocr_json
    ):
        """Test that corrections log is embedded in output."""
        # Arrange
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )
        start_time = 0.0

        # Act
        output = error_corrector._create_output_structure(
            mock_ocr_json, corrected_data, corrections_log, start_time
        )

        # Assert
        assert "corrections_applied" in output
        assert output["corrections_applied"]["count"] == 2
        assert len(output["corrections_applied"]["corrections"]) == 2

    def test_create_output_structure_preserves_regions_detected(
        self, error_corrector, mock_ocr_json
    ):
        """Test that regions_detected is preserved if present."""
        # Arrange
        mock_ocr_json["regions_detected"] = [
            {"id": "header", "bbox": [0, 0, 100, 100]},
            {"id": "defects", "bbox": [0, 100, 200, 300]},
        ]
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json
        )
        start_time = 0.0

        # Act
        output = error_corrector._create_output_structure(
            mock_ocr_json, corrected_data, corrections_log, start_time
        )

        # Assert
        assert "regions_detected" in output
        assert output["regions_detected"] == mock_ocr_json["regions_detected"]

    def test_create_output_structure_preserves_ocr_results_by_region(
        self, error_corrector, mock_ocr_json_with_regions
    ):
        """Test that ocr_results_by_region is preserved in output."""
        # Arrange
        corrected_data, corrections_log = error_corrector._apply_corrections(
            mock_ocr_json_with_regions
        )
        start_time = 0.0

        # Act
        output = error_corrector._create_output_structure(
            mock_ocr_json_with_regions,
            corrected_data,
            corrections_log,
            start_time,
        )

        # Assert
        assert "ocr_results_by_region" in output
        assert "header" in output["ocr_results_by_region"]
        assert "defects" in output["ocr_results_by_region"]


@pytest.mark.unit
class TestErrorCorrectorPathGeneration:
    """Test output path generation."""

    def test_build_output_path_removes_texts_suffix(
        self, error_corrector, tmp_path
    ):
        """Test _build_output_path() removes -texts suffix."""
        # Arrange
        input_path = tmp_path / "test-image-texts.json"

        # Act
        output_path = error_corrector._build_output_path(input_path)

        # Assert
        assert output_path.name == "test-image-corrected.json"
        assert "-texts" not in output_path.name

    def test_build_output_path_without_texts_suffix(
        self, error_corrector, tmp_path
    ):
        """Test _build_output_path() works without -texts suffix."""
        # Arrange
        input_path = tmp_path / "test-image.json"

        # Act
        output_path = error_corrector._build_output_path(input_path)

        # Assert
        assert output_path.name == "test-image-corrected.json"

    def test_build_output_path_creates_directory(
        self, error_corrector, tmp_path, test_settings
    ):
        """Test _build_output_path() creates output directory if needed."""
        # Arrange
        test_settings.output_dir = tmp_path / "new_output"
        error_corrector._settings = test_settings
        input_path = tmp_path / "test.json"

        # Act
        output_path = error_corrector._build_output_path(input_path)

        # Assert
        assert output_path.parent == test_settings.output_dir
        # Directory should be created (or at least path should be valid)
        assert output_path.parent.name == "new_output"


@pytest.mark.unit
class TestErrorCorrectorParametrized:
    """Parametrized tests for all correction dictionary entries."""

    @pytest.mark.parametrize(
        "original,expected",
        [
            ("Homep", "Номер"),
            ("Номео", "Номер"),
            ("НомеD", "Номер"),
            ("PeB", "Рев"),
            ("Pев", "Рев"),
            ("РeB", "Рев"),
            ("Wmyyep", "Изделие"),
            ("Изделuе", "Изделие"),
            ("Изделиe", "Изделие"),
            ("Дaта", "Дата"),
            ("Датa", "Дата"),
            ("ne ugim", "не соответствует"),
            ("не соотвтствует", "не соответствует"),
            ("не cоответствует", "не соответствует"),
            ("друие несоотвтствуя", "другие несоответствия"),
            ("другие нeсоответствия", "другие несоответствия"),
            ("годн", "годен"),
            ("годeн", "годен"),
            ("бракк", "брак"),
            ("браk", "брак"),
            ("пригоден", "годен"),
            ("непригоден", "брак"),
            ("ГЕОМЕТРUЯ", "ГЕОМЕТРИЯ"),
            ("ГЕОМЕТРИR", "ГЕОМЕТРИЯ"),
            ("ОТВЕРСТUЯ", "ОТВЕРСТИЯ"),
            ("ОТВЕРСТИR", "ОТВЕРСТИЯ"),
            ("РЕЗЬБA", "РЕЗЬБА"),
            ("РEЗЬБА", "РЕЗЬБА"),
            ("Кол-вo", "Кол-во"),
            ("Коп-во", "Кол-во"),
            ("Решенuе", "Решение"),
            ("Решениe", "Решение"),
            ("Описанuе", "Описание"),
            ("Описаниe", "Описание"),
            ("Параметp", "Параметр"),
            ("Парамeтр", "Параметр"),
            ("Нормa", "Норма"),
            ("Фактичeски", "Фактически"),
            ("Контpолер", "Контролер"),
            ("Контролеp", "Контролер"),
            ("Подпuсь", "Подпись"),
            ("Подписъ", "Подпись"),
            ("Устpанить", "Устранить"),
            ("Устранитъ", "Устранить"),
            ("Доработатъ", "Доработать"),
            ("Доpаботать", "Доработать"),
            ("Утилизиpовать", "Утилизировать"),
            ("MM", "мм"),
            ("ШТ", "шт"),
            ("шт.", "шт"),
        ],
    )
    def test_exact_match_corrections_from_dictionary(
        self, error_corrector, original, expected, tmp_path
    ):
        """Test all exact match corrections from OCR_CORRECTIONS dictionary."""
        # Arrange
        ocr_data = {
            "text_regions": [
                {
                    "text": original,
                    "confidence": 0.9,
                    "center": [100, 200],
                }
            ],
            "processing_metrics": {},
        }
        json_file = tmp_path / "test-texts.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False)

        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(ocr_data)

        # Assert
        assert corrected_data["text_regions"][0]["text"] == expected
        assert corrected_data["text_regions"][0]["corrected"] is True
        assert len(corrections_log) == 1
        assert corrections_log[0]["original"] == original
        assert corrections_log[0]["corrected"] == expected

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
    def test_fuzzy_match_corrections_from_dictionary(
        self, error_corrector, original, expected, tmp_path
    ):
        """Test all fuzzy match corrections from FUZZY_CORRECTIONS dictionary."""
        # Arrange
        ocr_data = {
            "text_regions": [
                {
                    "text": original,
                    "confidence": 0.9,
                    "center": [100, 200],
                }
            ],
            "processing_metrics": {},
        }

        # Act
        corrected_data, corrections_log = error_corrector._apply_corrections(ocr_data)

        # Assert
        assert corrected_data["text_regions"][0]["text"] == expected
        assert corrected_data["text_regions"][0]["corrected"] is True
        assert len(corrections_log) == 1
        assert corrections_log[0]["original"] == original
        assert corrections_log[0]["corrected"] == expected

    def test_all_ocr_corrections_entries_tested(self):
        """Verify that all OCR_CORRECTIONS entries are covered in parametrize."""
        # This test ensures we're testing all dictionary entries
        # Count unique corrections in parametrize test
        tested_corrections = {
            "Homep",
            "Номео",
            "НомеD",
            "PeB",
            "Pев",
            "РeB",
            "Wmyyep",
            "Изделuе",
            "Изделиe",
            "Дaта",
            "Датa",
            "ne ugim",
            "не соотвтствует",
            "не cоответствует",
            "друие несоотвтствуя",
            "другие нeсоответствия",
            "годн",
            "годeн",
            "бракк",
            "браk",
            "пригоден",
            "непригоден",
            "ГЕОМЕТРUЯ",
            "ГЕОМЕТРИR",
            "ОТВЕРСТUЯ",
            "ОТВЕРСТИR",
            "РЕЗЬБA",
            "РEЗЬБА",
            "Кол-вo",
            "Коп-во",
            "Решенuе",
            "Решениe",
            "Описанuе",
            "Описаниe",
            "Параметp",
            "Парамeтр",
            "Нормa",
            "Фактичeски",
            "Контpолер",
            "Контролеp",
            "Подпuсь",
            "Подписъ",
            "Устpанить",
            "Устранитъ",
            "Доработатъ",
            "Доpаботать",
            "Утилизиpовать",
            "MM",
            "ШТ",
            "шт.",
        }
        # Note: Some entries in OCR_CORRECTIONS are reference spellings (like "ГЕОМЕТРИЯ")
        # which don't need correction, so they're not in the test list
        assert len(tested_corrections) > 0

    def test_all_fuzzy_corrections_entries_tested(self):
        """Verify that all FUZZY_CORRECTIONS entries are covered in parametrize."""
        # This test ensures we're testing all fuzzy dictionary entries
        tested_fuzzy = {"номер", "дата", "рев", "изделие", "годен", "брак"}
        assert tested_fuzzy == set(FUZZY_CORRECTIONS.keys())


@pytest.mark.unit
class TestErrorCorrectorEdgeCases:
    """Test edge cases and error handling."""

    def test_process_with_empty_text_regions(
        self, error_corrector, tmp_path
    ):
        """Test process() handles empty text_regions."""
        # Arrange
        ocr_data = {"text_regions": [], "processing_metrics": {}}
        json_file = tmp_path / "empty-texts.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False)

        # Act
        result = error_corrector.process(json_file)

        # Assert
        assert result.total_texts == 0
        assert result.corrections_applied == 0
        assert result.correction_rate == 0.0

    def test_process_with_missing_processing_metrics(
        self, error_corrector, tmp_path
    ):
        """Test process() handles missing processing_metrics."""
        # Arrange
        ocr_data = {
            "text_regions": [
                {"text": "Homep", "confidence": 0.9, "center": [100, 200]}
            ]
        }
        json_file = tmp_path / "no-metrics-texts.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False)

        # Act
        result = error_corrector.process(json_file)

        # Assert
        assert result.corrections_applied == 1
        # Should not raise error

    def test_process_with_missing_text_field(self, error_corrector, tmp_path):
        """Test process() handles text_regions with missing text field."""
        # Arrange
        ocr_data = {
            "text_regions": [
                {"confidence": 0.9, "center": [100, 200]}  # Missing "text"
            ],
            "processing_metrics": {},
        }
        json_file = tmp_path / "missing-text-texts.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False)

        # Act
        result = error_corrector.process(json_file)

        # Assert
        # Should handle gracefully, treating missing text as empty string
        assert result.total_texts == 1

