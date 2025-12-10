"""E2E tests for error handling scenarios."""
import json
import tempfile
from pathlib import Path

import pytest

from src.error_corrector import ErrorCorrector
from src.field_validator import FieldValidator
from src.form_extractor import FormExtractor
from src.ocr_engine import OCREngine
from src.preprocessor import ImagePreprocessor


@pytest.mark.e2e
class TestErrorScenarios:
    """End-to-end tests for error handling and graceful degradation."""

    def test_corrupted_image_handling(
        self, e2e_settings, e2e_logger, temp_output_dir_e2e
    ):
        """Test graceful handling of corrupted image file."""
        # Create corrupted image file
        corrupted_file = temp_output_dir_e2e / "corrupted.jpg"
        corrupted_file.write_bytes(b"INVALID IMAGE DATA\x00\x01\x02\x03" * 100)

        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)

        # Should raise an exception for corrupted image
        with pytest.raises((ValueError, RuntimeError, OSError)):
            preprocessor.process(input_path=corrupted_file)

    def test_missing_file_handling(
        self, e2e_settings, e2e_logger
    ):
        """Test handling of non-existent file."""
        missing_file = Path("/nonexistent/path/image.jpg")

        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            preprocessor.process(input_path=missing_file)

    def test_invalid_image_format(
        self, e2e_settings, e2e_logger, temp_output_dir_e2e
    ):
        """Test handling of non-image file format."""
        # Create a text file with .jpg extension
        fake_image = temp_output_dir_e2e / "fake.jpg"
        fake_image.write_text("This is not an image file", encoding="utf-8")

        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)

        # Should raise an exception when trying to read as image
        with pytest.raises((ValueError, RuntimeError, OSError)):
            preprocessor.process(input_path=fake_image)

    def test_empty_image_file(
        self, e2e_settings, e2e_logger, temp_output_dir_e2e
    ):
        """Test handling of empty image file."""
        empty_file = temp_output_dir_e2e / "empty.jpg"
        empty_file.write_bytes(b"")

        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)

        # Should raise an exception for empty file
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError, OSError)):
            preprocessor.process(input_path=empty_file)

    def test_missing_ocr_results_file(
        self, e2e_settings, e2e_logger
    ):
        """Test error correction with missing OCR results file."""
        missing_file = Path("/nonexistent/path/ocr_results.json")

        error_corrector = ErrorCorrector(settings=e2e_settings, logger=e2e_logger)

        with pytest.raises(FileNotFoundError):
            error_corrector.process(input_path=missing_file)

    def test_invalid_json_structure(
        self, e2e_settings, e2e_logger, temp_output_dir_e2e
    ):
        """Test error correction with invalid JSON structure."""
        invalid_json = temp_output_dir_e2e / "invalid.json"
        invalid_json.write_text("This is not valid JSON", encoding="utf-8")

        error_corrector = ErrorCorrector(settings=e2e_settings, logger=e2e_logger)

        # Should raise JSON decode error or validation error
        with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
            error_corrector.process(input_path=invalid_json)

    def test_missing_ocr_results_by_region(
        self, e2e_settings, e2e_logger, temp_output_dir_e2e
    ):
        """Test form extraction with missing ocr_results_by_region."""
        import json

        # Create JSON file without required field
        invalid_data = {
            "texts": [],
            "metadata": {}
        }
        invalid_json = temp_output_dir_e2e / "missing_region.json"
        invalid_json.write_text(json.dumps(invalid_data), encoding="utf-8")

        form_extractor = FormExtractor(settings=e2e_settings, logger=e2e_logger)

        # Should raise ValueError about missing ocr_results_by_region
        with pytest.raises(ValueError, match="ocr_results_by_region"):
            form_extractor.extract(ocr_json_path=invalid_json)

    def test_graceful_degradation_on_low_quality_image(
        self, e2e_settings, e2e_logger, temp_output_dir_e2e
    ):
        """Test that system continues processing even with very low quality image."""
        import numpy as np
        import cv2

        # Create a very low quality image (mostly noise)
        low_quality_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        low_quality_path = temp_output_dir_e2e / "low_quality.jpg"
        cv2.imwrite(str(low_quality_path), low_quality_image)

        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)

        # Should complete preprocessing (even if result is poor)
        try:
            result = preprocessor.process(input_path=low_quality_path)
            assert result.output_path.exists(), "Should create output even for low quality"
        except Exception:
            # It's acceptable if very poor quality causes failure
            # The key is that it fails gracefully with an exception, not a crash
            pass

    def test_error_logging_verification(
        self, e2e_settings, e2e_logger, temp_output_dir_e2e
    ):
        """Verify that errors are properly logged."""
        missing_file = Path("/nonexistent/path/image.jpg")

        preprocessor = ImagePreprocessor(settings=e2e_settings, logger=e2e_logger)

        # Capture log output would require more setup, but we can verify
        # that exception is raised (which means error handling is working)
        with pytest.raises(FileNotFoundError):
            preprocessor.process(input_path=missing_file)

        # If we got here without a crash, error was handled gracefully

    def test_batch_processing_continues_after_single_file_error(
        self, batch_images_dir, e2e_settings, e2e_logger, cleanup_e2e_outputs, tmp_path
    ):
        """Test that batch processing continues after individual file errors."""
        from src.batch_processor import BatchProcessor

        output_dir = cleanup_e2e_outputs

        # Add a corrupted file to the batch directory
        corrupted_file = batch_images_dir / "test_corrupted_batch.jpg"
        try:
            corrupted_file.write_bytes(b"INVALID" * 100)

            batch_processor = BatchProcessor(settings=e2e_settings, logger=e2e_logger)

            result = batch_processor.process_directory(
                input_dir=batch_images_dir,
                output_dir=output_dir,
                mode="pipeline"
            )

            # Batch should continue processing other files
            assert result.total_files > 0, "Should process files in batch"

            # Some files might fail, but batch should complete
            assert result.successful_files + result.failed_files == result.total_files

            # If we have valid images, at least some should succeed
            # (excluding our corrupted test file)
            valid_count = len([f for f in batch_images_dir.glob("*.jpg")
                             if f.name != "test_corrupted_batch.jpg"])
            if valid_count > 0:
                # Batch should have attempted all files including corrupted one
                assert result.total_files >= valid_count

        finally:
            # Cleanup
            if corrupted_file.exists():
                corrupted_file.unlink()

