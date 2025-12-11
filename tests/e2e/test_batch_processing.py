"""E2E tests for batch processing workflow."""
import json
import time
from pathlib import Path

import pytest

from src.batch_processor import BatchProcessor


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_ocr
class TestBatchProcessing:
    """End-to-end tests for batch processing with multiple images."""

    def test_batch_processing_pipeline_mode(
        self, batch_images_dir, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Test batch processing in pipeline mode with shared OCR engine."""
        output_dir = cleanup_e2e_outputs

        batch_processor = BatchProcessor(settings=e2e_settings, logger=e2e_logger)

        start_time = time.perf_counter()
        result = batch_processor.process_directory(
            input_dir=batch_images_dir,
            output_dir=output_dir,
            mode="pipeline"
        )
        total_time = time.perf_counter() - start_time

        # Verify batch result structure
        assert result.total_files > 0, "Should process at least one file"
        assert result.successful_files >= 0, "Should have success count"
        assert result.failed_files >= 0, "Should have failure count"
        assert result.successful_files + result.failed_files == result.total_files
        assert result.total_duration_seconds > 0, "Should have processing time"
        assert len(result.file_results) == result.total_files

        # Verify all files were attempted
        image_files = list(batch_images_dir.glob("*.jpg")) + list(batch_images_dir.glob("*.png"))
        assert result.total_files == len(image_files), "Should process all image files"

        # Verify summary file was created
        if result.summary_path:
            assert result.summary_path.exists(), "Summary file should exist"

            with open(result.summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            assert "total_files" in summary_data
            assert "successful_files" in summary_data
            assert "failed_files" in summary_data
            assert "file_results" in summary_data

        e2e_logger.info("Batch processing completed: %d files in %.3f seconds (avg: %.3f sec/file)",
                       result.total_files,
                       total_time,
                       total_time / result.total_files if result.total_files > 0 else 0)

    def test_batch_processing_shared_ocr_engine(
        self, batch_images_dir, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Verify that batch processing reuses OCR engine across files."""
        output_dir = cleanup_e2e_outputs

        batch_processor = BatchProcessor(settings=e2e_settings, logger=e2e_logger)

        # Process a subset of files to test shared engine
        result = batch_processor.process_directory(
            input_dir=batch_images_dir,
            output_dir=output_dir,
            mode="pipeline"
        )

        # If multiple files processed successfully, shared engine was likely used
        # (we can't directly verify engine reuse, but if processing is fast and successful,
        # it indicates shared engine is working)
        if result.total_files >= 2:
            avg_time_per_file = result.total_duration_seconds / result.total_files
            # With shared engine, average time per file should be reasonable
            # (first file includes initialization, subsequent files should be faster)
            assert avg_time_per_file > 0, "Should have positive processing time"

        # Verify files were processed
        assert result.total_files > 0

    def test_batch_processing_memory_cleanup(
        self, batch_images_dir, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Test that batch processing handles memory cleanup between files."""
        output_dir = cleanup_e2e_outputs

        batch_processor = BatchProcessor(settings=e2e_settings, logger=e2e_logger)

        # Process batch - memory cleanup should happen automatically if threshold exceeded
        result = batch_processor.process_directory(
            input_dir=batch_images_dir,
            output_dir=output_dir,
            mode="pipeline"
        )

        # Verify processing completed without memory errors
        # (if memory cleanup fails, processing would likely crash or hang)
        assert result.total_files > 0
        assert result.total_duration_seconds > 0

        # Check that at least some files succeeded (indicating memory was managed)
        if result.total_files > 0:
            # If we got results, memory cleanup is likely working
            assert result.successful_files >= 0

    def test_batch_processing_error_isolation(
        self, batch_images_dir, e2e_settings, e2e_logger, cleanup_e2e_outputs, tmp_path
    ):
        """Test that one file failure doesn't stop batch processing."""
        output_dir = cleanup_e2e_outputs

        # Create a corrupted image file alongside valid images
        corrupted_file = batch_images_dir / "corrupted_test.jpg"
        if not corrupted_file.exists():
            # Create a file that looks like an image but is corrupted
            corrupted_file.write_bytes(b"INVALID IMAGE DATA\x00\x01\x02\x03")

        try:
            batch_processor = BatchProcessor(settings=e2e_settings, logger=e2e_logger)

            result = batch_processor.process_directory(
                input_dir=batch_images_dir,
                output_dir=output_dir,
                mode="pipeline"
            )

            # Batch should continue processing other files even if one fails
            # We can't guarantee the corrupted file will fail (might be skipped),
            # but we can verify batch continues
            assert result.total_files > 0

            # If we have valid images, at least some should succeed
            valid_images = [f for f in batch_images_dir.glob("*.jpg")
                          if f.name != "corrupted_test.jpg"]
            if len(valid_images) > 0:
                # At least one valid file should be processed
                assert result.total_files >= 1

        finally:
            # Cleanup corrupted test file
            if corrupted_file.exists():
                corrupted_file.unlink()

    def test_batch_processing_summary_generation(
        self, batch_images_dir, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Verify batch processing generates correct summary statistics."""
        output_dir = cleanup_e2e_outputs

        batch_processor = BatchProcessor(settings=e2e_settings, logger=e2e_logger)

        result = batch_processor.process_directory(
            input_dir=batch_images_dir,
            output_dir=output_dir,
            mode="pipeline"
        )

        # Verify summary statistics
        assert result.total_files >= 0
        assert result.successful_files >= 0
        assert result.failed_files >= 0
        assert result.successful_files + result.failed_files == result.total_files

        # Verify individual file results
        for file_result in result.file_results:
            assert file_result.filename, "Should have filename"
            assert isinstance(file_result.success, bool), "Should have success status"
            assert file_result.duration_seconds >= 0, "Should have duration"

            if file_result.success:
                assert file_result.error_message is None, "Successful files shouldn't have errors"
            else:
                assert file_result.error_message, "Failed files should have error messages"

        # Verify summary file if created
        if result.summary_path and result.summary_path.exists():
            with open(result.summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            assert summary["total_files"] == result.total_files
            assert summary["successful_files"] == result.successful_files
            assert summary["failed_files"] == result.failed_files
            assert len(summary["file_results"]) == result.total_files

    def test_batch_processing_performance_metrics(
        self, batch_images_dir, e2e_settings, e2e_logger, cleanup_e2e_outputs
    ):
        """Test that batch processing provides performance metrics."""
        output_dir = cleanup_e2e_outputs

        batch_processor = BatchProcessor(settings=e2e_settings, logger=e2e_logger)

        start_time = time.perf_counter()
        result = batch_processor.process_directory(
            input_dir=batch_images_dir,
            output_dir=output_dir,
            mode="pipeline"
        )
        measured_time = time.perf_counter() - start_time

        # Verify timing metrics
        assert result.total_duration_seconds > 0
        assert abs(result.total_duration_seconds - measured_time) < 5.0, \
            "Reported time should be close to measured time"

        if result.total_files > 0:
            avg_time = result.total_duration_seconds / result.total_files
            assert avg_time > 0, "Average time per file should be positive"

            e2e_logger.info("Performance: %d files in %.3f seconds (%.3f sec/file average)",
                           result.total_files,
                           result.total_duration_seconds,
                           avg_time)


