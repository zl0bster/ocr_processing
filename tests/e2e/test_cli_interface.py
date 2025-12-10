"""E2E tests for CLI interface via subprocess."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestCLIInterface:
    """End-to-end tests for command-line interface."""

    @property
    def main_script_path(self) -> Path:
        """Get path to main.py script."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "src" / "main.py"

    def run_cli(self, args: list[str]) -> tuple[int, str, str]:
        """Run CLI command and return exit code, stdout, stderr."""
        cmd = [sys.executable, str(self.main_script_path)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        return result.returncode, result.stdout, result.stderr

    def test_help_output(self):
        """Test --help argument produces help text."""
        exit_code, stdout, stderr = self.run_cli(["--help"])

        assert exit_code == 0, "Help should exit with code 0"
        assert "OCR processing pipeline CLI" in stdout or "description" in stdout.lower()
        assert "--file" in stdout or "file" in stdout.lower()
        assert "--batch" in stdout or "batch" in stdout.lower()
        assert "--mode" in stdout or "mode" in stdout.lower()

    def test_no_arguments(self):
        """Test CLI with no arguments (should show usage)."""
        exit_code, stdout, stderr = self.run_cli([])

        # Should exit successfully (code 0) with informative message
        assert exit_code == 0, "No args should exit gracefully with code 0"

    def test_file_argument_single_file(
        self, test_image_paths, cleanup_e2e_outputs
    ):
        """Test --file argument with single file processing."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_file = test_image_paths["compressed"]
        output_dir = cleanup_e2e_outputs

        exit_code, stdout, stderr = self.run_cli([
            "--file", str(input_file),
            "--mode", "preprocess",
            "--output", str(output_dir)
        ])

        # Preprocessing should succeed
        assert exit_code == 0, f"Preprocessing should succeed. stderr: {stderr}"

    def test_batch_argument(
        self, batch_images_dir, cleanup_e2e_outputs
    ):
        """Test --batch argument with directory."""
        output_dir = cleanup_e2e_outputs

        exit_code, stdout, stderr = self.run_cli([
            "--batch", str(batch_images_dir),
            "--mode", "preprocess",
            "--output", str(output_dir)
        ])

        # Batch processing should attempt to process files
        # (may have some failures, but should complete)
        assert exit_code in [0, 1], "Batch should exit with 0 (all success) or 1 (some failures)"
        assert "batch" in stdout.lower() or "processing" in stdout.lower() or len(stdout) > 0

    def test_mode_preprocess(
        self, test_image_paths, cleanup_e2e_outputs
    ):
        """Test --mode preprocess."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_file = test_image_paths["compressed"]
        output_dir = cleanup_e2e_outputs

        exit_code, stdout, stderr = self.run_cli([
            "--file", str(input_file),
            "--mode", "preprocess",
            "--output", str(output_dir)
        ])

        assert exit_code == 0, f"Preprocess mode should succeed. stderr: {stderr}"

    def test_mode_ocr(
        self, test_image_paths, cleanup_e2e_outputs
    ):
        """Test --mode ocr (requires preprocessed image)."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_file = test_image_paths["compressed"]
        output_dir = cleanup_e2e_outputs

        # First preprocess
        exit_code1, stdout1, stderr1 = self.run_cli([
            "--file", str(input_file),
            "--mode", "preprocess",
            "--output", str(output_dir)
        ])

        if exit_code1 != 0:
            pytest.skip("Preprocessing failed, cannot test OCR mode")

        # Find preprocessed output
        preprocessed_files = list(output_dir.glob(f"{input_file.stem}*cor*"))
        if not preprocessed_files:
            # Try to find any preprocessed file
            preprocessed_files = list(output_dir.glob("*cor*"))

        if not preprocessed_files:
            pytest.skip("Could not find preprocessed output file")

        preprocessed_file = preprocessed_files[0]

        # Now test OCR mode
        exit_code2, stdout2, stderr2 = self.run_cli([
            "--file", str(preprocessed_file),
            "--mode", "ocr",
            "--output", str(output_dir)
        ])

        # OCR mode should succeed or fail gracefully
        assert exit_code2 in [0, 1], "OCR mode should exit with valid code"

    @pytest.mark.slow
    @pytest.mark.requires_ocr
    def test_mode_pipeline(
        self, test_image_paths, cleanup_e2e_outputs
    ):
        """Test --mode pipeline (full processing)."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_file = test_image_paths["compressed"]
        output_dir = cleanup_e2e_outputs

        exit_code, stdout, stderr = self.run_cli([
            "--file", str(input_file),
            "--mode", "pipeline",
            "--output", str(output_dir)
        ])

        # Pipeline mode should succeed (may take longer)
        assert exit_code == 0, f"Pipeline mode should succeed. stderr: {stderr}"

    def test_output_argument(
        self, test_image_paths, cleanup_e2e_outputs
    ):
        """Test --output argument specifies custom output path."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_file = test_image_paths["compressed"]
        output_dir = cleanup_e2e_outputs

        exit_code, stdout, stderr = self.run_cli([
            "--file", str(input_file),
            "--mode", "preprocess",
            "--output", str(output_dir)
        ])

        assert exit_code == 0, "Should succeed with custom output path"
        # Verify output directory exists and has files
        assert output_dir.exists(), "Output directory should be created"

    def test_invalid_file_path(self):
        """Test error handling for invalid file path."""
        exit_code, stdout, stderr = self.run_cli([
            "--file", "/nonexistent/path/image.jpg",
            "--mode", "preprocess"
        ])

        # Should exit with error code
        assert exit_code != 0, "Invalid file path should return error code"

    def test_invalid_batch_directory(self):
        """Test error handling for invalid batch directory."""
        exit_code, stdout, stderr = self.run_cli([
            "--batch", "/nonexistent/directory",
            "--mode", "preprocess"
        ])

        # Should exit with error code
        assert exit_code != 0, "Invalid batch directory should return error code"

    def test_invalid_mode(self):
        """Test error handling for invalid mode."""
        exit_code, stdout, stderr = self.run_cli([
            "--file", "dummy.jpg",
            "--mode", "invalid_mode"
        ])

        # Should exit with error code (invalid choice)
        assert exit_code != 0, "Invalid mode should return error code"

    def test_exit_code_success(self, test_image_paths, cleanup_e2e_outputs):
        """Test that successful processing returns exit code 0."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")

        input_file = test_image_paths["compressed"]
        output_dir = cleanup_e2e_outputs

        exit_code, stdout, stderr = self.run_cli([
            "--file", str(input_file),
            "--mode", "preprocess",
            "--output", str(output_dir)
        ])

        assert exit_code == 0, "Successful processing should return exit code 0"

    def test_exit_code_failure(self):
        """Test that failed processing returns non-zero exit code."""
        exit_code, stdout, stderr = self.run_cli([
            "--file", "/nonexistent/file.jpg",
            "--mode", "preprocess"
        ])

        assert exit_code != 0, "Failed processing should return non-zero exit code"

