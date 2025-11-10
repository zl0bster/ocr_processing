from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config.settings import Settings
from error_corrector import ErrorCorrector
from field_validator import FieldValidator
from ocr_engine import OCREngine
from preprocessor import ImagePreprocessor


@dataclass(frozen=True)
class FileResult:
    """Result information for a single file in batch processing."""
    
    filename: str
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None
    texts_found: int = 0
    average_confidence: float = 0.0
    corrections_applied: int = 0
    fields_validated: int = 0


@dataclass(frozen=True)
class BatchResult:
    """Result information for batch processing."""
    
    total_files: int
    successful_files: int
    failed_files: int
    total_duration_seconds: float
    file_results: List[FileResult]
    summary_path: Optional[Path] = None


class BatchProcessor:
    """Batch processor for multiple images with shared OCR engine.
    
    Processes entire directories of images using a single OCR engine instance,
    significantly improving performance by eliminating repeated model loading.
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        mode: str = "pipeline"
    ) -> BatchResult:
        """Process all images in a directory using shared OCR engine.
        
        Args:
            input_dir: Directory containing images to process
            output_dir: Optional output directory (defaults to settings.output_dir)
            mode: Processing mode - "pipeline", "ocr", "preprocess", or "correction"
            
        Returns:
            BatchResult with aggregated metrics
        """
        start_time = time.perf_counter()
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            self._logger.warning("No image files found in directory '%s'", input_dir)
            return BatchResult(
                total_files=0,
                successful_files=0,
                failed_files=0,
                total_duration_seconds=time.perf_counter() - start_time,
                file_results=[]
            )
        
        self._logger.info("Found %d image files in '%s'", len(image_files), input_dir)
        
        # Process based on mode
        if mode == "pipeline":
            file_results = self._process_pipeline_batch(image_files)
        elif mode == "ocr":
            file_results = self._process_ocr_batch(image_files)
        elif mode == "preprocess":
            file_results = self._process_preprocess_batch(image_files)
        elif mode == "correction":
            file_results = self._process_correction_batch(image_files)
        else:
            raise ValueError(f"Unknown processing mode: {mode}")
        
        total_duration = time.perf_counter() - start_time
        successful = sum(1 for r in file_results if r.success)
        failed = len(file_results) - successful
        
        # Create batch summary
        batch_result = BatchResult(
            total_files=len(file_results),
            successful_files=successful,
            failed_files=failed,
            total_duration_seconds=total_duration,
            file_results=file_results
        )
        
        # Save summary report
        summary_path = self._save_batch_summary(batch_result, output_dir)
        batch_result = BatchResult(
            total_files=batch_result.total_files,
            successful_files=batch_result.successful_files,
            failed_files=batch_result.failed_files,
            total_duration_seconds=batch_result.total_duration_seconds,
            file_results=batch_result.file_results,
            summary_path=summary_path
        )
        
        # Log summary
        self._logger.info("=== Batch Processing Summary ===")
        self._logger.info("Total files: %d", batch_result.total_files)
        self._logger.info("Successful: %d", batch_result.successful_files)
        self._logger.info("Failed: %d", batch_result.failed_files)
        self._logger.info("Total time: %.3f seconds", total_duration)
        self._logger.info("Average time per file: %.3f seconds", 
                         total_duration / len(file_results) if file_results else 0)
        if summary_path:
            self._logger.info("Summary report: %s", summary_path)
        
        return batch_result

    def _process_pipeline_batch(self, image_files: List[Path]) -> List[FileResult]:
        """Process batch with full pipeline using shared OCR engine."""
        file_results = []
        
        # Create shared components
        preprocessor = ImagePreprocessor(settings=self._settings, logger=self._logger)
        error_corrector = ErrorCorrector(settings=self._settings, logger=self._logger)
        field_validator = FieldValidator(settings=self._settings, logger=self._logger)
        
        # Use shared OCR engine for all files
        with OCREngine(settings=self._settings, logger=self._logger) as ocr_engine:
            for image_file in image_files:
                self._logger.info("Processing %s...", image_file.name)
                file_start = time.perf_counter()
                
                try:
                    # Step 1: Preprocessing
                    preprocess_result = preprocessor.process(input_path=image_file)
                    self._logger.info("  Step 1/4: Preprocessing completed in %.3f seconds. Output: %s",
                                     preprocess_result.duration_seconds,
                                     preprocess_result.output_path)
                    if preprocess_result.deskew_angle is not None:
                        self._logger.debug("  Deskew angle: %.3f degrees", preprocess_result.deskew_angle)
                    
                    # Step 2: OCR with shared engine
                    ocr_result = ocr_engine.process(input_path=preprocess_result.output_path)
                    self._logger.info("  Step 2/4: OCR completed in %.3f seconds. Output: %s",
                                     ocr_result.duration_seconds,
                                     ocr_result.output_path)
                    self._logger.info("  Found %d text regions (avg confidence: %.3f)",
                                     ocr_result.total_texts_found,
                                     ocr_result.average_confidence)
                    if ocr_result.low_confidence_count > 0:
                        self._logger.warning("  %d text regions have low confidence",
                                            ocr_result.low_confidence_count)
                    
                    # Step 3: Error correction
                    correction_result = error_corrector.process(input_path=ocr_result.output_path)
                    self._logger.info("  Step 3/4: Error correction completed in %.3f seconds. Output: %s",
                                     correction_result.duration_seconds,
                                     correction_result.output_path)
                    self._logger.info("  Applied %d corrections (%.1f%%)",
                                     correction_result.corrections_applied,
                                     correction_result.correction_rate * 100)
                    
                    # Step 4: Field validation
                    validation_result = field_validator.process(
                        input_path=correction_result.output_path
                    )
                    self._logger.info("  Step 4/4: Validation completed in %.3f seconds. Output: %s",
                                     validation_result.duration_seconds,
                                     validation_result.output_path)
                    self._logger.info("  Validated %d/%d fields (%.1f%%)",
                                     validation_result.validated_fields,
                                     validation_result.total_fields,
                                     validation_result.validation_rate * 100)
                    if validation_result.failed_validations > 0:
                        self._logger.warning("  %d validation failures detected",
                                            validation_result.failed_validations)
                    
                    file_duration = time.perf_counter() - file_start
                    
                    file_results.append(FileResult(
                        filename=image_file.name,
                        success=True,
                        duration_seconds=file_duration,
                        texts_found=ocr_result.total_texts_found,
                        average_confidence=ocr_result.average_confidence,
                        corrections_applied=correction_result.corrections_applied,
                        fields_validated=validation_result.validated_fields
                    ))
                    
                    self._logger.info("✓ %s completed in %.3f seconds", 
                                     image_file.name, file_duration)
                    
                except Exception as e:
                    file_duration = time.perf_counter() - file_start
                    error_msg = str(e)
                    
                    file_results.append(FileResult(
                        filename=image_file.name,
                        success=False,
                        duration_seconds=file_duration,
                        error_message=error_msg
                    ))
                    
                    self._logger.error("✗ %s failed: %s", image_file.name, error_msg)
        
        return file_results

    def _process_ocr_batch(self, image_files: List[Path]) -> List[FileResult]:
        """Process batch with OCR only using shared engine."""
        file_results = []
        
        # Use shared OCR engine for all files
        with OCREngine(settings=self._settings, logger=self._logger) as ocr_engine:
            for image_file in image_files:
                self._logger.info("Processing %s...", image_file.name)
                file_start = time.perf_counter()
                
                try:
                    ocr_result = ocr_engine.process(input_path=image_file)
                    self._logger.info("  OCR completed in %.3f seconds. Output: %s",
                                     ocr_result.duration_seconds,
                                     ocr_result.output_path)
                    self._logger.info("  Found %d text regions (avg confidence: %.3f)",
                                     ocr_result.total_texts_found,
                                     ocr_result.average_confidence)
                    if ocr_result.low_confidence_count > 0:
                        self._logger.warning("  %d regions with low confidence",
                                            ocr_result.low_confidence_count)
                    file_duration = time.perf_counter() - file_start
                    
                    file_results.append(FileResult(
                        filename=image_file.name,
                        success=True,
                        duration_seconds=file_duration,
                        texts_found=ocr_result.total_texts_found,
                        average_confidence=ocr_result.average_confidence
                    ))
                    
                    self._logger.info("✓ %s completed in %.3f seconds", 
                                     image_file.name, file_duration)
                    
                except Exception as e:
                    file_duration = time.perf_counter() - file_start
                    error_msg = str(e)
                    
                    file_results.append(FileResult(
                        filename=image_file.name,
                        success=False,
                        duration_seconds=file_duration,
                        error_message=error_msg
                    ))
                    
                    self._logger.error("✗ %s failed: %s", image_file.name, error_msg)
        
        return file_results

    def _process_preprocess_batch(self, image_files: List[Path]) -> List[FileResult]:
        """Process batch with preprocessing only."""
        file_results = []
        preprocessor = ImagePreprocessor(settings=self._settings, logger=self._logger)
        
        for image_file in image_files:
            self._logger.info("Processing %s...", image_file.name)
            file_start = time.perf_counter()
            
            try:
                preprocess_result = preprocessor.process(input_path=image_file)
                self._logger.info("  Preprocessing completed in %.3f seconds. Output: %s",
                                 preprocess_result.duration_seconds,
                                 preprocess_result.output_path)
                if preprocess_result.deskew_angle is not None:
                    self._logger.debug("  Deskew angle: %.3f degrees", preprocess_result.deskew_angle)
                file_duration = time.perf_counter() - file_start
                
                file_results.append(FileResult(
                    filename=image_file.name,
                    success=True,
                    duration_seconds=file_duration
                ))
                
                self._logger.info("✓ %s completed in %.3f seconds", 
                                 image_file.name, file_duration)
                
            except Exception as e:
                file_duration = time.perf_counter() - file_start
                error_msg = str(e)
                
                file_results.append(FileResult(
                    filename=image_file.name,
                    success=False,
                    duration_seconds=file_duration,
                    error_message=error_msg
                ))
                
                self._logger.error("✗ %s failed: %s", image_file.name, error_msg)
        
        return file_results

    def _process_correction_batch(self, image_files: List[Path]) -> List[FileResult]:
        """Process batch with correction and validation only."""
        file_results = []
        error_corrector = ErrorCorrector(settings=self._settings, logger=self._logger)
        field_validator = FieldValidator(settings=self._settings, logger=self._logger)
        
        for image_file in image_files:
            self._logger.info("Processing %s...", image_file.name)
            file_start = time.perf_counter()
            
            try:
                correction_result = error_corrector.process(input_path=image_file)
                self._logger.info("  Error correction completed in %.3f seconds",
                                 correction_result.duration_seconds)
                self._logger.info("  Applied %d corrections", correction_result.corrections_applied)
                
                validation_result = field_validator.process(
                    input_path=correction_result.output_path
                )
                self._logger.info("  Validation completed in %.3f seconds. Output: %s",
                                 validation_result.duration_seconds,
                                 validation_result.output_path)
                self._logger.info("  Validated %d/%d fields",
                                 validation_result.validated_fields,
                                 validation_result.total_fields)
                file_duration = time.perf_counter() - file_start
                
                file_results.append(FileResult(
                    filename=image_file.name,
                    success=True,
                    duration_seconds=file_duration,
                    corrections_applied=correction_result.corrections_applied,
                    fields_validated=validation_result.validated_fields
                ))
                
                self._logger.info("✓ %s completed in %.3f seconds", 
                                 image_file.name, file_duration)
                
            except Exception as e:
                file_duration = time.perf_counter() - file_start
                error_msg = str(e)
                
                file_results.append(FileResult(
                    filename=image_file.name,
                    success=False,
                    duration_seconds=file_duration,
                    error_message=error_msg
                ))
                
                self._logger.error("✗ %s failed: %s", image_file.name, error_msg)
        
        return file_results

    def _save_batch_summary(self, batch_result: BatchResult, output_dir: Optional[Path]) -> Path:
        """Save batch processing summary to JSON file."""
        import json
        from datetime import datetime
        
        output_path = output_dir or self._settings.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_path / f"batch_summary_{timestamp}.json"
        
        summary_data = {
            "batch_info": {
                "processing_date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "total_files": batch_result.total_files,
                "successful_files": batch_result.successful_files,
                "failed_files": batch_result.failed_files,
                "total_duration_seconds": round(batch_result.total_duration_seconds, 3),
                "average_duration_seconds": round(
                    batch_result.total_duration_seconds / batch_result.total_files
                    if batch_result.total_files > 0 else 0, 3
                )
            },
            "file_results": [
                {
                    "filename": result.filename,
                    "success": result.success,
                    "duration_seconds": round(result.duration_seconds, 3),
                    "texts_found": result.texts_found,
                    "average_confidence": round(result.average_confidence, 3),
                    "corrections_applied": result.corrections_applied,
                    "fields_validated": result.fields_validated,
                    "error_message": result.error_message
                }
                for result in batch_result.file_results
            ]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        return summary_file

