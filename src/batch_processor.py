from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from config.settings import Settings
from error_corrector import ErrorCorrector
from field_validator import FieldValidator
from form_extractor import FormExtractor
from ocr_engine import OCREngine
from ocr_engine_pool import OCREnginePool
from preprocessor import ImagePreprocessor
from region_detector import RegionDetector
from utils.json_utils import convert_numpy_types
from utils.memory_monitor import MemoryMonitor


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
    extraction_result_path: Optional[Path] = None


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
        self._memory_monitor = MemoryMonitor(logger)
        
        # Initialize OCR engine pool if enabled, otherwise use single engine
        self._pool: Optional[OCREnginePool] = None
        if self._settings.ocr_pool_enabled:
            self._pool = OCREnginePool(
                settings=self._settings,
                logger=self._logger,
            )
            self._logger.info("OCR engine pool enabled for batch processing")
        else:
            self._logger.info("OCR engine pool disabled, using single engine mode")
    
    def close(self) -> None:
        """Close the OCR engine pool and release resources."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

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
        """Process batch with full pipeline and automatic OCR engine reload."""
        file_results = []
        
        self._logger.info("=== Starting batch processing ===")
        initial_memory = self._memory_monitor.log_memory("batch start", level="INFO")
        
        # Create shared components
        preprocessor = ImagePreprocessor(settings=self._settings, logger=self._logger)
        error_corrector = ErrorCorrector(settings=self._settings, logger=self._logger)
        field_validator = FieldValidator(settings=self._settings, logger=self._logger)
        form_extractor = FormExtractor(settings=self._settings, logger=self._logger)
        region_detector = RegionDetector(settings=self._settings, logger=self._logger)
        
        # Use pool if enabled, otherwise create single engine
        if self._pool is not None:
            # Pool mode: engines are acquired/released per file
            try:
                for image_file in image_files:
                    self._logger.info("Processing %s...", image_file.name)
                    file_start = time.perf_counter()
                    file_mem_start = self._memory_monitor.log_memory(
                        f"before {image_file.name}", 
                        level="DEBUG"
                    )
                    
                    # Acquire engine from pool (automatically released after use)
                    with self._pool.acquire() as ocr_engine:
                        try:
                            # Step 1: Preprocessing
                            preprocess_result = preprocessor.process(input_path=image_file)
                            self._logger.info("  Step 1/5: Preprocessing completed in %.3f seconds. Output: %s",
                                             preprocess_result.duration_seconds,
                                             preprocess_result.output_path)
                            if preprocess_result.deskew_angle is not None:
                                self._logger.debug("  Deskew angle: %.3f degrees", preprocess_result.deskew_angle)
                            
                            # Step 2: OCR with shared engine (regional processing)
                            try:
                                # Detect regions
                                preprocessed_image = cv2.imread(str(preprocess_result.output_path), cv2.IMREAD_COLOR)
                                if preprocessed_image is None:
                                    raise ValueError(
                                        f"Unable to read preprocessed image '{preprocess_result.output_path}'"
                                    )
                                
                                regions = region_detector.detect_zones(preprocessed_image)
                                self._logger.debug(
                                    "  Region detector identified %d zones using method '%s'",
                                    len(regions),
                                    regions[0].detection_method if regions else "none",
                                )
                                
                                # Process regions
                                ocr_result = ocr_engine.process_regions(
                                    image_path=preprocess_result.output_path,
                                    regions=regions,
                                )
                                self._logger.info("  Step 2/5: OCR completed in %.3f seconds. Output: %s",
                                                 ocr_result.duration_seconds,
                                                 ocr_result.output_path)
                                self._logger.info("  Found %d text regions (avg confidence: %.3f)",
                                                 ocr_result.total_texts_found,
                                                 ocr_result.average_confidence)
                                
                                # Детальное логирование результатов OCR
                                self._logger.debug("OCR result details:")
                                self._logger.debug("  Output path: %s", ocr_result.output_path)
                                self._logger.debug("  Duration: %.3f seconds", ocr_result.duration_seconds)
                                self._logger.debug("  Texts found: %d", ocr_result.total_texts_found)
                                self._logger.debug("  Avg confidence: %.3f", ocr_result.average_confidence)
                                self._logger.debug("  Low confidence count: %d", ocr_result.low_confidence_count)
                                
                                if ocr_result.total_texts_found == 0:
                                    self._logger.warning("  ⚠ OCR found ZERO text regions - check preprocessing quality")
                                
                                if ocr_result.low_confidence_count > 0:
                                    self._logger.warning("  %d text regions have low confidence",
                                                        ocr_result.low_confidence_count)
                            except Exception as ocr_error:
                                self._logger.error(
                                    "  Step 2/5: OCR FAILED - %s: %s",
                                    type(ocr_error).__name__,
                                    str(ocr_error)
                                )
                                raise  # Re-raise to be caught by outer exception handler
                            
                            # Step 3: Error correction
                            correction_result = error_corrector.process(input_path=ocr_result.output_path)
                            self._logger.info("  Step 3/5: Error correction completed in %.3f seconds. Output: %s",
                                             correction_result.duration_seconds,
                                             correction_result.output_path)
                            self._logger.info("  Applied %d corrections (%.1f%%)",
                                             correction_result.corrections_applied,
                                             correction_result.correction_rate * 100)
                            
                            # Step 4: Field validation
                            validation_result = field_validator.process(
                                input_path=correction_result.output_path
                            )
                            self._logger.info("  Step 4/5: Validation completed in %.3f seconds. Output: %s",
                                             validation_result.duration_seconds,
                                             validation_result.output_path)
                            self._logger.info("  Validated %d/%d fields (%.1f%%)",
                                             validation_result.validated_fields,
                                             validation_result.total_fields,
                                             validation_result.validation_rate * 100)
                            if validation_result.failed_validations > 0:
                                self._logger.warning("  %d validation failures detected",
                                                    validation_result.failed_validations)
                            
                            # Step 5: Data extraction
                            extraction_result = form_extractor.extract(
                                ocr_json_path=validation_result.output_path
                            )
                            self._logger.info("  Step 5/5: Data extraction completed in %.3f seconds. Output: %s",
                                             extraction_result.duration_seconds,
                                             extraction_result.output_path)
                            self._logger.info("  Extracted %d header fields, %d defect blocks, %d analysis rows",
                                             extraction_result.header_fields_extracted,
                                             extraction_result.defect_blocks_found,
                                             extraction_result.analysis_rows_found)
                            if extraction_result.mandatory_fields_missing > 0:
                                self._logger.warning("  %d mandatory fields are missing",
                                                    extraction_result.mandatory_fields_missing)
                            
                            file_duration = time.perf_counter() - file_start
                            
                            file_results.append(FileResult(
                                filename=image_file.name,
                                success=True,
                                duration_seconds=file_duration,
                                texts_found=ocr_result.total_texts_found,
                                average_confidence=ocr_result.average_confidence,
                                corrections_applied=correction_result.corrections_applied,
                                fields_validated=validation_result.validated_fields,
                                extraction_result_path=extraction_result.output_path
                            ))
                            
                            self._logger.info("✓ %s completed in %.3f seconds", 
                                             image_file.name, file_duration)
                            
                            # Log memory usage and cleanup
                            self._memory_monitor.log_memory_delta(
                                file_mem_start, 
                                f"after {image_file.name}"
                            )
                            gc.collect()
                            self._memory_monitor.log_memory("after gc.collect()", level="DEBUG")
                            
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
            finally:
                # Log pool statistics
                if self._pool is not None:
                    pool_stats = self._pool.get_statistics()
                    self._logger.info(
                        "Pool statistics: Total engines: %d, "
                        "Files processed: %d, Total restarts: %d",
                        pool_stats.total_engines,
                        pool_stats.total_files_processed,
                        pool_stats.total_restarts,
                    )
        else:
            # Single engine mode (backward compatibility)
            ocr_engine = OCREngine(settings=self._settings, logger=self._logger)
            engine_reload_count = 0
            
            try:
                for image_file in image_files:
                    self._logger.info("Processing %s...", image_file.name)
                    file_start = time.perf_counter()
                    file_mem_start = self._memory_monitor.log_memory(
                        f"before {image_file.name}", 
                        level="DEBUG"
                    )
                    
                    # Check memory and reload OCR engine if necessary
                    current_memory_mb = self._memory_monitor.get_memory_mb()
                    if current_memory_mb > self._settings.ocr_memory_reload_threshold_mb:
                        self._logger.warning(
                            "Memory usage (%.1f MB) exceeds threshold (%d MB). "
                            "Reloading OCR engine to free memory...",
                            current_memory_mb,
                            self._settings.ocr_memory_reload_threshold_mb
                        )
                        
                        # Close old engine and create new one
                        ocr_engine.close()
                        gc.collect()
                        mem_after_gc = self._memory_monitor.log_memory("after engine close + gc", level="INFO")
                        
                        ocr_engine = OCREngine(settings=self._settings, logger=self._logger)
                        engine_reload_count += 1
                        mem_after_reload = self._memory_monitor.log_memory(
                            f"after engine reload #{engine_reload_count}", 
                            level="INFO"
                        )
                        self._logger.info(
                            "OCR engine reloaded successfully (reload #%d). "
                            "Memory freed: %.1f MB",
                            engine_reload_count,
                            current_memory_mb - mem_after_reload
                        )
                    
                    try:
                        # Step 1: Preprocessing
                        preprocess_result = preprocessor.process(input_path=image_file)
                        self._logger.info("  Step 1/5: Preprocessing completed in %.3f seconds. Output: %s",
                                         preprocess_result.duration_seconds,
                                         preprocess_result.output_path)
                        if preprocess_result.deskew_angle is not None:
                            self._logger.debug("  Deskew angle: %.3f degrees", preprocess_result.deskew_angle)
                        
                        # Step 2: OCR with shared engine (regional processing)
                        try:
                            # Detect regions
                            preprocessed_image = cv2.imread(str(preprocess_result.output_path), cv2.IMREAD_COLOR)
                            if preprocessed_image is None:
                                raise ValueError(
                                    f"Unable to read preprocessed image '{preprocess_result.output_path}'"
                                )
                            
                            regions = region_detector.detect_zones(preprocessed_image)
                            self._logger.debug(
                                "  Region detector identified %d zones using method '%s'",
                                len(regions),
                                regions[0].detection_method if regions else "none",
                            )
                            
                            # Process regions
                            ocr_result = ocr_engine.process_regions(
                                image_path=preprocess_result.output_path,
                                regions=regions,
                            )
                            self._logger.info("  Step 2/5: OCR completed in %.3f seconds. Output: %s",
                                             ocr_result.duration_seconds,
                                             ocr_result.output_path)
                            self._logger.info("  Found %d text regions (avg confidence: %.3f)",
                                             ocr_result.total_texts_found,
                                             ocr_result.average_confidence)
                            
                            # Детальное логирование результатов OCR
                            self._logger.debug("OCR result details:")
                            self._logger.debug("  Output path: %s", ocr_result.output_path)
                            self._logger.debug("  Duration: %.3f seconds", ocr_result.duration_seconds)
                            self._logger.debug("  Texts found: %d", ocr_result.total_texts_found)
                            self._logger.debug("  Avg confidence: %.3f", ocr_result.average_confidence)
                            self._logger.debug("  Low confidence count: %d", ocr_result.low_confidence_count)
                            
                            if ocr_result.total_texts_found == 0:
                                self._logger.warning("  ⚠ OCR found ZERO text regions - check preprocessing quality")
                            
                            if ocr_result.low_confidence_count > 0:
                                self._logger.warning("  %d text regions have low confidence",
                                                    ocr_result.low_confidence_count)
                        except Exception as ocr_error:
                            self._logger.error(
                                "  Step 2/5: OCR FAILED - %s: %s",
                                type(ocr_error).__name__,
                                str(ocr_error)
                            )
                            raise  # Re-raise to be caught by outer exception handler
                        
                        # Step 3: Error correction
                        correction_result = error_corrector.process(input_path=ocr_result.output_path)
                        self._logger.info("  Step 3/5: Error correction completed in %.3f seconds. Output: %s",
                                         correction_result.duration_seconds,
                                         correction_result.output_path)
                        self._logger.info("  Applied %d corrections (%.1f%%)",
                                         correction_result.corrections_applied,
                                         correction_result.correction_rate * 100)
                        
                        # Step 4: Field validation
                        validation_result = field_validator.process(
                            input_path=correction_result.output_path
                        )
                        self._logger.info("  Step 4/5: Validation completed in %.3f seconds. Output: %s",
                                         validation_result.duration_seconds,
                                         validation_result.output_path)
                        self._logger.info("  Validated %d/%d fields (%.1f%%)",
                                         validation_result.validated_fields,
                                         validation_result.total_fields,
                                         validation_result.validation_rate * 100)
                        if validation_result.failed_validations > 0:
                            self._logger.warning("  %d validation failures detected",
                                                validation_result.failed_validations)
                        
                        # Step 5: Data extraction
                        extraction_result = form_extractor.extract(
                            ocr_json_path=validation_result.output_path
                        )
                        self._logger.info("  Step 5/5: Data extraction completed in %.3f seconds. Output: %s",
                                         extraction_result.duration_seconds,
                                         extraction_result.output_path)
                        self._logger.info("  Extracted %d header fields, %d defect blocks, %d analysis rows",
                                         extraction_result.header_fields_extracted,
                                         extraction_result.defect_blocks_found,
                                         extraction_result.analysis_rows_found)
                        if extraction_result.mandatory_fields_missing > 0:
                            self._logger.warning("  %d mandatory fields are missing",
                                                extraction_result.mandatory_fields_missing)
                        
                        file_duration = time.perf_counter() - file_start
                        
                        file_results.append(FileResult(
                            filename=image_file.name,
                            success=True,
                            duration_seconds=file_duration,
                            texts_found=ocr_result.total_texts_found,
                            average_confidence=ocr_result.average_confidence,
                            corrections_applied=correction_result.corrections_applied,
                            fields_validated=validation_result.validated_fields,
                            extraction_result_path=extraction_result.output_path
                        ))
                        
                        self._logger.info("✓ %s completed in %.3f seconds", 
                                         image_file.name, file_duration)
                        
                        # Log memory usage and cleanup
                        self._memory_monitor.log_memory_delta(
                            file_mem_start, 
                            f"after {image_file.name}"
                        )
                        gc.collect()
                        self._memory_monitor.log_memory("after gc.collect()", level="DEBUG")
                        
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
            finally:
                # Clean up OCR engine at the end
                if ocr_engine is not None:
                    ocr_engine.close()
                
                if engine_reload_count > 0:
                    self._logger.info(
                        "Total OCR engine reloads during batch: %d",
                        engine_reload_count
                    )
        
        # Final memory report
        self._logger.info("=== Batch processing completed ===")
        self._memory_monitor.log_memory_delta(initial_memory, "total batch")
        
        return file_results

    def _process_ocr_batch(self, image_files: List[Path]) -> List[FileResult]:
        """Process batch with OCR only and automatic OCR engine reload."""
        file_results = []
        
        initial_memory = self._memory_monitor.log_memory("OCR batch start", level="INFO")
        
        # Use pool if enabled, otherwise create single engine
        if self._pool is not None:
            # Pool mode: engines are acquired/released per file
            try:
                for image_file in image_files:
                    self._logger.info("Processing %s...", image_file.name)
                    file_start = time.perf_counter()
                    file_mem_start = self._memory_monitor.log_memory(
                        f"before {image_file.name}", 
                        level="DEBUG"
                    )
                    
                    # Acquire engine from pool (automatically released after use)
                    with self._pool.acquire() as ocr_engine:
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
                            
                            # Log memory usage and cleanup
                            self._memory_monitor.log_memory_delta(
                                file_mem_start, 
                                f"after {image_file.name}"
                            )
                            gc.collect()
                            self._memory_monitor.log_memory("after gc.collect()", level="DEBUG")
                            
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
            finally:
                # Log pool statistics
                if self._pool is not None:
                    pool_stats = self._pool.get_statistics()
                    self._logger.info(
                        "Pool statistics: Total engines: %d, "
                        "Files processed: %d, Total restarts: %d",
                        pool_stats.total_engines,
                        pool_stats.total_files_processed,
                        pool_stats.total_restarts,
                    )
        else:
            # Single engine mode (backward compatibility)
            ocr_engine = OCREngine(settings=self._settings, logger=self._logger)
            engine_reload_count = 0
            
            try:
                for image_file in image_files:
                    self._logger.info("Processing %s...", image_file.name)
                    file_start = time.perf_counter()
                    file_mem_start = self._memory_monitor.log_memory(
                        f"before {image_file.name}", 
                        level="DEBUG"
                    )
                    
                    # Check memory and reload OCR engine if necessary
                    current_memory_mb = self._memory_monitor.get_memory_mb()
                    if current_memory_mb > self._settings.ocr_memory_reload_threshold_mb:
                        self._logger.warning(
                            "Memory usage (%.1f MB) exceeds threshold (%d MB). "
                            "Reloading OCR engine to free memory...",
                            current_memory_mb,
                            self._settings.ocr_memory_reload_threshold_mb
                        )
                        
                        # Close old engine and create new one
                        ocr_engine.close()
                        gc.collect()
                        mem_after_gc = self._memory_monitor.log_memory("after engine close + gc", level="INFO")
                        
                        ocr_engine = OCREngine(settings=self._settings, logger=self._logger)
                        engine_reload_count += 1
                        mem_after_reload = self._memory_monitor.log_memory(
                            f"after engine reload #{engine_reload_count}", 
                            level="INFO"
                        )
                        self._logger.info(
                            "OCR engine reloaded successfully (reload #%d). "
                            "Memory freed: %.1f MB",
                            engine_reload_count,
                            current_memory_mb - mem_after_reload
                        )
                    
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
                        
                        # Log memory usage and cleanup
                        self._memory_monitor.log_memory_delta(
                            file_mem_start, 
                            f"after {image_file.name}"
                        )
                        gc.collect()
                        self._memory_monitor.log_memory("after gc.collect()", level="DEBUG")
                        
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
            finally:
                # Clean up OCR engine at the end
                if ocr_engine is not None:
                    ocr_engine.close()
                
                if engine_reload_count > 0:
                    self._logger.info(
                        "Total OCR engine reloads during batch: %d",
                        engine_reload_count
                    )
        
        # Final memory report
        self._memory_monitor.log_memory_delta(initial_memory, "OCR batch total")
        
        return file_results

    def _process_preprocess_batch(self, image_files: List[Path]) -> List[FileResult]:
        """Process batch with preprocessing only."""
        file_results = []
        preprocessor = ImagePreprocessor(settings=self._settings, logger=self._logger)
        
        initial_memory = self._memory_monitor.log_memory("Preprocess batch start", level="INFO")
        
        for image_file in image_files:
            self._logger.info("Processing %s...", image_file.name)
            file_start = time.perf_counter()
            file_mem_start = self._memory_monitor.log_memory(
                f"before {image_file.name}", 
                level="DEBUG"
            )
            
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
                
                # Log memory usage and cleanup
                self._memory_monitor.log_memory_delta(
                    file_mem_start, 
                    f"after {image_file.name}"
                )
                gc.collect()
                self._memory_monitor.log_memory("after gc.collect()", level="DEBUG")
                
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
        
        # Final memory report
        self._memory_monitor.log_memory_delta(initial_memory, "Preprocess batch total")
        
        return file_results

    def _process_correction_batch(self, image_files: List[Path]) -> List[FileResult]:
        """Process batch with correction and validation only."""
        file_results = []
        error_corrector = ErrorCorrector(settings=self._settings, logger=self._logger)
        field_validator = FieldValidator(settings=self._settings, logger=self._logger)
        
        initial_memory = self._memory_monitor.log_memory("Correction batch start", level="INFO")
        
        for image_file in image_files:
            self._logger.info("Processing %s...", image_file.name)
            file_start = time.perf_counter()
            file_mem_start = self._memory_monitor.log_memory(
                f"before {image_file.name}", 
                level="DEBUG"
            )
            
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
                
                # Log memory usage and cleanup
                self._memory_monitor.log_memory_delta(
                    file_mem_start, 
                    f"after {image_file.name}"
                )
                gc.collect()
                self._memory_monitor.log_memory("after gc.collect()", level="DEBUG")
                
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
        
        # Final memory report
        self._memory_monitor.log_memory_delta(initial_memory, "Correction batch total")
        
        return file_results

    def _save_batch_summary(self, batch_result: BatchResult, output_dir: Optional[Path]) -> Path:
        """Save batch processing summary to JSON file."""
        import json
        from datetime import datetime
        
        output_path = output_dir or self._settings.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_path / f"batch_summary_{timestamp}.json"
        
        # Mandatory fields to extract from header
        mandatory_fields = [
            "act_number",
            "act_date",
            "template_revision",
            "part_line_number",
            "quantity_checked",
            "control_type",
            "inspector_name",
        ]
        
        file_results_data = []
        for result in batch_result.file_results:
            file_data = {
                "filename": result.filename,
                "success": result.success,
                "duration_seconds": round(result.duration_seconds, 3),
                "texts_found": result.texts_found,
                "average_confidence": round(result.average_confidence, 3),
                "corrections_applied": result.corrections_applied,
                "fields_validated": result.fields_validated,
                "error_message": result.error_message,
            }
            
            # Extract required fields if extraction result exists
            if result.success and result.extraction_result_path:
                required_fields = self._extract_required_fields(
                    result.extraction_result_path, mandatory_fields
                )
                if required_fields is not None:
                    file_data["required_fields"] = required_fields
                else:
                    # If extraction file cannot be read, mark all fields as missing
                    file_data["required_fields"] = {
                        field: {"value": None, "confidence": 0.0, "suspicious": False}
                        for field in mandatory_fields
                    }
            else:
                # If extraction failed or path missing, mark all fields as missing
                file_data["required_fields"] = {
                    field: {"value": None, "confidence": 0.0, "suspicious": False}
                    for field in mandatory_fields
                }
            
            file_results_data.append(file_data)
        
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
            "file_results": file_results_data
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
        
        return summary_file
    
    def _extract_required_fields(
        self, extraction_result_path: Path, mandatory_fields: List[str]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Extract required fields from extraction result JSON file.
        
        Args:
            extraction_result_path: Path to -data.json file
            mandatory_fields: List of mandatory field names to extract
            
        Returns:
            Dictionary mapping field names to their values, confidence, and suspicious flag,
            or None if file cannot be read
        """
        import json
        
        if not extraction_result_path.exists():
            self._logger.warning(
                "Extraction result file not found: %s", extraction_result_path
            )
            return None
        
        try:
            with open(extraction_result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self._logger.warning(
                "Failed to load extraction result from %s: %s",
                extraction_result_path,
                str(e)
            )
            return None
        
        # Extract header section
        header = data.get("header", {})
        if not header:
            self._logger.warning(
                "No header section found in extraction result: %s",
                extraction_result_path
            )
            # Return all fields as missing
            return {
                field: {"value": None, "confidence": 0.0, "suspicious": False}
                for field in mandatory_fields
            }
        
        # Extract required fields
        required_fields = {}
        for field_name in mandatory_fields:
            field_data = header.get(field_name)
            
            if field_data is None:
                # Field is missing
                required_fields[field_name] = {
                    "value": None,
                    "confidence": 0.0,
                    "suspicious": False,
                }
            else:
                # Extract value, confidence, and suspicious flag
                value = field_data.get("value")
                confidence = field_data.get("confidence", 0.0)
                suspicious = field_data.get("suspicious", False)
                
                required_fields[field_name] = {
                    "value": value,
                    "confidence": round(confidence, 3),
                    "suspicious": suspicious,
                }
        
        return required_fields

