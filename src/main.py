from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import cv2

from config.settings import Settings
from preprocessor import ImagePreprocessor
from ocr_engine import OCREngine
from error_corrector import ErrorCorrector
from field_validator import FieldValidator
from form_extractor import FormExtractor
from batch_processor import BatchProcessor
from region_detector import RegionDetector

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="OCR processing pipeline CLI. Supports preprocessing and text recognition.",
    )
    parser.add_argument("--file", type=str, help="Path to the image file for processing.")
    parser.add_argument(
        "--batch",
        "--directory",
        dest="batch_dir",
        type=str,
        help="Path to directory for batch processing of multiple images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to store processed results. "
        "Defaults to the configured output directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "ocr", "correction", "pipeline", "batch"],
        default="pipeline",
        help="Processing mode: preprocess only, OCR only, correction only, full pipeline, or batch (default: pipeline).",
    )
    return parser


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure logger with console and rotating file handlers."""
    logger = logging.getLogger("image_ocr")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = log_dir / log_filename

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=settings.log_max_size_mb * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logger.level)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.debug("Logging initialized. Log file: %s", log_path)
    return logger


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings()
    logger = setup_logging(settings)

    # Check for batch mode
    if args.batch_dir or args.mode == "batch":
        if not args.batch_dir:
            logger.error("Batch mode requires --batch or --directory argument with path to directory.")
            return 1
        batch_path = Path(args.batch_dir)
        if not batch_path.exists():
            logger.error("Batch directory '%s' not found.", batch_path)
            return 1
        if not batch_path.is_dir():
            logger.error("Batch path '%s' is not a directory.", batch_path)
            return 1
        
        try:
            # Determine mode - default to pipeline for batch
            batch_mode = args.mode if args.mode != "batch" else "pipeline"
            return _run_batch(batch_path, args.output, batch_mode, settings, logger)
        except Exception as exc:
            logger.error("Batch processing failed: %s", exc, exc_info=logger.level == logging.DEBUG)
            return 1

    # Single file mode
    if not args.file:
        logger.info("No input file or batch directory provided. Use --file or --batch to specify input.")
        return 0

    input_path = Path(args.file)
    if not input_path.exists():
        logger.error("Input file '%s' not found.", input_path)
        return 1

    try:
        if args.mode == "preprocess":
            return _run_preprocessing(input_path, args.output, settings, logger)
        elif args.mode == "ocr":
            return _run_ocr(input_path, args.output, settings, logger)
        elif args.mode == "correction":
            return _run_correction(input_path, args.output, settings, logger)
        elif args.mode == "pipeline" or args.mode == "batch":
            return _run_pipeline(input_path, args.output, settings, logger)
        else:
            logger.error("Unknown processing mode: %s", args.mode)
            return 1
            
    except Exception as exc:
        logger.error("Processing failed: %s", exc, exc_info=logger.level == logging.DEBUG)
        return 1


def _run_preprocessing(input_path: Path, output_path: Optional[str], settings: Settings, logger: logging.Logger) -> int:
    """Run preprocessing only."""
    output_path_obj = Path(output_path) if output_path else None
    preprocessor = ImagePreprocessor(settings=settings, logger=logger)
    
    result = preprocessor.process(input_path=input_path, output_path=output_path_obj)
    
    logger.info(
        "Preprocessing succeeded. Output: %s (elapsed %.3f seconds)",
        result.output_path,
        result.duration_seconds,
    )
    if result.deskew_angle is not None:
        logger.info("Deskew angle applied: %.3f degrees", result.deskew_angle)
    
    return 0


def _run_ocr(input_path: Path, output_path: Optional[str], settings: Settings, logger: logging.Logger) -> int:
    """Run OCR only (assumes image is already preprocessed)."""
    output_path_obj = Path(output_path) if output_path else None
    ocr_engine = OCREngine(settings=settings, logger=logger)
    
    result = ocr_engine.process(input_path=input_path, output_path=output_path_obj)
    
    logger.info(
        "OCR processing succeeded. Output: %s (elapsed %.3f seconds)",
        result.output_path,
        result.duration_seconds,
    )
    logger.info(
        "Found %d text regions with average confidence %.3f",
        result.total_texts_found,
        result.average_confidence,
    )
    
    if result.low_confidence_count > 0:
        logger.warning(
            "%d text regions have low confidence",
            result.low_confidence_count,
        )
    
    return 0


def _run_correction(input_path: Path, output_path: Optional[str], settings: Settings, logger: logging.Logger) -> int:
    """Run error correction and validation on existing OCR results."""
    logger.info("Starting error correction and validation...")
    
    # Step 1: Error correction
    logger.info("Step 1: Applying error corrections...")
    error_corrector = ErrorCorrector(settings=settings, logger=logger)
    correction_result = error_corrector.process(input_path=input_path)
    
    logger.info(
        "Error correction completed in %.3f seconds. Output: %s",
        correction_result.duration_seconds,
        correction_result.output_path,
    )
    logger.info(
        "Applied %d corrections to %d texts (%.1f%%)",
        correction_result.corrections_applied,
        correction_result.total_texts,
        correction_result.correction_rate * 100
    )
    
    # Step 2: Field validation
    logger.info("Step 2: Validating fields...")
    field_validator = FieldValidator(settings=settings, logger=logger)
    
    # Use custom output path if provided, otherwise validate in place
    validation_output_path = Path(output_path) if output_path else None
    validation_result = field_validator.process(
        input_path=correction_result.output_path,
        output_path=validation_output_path
    )
    
    logger.info(
        "Field validation completed in %.3f seconds. Output: %s",
        validation_result.duration_seconds,
        validation_result.output_path,
    )
    
    # Final summary
    total_time = correction_result.duration_seconds + validation_result.duration_seconds
    logger.info("=== Correction & Validation Summary ===")
    logger.info("Total processing time: %.3f seconds", total_time)
    logger.info("Corrections applied: %d", correction_result.corrections_applied)
    logger.info("Fields validated: %d/%d (%.1f%%)",
                validation_result.validated_fields,
                validation_result.total_fields,
                validation_result.validation_rate * 100)
    
    if validation_result.failed_validations > 0:
        logger.warning(
            "%d validation failures detected - review output file",
            validation_result.failed_validations,
        )
    
    logger.info("Final output: %s", validation_result.output_path)
    
    return 0


def _run_pipeline(input_path: Path, output_path: Optional[str], settings: Settings, logger: logging.Logger) -> int:
    """Run full pipeline: preprocessing, OCR, error correction, and validation."""
    logger.info("Starting full processing pipeline...")
    
    # Step 1: Preprocessing
    logger.info("Step 1: Image preprocessing...")
    preprocessor = ImagePreprocessor(settings=settings, logger=logger)
    preprocess_result = preprocessor.process(input_path=input_path)
    
    logger.info(
        "Preprocessing completed in %.3f seconds. Output: %s",
        preprocess_result.duration_seconds,
        preprocess_result.output_path,
    )
    
    # Step 2: OCR on preprocessed image
    logger.info("Step 2: OCR text recognition...")
    region_detector = RegionDetector(settings=settings, logger=logger)

    preprocessed_image = cv2.imread(str(preprocess_result.output_path), cv2.IMREAD_COLOR)
    if preprocessed_image is None:
        raise ValueError(
            f"Unable to read preprocessed image '{preprocess_result.output_path}'"
        )

    regions = region_detector.detect_zones(preprocessed_image)
    logger.info(
        "Region detector identified %d zones using method '%s'",
        len(regions),
        regions[0].detection_method if regions else "none",
    )
    ocr_engine = OCREngine(settings=settings, logger=logger)
    ocr_result = ocr_engine.process_regions(
        image_path=preprocess_result.output_path,
        regions=regions,
    )
    
    logger.info(
        "OCR processing completed in %.3f seconds. Output: %s",
        ocr_result.duration_seconds,
        ocr_result.output_path,
    )
    
    # Step 3: Error correction
    logger.info("Step 3: Applying error corrections...")
    error_corrector = ErrorCorrector(settings=settings, logger=logger)
    correction_result = error_corrector.process(input_path=ocr_result.output_path)
    
    logger.info(
        "Error correction completed in %.3f seconds. Output: %s",
        correction_result.duration_seconds,
        correction_result.output_path,
    )
    
    # Step 4: Field validation
    logger.info("Step 4: Validating fields...")
    field_validator = FieldValidator(settings=settings, logger=logger)
    
    # For pipeline mode, use custom output path if provided for final results only
    final_output_path = Path(output_path) if output_path else None
    validation_result = field_validator.process(
        input_path=correction_result.output_path,
        output_path=final_output_path
    )
    
    logger.info(
        "Field validation completed in %.3f seconds. Output: %s",
        validation_result.duration_seconds,
        validation_result.output_path,
    )
    
    # Step 5: Data extraction
    logger.info("Step 5: Extracting structured data...")
    form_extractor = FormExtractor(settings=settings, logger=logger)
    
    extraction_result = form_extractor.extract(
        ocr_json_path=validation_result.output_path,
        output_path=final_output_path  # Use final_output_path if provided
    )
    
    logger.info(
        "Data extraction completed in %.3f seconds. Output: %s",
        extraction_result.duration_seconds,
        extraction_result.output_path
    )
    logger.info(
        "Extracted %d header fields, %d defect blocks, %d analysis rows",
        extraction_result.header_fields_extracted,
        extraction_result.defect_blocks_found,
        extraction_result.analysis_rows_found
    )
    
    if extraction_result.mandatory_fields_missing > 0:
        logger.warning(
            "%d mandatory fields are missing",
            extraction_result.mandatory_fields_missing
        )
    
    # Final summary
    total_time = (preprocess_result.duration_seconds + ocr_result.duration_seconds + 
                  correction_result.duration_seconds + validation_result.duration_seconds +
                  extraction_result.duration_seconds)
    
    logger.info("=== Pipeline Summary ===")
    logger.info("Total processing time: %.3f seconds", total_time)
    logger.info("Preprocessed image: %s", preprocess_result.output_path)
    logger.info("OCR results: %s", ocr_result.output_path)
    logger.info("Text regions found: %d (avg confidence: %.3f)", 
                ocr_result.total_texts_found, ocr_result.average_confidence)
    logger.info("Corrections applied: %d", correction_result.corrections_applied)
    logger.info("Fields validated: %d/%d (%.1f%%)",
                validation_result.validated_fields,
                validation_result.total_fields,
                validation_result.validation_rate * 100)
    logger.info("Header fields extracted: %d", extraction_result.header_fields_extracted)
    logger.info("Defect blocks found: %d", extraction_result.defect_blocks_found)
    logger.info("Analysis rows found: %d", extraction_result.analysis_rows_found)
    logger.info("Final output: %s", extraction_result.output_path)
    
    if preprocess_result.deskew_angle is not None:
        logger.info("Deskew angle applied: %.3f degrees", preprocess_result.deskew_angle)
    
    if ocr_result.low_confidence_count > 0:
        logger.warning(
            "%d text regions have low confidence",
            ocr_result.low_confidence_count,
        )
    
    if validation_result.failed_validations > 0:
        logger.warning(
            "%d validation failures detected",
            validation_result.failed_validations,
        )
    
    return 0


def _run_batch(batch_dir: Path, output_path: Optional[str], mode: str, settings: Settings, logger: logging.Logger) -> int:
    """Run batch processing on directory of images."""
    logger.info("Starting batch processing of directory: %s", batch_dir)
    logger.info("Processing mode: %s", mode)
    
    output_path_obj = Path(output_path) if output_path else None
    batch_processor = BatchProcessor(settings=settings, logger=logger)
    
    try:
        result = batch_processor.process_directory(
            input_dir=batch_dir,
            output_dir=output_path_obj,
            mode=mode
        )
        
        # Log final results
        logger.info("=== Batch Processing Complete ===")
        logger.info("Total files processed: %d", result.total_files)
        logger.info("Successful: %d", result.successful_files)
        logger.info("Failed: %d", result.failed_files)
        logger.info("Total time: %.3f seconds", result.total_duration_seconds)
        
        if result.total_files > 0:
            logger.info("Average time per file: %.3f seconds", 
                       result.total_duration_seconds / result.total_files)
        
        if result.summary_path:
            logger.info("Batch summary saved to: %s", result.summary_path)
        
        # Return error code if any files failed
        return 1 if result.failed_files > 0 else 0
    finally:
        # Clean up pool resources
        batch_processor.close()


if __name__ == "__main__":
    sys.exit(main())

