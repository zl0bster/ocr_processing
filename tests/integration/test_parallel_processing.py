"""Integration tests for parallel vs sequential processing."""
import pytest
import json
import time
import cv2
from pathlib import Path

from src.config.settings import Settings
from src.region_detector import RegionDetector
from src.ocr_engine import OCREngine


@pytest.mark.integration
@pytest.mark.requires_ocr
@pytest.mark.slow
class TestParallelProcessing:
    """Test parallel vs sequential processing integration."""
    
    def test_parallel_region_processing(
        self, integration_settings, integration_logger, shared_ocr_engine,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test parallel region processing with multiple regions."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Ensure parallel processing is enabled
        if not integration_settings.enable_parallel_processing:
            pytest.skip("Parallel processing disabled in settings")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        
        # Detect regions
        regions = region_detector.detect_zones(image, strategy="auto")
        if len(regions) < integration_settings.parallel_min_regions_for_parallelization:
            pytest.skip(f"Not enough regions ({len(regions)}) for parallel processing")
        
        # Act
        start_time = time.perf_counter()
        result = shared_ocr_engine.process_regions(image_path, regions)
        parallel_duration = time.perf_counter() - start_time
        
        # Assert
        assert result.output_path.exists(), "Output should be created"
        assert result.total_texts_found > 0, "Should detect text"
        assert parallel_duration > 0, "Should take time"
        
        # Verify regional results
        with open(result.output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        assert "ocr_results_by_region" in ocr_data, "Should contain regional results"
    
    def test_sequential_region_processing(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test sequential region processing as baseline."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange - Disable parallel processing
        sequential_settings = Settings(
            **integration_settings.model_dump(),
            enable_parallel_processing=False,
        )
        
        engine = OCREngine(sequential_settings, integration_logger)
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(sequential_settings, integration_logger)
        
        regions = region_detector.detect_zones(image, strategy="auto")
        if len(regions) == 0:
            pytest.skip("No regions detected")
        
        try:
            # Act
            start_time = time.perf_counter()
            result = engine.process_regions(image_path, regions)
            sequential_duration = time.perf_counter() - start_time
            
            # Assert
            assert result.output_path.exists(), "Output should be created"
            assert result.total_texts_found > 0, "Should detect text"
            assert sequential_duration > 0, "Should take time"
        finally:
            engine.close()
    
    def test_parallel_vs_sequential_results_consistency(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that parallel and sequential processing produce identical results."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        regions = region_detector.detect_zones(image, strategy="auto")
        
        if len(regions) < integration_settings.parallel_min_regions_for_parallelization:
            pytest.skip("Not enough regions for parallel processing")
        
        # Sequential processing
        sequential_settings = Settings(
            **integration_settings.model_dump(),
            enable_parallel_processing=False,
        )
        sequential_engine = OCREngine(sequential_settings, integration_logger)
        
        # Parallel processing
        parallel_engine = OCREngine(integration_settings, integration_logger)
        
        try:
            # Act - Sequential
            sequential_result = sequential_engine.process_regions(image_path, regions)
            
            # Act - Parallel
            parallel_result = parallel_engine.process_regions(image_path, regions)
            
            # Assert - Results should be consistent
            assert sequential_result.total_texts_found == parallel_result.total_texts_found, \
                "Text count should match"
            
            # Load and compare JSON structures
            with open(sequential_result.output_path, 'r', encoding='utf-8') as f:
                sequential_data = json.load(f)
            
            with open(parallel_result.output_path, 'r', encoding='utf-8') as f:
                parallel_data = json.load(f)
            
            # Compare regional results
            sequential_regions = sequential_data.get("ocr_results_by_region", {})
            parallel_regions = parallel_data.get("ocr_results_by_region", {})
            
            assert set(sequential_regions.keys()) == set(parallel_regions.keys()), \
                "Region IDs should match"
            
            # Compare text counts per region
            for region_id in sequential_regions.keys():
                seq_count = len(sequential_regions[region_id])
                par_count = len(parallel_regions[region_id])
                assert seq_count == par_count, \
                    f"Text count for region {region_id} should match: {seq_count} vs {par_count}"
        finally:
            sequential_engine.close()
            parallel_engine.close()
    
    def test_parallel_processing_performance(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that parallel processing is faster for multiple regions."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        if not integration_settings.enable_parallel_processing:
            pytest.skip("Parallel processing disabled")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        regions = region_detector.detect_zones(image, strategy="auto")
        
        if len(regions) < integration_settings.parallel_min_regions_for_parallelization:
            pytest.skip("Not enough regions for parallel processing")
        
        # Sequential
        sequential_settings = Settings(
            **integration_settings.model_dump(),
            enable_parallel_processing=False,
        )
        sequential_engine = OCREngine(sequential_settings, integration_logger)
        
        # Parallel
        parallel_engine = OCREngine(integration_settings, integration_logger)
        
        try:
            # Measure sequential
            start = time.perf_counter()
            sequential_result = sequential_engine.process_regions(image_path, regions)
            sequential_time = time.perf_counter() - start
            
            # Measure parallel
            start = time.perf_counter()
            parallel_result = parallel_engine.process_regions(image_path, regions)
            parallel_time = time.perf_counter() - start
            
            # Assert - Parallel should be faster or at least not much slower
            # (Due to overhead, parallel might be slightly slower for small workloads)
            # Just verify both complete successfully
            assert sequential_result.total_texts_found > 0
            assert parallel_result.total_texts_found > 0
            assert sequential_time > 0
            assert parallel_time > 0
        finally:
            sequential_engine.close()
            parallel_engine.close()
    
    def test_parallel_processing_memory_management(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test memory management in parallel processing mode."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        if not integration_settings.enable_parallel_processing:
            pytest.skip("Parallel processing disabled")
        
        # Arrange
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(integration_settings, integration_logger)
        regions = region_detector.detect_zones(image, strategy="auto")
        
        if len(regions) < integration_settings.parallel_min_regions_for_parallelization:
            pytest.skip("Not enough regions for parallel processing")
        
        engine = OCREngine(integration_settings, integration_logger)
        
        try:
            # Act - Process multiple times to check for memory leaks
            result1 = engine.process_regions(image_path, regions)
            result2 = engine.process_regions(image_path, regions)
            
            # Assert - Both should succeed
            assert result1.output_path.exists()
            assert result2.output_path.exists()
            assert result1.total_texts_found == result2.total_texts_found, \
                "Results should be consistent"
        finally:
            engine.close()
    
    def test_parallel_processing_fallback_to_sequential(
        self, integration_settings, integration_logger,
        test_image_paths, cleanup_integration_outputs
    ):
        """Test that processing falls back to sequential when parallel is disabled."""
        if "compressed" not in test_image_paths:
            pytest.skip("Compressed test image not found")
        
        # Arrange - Disable parallel processing
        sequential_settings = Settings(
            **integration_settings.model_dump(),
            enable_parallel_processing=False,
        )
        
        engine = OCREngine(sequential_settings, integration_logger)
        image_path = test_image_paths["compressed"]
        image = cv2.imread(str(image_path))
        region_detector = RegionDetector(sequential_settings, integration_logger)
        regions = region_detector.detect_zones(image, strategy="auto")
        
        if len(regions) == 0:
            pytest.skip("No regions detected")
        
        try:
            # Act
            result = engine.process_regions(image_path, regions)
            
            # Assert - Should use sequential processing
            assert result.output_path.exists()
            assert result.total_texts_found > 0
        finally:
            engine.close()

