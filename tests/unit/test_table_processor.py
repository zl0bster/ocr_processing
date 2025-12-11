"""Unit tests for TableProcessor module."""

import logging
import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from concurrent.futures import ProcessPoolExecutor, Future

from src.table_processor import TableProcessor, TableCell
from src.table_detector import TableGrid
from src.config.settings import Settings


@pytest.fixture
def mock_logger():
    """Mock logging.Logger instance."""
    logger = MagicMock(spec=logging.Logger)
    logger.level = logging.WARNING
    return logger


@pytest.fixture
def table_settings():
    """Settings with table processing parameters."""
    return Settings(
        enable_parallel_processing=True,
        parallel_min_cells_for_parallelization=10,
        parallel_cells_workers=2,
        table_cell_margin=2,
        table_cell_preprocess=True,
        table_cell_min_width=20,
        table_cell_min_height=15,
    )


@pytest.fixture
def table_settings_no_parallel():
    """Settings with parallel processing disabled."""
    return Settings(
        enable_parallel_processing=False,
        table_cell_margin=2,
        table_cell_preprocess=False,
        table_cell_min_width=20,
        table_cell_min_height=15,
    )


@pytest.fixture
def mock_ocr_engine():
    """Mock OCR engine with required methods."""
    mock_engine = MagicMock()
    
    # Mock _resize_for_ocr to return image and scale
    def mock_resize(image):
        return image, 1.0
    mock_engine._resize_for_ocr = Mock(side_effect=mock_resize)
    
    # Mock _run_paddle_ocr to return mock results
    def mock_run_ocr(image):
        return [[([[10, 10], [50, 10], [50, 30], [10, 30]], ("test_text", 0.9))]]
    mock_engine._run_paddle_ocr = Mock(side_effect=mock_run_ocr)
    
    # Mock _process_ocr_results to return detections
    def mock_process_results(ocr_results, scale=1.0):
        from src.ocr_engine import TextDetection
        return [
            TextDetection(
                text="test_text",
                confidence=0.9,
                bbox=[[10, 10], [50, 10], [50, 30], [10, 30]],
                center_x=30,
                center_y=20,
            )
        ]
    mock_engine._process_ocr_results = Mock(side_effect=mock_process_results)
    
    return mock_engine


@pytest.fixture
def mock_table_grid():
    """Create mock TableGrid with predefined structure."""
    return TableGrid(
        rows=[0, 100, 200, 300, 400],
        cols=[0, 150, 300, 450],
        confidence=0.85,
        num_rows=4,
        num_cols=3,
        method="morphology",
    )


@pytest.fixture
def mock_table_image():
    """Create synthetic table region image."""
    image = np.ones((400, 450, 3), dtype=np.uint8) * 255
    
    # Draw some text-like content in cells
    cv2.rectangle(image, (10, 10), (140, 90), (200, 200, 200), -1)
    cv2.rectangle(image, (160, 10), (290, 90), (200, 200, 200), -1)
    cv2.rectangle(image, (310, 10), (440, 90), (200, 200, 200), -1)
    
    return image


@pytest.fixture
def column_mapping():
    """Column index to field name mapping."""
    return {
        0: "row_number",
        1: "parameter",
        2: "value",
    }


@pytest.mark.unit
class TestTableProcessorCellExtraction:
    """Test cell extraction workflow."""

    def test_extract_cells_with_valid_grid(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_grid, mock_table_image
    ):
        """Test extract_cells() with detected grid."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        
        # Act
        cells = processor.extract_cells(mock_table_image, mock_table_grid)
        
        # Assert
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(cell, TableCell) for cell in cells)
        assert all(cell.text is not None for cell in cells)

    def test_extract_cells_with_column_mapping(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_grid, mock_table_image, column_mapping
    ):
        """Test extract_cells() with column mapping."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        
        # Act
        cells = processor.extract_cells(mock_table_image, mock_table_grid, column_mapping)
        
        # Assert
        assert len(cells) > 0
        # Check that cells have field names from mapping
        cells_with_mapping = [c for c in cells if c.field_name is not None]
        assert len(cells_with_mapping) > 0

    def test_extract_cells_with_none_image(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_grid
    ):
        """Test extract_cells() with None image."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        
        # Act
        cells = processor.extract_cells(None, mock_table_grid)
        
        # Assert
        assert cells == []
        mock_logger.warning.assert_called()

    def test_extract_cells_with_empty_image(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_grid
    ):
        """Test extract_cells() with empty image."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        empty_image = np.array([])
        
        # Act
        cells = processor.extract_cells(empty_image, mock_table_grid)
        
        # Assert
        assert cells == []
        mock_logger.warning.assert_called()


@pytest.mark.unit
class TestTableProcessorParallelDecision:
    """Test parallel vs sequential processing decision."""

    def test_should_use_parallel_processing_with_large_table(
        self, table_settings, mock_logger, mock_ocr_engine, mock_table_grid
    ):
        """Test _should_use_parallel_processing() with large table."""
        # Arrange
        table_settings.enable_parallel_processing = True
        # Create mock engine pool (required for parallel processing)
        mock_engine_pool = MagicMock()
        processor = TableProcessor(
            settings=table_settings,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
            engine_pool=mock_engine_pool,
        )
        # Grid with 4 rows * 3 cols = 12 cells (>= 10 threshold)
        
        # Act
        should_parallel = processor._should_use_parallel_processing(mock_table_grid)
        
        # Assert
        assert should_parallel is True

    def test_should_use_parallel_processing_with_small_table(
        self, table_settings, mock_logger, mock_ocr_engine
    ):
        """Test _should_use_parallel_processing() with small table."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        small_grid = TableGrid(
            rows=[0, 100, 200],
            cols=[0, 150],
            confidence=0.85,
            num_rows=2,
            num_cols=1,
            method="morphology",
        )  # 2 rows * 1 col = 2 cells (< 10 threshold)
        
        # Act
        should_parallel = processor._should_use_parallel_processing(small_grid)
        
        # Assert
        assert should_parallel is False

    def test_should_use_parallel_processing_when_disabled(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_grid
    ):
        """Test _should_use_parallel_processing() when disabled in settings."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        
        # Act
        should_parallel = processor._should_use_parallel_processing(mock_table_grid)
        
        # Assert
        assert should_parallel is False


@pytest.mark.unit
class TestTableProcessorSequential:
    """Test sequential cell processing."""

    def test_extract_cells_sequential(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_grid, mock_table_image
    ):
        """Test _extract_cells_sequential() flow."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        
        # Act
        cells = processor._extract_cells_sequential(mock_table_image, mock_table_grid, None)
        
        # Assert
        assert isinstance(cells, list)
        assert len(cells) > 0
        assert all(isinstance(cell, TableCell) for cell in cells)
        mock_logger.info.assert_called()

    def test_extract_single_cell(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_image, column_mapping
    ):
        """Test _extract_single_cell() with valid cell."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        
        # Act
        cell = processor._extract_single_cell(
            mock_table_image,
            row_idx=0,
            col_idx=0,
            y_start=0,
            y_end=100,
            x_start=0,
            x_end=150,
            column_mapping=column_mapping,
        )
        
        # Assert
        assert cell is not None
        assert isinstance(cell, TableCell)
        assert cell.row_idx == 0
        assert cell.col_idx == 0
        assert cell.field_name == "row_number"

    def test_extract_single_cell_too_small(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, mock_table_image
    ):
        """Test _extract_single_cell() with cell too small."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        
        # Act - cell is only 10x10 pixels (below minimum)
        cell = processor._extract_single_cell(
            mock_table_image,
            row_idx=0,
            col_idx=0,
            y_start=0,
            y_end=10,
            x_start=0,
            x_end=10,
            column_mapping=None,
        )
        
        # Assert
        assert cell is None
        mock_logger.debug.assert_called()


@pytest.mark.unit
class TestTableProcessorParallel:
    """Test parallel cell processing (mocked)."""

    @patch("src.table_processor.ThreadPoolExecutor")
    def test_extract_cells_parallel(
        self,
        mock_executor_class,
        table_settings,
        mock_logger,
        mock_ocr_engine,
        mock_table_grid,
        mock_table_image,
    ):
        """Test _extract_cells_parallel() workflow."""
        # Arrange
        from src.parallel_ocr_worker import CellOCRResult
        
        # Mock ThreadPoolExecutor
        mock_executor = MagicMock()
        mock_future = MagicMock(spec=Future)
        mock_future.result.return_value = CellOCRResult(
            row_idx=0,
            col_idx=0,
            x_start=0,
            y_start=0,
            x_end=150,
            y_end=100,
            text="test_text",
            confidence=0.9,
            field_name=None,
        )
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor.submit.return_value = mock_future
        
        mock_executor_class.return_value = mock_executor
        
        # Mock as_completed and recognize_cell_worker
        with patch("src.table_processor.as_completed", return_value=[mock_future]):
            with patch("src.parallel_ocr_worker.recognize_cell_worker") as mock_worker:
                def mock_worker_func(task, params):
                    return CellOCRResult(
                        row_idx=task.row_idx,
                        col_idx=task.col_idx,
                        x_start=task.x_start,
                        y_start=task.y_start,
                        x_end=task.x_end,
                        y_end=task.y_end,
                        text="test_text",
                        confidence=0.9,
                        field_name=task.field_name,
                    )
                mock_worker.side_effect = mock_worker_func
                
                # Create mock engine pool for parallel processing
                mock_engine_pool = MagicMock()
                processor = TableProcessor(
                    settings=table_settings,
                    logger=mock_logger,
                    ocr_engine=mock_ocr_engine,
                    engine_pool=mock_engine_pool,
                )
                
                # Act
                cells = processor._extract_cells_parallel(mock_table_image, mock_table_grid, None)
                
                # Assert
                assert isinstance(cells, list)
                mock_logger.info.assert_called()

    @patch("src.table_processor.ThreadPoolExecutor")
    def test_extract_cells_parallel_fallback_on_error(
        self,
        mock_executor_class,
        table_settings,
        mock_logger,
        mock_ocr_engine,
        mock_table_grid,
        mock_table_image,
    ):
        """Test _extract_cells_parallel() falls back to sequential on error."""
        # Arrange
        mock_executor = MagicMock()
        mock_executor.__enter__.side_effect = Exception("Parallel processing failed")
        mock_executor_class.return_value = mock_executor
        
        # Create mock engine pool for parallel processing
        mock_engine_pool = MagicMock()
        processor = TableProcessor(
            settings=table_settings,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
            engine_pool=mock_engine_pool,
        )
        
        # Act
        cells = processor._extract_cells_parallel(mock_table_image, mock_table_grid, None)
        
        # Assert
        # Should fall back to sequential
        assert isinstance(cells, list)
        mock_logger.error.assert_called()


@pytest.mark.unit
class TestTableProcessorBoundaries:
    """Test boundary building methods."""

    def test_build_row_boundaries(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine
    ):
        """Test _build_row_boundaries() from horizontal lines."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        row_lines = [100, 200, 300]
        image_height = 400
        
        # Act
        boundaries = processor._build_row_boundaries(row_lines, image_height)
        
        # Assert
        assert boundaries[0] == 0
        assert boundaries[-1] == image_height
        assert len(boundaries) == len(row_lines) + 1

    def test_build_col_boundaries(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine
    ):
        """Test _build_col_boundaries() from vertical lines."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        col_lines = [150, 300, 450]
        image_width = 600
        
        # Act
        boundaries = processor._build_col_boundaries(col_lines, image_width)
        
        # Assert
        assert boundaries[0] == 0
        assert boundaries[-1] == image_width
        assert len(boundaries) == len(col_lines) + 1


@pytest.mark.unit
class TestTableProcessorCellOperations:
    """Test cell-level operations."""

    def test_preprocess_cell(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine
    ):
        """Test _preprocess_cell() enhancement."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        cell_image = np.ones((50, 100, 3), dtype=np.uint8) * 128
        
        # Act
        preprocessed = processor._preprocess_cell(cell_image)
        
        # Assert
        assert preprocessed is not None
        assert preprocessed.shape == cell_image.shape

    def test_ocr_cell(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine
    ):
        """Test _ocr_cell() with mocked OCR engine."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        cell_image = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        # Act
        text, confidence = processor._ocr_cell(cell_image)
        
        # Assert
        assert isinstance(text, str)
        assert 0.0 <= confidence <= 1.0
        mock_ocr_engine._resize_for_ocr.assert_called()
        mock_ocr_engine._run_paddle_ocr.assert_called()

    def test_ocr_cell_with_exception(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine
    ):
        """Test _ocr_cell() handles OCR exceptions."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        mock_ocr_engine._resize_for_ocr.side_effect = Exception("OCR failed")
        cell_image = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        # Act
        text, confidence = processor._ocr_cell(cell_image)
        
        # Assert
        assert text == ""
        assert confidence == 0.0
        mock_logger.warning.assert_called()


@pytest.mark.unit
class TestTableProcessorRowStructuring:
    """Test table row structuring."""

    def test_build_table_rows(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, column_mapping
    ):
        """Test build_table_rows() converts cells to row format."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        cells = [
            TableCell(
                row_idx=0,
                col_idx=0,
                x_start=0,
                y_start=0,
                x_end=150,
                y_end=100,
                text="row1",
                confidence=0.9,
                field_name="row_number",
            ),
            TableCell(
                row_idx=0,
                col_idx=1,
                x_start=150,
                y_start=0,
                x_end=300,
                y_end=100,
                text="param1",
                confidence=0.85,
                field_name="parameter",
            ),
            TableCell(
                row_idx=1,
                col_idx=0,
                x_start=0,
                y_start=100,
                x_end=150,
                y_end=200,
                text="row2",
                confidence=0.9,
                field_name="row_number",
            ),
        ]
        
        # Act
        rows = processor.build_table_rows(cells, column_mapping)
        
        # Assert
        assert isinstance(rows, list)
        assert len(rows) == 2  # Two rows
        assert rows[0]["row_index"] == 0
        assert rows[1]["row_index"] == 1
        assert "row_number" in rows[0]
        assert "parameter" in rows[0]

    def test_build_table_rows_with_row_grouping(
        self, table_settings_no_parallel, mock_logger, mock_ocr_engine, column_mapping
    ):
        """Test build_table_rows() groups cells by row_idx."""
        # Arrange
        processor = TableProcessor(
            settings=table_settings_no_parallel,
            logger=mock_logger,
            ocr_engine=mock_ocr_engine,
        )
        cells = [
            TableCell(0, 0, 0, 0, 150, 100, "cell00", 0.9, "row_number"),
            TableCell(0, 1, 150, 0, 300, 100, "cell01", 0.9, "parameter"),
            TableCell(1, 0, 0, 100, 150, 200, "cell10", 0.9, "row_number"),
        ]
        
        # Act
        rows = processor.build_table_rows(cells, column_mapping)
        
        # Assert
        assert len(rows) == 2
        assert len([k for k in rows[0].keys() if k != "row_index"]) == 2
        assert len([k for k in rows[1].keys() if k != "row_index"]) == 1
