---
title: "Testing Vision and Strategy"
version: "1.0"
last_updated: "2025-01-26"
tags: ["testing", "pytest", "ocr", "image-processing", "best-practices"]
complexity: "medium"
llm_relevance: "high"
domain: "testing"
---

# Testing Vision and Strategy

## 🎯 Context for LLM

This document describes the testing strategy for the OCR image processing project, including approaches to unit, integration, and end-to-end testing, test structure, and best practices for all components of the image processing pipeline.

## 🤖 Instructions for LLM Agent

### When writing tests:
1. Follow the test file structure described in section 3
2. Use fixtures for test isolation and code reuse
3. Apply mocks for external dependencies (PaddleOCR, file system)
4. Write tests in AAA format (Arrange-Act-Assert)
5. Use parametrize to cover multiple cases with one test
6. Use actual test images from `images/test_images/` for integration tests

### Related documents:
- **[Technical Vision](vision.md)** - overall system architecture
- **[Task List](tasklist.md)** - development plan and progress

---

## 1. General Testing Strategy

### 1.1. Testing Pyramid

The project follows the classic testing pyramid adapted for image processing:

```
        ╱╲
       ╱  ╲        E2E Tests (5-10%)
      ╱────╲       - Full pipeline with real test images
     ╱      ╲      - Critical processing flows
    ╱────────╲ 
   ╱          ╲    Integration Tests (20-30%)
  ╱────────────╲   - Real PaddleOCR integration
 ╱              ╲  - Component interactions
╱────────────────╲ - Image processing pipelines
╲                ╱
 ╲──────────────╱  Unit Tests (60-70%)
  ╲            ╱   - Isolated function testing
   ╲──────────╱    - Business logic
    ╲        ╱     - Data validation
     ╲──────╱
```

**Target coverage metrics:**
- **Unit tests**: 80%+ code coverage
- **Integration tests**: 60%+ coverage of critical integrations
- **E2E tests**: 100% coverage of critical user scenarios

### 1.2. Testing Principles

#### FIRST Principles
- **F**ast - Tests execute quickly (unit < 100ms, integration < 2s)
- **I**solated - Tests are independent of each other
- **R**epeatable - Deterministic results on repeated runs
- **S**elf-validating - Test clearly shows pass/fail without manual inspection
- **T**imely - Tests written alongside code (TDD approach)

#### AAA Pattern (Arrange-Act-Assert)
All tests follow this structure:
```python
def test_preprocessor_applies_deskew():
    # Arrange - prepare data and environment
    image = cv2.imread("test_image.jpg")
    preprocessor = ImagePreprocessor(settings, logger)
    
    # Act - execute the action being tested
    result = preprocessor.process(image_path)
    
    # Assert - verify the result
    assert result.deskew_angle is not None
    assert abs(result.deskew_angle) < 45.0
```

### 1.3. Test Isolation

**Golden rule:** Each test must be completely isolated and not depend on other tests.

**Isolation rules:**
- ✅ Use fixtures for setup/teardown
- ✅ Create fresh test images for each test (or use fixtures)
- ✅ Mock external dependencies (PaddleOCR, file system)
- ✅ Clean up temporary files after each test
- ❌ Don't use shared state between tests
- ❌ Don't rely on test execution order

---

## 2. Framework and Tools

### 2.1. Testing Stack

| Tool | Purpose | Version |
|:---|:---|:---|
| **pytest** | Main testing framework | 7.4+ |
| **pytest-cov** | Code coverage measurement | 4.1+ |
| **pytest-mock** | Simplified mocking | 3.11+ |
| **opencv-python** | Image manipulation in tests | 4.8+ |
| **numpy** | Array operations for test images | 1.24+ |
| **freezegun** | Time freezing for log testing | 1.2+ |

**NOT NEEDED for this project:**
- ❌ **pytest-asyncio** - no async operations
- ❌ **pytest-postgresql** - no database
- ❌ **fakeredis** - no Redis
- ❌ **factory-boy** - no complex DB models
- ❌ **respx** - no HTTP requests
- ❌ **aiogram-tests** - no Telegram bot

### 2.2. Installing Dependencies

```bash
# Development dependencies
pip install pytest pytest-cov pytest-mock freezegun
```

### 2.3. pytest Configuration

**pytest.ini**
```ini
[pytest]
# Test discovery patterns
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

# Minimum coverage requirement
addopts = 
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=75
    --strict-markers
    --tb=short

# Test markers
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (with real OCR)
    e2e: End-to-end tests (full pipeline)
    slow: Slow tests (> 1 second)
    requires_ocr: Tests requiring PaddleOCR
    requires_gpu: Tests requiring GPU (optional)

# Test timeout
timeout = 60
```

---

## 3. Test Structure

### 3.1. File Organization

```
tests/
├── conftest.py                   # Global fixtures
├── pytest.ini                    # pytest configuration
├── __init__.py
│
├── fixtures/                     # Shared test fixtures
│   ├── __init__.py
│   ├── image_fixtures.py        # Test image loading
│   ├── ocr_fixtures.py          # Mock OCR results
│   └── config_fixtures.py       # Settings overrides
│
├── unit/                          # Unit tests (60-70%)
│   ├── conftest.py               # Fixtures for unit tests
│   ├── test_preprocessor.py     # Image preprocessing
│   ├── test_perspective_corrector.py
│   ├── test_region_detector.py   # Zone detection
│   ├── test_ocr_engine.py       # OCR logic (mocked)
│   ├── test_error_corrector.py  # Corrections dictionary
│   ├── test_field_validator.py  # Validation rules
│   ├── test_form_extractor.py   # Data extraction
│   ├── test_table_detector.py
│   ├── test_table_processor.py
│   ├── test_batch_processor.py
│   └── config/
│       ├── test_settings.py
│       ├── test_corrections.py
│       └── test_validation_rules.py
│
├── integration/                  # Integration tests (20-30%)
│   ├── conftest.py               # Fixtures for integration tests
│   ├── test_preprocessing_pipeline.py
│   ├── test_ocr_pipeline.py     # Real PaddleOCR
│   ├── test_correction_validation_flow.py
│   ├── test_extraction_flow.py
│   └── test_parallel_processing.py
│
└── e2e/                          # End-to-end tests (5-10%)
    ├── conftest.py               # Fixtures for e2e tests
    ├── test_full_pipeline.py    # End-to-end with test images
    ├── test_batch_processing.py
    └── test_error_scenarios.py
```

### 3.2. Naming Conventions

**Test files:**
- Prefix: `test_` or suffix `_test.py`
- Example: `test_preprocessor.py`, `ocr_engine_test.py`

**Test functions:**
```python
# ✅ Good - descriptive name with context
def test_preprocessor_applies_deskew_for_rotated_image():
    pass

def test_region_detector_falls_back_to_template_when_adaptive_fails():
    pass

def test_error_corrector_applies_dictionary_corrections():
    pass

# ❌ Bad - unclear name
def test_process():
    pass

def test_detection():
    pass
```

**Naming pattern:** `test_[component]_[action]_[expected_result]`

---

## 4. Component-Specific Testing

### 4.1. Image Preprocessing Testing

#### 4.1.1. Preprocessor Testing

**Strategy:**
- Test perspective correction with known angles
- Test deskew with synthetic rotated images
- Test adaptive scaling for different resolutions
- Test enhancement (CLAHE, denoising, binarization)
- Mock image loading failures

**Example test:**
```python
import pytest
import cv2
import numpy as np
from pathlib import Path

from preprocessor import ImagePreprocessor
from config.settings import Settings

@pytest.fixture
def test_settings():
    """Settings optimized for testing."""
    return Settings(
        enable_perspective_correction=True,
        enable_deskew=True,
        enable_denoising=False,  # Faster for tests
        ocr_use_gpu=False
    )

@pytest.fixture
def synthetic_skewed_image():
    """Generate synthetic skewed document for testing."""
    # Create white image with text-like lines
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    # Add horizontal lines (simulating text)
    cv2.line(image, (50, 200), (750, 250), (0, 0, 0), 2)
    cv2.line(image, (50, 400), (750, 450), (0, 0, 0), 2)
    # Rotate to create skew
    center = (400, 500)
    rotation_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    skewed = cv2.warpAffine(image, rotation_matrix, (800, 1000))
    return skewed

@pytest.mark.unit
def test_preprocessor_detects_and_corrects_skew(test_settings, synthetic_skewed_image, tmp_path):
    """Test preprocessor detects and corrects image skew."""
    # Arrange
    logger = logging.getLogger("test")
    preprocessor = ImagePreprocessor(settings=test_settings, logger=logger)
    input_path = tmp_path / "skewed.jpg"
    cv2.imwrite(str(input_path), synthetic_skewed_image)
    
    # Act
    result = preprocessor.process(input_path=input_path)
    
    # Assert
    assert result.deskew_angle is not None
    assert abs(result.deskew_angle) < 10.0  # Should detect small skew
    assert result.output_path.exists()
```

**Testing checklist:**
- [ ] Perspective correction with various angles
- [ ] Deskew detection and correction
- [ ] Adaptive scaling for low/high resolution images
- [ ] Image enhancement (CLAHE, denoising)
- [ ] Binarization modes (Otsu, adaptive)
- [ ] Error handling for invalid images
- [ ] File path generation and output

#### 4.1.2. Perspective Corrector Testing

**Strategy:**
- Test document contour detection
- Test corner detection and ordering
- Test perspective transformation
- Test with images without clear document boundaries

**Example test:**
```python
@pytest.mark.unit
def test_perspective_corrector_detects_document_contour():
    """Test perspective corrector detects document boundaries."""
    # Arrange
    # Create image with clear rectangular document
    image = create_test_document_image()
    corrector = PerspectiveCorrector(settings, logger)
    
    # Act
    corrected, was_corrected = corrector.correct(image)
    
    # Assert
    assert was_corrected is True
    assert corrected.shape == image.shape  # Same dimensions
```

### 4.2. Region Detection Testing

#### 4.2.1. RegionDetector Testing

**Strategy:**
- Test adaptive line detection strategy
- Test text-based projection strategy
- Test template-based fallback
- Test with different form layouts
- Mock template loading

**Example test:**
```python
@pytest.mark.unit
def test_region_detector_uses_adaptive_strategy_first():
    """Test RegionDetector tries adaptive strategy before fallback."""
    # Arrange
    image = load_test_image("034_compr.jpg")
    detector = RegionDetector(settings, logger)
    
    # Act
    regions = detector.detect_zones(image, strategy="adaptive")
    
    # Assert
    assert len(regions) >= 3  # header, defects, analysis
    assert all(r.detection_method == "adaptive" for r in regions)
```

**Testing checklist:**
- [ ] Adaptive line detection finds horizontal separators
- [ ] Text-based projection detects regions without borders
- [ ] Template fallback works when adaptive fails
- [ ] Region coordinates are normalized (0.0-1.0)
- [ ] Region confidence scores are calculated
- [ ] Error handling for empty/invalid images

### 4.3. OCR Engine Testing

#### 4.3.1. Unit Tests (Mocked OCR)

**Strategy:**
- Mock PaddleOCR to avoid slow initialization
- Test result parsing logic
- Test confidence filtering
- Test region-based processing coordination
- Test parallel processing decision logic

**Example test:**
```python
from unittest.mock import MagicMock, patch

@pytest.mark.unit
@patch('ocr_engine.PaddleOCR')
def test_ocr_engine_filters_low_confidence_results(mock_paddleocr):
    """Test OCR engine filters results below confidence threshold."""
    # Arrange
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], ('Text1', 0.95)),
            ([[0, 60], [100, 60], [100, 110], [0, 110]], ('Text2', 0.3)),  # Low confidence
        ]
    ]
    mock_paddleocr.return_value = mock_ocr
    
    engine = OCREngine(settings, logger)
    
    # Act
    result = engine.process(image_path)
    
    # Assert
    assert result.total_texts_found == 1  # Only high confidence
    assert result.low_confidence_count == 1
```

#### 4.3.2. Integration Tests (Real OCR)

**Strategy:**
- Test with actual test images from `images/test_images/`
- Measure accuracy and performance
- Test memory management
- Test parallel vs sequential processing

**Example test:**
```python
@pytest.mark.integration
@pytest.mark.requires_ocr
@pytest.mark.slow
def test_ocr_engine_processes_test_image_034():
    """Test OCR engine processes actual test image."""
    # Arrange
    image_path = Path("images/test_images/034_compr.jpg")
    engine = OCREngine(settings, logger)
    
    # Act
    result = engine.process(image_path)
    
    # Assert
    assert result.total_texts_found > 0
    assert result.average_confidence > 0.5
    assert result.output_path.exists()
```

**Testing checklist:**
- [ ] OCR result parsing (standard PaddleOCR format)
- [ ] Confidence threshold filtering
- [ ] Region-based processing
- [ ] Memory cleanup after processing
- [ ] Parallel processing for large tables
- [ ] Error handling for corrupted images
- [ ] Context manager (resource cleanup)

### 4.4. Error Correction Testing

#### 4.4.1. ErrorCorrector Testing

**Strategy:**
- Test correction dictionary application
- Test correction logging
- Test case variations
- Test multiple corrections per text

**Example test:**
```python
@pytest.mark.unit
def test_error_corrector_applies_dictionary_corrections():
    """Test ErrorCorrector applies corrections from dictionary."""
    # Arrange
    ocr_data = {
        "text_regions": [
            {"text": "Homep", "confidence": 0.9},  # Should become "Номер"
            {"text": "PeB", "confidence": 0.85},   # Should become "Рев"
        ]
    }
    corrector = ErrorCorrector(settings, logger)
    
    # Act
    result = corrector.process(ocr_json_path)
    
    # Assert
    assert result.corrections_applied == 2
    # Verify corrections in output JSON
```

**Testing checklist:**
- [ ] Exact match corrections
- [ ] Case-insensitive fuzzy corrections
- [ ] Correction logging and metadata
- [ ] Multiple corrections in single text
- [ ] No false corrections (correct text unchanged)

### 4.5. Field Validation Testing

#### 4.5.1. FieldValidator Testing

**Strategy:**
- Test validation rules for each field type
- Test date format validation
- Test numeric field validation
- Test mandatory field checks

**Example test:**
```python
@pytest.mark.parametrize("act_number,expected_valid", [
    ("001/2025", True),
    ("123/24", True),
    ("abc/2025", False),
    ("001-2025", False),
    ("", False),  # Empty
])
@pytest.mark.unit
def test_field_validator_validates_act_number_format(act_number, expected_valid):
    """Test FieldValidator validates act number format."""
    # Arrange
    validator = FieldValidator(settings, logger)
    field_data = {"act_number": {"value": act_number}}
    
    # Act
    result = validator.validate_field("act_number", field_data["act_number"])
    
    # Assert
    assert result.is_valid == expected_valid
```

**Testing checklist:**
- [ ] Act number format (XXX/YY)
- [ ] Date format (DD.MM.YYYY)
- [ ] Quantity validation (positive integers)
- [ ] Measurement validation (decimals)
- [ ] Status validation (allowed values)
- [ ] Mandatory field detection
- [ ] Confidence-based suspicious flagging

### 4.6. Form Extraction Testing

#### 4.6.1. FormExtractor Testing

**Strategy:**
- Test header extraction with various layouts
- Test sticker detection (priority source)
- Test defect block parsing
- Test analysis section extraction
- Test table data vs flat OCR data handling
- Test suspicious value flagging

**Example test:**
```python
@pytest.mark.unit
def test_form_extractor_detects_sticker_as_priority_source():
    """Test FormExtractor prioritizes sticker data over handwritten."""
    # Arrange
    ocr_data = {
        "ocr_results_by_region": {
            "header": [
                {"text": "11962", "confidence": 0.95, "center_x": 200, "center_y": 150},  # Sticker
                {"text": "12345", "confidence": 0.7, "center_x": 300, "center_y": 200},   # Handwritten
            ]
        }
    }
    extractor = FormExtractor(settings, logger)
    
    # Act
    result = extractor.extract(ocr_json_path)
    
    # Assert
    assert result.header_data.sticker_data.part_line_number.value == "11962"
    # Handwritten value should not override sticker
```

**Testing checklist:**
- [ ] Header field extraction (act number, date, inspector)
- [ ] Sticker detection and priority
- [ ] Defect block classification (geometry/holes/surface)
- [ ] Defect row grouping by Y-coordinate
- [ ] Analysis section parsing
- [ ] Table data handling (when table detection enabled)
- [ ] Mandatory field validation
- [ ] Suspicious value flagging (low confidence)

### 4.7. Table Processing Testing

#### 4.7.1. TableDetector and TableProcessor Testing

**Strategy:**
- Test table grid detection
- Test cell extraction
- Test column mapping
- Test parallel cell processing
- Test fallback to flat OCR

**Example test:**
```python
@pytest.mark.unit
def test_table_processor_extracts_cells_from_grid():
    """Test TableProcessor extracts individual cells from detected grid."""
    # Arrange
    image = load_test_image("034_compr.jpg")
    grid = TableGrid(rows=5, cols=4, ...)  # Detected grid
    processor = TableProcessor(settings, logger, ocr_engine)
    
    # Act
    cells = processor.extract_cells(image, grid)
    
    # Assert
    assert len(cells) == 20  # 5 rows × 4 cols
    assert all(cell.text is not None for cell in cells)
```

**Testing checklist:**
- [ ] Table grid detection (horizontal/vertical lines)
- [ ] Cell extraction with coordinates
- [ ] Column mapping from templates
- [ ] Parallel cell processing (for large tables)
- [ ] Fallback to flat OCR when detection fails
- [ ] Cell-level preprocessing

### 4.8. Batch Processing Testing

#### 4.8.1. BatchProcessor Testing

**Strategy:**
- Test shared OCR engine pattern
- Test memory cleanup between files
- Test error isolation (graceful degradation)
- Test summary generation

**Example test:**
```python
@pytest.mark.integration
@pytest.mark.slow
def test_batch_processor_reuses_ocr_engine():
    """Test BatchProcessor reuses OCR engine for performance."""
    # Arrange
    input_dir = Path("images/batch1")
    processor = BatchProcessor(settings, logger)
    
    # Act
    result = processor.process_directory(input_dir, mode="pipeline")
    
    # Assert
    assert result.successful_files == result.total_files
    assert result.total_duration_seconds < (result.total_files * 10)  # Faster than individual
```

**Testing checklist:**
- [ ] Shared OCR engine initialization
- [ ] Memory cleanup between files
- [ ] Error isolation (one file failure doesn't stop batch)
- [ ] Summary JSON generation
- [ ] Progress logging
- [ ] Graceful degradation on errors

---

## 5. Fixtures and Test Data Management

### 5.1. Global Fixtures (conftest.py)

**Main `tests/conftest.py`:**
```python
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from config.settings import Settings

@pytest.fixture
def test_settings():
    """Settings optimized for fast testing."""
    return Settings(
        enable_parallel_processing=False,
        ocr_use_gpu=False,
        enable_denoising=False,
        ocr_confidence_threshold=0.5,
        log_level="WARNING"  # Reduce log noise
    )

@pytest.fixture
def test_image_034():
    """Load actual test image 034_compr.jpg."""
    path = Path("images/test_images/034_compr.jpg")
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return cv2.imread(str(path))

@pytest.fixture
def test_image_034_full():
    """Load full resolution test image 034.jpg."""
    path = Path("images/test_images/034.jpg")
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return cv2.imread(str(path))

@pytest.fixture
def synthetic_skewed_image():
    """Generate synthetic skewed document for testing."""
    image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    # Add horizontal lines
    cv2.line(image, (50, 200), (750, 250), (0, 0, 0), 2)
    cv2.line(image, (50, 400), (750, 450), (0, 0, 0), 2)
    # Rotate to create skew
    center = (400, 500)
    rotation_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (800, 1000))

@pytest.fixture
def mock_ocr_response():
    """Mock PaddleOCR response structure."""
    return [
        [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], ('ГЕОМЕТРИЯ', 0.95)),
            ([[0, 60], [100, 60], [100, 110], [0, 110]], ('ОТВЕРСТИЯ', 0.92)),
        ]
    ]

@pytest.fixture
def mock_ocr_engine(mock_ocr_response):
    """Mock OCR engine for unit tests."""
    mock_engine = MagicMock()
    mock_engine.ocr.return_value = mock_ocr_response
    return mock_engine
```

### 5.2. Image Fixtures

**`tests/fixtures/image_fixtures.py`:**
```python
import cv2
import numpy as np
from pathlib import Path

def create_test_document_image(width=800, height=1000):
    """Create synthetic document image for testing."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Add header section
    cv2.rectangle(image, (50, 50), (750, 200), (0, 0, 0), 2)
    # Add defects section
    cv2.rectangle(image, (50, 250), (750, 600), (0, 0, 0), 2)
    # Add analysis section
    cv2.rectangle(image, (50, 650), (750, 950), (0, 0, 0), 2)
    return image

def create_rotated_image(image, angle):
    """Rotate image by specified angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))
```

### 5.3. OCR Result Fixtures

**`tests/fixtures/ocr_fixtures.py`:**
```python
def create_mock_ocr_result_by_region():
    """Create mock OCR results with regional structure."""
    return {
        "ocr_results_by_region": {
            "header": [
                {
                    "text": "034/25",
                    "confidence": 0.995,
                    "bbox": [[100, 50], [200, 50], [200, 80], [100, 80]],
                    "center_x": 150,
                    "center_y": 65
                },
                # ... more header texts
            ],
            "defects": [
                {
                    "text": "ГЕОМЕТРИЯ",
                    "confidence": 0.95,
                    "bbox": [[50, 250], [300, 250], [300, 280], [50, 280]],
                    "center_x": 175,
                    "center_y": 265
                },
                # ... more defect texts
            ],
            "analysis": [
                # ... analysis texts
            ]
        }
    }
```

### 5.4. Parametrize for Multiple Cases

**Example:**
```python
@pytest.mark.parametrize(
    "resolution,expected_scaling",
    [
        ((1920, 1080), True),   # 2.1 MP - should scale
        ((2560, 1440), True),   # 3.7 MP - should scale
        ((3264, 2448), False), # 8.0 MP - no scaling
        ((4000, 3000), False),  # 12 MP - no scaling
    ]
)
@pytest.mark.unit
def test_preprocessor_adaptive_scaling(resolution, expected_scaling):
    """Test adaptive scaling based on image resolution."""
    # Arrange
    width, height = resolution
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    preprocessor = ImagePreprocessor(settings, logger)
    
    # Act
    result = preprocessor._scale_image(image)
    
    # Assert
    if expected_scaling:
        assert result.shape != image.shape
    else:
        assert result.shape == image.shape
```

---

## 6. CI/CD Integration

### 6.1. GitHub Actions Workflow

**.github/workflows/tests.yml:**
```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
    
    - name: Install Python dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock freezegun
    
    - name: Run unit tests
      run: |
        pytest tests/unit -v --cov=src --cov-report=xml --cov-report=term
    
    - name: Run integration tests
      run: |
        pytest tests/integration -v -m "not slow"
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        fail_ci_if_error: true
```

### 6.2. Pre-commit Hooks

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: Run unit tests
        entry: pytest tests/unit -v
        language: system
        pass_filenames: false
        always_run: true
      
      - id: pytest-check-coverage
        name: Check test coverage
        entry: pytest tests/unit --cov=src --cov-fail-under=75
        language: system
        pass_filenames: false
        always_run: true
```

### 6.3. Testing at Different Stages

**Development (local):**
```bash
# Fast unit tests only
pytest tests/unit -v -m "not slow"

# All tests except E2E
pytest tests/unit tests/integration -v

# Specific test file
pytest tests/unit/test_preprocessor.py -v

# Tests by marker
pytest -m "unit and not slow" -v
```

**CI/CD (automated):**
```bash
# Stage 1: Fast feedback (< 1 min)
pytest tests/unit -m "not slow" -v

# Stage 2: Full unit + integration (< 5 min)
pytest tests/unit tests/integration -v

# Stage 3: E2E tests (only on main branch)
pytest tests/e2e -v
```

**Pre-deployment (staging):**
```bash
# Full test suite with coverage
pytest tests/ -v --cov=src --cov-report=html

# Smoke tests on staging
pytest tests/e2e -m "smoke" -v
```

---

## 7. Best Practices and Anti-patterns

### 7.1. DOs (Do This)

✅ **Write tests alongside code (TDD)**
```python
# 1. Write test first
def test_calculate_confidence_average():
    confidences = [0.9, 0.8, 0.7]
    assert calculate_average_confidence(confidences) == 0.8

# 2. Then implement
def calculate_average_confidence(confidences: list[float]) -> float:
    return sum(confidences) / len(confidences)
```

✅ **Use descriptive test names**
```python
# Good
def test_preprocessor_applies_deskew_for_5_degree_rotation():
    pass

# Bad
def test_deskew():
    pass
```

✅ **Isolate tests with fixtures**
```python
@pytest.fixture
def clean_test_image(tmp_path):
    """Ensure clean test image for each test."""
    image_path = tmp_path / "test.jpg"
    cv2.imwrite(str(image_path), test_image)
    yield image_path
    # Cleanup handled by tmp_path
```

✅ **Mock external dependencies (PaddleOCR)**
```python
@patch('ocr_engine.PaddleOCR')
def test_ocr_processing(mock_paddleocr):
    mock_paddleocr.return_value.ocr.return_value = mock_result
    # Test logic here
```

✅ **Test edge cases**
```python
@pytest.mark.parametrize("confidence,expected", [
    (0.0, False),      # minimum
    (1.0, True),      # maximum
    (0.49, False),    # below threshold
    (0.51, True),     # above threshold
])
def test_confidence_threshold_filtering(confidence, expected):
    result = filter_by_confidence(confidence, threshold=0.5)
    assert result == expected
```

### 7.2. DON'Ts (Don't Do This)

❌ **Don't use sleep in tests**
```python
# Bad
def test_async_operation():
    start_operation()
    time.sleep(5)  # ❌ Fragile test
    assert operation_complete()

# Good - use proper synchronization or mocks
def test_async_operation():
    result = await async_operation()
    assert result.status == "complete"
```

❌ **Don't test implementation details**
```python
# Bad - tests internal implementation
def test_internal_cache_structure():
    obj = MyClass()
    assert obj._cache == {}  # ❌ Fragile to refactoring

# Good - tests behavior
def test_caching_improves_performance():
    obj = MyClass()
    first_call = obj.expensive_operation()
    second_call = obj.expensive_operation()  # Should be faster
    assert first_call == second_call
```

❌ **Don't create dependencies between tests**
```python
# Bad
def test_step_1():
    global shared_data
    shared_data = {"key": "value"}  # ❌ Shared state

def test_step_2():
    assert shared_data["key"] == "value"  # ❌ Depends on test_step_1

# Good - each test independent
@pytest.fixture
def test_data():
    return {"key": "value"}

def test_step_1(test_data):
    assert test_data["key"] == "value"

def test_step_2(test_data):
    assert test_data["key"] == "value"
```

❌ **Don't test PaddleOCR accuracy itself**
```python
# Bad - testing external library
def test_paddleocr_recognizes_text():
    ocr = PaddleOCR()
    result = ocr.ocr("test_image.jpg")
    assert result[0][0][1][0] == "expected_text"  # ❌ Tests PaddleOCR, not our code

# Good - test our logic
def test_ocr_engine_filters_low_confidence():
    # Mock PaddleOCR, test our filtering logic
    pass
```

❌ **Don't create huge test images**
```python
# Bad - large files slow down tests
def test_with_10mb_image():
    image = load_10mb_image()  # ❌ Too large

# Good - use reasonable sizes or synthetic images
def test_with_synthetic_image():
    image = create_test_image(800, 1000)  # ✅ Small and fast
```

### 7.3. Code Review Checklist

When reviewing tests, check:

- [ ] **Coverage**: New code is covered by tests (75%+ target)
- [ ] **Isolation**: Tests are independent and don't affect each other
- [ ] **Naming**: Test names describe what is being tested
- [ ] **AAA Pattern**: Arrange-Act-Assert is clearly separated
- [ ] **Mocking**: External dependencies (PaddleOCR) are mocked in unit tests
- [ ] **Edge cases**: Boundary conditions are tested
- [ ] **Performance**: Unit tests are fast (< 100ms)
- [ ] **Documentation**: Complex tests have docstrings

---

## 8. Debugging and Troubleshooting

### 8.1. Running Tests in Debug Mode

```bash
# Verbose output with full traceback
pytest tests/ -vv --tb=long

# Stop on first failure
pytest tests/ -x

# Run specific test in debug
pytest tests/unit/test_preprocessor.py::test_deskew -vv --pdb

# Print statements in output
pytest tests/ -s
```

### 8.2. Debugging Failed Tests

**Using pytest.set_trace():**
```python
def test_complex_logic():
    # Arrange
    data = prepare_test_data()
    
    # Act
    result = complex_function(data)
    
    # Debug point
    pytest.set_trace()  # Starts pdb debugger
    
    # Assert
    assert result.status == "success"
```

**Using --pdb flag:**
```bash
# Automatically starts debugger on failures
pytest tests/ --pdb
```

### 8.3. Common Issues and Solutions

**Issue: Tests work locally but fail in CI**
```python
# Cause: Path dependencies
# Bad:
def test_load_image():
    image = cv2.imread("images/test.jpg")  # ❌ Hard-coded path

# Good:
def test_load_image(tmp_path):
    image_path = tmp_path / "test.jpg"
    cv2.imwrite(str(image_path), test_image)
    image = cv2.imread(str(image_path))  # ✅ Uses fixture
```

**Issue: Flaky tests (unstable)**
```python
# Cause: Non-deterministic OCR results
# Bad:
def test_ocr_accuracy():
    result = ocr_engine.process(image)
    assert result.text == "exact_text"  # ❌ May vary

# Good:
def test_ocr_confidence():
    result = ocr_engine.process(image)
    assert result.confidence > 0.8  # ✅ Tests confidence, not exact text
```

**Issue: Slow tests**
```python
# Cause: Real OCR calls in unit tests
# Bad:
def test_ocr_processing():
    ocr = PaddleOCR()  # ❌ Slow initialization
    result = ocr.ocr(image)

# Good:
@patch('ocr_engine.PaddleOCR')
def test_ocr_processing(mock_paddleocr):
    mock_paddleocr.return_value.ocr.return_value = mock_result  # ✅ Fast mock
```

---

## 9. Metrics and Reporting

### 9.1. Coverage Reports

**Generate HTML report:**
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Terminal report:**
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

**XML for CI integration:**
```bash
pytest tests/ --cov=src --cov-report=xml
```

### 9.2. Test Execution Metrics

**Slowest tests report:**
```bash
pytest tests/ --durations=10  # Top 10 slowest tests
```

**Test results summary:**
```bash
pytest tests/ -v --tb=short --junit-xml=test-results.xml
```

### 9.3. Target Quality Metrics

| Metric | Target | Criticality |
|:---|:---:|:---:|
| **Code Coverage** | 75%+ | 🔴 High |
| **Unit Test Pass Rate** | 100% | 🔴 High |
| **Integration Test Pass Rate** | 95%+ | 🟡 Medium |
| **E2E Test Pass Rate** | 90%+ | 🟡 Medium |
| **Unit Test Execution Time** | < 2 min | 🟢 Low |
| **Full Test Suite Time** | < 10 min | 🟡 Medium |
| **Test Flakiness Rate** | < 1% | 🔴 High |

**Priority components for high coverage:**
- `error_corrector.py` - 90%+ (business logic)
- `field_validator.py` - 90%+ (validation rules)
- `form_extractor.py` - 85%+ (complex extraction)
- `region_detector.py` - 80%+ (multiple strategies)
- `preprocessor.py` - 80%+ (image processing)

---

## 10. Key Differences from testing_rules.md

### 10.1. NOT APPLICABLE to This Project

**Removed from testing_rules.md:**
- ❌ Telegram bot testing (no bot in this project)
- ❌ Database testing (no PostgreSQL, Redis, Qdrant)
- ❌ LLM/RAG testing (no AI providers)
- ❌ Session management testing (stateless system)
- ❌ Middleware testing (no web framework)
- ❌ WebSocket testing (no real-time features)

### 10.2. APPLICABLE and ADAPTED

**Kept and adapted from testing_rules.md:**
- ✅ Testing pyramid (adjusted ratios for image processing)
- ✅ AAA pattern (Arrange-Act-Assert)
- ✅ Test isolation and fixtures
- ✅ Mocking external dependencies (PaddleOCR instead of APIs)
- ✅ Performance testing (batch processing optimization)
- ✅ Error handling testing
- ✅ Configuration testing

### 10.3. NEW for OCR Project

**Added specifically for image processing:**
- ✅ Image fixture management
- ✅ Synthetic image generation for testing
- ✅ OCR result mocking strategies
- ✅ Region detection testing
- ✅ Table processing testing
- ✅ Memory management testing (batch processing)

---

## 11. Implementation Examples

### 11.1. Testing Perspective Correction

```python
@pytest.mark.unit
def test_perspective_corrector_with_known_angle():
    """Test perspective correction with known transformation angle."""
    # Arrange
    # Create image with known perspective distortion
    original = create_rectangular_document()
    distorted = apply_perspective_transform(original, angle=15.0)
    corrector = PerspectiveCorrector(settings, logger)
    
    # Act
    corrected, was_corrected = corrector.correct(distorted)
    
    # Assert
    assert was_corrected is True
    # Verify corners are more rectangular after correction
    corners = detect_corners(corrected)
    assert is_rectangular(corners, tolerance=5.0)
```

### 11.2. Testing Error Correction Dictionary

```python
@pytest.mark.parametrize("input_text,expected_output", [
    ("Homep", "Номер"),
    ("PeB", "Рев"),
    ("Wmyyep", "Изделие"),
    ("CorrectText", "CorrectText"),  # No correction needed
])
@pytest.mark.unit
def test_error_corrector_applies_dictionary(input_text, expected_output):
    """Test error corrector applies corrections from dictionary."""
    # Arrange
    from config.corrections import get_correction
    
    # Act
    corrected, was_corrected = get_correction(input_text)
    
    # Assert
    assert corrected == expected_output
    if input_text != expected_output:
        assert was_corrected is True
```

### 11.3. Testing Validation Rules with Parametrize

```python
@pytest.mark.parametrize("date_str,expected_valid", [
    ("15.10.2025", True),
    ("1.1.2025", True),
    ("15/10/2025", False),  # Wrong separator
    ("2025-10-15", False),  # Wrong format
    ("32.10.2025", False), # Invalid day
])
@pytest.mark.unit
def test_field_validator_validates_date_format(date_str, expected_valid):
    """Test FieldValidator validates date format."""
    # Arrange
    from config.validation_rules import get_rule
    rule = get_rule("date")
    
    # Act
    is_valid, error_msg = rule.validate(date_str)
    
    # Assert
    assert is_valid == expected_valid
```

### 11.4. Testing Form Extraction with Mock OCR Data

```python
@pytest.mark.unit
def test_form_extractor_extracts_header_fields():
    """Test FormExtractor extracts header fields from OCR results."""
    # Arrange
    ocr_data = {
        "ocr_results_by_region": {
            "header": [
                {"text": "034/25", "confidence": 0.995, "center_x": 150, "center_y": 65},
                {"text": "05.11.2025", "confidence": 0.998, "center_x": 200, "center_y": 65},
                {"text": "Денисова Л.В", "confidence": 0.72, "center_x": 300, "center_y": 100},
            ]
        }
    }
    extractor = FormExtractor(settings, logger)
    
    # Act
    result = extractor.extract(ocr_json_path)
    
    # Assert
    assert result.header_data.act_number.value == "034/25"
    assert result.header_data.act_date.value == "05.11.2025"
    assert result.header_data.inspector_name.value == "Денисова Л.В"
    assert result.header_data.inspector_name.suspicious is True  # Low confidence
```

### 11.5. Testing Batch Processing with Memory Cleanup

```python
@pytest.mark.integration
@pytest.mark.slow
def test_batch_processor_cleans_memory_between_files():
    """Test BatchProcessor cleans memory between files."""
    # Arrange
    import psutil
    import os
    process = psutil.Process(os.getpid())
    input_dir = Path("images/batch1")
    processor = BatchProcessor(settings, logger)
    
    # Act
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    result = processor.process_directory(input_dir, mode="pipeline")
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Assert
    assert result.successful_files > 0
    # Memory should not grow excessively (allow 200MB overhead)
    memory_growth = final_memory - initial_memory
    assert memory_growth < 200  # MB
```

---

## 12. Development Workflow

### 12.1. Before Creating Tests

- ✅ Review `vision.md` for architecture
- ✅ Understand component responsibilities
- ✅ Identify testable behaviors (not implementation details)
- ✅ Plan test data needs (images, mock OCR results)

### 12.2. During Test Creation

- Write test first (TDD when possible)
- Use descriptive test names
- Follow AAA pattern
- Mock external dependencies (PaddleOCR)
- Test edge cases and errors
- Use fixtures for test data

### 12.3. After Test Creation

- Run full test suite
- Check coverage report
- Review for flaky tests
- Update documentation if needed
- Ensure tests are fast (< 100ms for unit tests)

---

## 13. Related Documents

- **[Technical Vision](vision.md)** - overall system architecture
- **[Task List](tasklist.md)** - development plan and progress
- **[Testing Rules (Source)](../todo/testing_rules.md)** - original testing principles (adapted)

---

## 14. Change History

### Version 1.0 (2025-01-26)
**Document creation:**
- Defined general testing strategy (Testing Pyramid)
- Described framework and tools (pytest ecosystem)
- Defined test structure (unit/integration/e2e)
- Added component-specific testing approaches
- Defined best practices and anti-patterns
- Added CI/CD integration
- Defined target quality metrics
- Adapted principles from testing_rules.md for OCR project

*Last updated: 2025-01-26*
*Documentation version: 1.0*


