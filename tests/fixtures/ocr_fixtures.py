"""OCR result fixtures for testing."""


def create_mock_ocr_result_by_region():
    """Create mock OCR results with regional structure.
    
    Returns:
        Dictionary with ocr_results_by_region structure matching expected output format
    """
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
                {
                    "text": "15.10.2025",
                    "confidence": 0.98,
                    "bbox": [[250, 50], [400, 50], [400, 80], [250, 80]],
                    "center_x": 325,
                    "center_y": 65
                },
                {
                    "text": "Иванов И.И.",
                    "confidence": 0.92,
                    "bbox": [[50, 100], [300, 100], [300, 130], [50, 130]],
                    "center_x": 175,
                    "center_y": 115
                },
            ],
            "defects": [
                {
                    "text": "ГЕОМЕТРИЯ",
                    "confidence": 0.95,
                    "bbox": [[50, 250], [300, 250], [300, 280], [50, 280]],
                    "center_x": 175,
                    "center_y": 265
                },
                {
                    "text": "ОТВЕРСТИЯ",
                    "confidence": 0.93,
                    "bbox": [[350, 250], [600, 250], [600, 280], [350, 280]],
                    "center_x": 475,
                    "center_y": 265
                },
                {
                    "text": "ПОВЕРХНОСТЬ",
                    "confidence": 0.91,
                    "bbox": [[50, 300], [400, 300], [400, 330], [50, 330]],
                    "center_x": 225,
                    "center_y": 315
                },
            ],
            "analysis": [
                {
                    "text": "Отклонения",
                    "confidence": 0.94,
                    "bbox": [[50, 650], [250, 650], [250, 680], [50, 680]],
                    "center_x": 150,
                    "center_y": 665
                },
                {
                    "text": "Использовать",
                    "confidence": 0.96,
                    "bbox": [[50, 750], [300, 750], [300, 780], [50, 780]],
                    "center_x": 175,
                    "center_y": 765
                },
            ]
        }
    }







