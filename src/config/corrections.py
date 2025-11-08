"""OCR error correction dictionary for Russian QC form text recognition.

This module contains common OCR errors detected by PaddleOCR when processing
Russian text on quality control forms. The corrections are applied during
post-processing to improve recognition accuracy.
"""

# Static dictionary mapping common OCR errors to correct text
OCR_CORRECTIONS = {
    # Header and metadata field labels
    "Homep": "Номер",
    "Номео": "Номер",
    "НомеD": "Номер",
    "PeB": "Рев",
    "Pев": "Рев",
    "РeB": "Рев",
    "Wmyyep": "Изделие",
    "Изделuе": "Изделие",
    "Изделиe": "Изделие",
    "Дaта": "Дата",
    "Датa": "Дата",
    
    # Status and decision terms
    "ne ugim": "не соответствует",
    "не соотвтствует": "не соответствует",
    "не cоответствует": "не соответствует",
    "друие несоотвтствуя": "другие несоответствия",
    "другие нeсоответствия": "другие несоответствия",
    "годн": "годен",
    "годeн": "годен",
    "бракк": "брак",
    "браk": "брак",
    "пригоден": "годен",
    "непригоден": "брак",
    
    # Technical terms and parameters
    "ГЕОМЕТРИЯ": "ГЕОМЕТРИЯ",  # Reference spelling
    "ГЕОМЕТРUЯ": "ГЕОМЕТРИЯ",
    "ГЕОМЕТРИR": "ГЕОМЕТРИЯ",
    "ОТВЕРСТИЯ": "ОТВЕРСТИЯ",  # Reference spelling
    "ОТВЕРСТUЯ": "ОТВЕРСТИЯ",
    "ОТВЕРСТИR": "ОТВЕРСТИЯ",
    "РЕЗЬБА": "РЕЗЬБА",  # Reference spelling
    "РЕЗЬБA": "РЕЗЬБА",
    "РEЗЬБА": "РЕЗЬБА",
    
    # Table headers and common field names
    "Кол-во": "Кол-во",
    "Кол-вo": "Кол-во",
    "Коп-во": "Кол-во",
    "Решение": "Решение",
    "Решенuе": "Решение",
    "Решениe": "Решение",
    "Описание": "Описание",
    "Описанuе": "Описание",
    "Описаниe": "Описание",
    "Параметр": "Параметр",
    "Параметp": "Параметр",
    "Парамeтр": "Параметр",
    "Норма": "Норма",
    "Нормa": "Норма",
    "Фактически": "Фактически",
    "Фактичeски": "Фактически",
    
    # Inspector and signature fields
    "Контролер": "Контролер",
    "Контpолер": "Контролер",
    "Контролеp": "Контролер",
    "Подпись": "Подпись",
    "Подпuсь": "Подпись",
    "Подписъ": "Подпись",
    
    # Actions and operations
    "Устранить": "Устранить",
    "Устpанить": "Устранить",
    "Устранитъ": "Устранить",
    "Доработать": "Доработать",
    "Доработатъ": "Доработать",
    "Доpаботать": "Доработать",
    "Утилизировать": "Утилизировать",
    "Утилизиpовать": "Утилизировать",
    
    # Common measurement units
    "мм": "мм",
    "MM": "мм",
    "шт": "шт",
    "ШТ": "шт",
    "шт.": "шт",
}

# Extended corrections for partial matches (case-insensitive)
# These are applied when exact match is not found
FUZZY_CORRECTIONS = {
    "номер": "Номер",
    "дата": "Дата",
    "рев": "Рев",
    "изделие": "Изделие",
    "годен": "годен",
    "брак": "брак",
}


def get_correction(text: str) -> tuple[str, bool]:
    """Get correction for OCR text if available.
    
    Args:
        text: Original OCR text
        
    Returns:
        Tuple of (corrected_text, was_corrected)
        - If correction found: returns corrected text and True
        - If no correction: returns original text and False
    """
    # Try exact match first
    if text in OCR_CORRECTIONS:
        return OCR_CORRECTIONS[text], True
    
    # Try fuzzy match (case-insensitive)
    text_lower = text.lower()
    if text_lower in FUZZY_CORRECTIONS:
        return FUZZY_CORRECTIONS[text_lower], True
    
    # No correction found
    return text, False


def apply_corrections_to_text_list(texts: list[str]) -> tuple[list[str], list[dict]]:
    """Apply corrections to a list of text strings.
    
    Args:
        texts: List of OCR text strings
        
    Returns:
        Tuple of (corrected_texts, correction_records)
        - corrected_texts: List with corrections applied
        - correction_records: List of dicts with correction details
    """
    corrected_texts = []
    correction_records = []
    
    for idx, text in enumerate(texts):
        corrected, was_corrected = get_correction(text)
        corrected_texts.append(corrected)
        
        if was_corrected:
            correction_records.append({
                "index": idx,
                "original": text,
                "corrected": corrected,
            })
    
    return corrected_texts, correction_records

