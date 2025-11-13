"""Data structures for form extraction using dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FieldValue:
    """Field with OCR metadata."""

    value: Optional[str] = None
    confidence: float = 0.0
    source: str = "header"  # header|sticker|defects|analysis
    validated: bool = False
    corrected: bool = False
    suspicious: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class StickerData:
    """Sticker data (priority source)."""

    part_line_number: Optional[FieldValue] = None
    part_designation: Optional[FieldValue] = None
    quantity_ordered: Optional[FieldValue] = None
    order_number: Optional[FieldValue] = None
    order_date: Optional[FieldValue] = None
    client: Optional[FieldValue] = None
    qr_code_detected: bool = False


@dataclass
class HeaderData:
    """Header section structured data."""

    # Document metadata
    act_number: Optional[FieldValue] = None
    act_date: Optional[FieldValue] = None
    template_revision: Optional[FieldValue] = None

    # Part details (sticker or handwritten)
    sticker_data: Optional[StickerData] = None
    part_line_number: Optional[FieldValue] = None
    part_designation: Optional[FieldValue] = None
    part_name: Optional[FieldValue] = None

    # Quantities
    quantity_ordered: Optional[FieldValue] = None
    quantity_checked: Optional[FieldValue] = None
    quantity_defective: Optional[FieldValue] = None
    quantity_passed: Optional[FieldValue] = None

    # Control
    control_type: Optional[FieldValue] = None
    inspector_name: Optional[FieldValue] = None


@dataclass
class DefectRow:
    """Single row in defects table."""

    row_number: Optional[str] = None
    parameter: Optional[str] = None
    fact_value: Optional[str] = None
    deviation: Optional[str] = None
    defect: Optional[str] = None
    quantity: Optional[str] = None
    confidence_avg: float = 0.0
    source_texts: List[str] = field(default_factory=list)


@dataclass
class DefectBlock:
    """Defects table block (geometry/holes/surface)."""

    block_type: str  # "geometry"|"holes"|"surface"|"other"
    block_title: str
    column_headers: List[str] = field(default_factory=list)
    rows: List[DefectRow] = field(default_factory=list)


@dataclass
class AnalysisRow:
    """Analysis deviations table row."""

    row_number: Optional[str] = None
    operation: Optional[str] = None
    cause: Optional[str] = None
    responsible: Optional[str] = None
    decision: Optional[str] = None
    confidence_avg: float = 0.0


@dataclass
class FinalDecision:
    """Final decision and signatures."""

    action: Optional[str] = None  # "использовать"|"доработать"|"не использовать"
    quantity: Optional[int] = None
    manager_name: Optional[str] = None
    otk_representative: Optional[str] = None


@dataclass
class AnalysisData:
    """Analysis section structured data."""

    deviations: List[AnalysisRow] = field(default_factory=list)
    final_decision: Optional[FinalDecision] = None


def dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict for JSON serialization.

    Args:
        obj: Dataclass instance, list, dict, or primitive value

    Returns:
        Dictionary or primitive value suitable for JSON serialization
    """
    if obj is None:
        return None
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {
            key: dataclass_to_dict(value) for key, value in obj.__dict__.items()
        }
    return obj



