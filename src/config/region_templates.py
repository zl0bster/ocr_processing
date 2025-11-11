"""Region template configuration and loader utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_REGION_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "otk_v1": [
        {
            "region_id": "header",
            "y_start_norm": 0.0,
            "y_end_norm": 0.33,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "defects",
            "y_start_norm": 0.33,
            "y_end_norm": 0.66,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "analysis",
            "y_start_norm": 0.66,
            "y_end_norm": 1.0,
            "detection_method": "template",
            "confidence": 1.0,
        },
    ],
    "otk_v2": [
        {
            "region_id": "header",
            "y_start_norm": 0.0,
            "y_end_norm": 0.28,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "defects",
            "y_start_norm": 0.28,
            "y_end_norm": 0.64,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "analysis",
            "y_start_norm": 0.64,
            "y_end_norm": 1.0,
            "detection_method": "template",
            "confidence": 1.0,
        },
    ],
    "otk_standard": [
        {
            "region_id": "header",
            "y_start_norm": 0.0,
            "y_end_norm": 0.33,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "defects",
            "y_start_norm": 0.33,
            "y_end_norm": 0.66,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "analysis",
            "y_start_norm": 0.66,
            "y_end_norm": 1.0,
            "detection_method": "template",
            "confidence": 1.0,
        },
    ],
    "otk_extended": [
        {
            "region_id": "header",
            "y_start_norm": 0.0,
            "y_end_norm": 0.28,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "defects",
            "y_start_norm": 0.28,
            "y_end_norm": 0.64,
            "detection_method": "template",
            "confidence": 1.0,
        },
        {
            "region_id": "analysis",
            "y_start_norm": 0.64,
            "y_end_norm": 1.0,
            "detection_method": "template",
            "confidence": 1.0,
        },
    ],
}


def load_region_templates(
    template_file: Optional[Path],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load region templates from JSON file.

    Args:
        template_file: Path to JSON file containing region template definitions.
        logger: Optional logger for diagnostic messages.

    Returns:
        Dictionary mapping template names to a list of region definitions.
    """
    if not template_file:
        return DEFAULT_REGION_TEMPLATES

    if not template_file.exists():
        if logger:
            logger.debug(
                "Region template file '%s' not found. Using built-in defaults.",
                template_file,
            )
        return DEFAULT_REGION_TEMPLATES

    try:
        with template_file.open("r", encoding="utf-8") as fp:
            raw_data = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        if logger:
            logger.warning(
                "Failed to load region templates from '%s': %s. Using defaults.",
                template_file,
                exc,
            )
        return DEFAULT_REGION_TEMPLATES

    if not isinstance(raw_data, dict):
        if logger:
            logger.warning(
                "Region template file '%s' must contain a JSON object. Using defaults.",
                template_file,
            )
        return DEFAULT_REGION_TEMPLATES

    validated_templates: Dict[str, List[Dict[str, Any]]] = {}

    for template_name, regions in raw_data.items():
        if not isinstance(template_name, str):
            if logger:
                logger.debug(
                    "Skipping template with non-string name: %r", template_name
                )
            continue
        if not isinstance(regions, Iterable):
            if logger:
                logger.debug(
                    "Skipping template '%s' because regions are not iterable",
                    template_name,
                )
            continue

        validated_regions: List[Dict[str, Any]] = []
        for region in regions:
            if not isinstance(region, dict):
                continue
            region_id = region.get("region_id")
            y_start = region.get("y_start_norm")
            y_end = region.get("y_end_norm")
            if (
                not isinstance(region_id, str)
                or not isinstance(y_start, (int, float))
                or not isinstance(y_end, (int, float))
            ):
                continue
            if not 0.0 <= float(y_start) < float(y_end) <= 1.0:
                continue

            validated_region = {
                "region_id": region_id,
                "y_start_norm": float(y_start),
                "y_end_norm": float(y_end),
                "detection_method": region.get("detection_method", "template"),
                "confidence": float(region.get("confidence", 1.0)),
            }
            validated_regions.append(validated_region)

        if validated_regions:
            validated_templates[template_name] = validated_regions

    if not validated_templates:
        if logger:
            logger.warning(
                "Region template file '%s' did not contain any valid templates. "
                "Using built-in defaults.",
                template_file,
            )
        return DEFAULT_REGION_TEMPLATES

    if logger:
        logger.debug(
            "Loaded %d region templates from '%s'",
            len(validated_templates),
            template_file,
        )
    return validated_templates

