"""JSON serialization utilities for handling numpy types."""

from __future__ import annotations

from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization.
    
    This function is designed to be used as the 'default' parameter in json.dump()
    to handle numpy types that are not JSON serializable by default.
    
    Args:
        obj: Object that may contain numpy types.
        
    Returns:
        Converted object with numpy types replaced by native Python types.
        
    Raises:
        TypeError: If object is not a numpy type (expected behavior for json.dump default).
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

