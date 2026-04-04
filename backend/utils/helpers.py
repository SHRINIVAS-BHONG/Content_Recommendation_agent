"""
utils/helpers.py — Shared utility functions used across the agent pipeline.

Keeps generic, reusable logic separate from node / service business logic.
"""

from typing import Any, Dict, List


def deduplicate(items: List[Any]) -> List[Any]:
    """
    Remove duplicates from a list while preserving insertion order.

    Args:
        items: Any list with potentially repeated values.

    Returns:
        New list with duplicates removed.
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_get(d: Dict, key: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from a dict, returning a default if missing or None.

    Args:
        d       : Source dictionary.
        key     : Key to look up.
        default : Fallback value (default: None).

    Returns:
        Value at key, or default.
    """
    value = d.get(key)
    return value if value is not None else default


def format_score(score: Any) -> float:
    """
    Coerce a score value to a float rounded to 2 decimal places.
    Returns 0.0 if the value cannot be converted.

    Args:
        score: Raw score value from ML module.

    Returns:
        Float score between 0.0 and 1.0 (or whatever scale the ML module uses).
    """
    try:
        return round(float(score), 2)
    except (TypeError, ValueError):
        return 0.0


def normalize_result(raw: Dict) -> Dict:
    """
    Normalize a single raw result dict from the ML module into the expected
    output schema. Fills missing fields with safe defaults.

    Expected output schema:
        {
            "title"    : str,
            "image"    : str  (URL),
            "synopsis" : str,
            "score"    : float,
            "genres"   : list[str]
        }

    Args:
        raw: A dict returned by the ML recommender (structure may vary).

    Returns:
        Normalized dict matching the frontend contract.
    """
    return {
        "title":    safe_get(raw, "title",    default="Unknown"),
        "image":    safe_get(raw, "image",    default=""),
        "synopsis": safe_get(raw, "synopsis", default="No synopsis available."),
        "score":    format_score(safe_get(raw, "score", default=0.0)),
        "genres":   safe_get(raw, "genres",   default=[]),
    }
