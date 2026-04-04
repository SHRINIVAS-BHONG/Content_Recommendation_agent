"""
services/query_parser.py — Extracts structured features from a raw user query.

Responsibilities:
    - extract_tags(query)    : Pull descriptive mood/genre keywords
    - detect_type(query)     : Identify if the user wants anime or manga
    - extract_reference(query): Find a reference title using "like X" pattern

Used by: nodes/process.py
"""

import re
from typing import List


# ── Tag keywords ──────────────────────────────────────────────────────────────

# Mood / genre keywords we scan for in the query
TAG_KEYWORDS: List[str] = [
    "dark", "romance", "action", "comedy", "horror", "thriller",
    "mystery", "fantasy", "sci-fi", "adventure", "psychological",
    "slice of life", "drama", "supernatural", "mecha", "sports",
    "historical", "military", "magic", "school", "isekai",
]


# ── Public functions ───────────────────────────────────────────────────────────

def extract_tags(query: str) -> List[str]:
    """
    Scan the query for known mood/genre keywords and return them as a list.

    Example:
        extract_tags("dark romance anime like Vampire Knight")
        → ["dark", "romance"]

    Args:
        query: Raw user input string.

    Returns:
        List of matched tag strings (lowercase, de-duplicated).
    """
    query_lower = query.lower()
    found: List[str] = []

    for tag in TAG_KEYWORDS:
        # Match whole-word occurrences to avoid partial matches
        pattern = r'\b' + re.escape(tag) + r'\b'
        if re.search(pattern, query_lower):
            found.append(tag)

    return found


def detect_type(query: str) -> str:
    """
    Determine whether the user is asking for anime or manga.
    Defaults to "anime" if neither is explicitly mentioned.

    Example:
        detect_type("dark manga recommendations") → "manga"
        detect_type("best action anime")          → "anime"

    Args:
        query: Raw user input string.

    Returns:
        "anime" or "manga"
    """
    query_lower = query.lower()

    if re.search(r'\bmanga\b', query_lower):
        return "manga"
    if re.search(r'\banime\b', query_lower):
        return "anime"

    return "anime"  # sensible default


def extract_reference(query: str) -> str:
    """
    Extract a reference title using the pattern "like <Title>".
    The reference is everything after "like" up to the next keyword boundary.

    Example:
        extract_reference("dark anime like Attack on Titan") → "Attack on Titan"
        extract_reference("recommend good action anime")     → ""

    Args:
        query: Raw user input string.

    Returns:
        Extracted reference title (title-cased) or empty string if not found.
    """
    # Match "like <anything>" — greedy until end of string or a comma/period
    match = re.search(r'\blike\s+([^,\.]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip().title()

    return ""
