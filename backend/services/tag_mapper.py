"""
services/tag_mapper.py — Dataset-aware tag expansion service.

Instead of a hand-written static dictionary, this service:
  1. Loads the actual anime/manga JSON datasets at startup (once, cached).
  2. Builds a co-occurrence index: tags that frequently appear together
     in the same entry are treated as "related".
  3. Falls back to a curated semantic map for mood words not in the dataset
     (e.g. "dark", "sad", "epic" — words users type but aren't dataset tags).

This means the expansion automatically reflects YOUR data — if "military"
co-occurs with "historical" and "drama" in the dataset, that relationship
is discovered automatically rather than hard-coded.

Used by: nodes/reasoning.py
"""

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

# ── Dataset paths ─────────────────────────────────────────────────────────────
# Walks up from this file:  services/ → agent/ → backend/ → project_root/
_BASE_DIR   = Path(__file__).resolve().parents[3]   # project root
_ANIME_PATH = _BASE_DIR / "data" / "refined_anime_dataset.json"
_MANGA_PATH = _BASE_DIR / "data" / "refined_manga_dataset.json"

# How many co-occurring tags to surface per input tag
_TOP_N_RELATED = 6

# ── Semantic fallback for mood/vibe words that users type ─────────────────────
# Maps user-typed mood words to real tags from refined_anime/manga_dataset.json
_SEMANTIC_FALLBACK: Dict[str, List[str]] = {
    "dark":       ["psychological", "horror", "gore", "drama", "seinen"],
    "sad":        ["drama", "tragedy", "romance", "slice of life"],
    "funny":      ["comedy", "parody", "gag humor", "slice of life"],
    "scary":      ["horror", "supernatural", "suspense", "mystery"],
    "emotional":  ["drama", "romance", "slice of life", "iyashikei"],
    "epic":       ["action", "adventure", "fantasy", "shounen", "military"],
    "slow":       ["slice of life", "iyashikei", "drama", "romance"],
    "cute":       ["slice of life", "comedy", "romance", "school", "cgdct"],
    "complex":    ["psychological", "mystery", "suspense", "avant garde"],
    "feel-good":  ["comedy", "slice of life", "romance", "iyashikei"],
    "violent":    ["action", "military", "gore", "horror", "seinen"],
    "romantic":   ["romance", "drama", "shoujo", "romantic subtext"],
    "futuristic": ["sci-fi", "mecha", "space", "time travel"],
    "magical":    ["mahou shoujo", "fantasy", "supernatural", "mythology"],
    "wholesome":  ["slice of life", "iyashikei", "comedy", "romance"],
    "melancholy": ["drama", "psychological", "slice of life", "iyashikei"],
    "uplifting":  ["sports", "comedy", "adventure", "shounen"],
    "depressing": ["drama", "psychological", "horror", "tragedy"],
    "happy":      ["comedy", "slice of life", "iyashikei", "romance"],
    "thriller":   ["suspense", "mystery", "psychological", "organized crime"],
    "crime":      ["organized crime", "detective", "mystery", "seinen"],
    "historical": ["historical", "samurai", "military", "drama"],
    "school":     ["school", "comedy", "romance", "slice of life"],
    "sports":     ["sports", "team sports", "combat sports", "drama"],
    "music":      ["performing arts", "showbiz", "idols (female)", "drama"],
    "cooking":    ["gourmet", "slice of life", "comedy"],
    "game":       ["strategy game", "high stakes game", "video game", "psychological"],
    "isekai":     ["isekai", "fantasy", "adventure", "reincarnation"],
    "vampire":    ["vampire", "supernatural", "horror", "romance"],
    "robot":      ["mecha", "sci-fi", "action", "military"],
    "magic":      ["mahou shoujo", "fantasy", "supernatural", "mythology"],
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_tag(raw: str) -> str:
    """
    Strip JSON-artifact characters from dataset tag strings.

    The dataset stores tags like: "['action'",  "'drama'",  "['shounen']"
    This cleans them to:           "action"      "drama"     "shounen"
    """
    return re.sub(r"[\[\]'\"\\]", "", str(raw)).strip().lower()


def _load_dataset(path: Path) -> List[dict]:
    """Load a JSON file; return [] if the file does not exist yet."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Handle {"anime": [...]} wrapper shapes
        return next((v for v in data.values() if isinstance(v, list)), [])
    return []


def _build_cooccurrence_index(entries: List[dict]) -> Dict[str, Dict[str, int]]:
    """
    Scan every dataset entry and record which tags appear together.

    For each entry:
        - collect all clean tags + genres into a set
        - for every pair (A, B) in that set, increment index[A][B]

    The result is a nested dict:  tag → {related_tag: co_occurrence_count}
    """
    index: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for entry in entries:
        raw = (entry.get("tags", []) or []) + (entry.get("genres", []) or [])
        clean_set = {_clean_tag(t) for t in raw if _clean_tag(t)}

        for tag_a in clean_set:
            for tag_b in clean_set:
                if tag_a != tag_b:
                    index[tag_a][tag_b] += 1

    return index


@lru_cache(maxsize=2)   # cache anime index + manga index — never rebuilt
def _get_index(media_type: str) -> Dict[str, Dict[str, int]]:
    """Return (and cache) the co-occurrence index for anime or manga."""
    path    = _ANIME_PATH if media_type == "anime" else _MANGA_PATH
    entries = _load_dataset(path)
    return _build_cooccurrence_index(entries)


def _top_related(
    tag: str, index: Dict[str, Dict[str, int]], n: int
) -> List[str]:
    """Return the n most frequently co-occurring tags for a given tag."""
    co = index.get(tag.lower(), {})
    if not co:
        return []
    return [t for t, _ in sorted(co.items(), key=lambda x: x[1], reverse=True)[:n]]


# ── Public API ────────────────────────────────────────────────────────────────

def expand_tags(tags: List[str], media_type: str = "anime") -> List[str]:
    """
    Expand a list of base tags using dataset co-occurrence data.

    Strategy:
      1. Query the co-occurrence index (built from your real JSON dataset).
      2. If a tag has no co-occurrence signal (e.g. a mood word like "dark"),
         fall back to the semantic map.
      3. Deduplicate while preserving insertion order.

    Args:
        tags       : Tags from the process node.
        media_type : "anime" or "manga" — picks the right dataset index.

    Returns:
        Expanded, de-duplicated tag list grounded in your actual dataset.

    Example (with real dataset loaded):
        expand_tags(["action", "military"], "anime")
        → ["action", "military", "drama", "adventure", "shounen",
           "historical", "fantasy", ...]
    """
    index    = _get_index(media_type)
    expanded = list(tags)   # originals first

    for tag in tags:
        related = _top_related(tag.lower(), index, _TOP_N_RELATED)

        # Fallback for mood words absent from the dataset
        if not related:
            related = _SEMANTIC_FALLBACK.get(tag.lower(), [])

        for r in related:
            if r not in expanded:
                expanded.append(r)

    return expanded


def get_reference_tags(reference: str, media_type: str = "anime") -> List[str]:
    """
    Find a reference title in the actual dataset and return its tags/genres.

    Searches across title, title_english, title_japanese, and title_synonyms.
    Returns the cleaned tag + genre set from the matched entry.

    Args:
        reference  : Title string from "like X" pattern (e.g. "Attack On Titan").
        media_type : "anime" or "manga".

    Returns:
        List of clean tags from the matched dataset entry, or [] if not found.

    Example:
        get_reference_tags("Fullmetal Alchemist Brotherhood", "anime")
        → ["drama", "manga", "adventure", "action", "military",
           "fantasy", "shounen"]
    """
    if not reference:
        return []

    path    = _ANIME_PATH if media_type == "anime" else _MANGA_PATH
    entries = _load_dataset(path)
    if not entries:
        return []

    ref_lower = reference.lower().strip()

    for entry in entries:
        # Collect every title variant for this entry
        candidates = [
            entry.get("title", ""),
            entry.get("title_english", ""),
            entry.get("title_japanese", ""),
        ] + list(entry.get("title_synonyms", []) or [])

        cleaned_candidates = [_clean_tag(str(c)) for c in candidates if c]

        # Exact match → then substring match
        if ref_lower in cleaned_candidates:
            match = entry
            break
        if any(ref_lower in c for c in cleaned_candidates):
            match = entry
            break
    else:
        return []   # no match found

    raw = (match.get("tags", []) or []) + (match.get("genres", []) or [])
    return list({_clean_tag(t) for t in raw if _clean_tag(t)})
