"""
services/query_parser.py — Extracts structured features from a raw user query.

Responsibilities:
    - extract_tags(query)           : Pull descriptive mood/genre keywords
    - detect_type(query)            : Identify if the user wants anime or manga
    - extract_reference(query)      : Find a reference title using "like X" pattern
    - extract_intent(query)         : Classify query purpose (rule-based)
    - compute_complexity_score(query): Score query complexity in [0.0, 1.0]
    - extract_semantic_hints(query) : Find nuanced preference phrases

Used by: nodes/process.py
"""

import re
from typing import List


# ── Tag keywords ──────────────────────────────────────────────────────────────

# All unique tags extracted from refined_anime_dataset.json + refined_manga_dataset.json
# (89 real dataset tags) PLUS common user-typed mood/vibe words that map via
# _SEMANTIC_FALLBACK in tag_mapper.py to real dataset tags.
TAG_KEYWORDS: List[str] = [
    # ── Real dataset tags ──────────────────────────────────────────────────
    # Action / Adventure
    "action", "adventure", "martial arts", "samurai", "military",
    "survival", "gore", "super power", "combat sports",
    # Drama / Emotion
    "drama", "romance", "slice of life", "iyashikei", "memoir",
    "romantic subtext",
    # Comedy
    "comedy", "parody", "gag humor",
    # Mature / Niche
    "ecchi", "harem", "reverse harem", "boys love", "girls love",
    "mahou shoujo", "magical sex shift",
    # Dark / Thriller
    "horror", "psychological", "suspense", "mystery", "detective",
    "organized crime", "delinquents", "vampire",
    # Fantasy / Sci-Fi
    "fantasy", "sci-fi", "supernatural", "mythology", "isekai",
    "reincarnation", "time travel", "space", "mecha",
    # Demographic
    "shounen", "shoujo", "seinen", "josei", "kids",
    # School / Slice
    "school", "childcare", "adult cast", "crossdressing",
    "otaku culture", "anthropomorphic", "pets",
    # Sports / Games
    "sports", "team sports", "racing", "strategy game",
    "high stakes game", "video game",
    # Arts / Culture
    "historical", "performing arts", "showbiz", "idols (female)",
    "idols (male)", "visual arts", "gourmet", "medical", "educational",
    "workplace",
    # Misc
    "award winning", "avant garde", "love polygon", "villainess", "cgdct",

    # ── User-typed mood/vibe words (mapped via _SEMANTIC_FALLBACK) ─────────
    "dark", "sad", "funny", "scary", "emotional", "epic", "slow",
    "cute", "complex", "feel-good", "violent", "romantic", "futuristic",
    "magical", "wholesome", "melancholy", "uplifting", "depressing",
    "happy", "thriller", "crime", "cooking", "game", "robot", "magic",
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


# ── Intent classification constants ───────────────────────────────────────────

# Primary mood words that signal a mood-driven search
_MOOD_WORDS: List[str] = [
    "dark", "sad", "feel-good", "happy", "melancholy",
    "uplifting", "depressing", "wholesome", "scary", "emotional",
    "violent", "romantic", "funny", "cute", "epic", "complex",
]

# Character trait phrases that signal a character-focused search
_CHARACTER_PHRASES: List[str] = [
    "strong female lead", "anti-hero", "protagonist",
    "villain", "character development", "villainess",
]

# Conjunctions that indicate a complex, multi-signal query
_CONJUNCTIONS: List[str] = ["and", "with", "but not"]

# ── Semantic hint phrases ─────────────────────────────────────────────────────

_SEMANTIC_HINT_PHRASES: List[str] = [
    # Narrative / writing style
    "character development", "plot twist", "slow burn", "coming of age",
    "redemption arc", "found family", "dark themes", "time travel",
    "reincarnation", "isekai", "survival",
    # Character types
    "strong female lead", "anti-hero", "villainess", "vampire",
    "mahou shoujo", "idol",
    # Tone / mood
    "psychological", "slice of life", "comedy", "romance", "tragedy",
    "horror", "mystery", "suspense", "gore",
    # Setting
    "historical", "military", "school", "space", "mecha",
    "supernatural", "mythology", "samurai",
    # Demographic hints
    "shounen", "shoujo", "seinen", "josei",
    # Niche
    "sports", "martial arts", "gourmet", "medical", "workplace",
    "strategy game", "high stakes game", "performing arts",
    "boys love", "girls love", "harem", "ecchi",
    "award winning", "avant garde",
]


# ── New public functions ───────────────────────────────────────────────────────

def extract_intent(query: str) -> str:
    """
    Classify the user's query intent using rule-based heuristics.

    Classification priority (first match wins):
        1. "find_similar"    — query contains a "like <Title>" reference pattern
        2. "mood_search"     — query is primarily mood words
        3. "character_search"— query mentions character trait phrases
        4. "complex_search"  — multiple genre/mood signals combined with conjunctions
        5. "genre_search"    — default fallback

    Example:
        extract_intent("dark anime like Attack on Titan") → "find_similar"
        extract_intent("sad and melancholy anime")        → "mood_search"
        extract_intent("strong female lead fantasy")      → "character_search"
        extract_intent("action and romance with comedy")  → "complex_search"
        extract_intent("fantasy adventure anime")         → "genre_search"

    Args:
        query: Raw user input string.

    Returns:
        One of: "find_similar", "mood_search", "character_search",
                "complex_search", "genre_search".
    """
    query_lower = query.lower()

    # 1. find_similar — "like X" reference present
    if re.search(r'\blike\s+\S', query_lower):
        return "find_similar"

    # 2. mood_search — primary mood words present
    for mood in _MOOD_WORDS:
        pattern = r'\b' + re.escape(mood) + r'\b'
        if re.search(pattern, query_lower):
            return "mood_search"

    # 3. character_search — character trait phrases present
    for phrase in _CHARACTER_PHRASES:
        if phrase in query_lower:
            return "character_search"

    # 4. complex_search — multiple genre/mood terms with conjunctions
    tag_matches = sum(
        1 for tag in TAG_KEYWORDS
        if re.search(r'\b' + re.escape(tag) + r'\b', query_lower)
    )
    has_conjunction = any(
        re.search(r'\b' + re.escape(conj) + r'\b', query_lower)
        for conj in _CONJUNCTIONS
    )
    if tag_matches >= 2 and has_conjunction:
        return "complex_search"

    # 5. Default
    return "genre_search"


def compute_complexity_score(query: str) -> float:
    """
    Compute a complexity score for the query in the range [0.0, 1.0].

    Scoring heuristic:
        Base score: 0.0
        +0.3  if word count > 8
        +0.2  if a "like <Title>" reference is present
        +0.2  if any semantic hint phrases are detected
        +0.3  if multiple genre/mood terms combined with conjunctions

    The result is clamped to [0.0, 1.0].

    Example:
        compute_complexity_score("dark anime") → 0.0
        compute_complexity_score("dark romance anime like Vampire Knight with "
                                 "strong character development and plot twists")
        → 1.0 (clamped)

    Args:
        query: Raw user input string.

    Returns:
        Float in [0.0, 1.0].
    """
    query_lower = query.lower()
    score = 0.0

    # +0.3 if word count > 8
    word_count = len(query.split())
    if word_count > 8:
        score += 0.3

    # +0.2 if reference present ("like <something>")
    if re.search(r'\blike\s+\S', query_lower):
        score += 0.2

    # +0.2 if semantic hint phrases detected
    hints = extract_semantic_hints(query)
    if hints:
        score += 0.2

    # +0.3 if multiple genre/mood terms with conjunctions
    tag_matches = sum(
        1 for tag in TAG_KEYWORDS
        if re.search(r'\b' + re.escape(tag) + r'\b', query_lower)
    )
    has_conjunction = any(
        re.search(r'\b' + re.escape(conj) + r'\b', query_lower)
        for conj in _CONJUNCTIONS
    )
    if tag_matches >= 2 and has_conjunction:
        score += 0.3

    # Clamp to [0.0, 1.0]
    return min(1.0, max(0.0, score))


def extract_semantic_hints(query: str) -> List[str]:
    """
    Scan the query for nuanced preference phrases and return all matches.

    Scanned phrases:
        "character development", "plot twist", "slow burn", "coming of age",
        "strong female lead", "anti-hero", "redemption arc", "found family",
        "dark themes", "psychological", "slice of life", "isekai",
        "romance", "comedy", "tragedy"

    Example:
        extract_semantic_hints("slow burn romance with character development")
        → ["slow burn", "romance", "character development"]

    Args:
        query: Raw user input string.

    Returns:
        List of matched hint phrases (in the order they appear in
        _SEMANTIC_HINT_PHRASES, de-duplicated).
    """
    query_lower = query.lower()
    return [phrase for phrase in _SEMANTIC_HINT_PHRASES if phrase in query_lower]
