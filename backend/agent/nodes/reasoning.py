"""
nodes/reasoning.py — Reasoning nodes: simple and deep query enrichment.

Responsibilities:
    simple_reasoning_node:
        - Expand base tags via dataset co-occurrence (tag_mapper.expand_tags)
        - If a reference title exists → fetch its actual tags from the dataset
        - Set search_strategy = "tag_only"
        - Build model_input with all required keys
        - Deduplicate tags
        - Append at least one entry to reasoning_trace

    deep_reasoning_node:
        - Expand base tags via dataset co-occurrence
        - Translate semantic_hints into additional tags
        - Look up reference synopsis from dataset when intent == "find_similar"
        - Set search_strategy based on intent mapping
        - Build full model_input payload
        - Deduplicate tags
        - Append multiple entries to reasoning_trace

Reads  : state["tags"], state["type"], state["reference"], state["intent"],
         state["semantic_hints"], state["reference_synopsis"]
Updates: state["tags"], state["search_strategy"], state["reference_synopsis"],
         state["model_input"], state["reasoning_trace"]

Uses: services/tag_mapper.py, utils/helpers.py
"""

from agent.state import AgentState
from services.tag_mapper import (
    expand_tags,
    get_reference_tags,
    _load_dataset,
    _ANIME_PATH,
    _MANGA_PATH,
)
from utils.helpers import deduplicate


# ── Semantic hint → real dataset tag mapping ─────────────────────────────────
# Keys are phrases users type; values are actual tags from the dataset.
# All tags verified against refined_anime_dataset.json + refined_manga_dataset.json

_HINT_TO_TAGS = {
    # Narrative / writing style
    "character development": ["coming of age", "drama", "slice of life"],
    "slow burn":             ["romance", "drama", "iyashikei"],
    "plot twist":            ["mystery", "psychological", "suspense"],
    "dark themes":           ["psychological", "horror", "gore", "drama"],
    "found family":          ["drama", "slice of life", "adventure"],
    "redemption arc":        ["drama", "action", "psychological"],
    "coming of age":         ["school", "drama", "slice of life", "shounen"],
    "time travel":           ["time travel", "sci-fi", "mystery"],
    "reincarnation":         ["reincarnation", "isekai", "fantasy"],
    "survival":              ["survival", "action", "horror", "gore"],
    # Character types
    "strong female lead":    ["action", "adventure", "shoujo", "mahou shoujo"],
    "anti-hero":             ["psychological", "seinen", "drama", "action"],
    "villainess":            ["villainess", "romance", "fantasy", "isekai"],
    "vampire":               ["vampire", "supernatural", "horror", "romance"],
    "mahou shoujo":          ["mahou shoujo", "fantasy", "drama"],
    "idol":                  ["idols (female)", "idols (male)", "performing arts", "showbiz"],
    # Tone / mood
    "psychological":         ["psychological", "mystery", "suspense", "horror"],
    "slice of life":         ["slice of life", "iyashikei", "comedy", "drama"],
    "comedy":                ["comedy", "parody", "gag humor", "slice of life"],
    "romance":               ["romance", "drama", "shoujo", "romantic subtext"],
    "tragedy":               ["drama", "psychological", "horror"],
    "horror":                ["horror", "supernatural", "gore", "mystery"],
    "mystery":               ["mystery", "detective", "suspense", "psychological"],
    "suspense":              ["suspense", "mystery", "thriller", "psychological"],
    "gore":                  ["gore", "horror", "action", "survival"],
    # Setting
    "historical":            ["historical", "samurai", "military", "drama"],
    "military":              ["military", "action", "historical", "drama"],
    "school":                ["school", "comedy", "romance", "slice of life"],
    "space":                 ["space", "sci-fi", "adventure", "mecha"],
    "mecha":                 ["mecha", "sci-fi", "action", "military"],
    "supernatural":          ["supernatural", "fantasy", "horror", "mystery"],
    "mythology":             ["mythology", "fantasy", "historical", "adventure"],
    "samurai":               ["samurai", "historical", "action", "drama"],
    # Demographic hints
    "shounen":               ["shounen", "action", "adventure", "comedy"],
    "shoujo":                ["shoujo", "romance", "drama", "slice of life"],
    "seinen":                ["seinen", "psychological", "drama", "action"],
    "josei":                 ["josei", "romance", "drama", "slice of life"],
    # Niche
    "sports":                ["sports", "team sports", "combat sports", "drama"],
    "martial arts":          ["martial arts", "action", "sports", "shounen"],
    "gourmet":               ["gourmet", "slice of life", "comedy"],
    "medical":               ["medical", "drama", "educational"],
    "workplace":             ["workplace", "slice of life", "drama", "comedy"],
    "strategy game":         ["strategy game", "high stakes game", "psychological"],
    "high stakes game":      ["high stakes game", "strategy game", "psychological", "thriller"],
    "performing arts":       ["performing arts", "showbiz", "drama", "romance"],
    "boys love":             ["boys love", "romance", "drama"],
    "girls love":            ["girls love", "romance", "drama"],
    "harem":                 ["harem", "romance", "comedy", "ecchi"],
    "ecchi":                 ["ecchi", "comedy", "romance", "harem"],
    "award winning":         ["award winning", "drama", "psychological"],
    "avant garde":           ["avant garde", "psychological", "drama"],
    "isekai":                ["isekai", "fantasy", "adventure", "reincarnation"],
}

# ── Intent → search_strategy mapping ─────────────────────────────────────────

_INTENT_TO_STRATEGY = {
    "find_similar":   "reference",
    "mood_search":    "semantic",
    "complex_search": "hybrid",
}


# ── Simple reasoning node ─────────────────────────────────────────────────────

def simple_reasoning_node(state: AgentState) -> AgentState:
    """
    Fast-path reasoning for low-complexity queries.

    Flow:
        1. Expand base tags using dataset co-occurrence.
        2. Merge reference title tags if reference is non-empty.
        3. Apply user personalization if authenticated.
        4. Deduplicate tags.
        5. Set search_strategy = "tag_only".
        6. Build model_input with all required keys.
        7. Append a trace entry.

    Args:
        state: AgentState with tags, type, reference, intent, semantic_hints,
               and reference_synopsis already populated by process_node.

    Returns:
        Updated AgentState with enriched tags, search_strategy, model_input,
        and reasoning_trace.
    """
    current_tags: list = state.get("tags", [])
    media_type: str    = state.get("type", "anime")
    reference: str     = state.get("reference", "")
    intent: str        = state.get("intent", "genre_search")
    semantic_hints: list = state.get("semantic_hints", [])
    reference_synopsis: str = state.get("reference_synopsis", "")
    reasoning_trace: list = list(state.get("reasoning_trace", []))
    
    # User context fields
    is_authenticated: bool = state.get("is_authenticated", False)
    user_preferences: dict = state.get("user_preferences") or {}
    conversation_context: dict = state.get("conversation_context") or {}

    # ── Step 1: Expand base tags using dataset co-occurrence ───────────────
    expanded = expand_tags(current_tags, media_type=media_type)

    # ── Step 2: Merge reference title tags if reference is non-empty ───────
    if reference:
        ref_tags = get_reference_tags(reference, media_type=media_type)
        expanded.extend(ref_tags)

    # ── Step 3: Apply user personalization if authenticated ────────────────
    if is_authenticated:
        # Add preferred genres as tags if not already present
        preferred_genres = user_preferences.get("preferred_genres", [])
        for genre in preferred_genres:
            if genre.lower() not in [tag.lower() for tag in expanded]:
                expanded.append(genre)
        
        # Remove avoided genres
        avoided_genres = user_preferences.get("avoided_genres", [])
        avoided_lower = [genre.lower() for genre in avoided_genres]
        expanded = [tag for tag in expanded if tag.lower() not in avoided_lower]
        
        reasoning_trace.append(
            f"[simple_reasoning_node] Applied user personalization: "
            f"added {len(preferred_genres)} preferred genres, "
            f"filtered {len(avoided_genres)} avoided genres."
        )

    # ── Step 4: Deduplicate while preserving order ─────────────────────────
    deduped_tags = deduplicate(expanded)

    # ── Step 5: Set search strategy ────────────────────────────────────────
    search_strategy = "tag_only"

    # ── Step 6: Build model_input with all required keys ───────────────────
    model_input = {
        "tags":               deduped_tags,
        "type":               media_type,
        "reference":          reference,
        "intent":             intent,
        "semantic_hints":     semantic_hints,
        "search_strategy":    search_strategy,
        "reference_synopsis": reference_synopsis,
        "complexity":         "simple",
        # User context for personalization
        "is_authenticated":   is_authenticated,
        "user_preferences":   user_preferences,
        "conversation_context": conversation_context,
    }

    # ── Step 7: Append trace entry ─────────────────────────────────────────
    reasoning_trace.append(
        f"[simple_reasoning_node] Expanded {len(current_tags)} base tag(s) to "
        f"{len(deduped_tags)} tag(s) via co-occurrence. "
        f"search_strategy={search_strategy!r}. "
        f"reference={reference!r}. "
        f"authenticated={is_authenticated}."
    )

    # ── Update state ───────────────────────────────────────────────────────
    state["tags"]             = deduped_tags
    state["search_strategy"]  = search_strategy
    state["model_input"]      = model_input
    state["reasoning_trace"]  = reasoning_trace

    return state


# ── Deep reasoning node ───────────────────────────────────────────────────────

def deep_reasoning_node(state: AgentState) -> AgentState:
    """
    Full multi-factor analysis for complex queries.

    Flow:
        1. Expand base tags using dataset co-occurrence.
        2. Translate semantic_hints into additional tags.
        3. Apply user personalization if authenticated.
        4. Look up reference synopsis from dataset when intent == "find_similar".
        5. Set search_strategy based on intent mapping.
        6. Build full model_input payload.
        7. Deduplicate tags.
        8. Append multiple trace entries.

    Args:
        state: AgentState with tags, type, reference, intent, semantic_hints,
               and reference_synopsis already populated by process_node.

    Returns:
        Updated AgentState with enriched tags, search_strategy,
        reference_synopsis, model_input, and reasoning_trace.
    """
    current_tags: list = state.get("tags", [])
    media_type: str    = state.get("type", "anime")
    reference: str     = state.get("reference", "")
    intent: str        = state.get("intent", "genre_search")
    semantic_hints: list = state.get("semantic_hints", [])
    reference_synopsis: str = state.get("reference_synopsis", "")
    reasoning_trace: list = list(state.get("reasoning_trace", []))
    
    # User context fields
    is_authenticated: bool = state.get("is_authenticated", False)
    user_preferences: dict = state.get("user_preferences") or {}
    conversation_context: dict = state.get("conversation_context") or {}

    # ── Step 1: Expand base tags using dataset co-occurrence ───────────────
    expanded = expand_tags(current_tags, media_type=media_type)
    reasoning_trace.append(
        f"[deep_reasoning_node] Step 1: Co-occurrence expansion — "
        f"{len(current_tags)} base tag(s) → {len(expanded)} expanded tag(s)."
    )

    # ── Step 2: Translate semantic_hints into additional tags ──────────────
    hint_tags_added = []
    for hint in semantic_hints:
        hint_lower = hint.lower()
        for phrase, tags in _HINT_TO_TAGS.items():
            if phrase in hint_lower:
                for t in tags:
                    if t not in expanded:
                        expanded.append(t)
                        hint_tags_added.append(t)

    reasoning_trace.append(
        f"[deep_reasoning_node] Step 2: Semantic hint translation — "
        f"hints={semantic_hints!r} → added tags={hint_tags_added!r}."
    )

    # ── Step 3: Apply user personalization if authenticated ────────────────
    personalization_applied = False
    if is_authenticated:
        # Add preferred genres as tags if not already present
        preferred_genres = user_preferences.get("preferred_genres", [])
        added_preferred = []
        for genre in preferred_genres:
            if genre.lower() not in [tag.lower() for tag in expanded]:
                expanded.append(genre)
                added_preferred.append(genre)
        
        # Remove avoided genres
        avoided_genres = user_preferences.get("avoided_genres", [])
        avoided_lower = [genre.lower() for genre in avoided_genres]
        original_count = len(expanded)
        expanded = [tag for tag in expanded if tag.lower() not in avoided_lower]
        filtered_count = original_count - len(expanded)
        
        # Consider conversation context for deduplication hints
        recent_results = conversation_context.get("recent_results", [])
        if recent_results:
            # Extract titles from recent results for deduplication later
            recent_titles = [result.get("title", "") for result in recent_results[-10:]]
            reasoning_trace.append(
                f"[deep_reasoning_node] Step 3b: Found {len(recent_titles)} recent results for deduplication."
            )
        
        personalization_applied = True
        reasoning_trace.append(
            f"[deep_reasoning_node] Step 3a: Applied user personalization — "
            f"added {len(added_preferred)} preferred genres, "
            f"filtered {filtered_count} avoided genre tags."
        )

    # ── Step 4: Reference synopsis lookup when intent == "find_similar" ────
    if intent == "find_similar":
        dataset_path = _ANIME_PATH if media_type == "anime" else _MANGA_PATH
        entries = _load_dataset(dataset_path)

        if not entries:
            # Dataset file is missing or empty — skip lookup, log warning
            reasoning_trace.append(
                f"[deep_reasoning_node] WARNING: Dataset file not found or empty "
                f"at {dataset_path}. Skipping reference synopsis lookup."
            )
        else:
            ref_lower = reference.lower().strip()
            found_entry = None

            for entry in entries:
                candidates = [
                    entry.get("title", ""),
                    entry.get("title_english", ""),
                    entry.get("title_japanese", ""),
                ] + list(entry.get("title_synonyms", []) or [])

                cleaned = [str(c).strip().lower() for c in candidates if c]

                if ref_lower in cleaned:
                    found_entry = entry
                    break
                if any(ref_lower in c for c in cleaned):
                    found_entry = entry
                    break

            if found_entry is not None:
                reference_synopsis = found_entry.get("synopsis", "") or ""
                reasoning_trace.append(
                    f"[deep_reasoning_node] Step 4: Found reference title "
                    f"{reference!r} in dataset. Synopsis length: "
                    f"{len(reference_synopsis)} chars."
                )
            else:
                reference_synopsis = ""
                reasoning_trace.append(
                    f"[deep_reasoning_node] WARNING: Reference title {reference!r} "
                    f"not found in dataset. Setting reference_synopsis to empty string."
                )

    # ── Step 5: Set search_strategy based on intent ────────────────────────
    search_strategy = _INTENT_TO_STRATEGY.get(intent, "tag_only")

    # ── Step 6: Deduplicate tags ───────────────────────────────────────────
    deduped_tags = deduplicate(expanded)

    # ── Step 7: Build full model_input payload ─────────────────────────────
    model_input = {
        "tags":               deduped_tags,
        "type":               media_type,
        "reference":          reference,
        "intent":             intent,
        "semantic_hints":     semantic_hints,
        "search_strategy":    search_strategy,
        "reference_synopsis": reference_synopsis,
        "complexity":         "complex",
        # User context for personalization
        "is_authenticated":   is_authenticated,
        "user_preferences":   user_preferences,
        "conversation_context": conversation_context,
        "personalization_applied": personalization_applied,
    }

    reasoning_trace.append(
        f"[deep_reasoning_node] Step 5: search_strategy={search_strategy!r} "
        f"(intent={intent!r}). Final tag count: {len(deduped_tags)}. "
        f"Personalization applied: {personalization_applied}."
    )

    # ── Update state ───────────────────────────────────────────────────────
    state["tags"]               = deduped_tags
    state["search_strategy"]    = search_strategy
    state["reference_synopsis"] = reference_synopsis
    state["model_input"]        = model_input
    state["reasoning_trace"]    = reasoning_trace

    return state
