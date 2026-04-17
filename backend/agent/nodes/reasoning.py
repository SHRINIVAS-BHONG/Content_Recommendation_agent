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


# ── Semantic hint → tag mapping ───────────────────────────────────────────────

_HINT_TO_TAGS = {
    "character development": ["coming of age", "growth"],
    "slow burn":             ["romance", "drama"],
    "plot twist":            ["mystery", "thriller"],
    "dark themes":           ["psychological", "horror"],
    "found family":          ["drama", "slice of life"],
    "redemption arc":        ["drama", "action"],
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
        3. Deduplicate tags.
        4. Set search_strategy = "tag_only".
        5. Build model_input with all required keys.
        6. Append a trace entry.

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

    # ── Step 1: Expand base tags using dataset co-occurrence ───────────────
    expanded = expand_tags(current_tags, media_type=media_type)

    # ── Step 2: Merge reference title tags if reference is non-empty ───────
    if reference:
        ref_tags = get_reference_tags(reference, media_type=media_type)
        expanded.extend(ref_tags)

    # ── Step 3: Deduplicate while preserving order ─────────────────────────
    deduped_tags = deduplicate(expanded)

    # ── Step 4: Set search strategy ────────────────────────────────────────
    search_strategy = "tag_only"

    # ── Step 5: Build model_input with all required keys ───────────────────
    model_input = {
        "tags":               deduped_tags,
        "type":               media_type,
        "reference":          reference,
        "intent":             intent,
        "semantic_hints":     semantic_hints,
        "search_strategy":    search_strategy,
        "reference_synopsis": reference_synopsis,
        "complexity":         "simple",
    }

    # ── Step 6: Append trace entry ─────────────────────────────────────────
    reasoning_trace.append(
        f"[simple_reasoning_node] Expanded {len(current_tags)} base tag(s) to "
        f"{len(deduped_tags)} tag(s) via co-occurrence. "
        f"search_strategy={search_strategy!r}. "
        f"reference={reference!r}."
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
        3. Look up reference synopsis from dataset when intent == "find_similar".
        4. Set search_strategy based on intent mapping.
        5. Build full model_input payload.
        6. Deduplicate tags.
        7. Append multiple trace entries.

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

    # ── Step 3: Reference synopsis lookup when intent == "find_similar" ────
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
                    f"[deep_reasoning_node] Step 3: Found reference title "
                    f"{reference!r} in dataset. Synopsis length: "
                    f"{len(reference_synopsis)} chars."
                )
            else:
                reference_synopsis = ""
                reasoning_trace.append(
                    f"[deep_reasoning_node] WARNING: Reference title {reference!r} "
                    f"not found in dataset. Setting reference_synopsis to empty string."
                )

    # ── Step 4: Set search_strategy based on intent ────────────────────────
    search_strategy = _INTENT_TO_STRATEGY.get(intent, "tag_only")

    # ── Step 5: Deduplicate tags ───────────────────────────────────────────
    deduped_tags = deduplicate(expanded)

    # ── Step 6: Build full model_input payload ─────────────────────────────
    model_input = {
        "tags":               deduped_tags,
        "type":               media_type,
        "reference":          reference,
        "intent":             intent,
        "semantic_hints":     semantic_hints,
        "search_strategy":    search_strategy,
        "reference_synopsis": reference_synopsis,
        "complexity":         "complex",
    }

    reasoning_trace.append(
        f"[deep_reasoning_node] Step 4: search_strategy={search_strategy!r} "
        f"(intent={intent!r}). Final tag count: {len(deduped_tags)}."
    )

    # ── Update state ───────────────────────────────────────────────────────
    state["tags"]               = deduped_tags
    state["search_strategy"]    = search_strategy
    state["reference_synopsis"] = reference_synopsis
    state["model_input"]        = model_input
    state["reasoning_trace"]    = reasoning_trace

    return state
