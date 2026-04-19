"""
nodes/refine.py — Refine node: adjusts model input for a retry cycle.

Responsibilities:
    refine_node:
        - Strategy 1 (low coverage): if quality_report["uncovered_tags"] is
          non-empty, add those tags to state["tags"] (deduplicated).
        - Strategy 2 (low score): if quality_report["avg_score"] < 6.0 AND
          state["search_strategy"] == "tag_only", switch search_strategy to
          "semantic".
        - Strategy 3 (second cycle): if state["refinement_count"] == 1,
          broaden by removing the last tag from state["tags"] (least-common
          heuristic) and adding a "strictness": "low" hint to model_input.
        - Rebuild state["model_input"] with updated tags, strategy, and all
          other enrichment fields from the current state.
        - Increment state["refinement_count"] by exactly 1.
        - Append at least one entry to state["reasoning_trace"] describing
          the strategy applied.

Reads  : state["quality_report"], state["tags"], state["model_input"],
         state["search_strategy"], state["refinement_count"]
Updates: state["tags"], state["search_strategy"], state["model_input"],
         state["refinement_count"], state["reasoning_trace"]

Uses: utils/helpers.py (deduplicate), agent/state.py (AgentState)
"""

from backend.agent.state import AgentState
from backend.utils.helpers import deduplicate


def refine_node(state: AgentState) -> AgentState:
    """
    Adjust the model input payload and prepare for a retry cycle.

    Applies up to three refinement strategies based on the quality report,
    then rebuilds model_input, increments refinement_count, and logs the
    strategy applied to reasoning_trace.

    Args:
        state: AgentState with quality_report, tags, model_input,
               search_strategy, and refinement_count already populated.

    Returns:
        Updated AgentState with adjusted tags, search_strategy, model_input,
        incremented refinement_count, and extended reasoning_trace.
    """
    quality_report: dict  = state.get("quality_report", {})
    tags: list            = list(state.get("tags", []))
    search_strategy: str  = state.get("search_strategy", "tag_only")
    refinement_count: int = state.get("refinement_count", 0)
    model_input: dict     = dict(state.get("model_input", {}))
    reasoning_trace: list = list(state.get("reasoning_trace", []))

    # Pull enrichment fields from current model_input (or state fallbacks)
    media_type: str          = model_input.get("type",               state.get("type", "anime"))
    reference: str           = model_input.get("reference",          state.get("reference", ""))
    intent: str              = model_input.get("intent",             state.get("intent", "genre_search"))
    semantic_hints: list     = model_input.get("semantic_hints",     state.get("semantic_hints", []))
    reference_synopsis: str  = model_input.get("reference_synopsis", state.get("reference_synopsis", ""))
    complexity: str          = model_input.get("complexity",         "complex")

    strategies_applied = []

    avg_score: float = quality_report.get("avg_score", 0.0)

    # ── Strategy 3: Second cycle — broaden by removing least-common tag ───
    # Applied first so that the removal targets the original tags, not tags
    # that are about to be added by Strategy 1.
    if refinement_count == 1:
        removed_tag = None
        if tags:
            removed_tag = tags[-1]
            tags = tags[:-1]
        model_input["strictness"] = "low"
        strategies_applied.append(
            f"Strategy 3 (second cycle): refinement_count=1 → removed last tag "
            f"{removed_tag!r}, set strictness='low'."
        )

    # ── Strategy 1: Low coverage — add uncovered tags ─────────────────────
    uncovered_tags: list = quality_report.get("uncovered_tags", [])
    if uncovered_tags:
        tags = deduplicate(tags + uncovered_tags)
        strategies_applied.append(
            f"Strategy 1 (low coverage): added uncovered tags {uncovered_tags!r} "
            f"→ tags now {len(tags)} item(s)."
        )

    # ── Strategy 2: Low score — switch to semantic search ─────────────────
    if avg_score < 6.0 and search_strategy == "tag_only":
        search_strategy = "semantic"
        strategies_applied.append(
            f"Strategy 2 (low score): avg_score={avg_score:.2f} < 6.0 and "
            f"search_strategy was 'tag_only' → switched to 'semantic'."
        )

    # ── Rebuild model_input with updated fields ───────────────────────────
    model_input.update({
        "tags":               tags,
        "type":               media_type,
        "reference":          reference,
        "intent":             intent,
        "semantic_hints":     semantic_hints,
        "search_strategy":    search_strategy,
        "reference_synopsis": reference_synopsis,
        "complexity":         complexity,
    })

    # ── Append trace entry ────────────────────────────────────────────────
    if strategies_applied:
        trace_entry = "[refine_node] " + " | ".join(strategies_applied)
    else:
        trace_entry = (
            f"[refine_node] No specific strategy triggered "
            f"(uncovered_tags=[], avg_score={avg_score:.2f}, "
            f"search_strategy={search_strategy!r}, "
            f"refinement_count={refinement_count}). "
            f"Rebuilt model_input with current state."
        )
    reasoning_trace.append(trace_entry)

    # ── Increment refinement_count ────────────────────────────────────────
    refinement_count += 1

    # ── Update state ──────────────────────────────────────────────────────
    state["tags"]              = tags
    state["search_strategy"]   = search_strategy
    state["model_input"]       = model_input
    state["refinement_count"]  = refinement_count
    state["reasoning_trace"]   = reasoning_trace

    return state
