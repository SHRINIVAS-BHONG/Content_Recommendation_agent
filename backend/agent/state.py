"""
state.py — Shared state object passed between all nodes in the agent graph.

Every node reads from this state, updates relevant fields, and returns it.
This ensures a single source of truth throughout the pipeline.
"""

from typing import TypedDict, List, Any, Dict, Optional


class AgentState(TypedDict):
    """
    Shared state structure used across all agent nodes.

    Fields:
        query              : Raw user input string (set at entry point, never modified)
        tags               : List of descriptive tags extracted/expanded from the query
        type               : Media type detected from query — "anime" or "manga"
        reference          : Reference title extracted from "like X" patterns
        results            : Final list of recommendation objects from the ML module

        intent             : Query purpose classification — "find_similar", "genre_search",
                             "mood_search", "character_search", or "complex_search"
        complexity_score   : Float in [0.0, 1.0] measuring query complexity
        semantic_hints     : Nuanced preference phrases extracted from the query

        search_strategy    : Strategy hint for the model — "tag_only", "semantic",
                             "hybrid", or "reference"
        reference_synopsis : Full synopsis of the reference title (from dataset)
        model_input        : Structured payload sent to model.recommend()

        refinement_count   : Number of refinement cycles completed (0–2)
        quality_report     : Quality metrics dict from evaluator node
        reasoning_trace    : Human-readable log of decisions made by each node
    """
    # ── Existing fields (unchanged) ──────────────────────────────────────
    query: str
    tags: List[str]
    type: str                    # "anime" | "manga"
    reference: str
    results: List[Any]

    # ── New: Query understanding ──────────────────────────────────────────
    intent: str                  # "find_similar" | "genre_search" | "mood_search"
                                 # | "character_search" | "complex_search"
    complexity_score: float      # 0.0–1.0; drives simple vs. deep routing
    semantic_hints: List[str]    # nuanced preferences extracted from query

    # ── New: Model input enrichment ───────────────────────────────────────
    search_strategy: str         # "tag_only" | "semantic" | "hybrid" | "reference"
    reference_synopsis: str      # synopsis of reference title looked up from dataset
    model_input: Dict            # the full payload sent to model.recommend()

    # ── New: Refinement loop ──────────────────────────────────────────────
    refinement_count: int        # 0, 1, or 2 — guards against infinite loops
    quality_report: Dict         # {coverage, diversity, avg_score, verdict, uncovered_tags}

    # ── Pagination ────────────────────────────────────────────────────────
    page: int                    # 0-indexed page for Load More

    # ── New: Explainability ───────────────────────────────────────────────
    reasoning_trace: List[str]   # human-readable log of decisions made

    # ── User context (authenticated users) ───────────────────────────────
    is_authenticated: bool                          # True for authenticated users
    user_id: Optional[str]                          # UUID string or None for anonymous
    user_preferences: Optional[Dict[str, Any]]      # Learned genre/content preferences
    conversation_context: Optional[Dict[str, Any]]  # Recent queries and results
    user_feedback_history: Optional[List[Dict[str, Any]]]  # Past feedback records
    personalization_weights: Optional[Dict[str, float]]    # Per-genre weight multipliers


def initial_state(
    query: str,
    page: int = 0,
    user_id: Optional[str] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    user_feedback_history: Optional[List[Dict[str, Any]]] = None,
) -> "AgentState":
    """
    Create a fresh state object with default empty values.
    Supports both authenticated and anonymous modes.

    Args:
        query: The raw user input string.
        page: 0-indexed page number for Load More pagination.
        user_id: Optional user UUID string. When provided the state is
                 initialised in authenticated mode.
        user_preferences: Optional dict with preferred_genres, avoided_genres,
                          content_types, personalization_level, etc.
        conversation_context: Optional dict with recent_queries, recent_results,
                              interaction_count, last_interaction_time.
        user_feedback_history: Optional list of past feedback dicts.

    Returns:
        An AgentState with the query set and all other fields initialized.
        When user_id is supplied, is_authenticated is True and personalization
        weights are derived from user_preferences; otherwise the state degrades
        gracefully to anonymous behaviour.
    """
    is_authenticated = user_id is not None

    # Build personalization weights from preferences when available
    personalization_weights: Optional[Dict[str, float]] = None
    if is_authenticated and user_preferences:
        weights: Dict[str, float] = {}
        for genre in user_preferences.get("preferred_genres", []):
            weights[genre] = 1.5   # boost preferred genres
        for genre in user_preferences.get("avoided_genres", []):
            weights[genre] = 0.3   # suppress avoided genres
        if weights:
            personalization_weights = weights

    return AgentState(
        query=query,
        tags=[],
        type="anime",           # default type
        reference="",
        results=[],
        # query understanding
        intent="genre_search",
        complexity_score=0.0,
        semantic_hints=[],
        # model input enrichment
        search_strategy="hybrid",
        reference_synopsis="",
        model_input={},
        # refinement loop
        refinement_count=0,
        quality_report={},
        # pagination
        page=page,
        # explainability
        reasoning_trace=[],
        # user context
        is_authenticated=is_authenticated,
        user_id=user_id,
        user_preferences=user_preferences,
        conversation_context=conversation_context,
        user_feedback_history=user_feedback_history,
        personalization_weights=personalization_weights,
    )
