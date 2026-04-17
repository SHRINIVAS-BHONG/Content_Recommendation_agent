"""
state.py — Shared state object passed between all nodes in the agent graph.

Every node reads from this state, updates relevant fields, and returns it.
This ensures a single source of truth throughout the pipeline.
"""

from typing import TypedDict, List, Any, Dict


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

    # ── New: Explainability ───────────────────────────────────────────────
    reasoning_trace: List[str]   # human-readable log of decisions made


def initial_state(query: str) -> AgentState:
    """
    Create a fresh state object with default empty values.
    Call this at the graph entry point before running nodes.

    Args:
        query: The raw user input string.

    Returns:
        An AgentState with the query set and all other fields initialized.
    """
    return AgentState(
        query=query,
        tags=[],
        type="anime",           # default type
        reference="",
        results=[],
        # new defaults
        intent="genre_search",
        complexity_score=0.0,
        semantic_hints=[],
        search_strategy="hybrid",
        reference_synopsis="",
        model_input={},
        refinement_count=0,
        quality_report={},
        reasoning_trace=[],
    )
