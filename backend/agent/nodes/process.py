"""
nodes/process.py — Node 1: Parse the raw user query into structured state fields.

Responsibilities:
    - Extract mood/genre tags from the query
    - Detect media type (anime / manga)
    - Extract a reference title (e.g. "like Attack on Titan")
    - Classify query intent
    - Compute complexity score
    - Extract semantic hint phrases

Reads  : state["query"]
Updates: state["tags"], state["type"], state["reference"],
         state["intent"], state["complexity_score"], state["semantic_hints"]

Uses: services/query_parser.py
"""

from backend.agent.state import AgentState
from backend.services.query_parser import (
    extract_tags,
    detect_type,
    extract_reference,
    extract_intent,
    compute_complexity_score,
    extract_semantic_hints,
)


def process_node(state: AgentState) -> AgentState:
    """
    Convert the raw query into structured data and update state.

    Flow:
        query → [query_parser] → tags + type + reference
                               + intent + complexity_score + semantic_hints
                               → state

    Args:
        state: Current AgentState (expects state["query"] to be set).

    Returns:
        Updated AgentState with tags, type, reference, intent,
        complexity_score, and semantic_hints populated.

    Raises:
        ValueError: If the query is empty or contains only whitespace.
    """
    query: str = state.get("query", "")

    if not query or not query.strip():
        raise ValueError("process_node: 'query' is missing or empty in state.")

    # ── Extract structured fields from the query ───────────────────────────
    tags             = extract_tags(query)              # ["dark", "romance", ...]
    media_type       = detect_type(query)               # "anime" or "manga"
    reference        = extract_reference(query)         # "Attack On Titan" or ""
    intent           = extract_intent(query)            # "find_similar" | ...
    complexity_score = compute_complexity_score(query)  # float in [0.0, 1.0]
    semantic_hints   = extract_semantic_hints(query)    # ["slow burn", ...]

    # ── Update state ───────────────────────────────────────────────────────
    state["tags"]             = tags
    state["type"]             = media_type
    state["reference"]        = reference
    state["intent"]           = intent
    state["complexity_score"] = complexity_score
    state["semantic_hints"]   = semantic_hints

    return state
