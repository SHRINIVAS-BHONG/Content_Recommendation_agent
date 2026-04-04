"""
nodes/process.py — Node 1: Parse the raw user query into structured state fields.

Responsibilities:
    - Extract mood/genre tags from the query
    - Detect media type (anime / manga)
    - Extract a reference title (e.g. "like Attack on Titan")

Reads  : state["query"]
Updates: state["tags"], state["type"], state["reference"]

Uses: services/query_parser.py
"""

from agent.state import AgentState
from agent.services.query_parser import extract_tags, detect_type, extract_reference


def process_node(state: AgentState) -> AgentState:
    """
    Convert the raw query into structured data and update state.

    Flow:
        query → [query_parser] → tags + type + reference → state

    Args:
        state: Current AgentState (expects state["query"] to be set).

    Returns:
        Updated AgentState with tags, type, and reference populated.
    """
    query: str = state.get("query", "")

    if not query:
        raise ValueError("process_node: 'query' is missing or empty in state.")

    # ── Extract structured fields from the query ───────────────────────────
    tags      = extract_tags(query)       # ["dark", "romance", ...]
    media_type = detect_type(query)        # "anime" or "manga"
    reference  = extract_reference(query)  # "Attack On Titan" or ""

    # ── Update state ───────────────────────────────────────────────────────
    state["tags"]      = tags
    state["type"]      = media_type
    state["reference"] = reference

    return state
