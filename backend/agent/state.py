"""
state.py — Shared state object passed between all nodes in the agent graph.

Every node reads from this state, updates relevant fields, and returns it.
This ensures a single source of truth throughout the pipeline.
"""

from typing import TypedDict, List, Any


class AgentState(TypedDict):
    """
    Shared state structure used across all agent nodes.

    Fields:
        query     : Raw user input string (set at entry point, never modified)
        tags      : List of descriptive tags extracted/expanded from the query
        type      : Media type detected from query — "anime" or "manga"
        reference : Reference title extracted from "like X" patterns
        results   : Final list of recommendation objects from the ML module
    """
    query: str
    tags: List[str]
    type: str
    reference: str
    results: List[Any]


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
        type="anime",       # default type
        reference="",
        results=[],
    )
