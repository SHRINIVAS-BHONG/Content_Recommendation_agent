"""
nodes/output.py — Node 4: Format raw ML results into the frontend contract.

Responsibilities:
    - Normalize each raw result dict into the expected output schema
    - Guarantee all required fields are present with safe defaults
    - Return the final, clean results list

Reads  : state["results"]
Updates: state["results"]  (replaces raw with formatted)

Uses: utils/helpers.py → normalize_result()

Output schema per result:
    {
        "title"    : str,
        "image"    : str   (URL),
        "synopsis" : str,
        "score"    : float,
        "genres"   : list[str]
    }
"""

from typing import List, Dict, Any
from agent.state import AgentState
from agent.utils.helpers import normalize_result


def output_node(state: AgentState) -> AgentState:
    """
    Transform raw ML results into the standardized frontend response format.

    Flow:
        state["results"] (raw) → normalize each item → state["results"] (clean)

    Args:
        state: Current AgentState (expects state["results"] to be a list).

    Returns:
        Updated AgentState where state["results"] contains normalized dicts.
    """
    raw_results: List[Any] = state.get("results", [])

    # ── Normalize every result to match the frontend schema ────────────────
    formatted: List[Dict[str, Any]] = [
        normalize_result(item)
        for item in raw_results
        if isinstance(item, dict)   # skip any non-dict entries gracefully
    ]

    # ── Update state with clean, formatted results ─────────────────────────
    state["results"] = formatted

    return state
