"""
nodes/output.py — Node 4: Format raw ML results into the frontend contract.

Responsibilities:
    - Normalize each raw result dict into the expected output schema
    - Guarantee all required fields are present with safe defaults
    - Attach reasoning_trace and refinement_count to the final response
    - Return the final, clean results list

Reads  : state["results"], state["reasoning_trace"], state["refinement_count"]
Updates: state["results"]  (replaces raw with formatted)

Uses: utils/helpers.py → normalize_result()

Output schema per result:
    {
        "title"            : str,
        "image"            : str   (URL),
        "synopsis"         : str,
        "score"            : float,
        "genres"           : list[str],
        "similarity_score" : float | None  (passed through if model provides it),
        "match_reason"     : str | None    (passed through if model provides it),
    }
"""

from typing import List, Dict, Any
from backend.agent.state import AgentState
from backend.utils.helpers import normalize_result


def output_node(state: AgentState) -> AgentState:
    """
    Transform raw ML results into the standardized frontend response format.

    Flow:
        state["results"] (raw) → normalize each item → state["results"] (clean)

    Also preserves state["reasoning_trace"] and state["refinement_count"] so
    the caller can return them alongside results.

    Args:
        state: Current AgentState (expects state["results"] to be a list).

    Returns:
        Updated AgentState where:
            - state["results"] contains normalized dicts
            - state["reasoning_trace"] is preserved from input
            - state["refinement_count"] is preserved from input
    """
    raw_results: List[Any] = state.get("results", [])

    # ── Normalize every result to match the frontend schema ────────────────
    formatted: List[Dict[str, Any]] = [
        normalize_result(item)
        for item in raw_results
        if isinstance(item, dict)   # skip any non-dict entries gracefully
    ]

    # ── Update state with clean, formatted results ─────────────────────────
    # reasoning_trace and refinement_count are already in state; preserve them.
    state["results"] = formatted

    return state
