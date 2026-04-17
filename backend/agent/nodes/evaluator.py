"""
nodes/evaluator.py — Evaluator node: quality assessment of recommendation results.

Responsibilities:
    evaluator_node:
        - Call quality_evaluator.evaluate_results to compute quality metrics
        - Store the full quality report in state["quality_report"]
        - Append at least one entry to state["reasoning_trace"] describing
          the verdict and key metrics (coverage, avg_score, result_count)

Reads  : state["results"], state["tags"], state["refinement_count"]
Updates: state["quality_report"], state["reasoning_trace"]

Uses: services/quality_evaluator.py
"""

from agent.state import AgentState
from services import quality_evaluator


def evaluator_node(state: AgentState) -> AgentState:
    """
    Self-reflection step: evaluate the quality of state["results"] and
    decide whether to accept or trigger a refinement cycle.

    Flow:
        1. Call quality_evaluator.evaluate_results with results, tags,
           and refinement_count.
        2. Store the returned quality report in state["quality_report"].
        3. Append a trace entry describing the verdict and key metrics.

    Args:
        state: AgentState with results, tags, and refinement_count already
               populated by recommend_node.

    Returns:
        Updated AgentState with quality_report set and reasoning_trace
        extended by at least one entry.
    """
    results: list = state.get("results", [])
    tags: list = state.get("tags", [])
    refinement_count: int = state.get("refinement_count", 0)
    reasoning_trace: list = list(state.get("reasoning_trace", []))

    # ── Step 1: Compute quality metrics ───────────────────────────────────
    report = quality_evaluator.evaluate_results(results, tags, refinement_count)

    # ── Step 2: Append trace entry with verdict and key metrics ───────────
    reasoning_trace.append(
        f"[evaluator_node] verdict={report['verdict']!r} | "
        f"result_count={report['result_count']} | "
        f"coverage={report['coverage']:.2f} | "
        f"avg_score={report['avg_score']:.2f} | "
        f"refinement_count={refinement_count}"
    )

    # ── Step 3: Update state ───────────────────────────────────────────────
    state["quality_report"] = report
    state["reasoning_trace"] = reasoning_trace

    return state
