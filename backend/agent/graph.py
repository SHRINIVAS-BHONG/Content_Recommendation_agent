"""
graph.py — LangGraph execution graph for the recommendation agent.

Defines the node pipeline and wires them together with conditional routing
and a bounded refinement loop:

    process_node
        ↓ route_query (conditional)
    simple_reasoning_node  OR  deep_reasoning_node
        ↓ (both converge)
    recommend_node
        ↓
    evaluator_node
        ↓ route_evaluation (conditional)
    refine_node  →  recommend_node  (loop, max 2 times)
        OR
    output_node  →  END

Usage (from FastAPI or any entry point):

    from agent.graph import run_agent

    result = run_agent("dark anime like Attack on Titan")
    print(result["results"])
    print(result["reasoning_trace"])
    print(result["refinement_count"])
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState, initial_state
from typing import Optional, Dict, Any, List
from agent.nodes.process   import process_node
from agent.nodes.reasoning import simple_reasoning_node, deep_reasoning_node
from agent.nodes.recommend import recommend_node
from agent.nodes.evaluator import evaluator_node
from agent.nodes.refine    import refine_node
from agent.nodes.output    import output_node

# Build once at import time — graph structure never changes at runtime
_compiled_graph = None


# ── Conditional edge functions ────────────────────────────────────────────────

def route_query(state: AgentState) -> str:
    """
    Route after process_node based on query complexity.

    Returns:
        "deep_reasoning_node"   when complexity_score >= 0.5
        "simple_reasoning_node" otherwise
    """
    if state["complexity_score"] >= 0.5:
        return "deep_reasoning_node"
    return "simple_reasoning_node"


def route_evaluation(state: AgentState) -> str:
    """
    Route after evaluator_node: accept results or trigger a refinement cycle.

    Returns:
        "refine_node"  when verdict == "needs_refinement" AND refinement_count < 2
        "output_node"  in all other cases
    """
    verdict = state["quality_report"].get("verdict", "quality_ok")
    if verdict == "needs_refinement" and state["refinement_count"] < 2:
        return "refine_node"
    return "output_node"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for the agent pipeline.

    Nodes are registered and wired with conditional edges for complexity-based
    routing and a bounded quality-evaluation refinement loop.

    Returns:
        A compiled LangGraph runnable — call .invoke(state) on it.
    """
    graph = StateGraph(AgentState)

    # ── Register all seven nodes ───────────────────────────────────────────
    graph.add_node("process_node",          process_node)
    graph.add_node("simple_reasoning_node", simple_reasoning_node)
    graph.add_node("deep_reasoning_node",   deep_reasoning_node)
    graph.add_node("recommend_node",        recommend_node)
    graph.add_node("evaluator_node",        evaluator_node)
    graph.add_node("refine_node",           refine_node)
    graph.add_node("output_node",           output_node)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("process_node")

    # ── Conditional routing: simple vs. complex reasoning ──────────────────
    graph.add_conditional_edges(
        "process_node",
        route_query,
        {
            "simple_reasoning_node": "simple_reasoning_node",
            "deep_reasoning_node":   "deep_reasoning_node",
        }
    )

    # ── Both reasoning paths feed into the model ───────────────────────────
    graph.add_edge("simple_reasoning_node", "recommend_node")
    graph.add_edge("deep_reasoning_node",   "recommend_node")
    graph.add_edge("recommend_node",        "evaluator_node")

    # ── Conditional routing: accept or refine loop ─────────────────────────
    graph.add_conditional_edges(
        "evaluator_node",
        route_evaluation,
        {
            "refine_node": "refine_node",
            "output_node": "output_node",
        }
    )

    # ── Refinement loops back to the model (tags already expanded) ─────────
    graph.add_edge("refine_node", "recommend_node")
    graph.add_edge("output_node", END)

    return graph.compile()


def _get_graph():
    """Return the singleton compiled graph, building it once if needed."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(
    query: str,
    user_id: Optional[str] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    user_feedback_history: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """
    Run the full agent pipeline for a user query.

    Supports both authenticated and anonymous modes. When user_id is provided
    the pipeline receives user context for personalised recommendations.

    Args:
        query: Raw user input string (e.g. "dark anime like Death Note").
        user_id: Optional authenticated user UUID string.
        user_preferences: Optional dict with preferred/avoided genres and
                          content type preferences.
        conversation_context: Optional dict with recent queries and results.
        user_feedback_history: Optional list of past feedback records.

    Returns:
        {
            "results":          list of normalised recommendation dicts,
            "reasoning_trace":  list of human-readable decision strings,
            "refinement_count": number of refinement cycles that occurred,
            "is_authenticated": bool indicating whether user context was used,
        }

    Raises:
        ValueError        : If query is empty or whitespace-only.
        FileNotFoundError : If the .pkl model file is missing.
        RuntimeError      : If any node in the pipeline fails.
    """
    if not query or not query.strip():
        raise ValueError("run_agent: query must be a non-empty string.")

    graph  = _get_graph()
    state  = initial_state(
        query=query,
        user_id=user_id,
        user_preferences=user_preferences,
        conversation_context=conversation_context,
        user_feedback_history=user_feedback_history,
    )
    result = graph.invoke(state)

    return {
        "results":          result.get("results", []),
        "reasoning_trace":  result.get("reasoning_trace", []),
        "refinement_count": result.get("refinement_count", 0),
        "is_authenticated": result.get("is_authenticated", False),
    }
