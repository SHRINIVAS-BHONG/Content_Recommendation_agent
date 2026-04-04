"""
graph.py — LangGraph execution graph for the recommendation agent.

Defines the node pipeline and wires them together:

    process → reasoning → recommend → output

The recommend node self-loads its pre-trained .pkl model — no external
injection needed. Just call run_agent(query) and the graph handles everything.

Usage (from FastAPI or any entry point):

    from agent.graph import run_agent

    result = run_agent("dark anime like Attack on Titan")
    print(result["results"])
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState, initial_state
from agent.nodes.process   import process_node
from agent.nodes.reasoning import reasoning_node
from agent.nodes.recommend import recommend_node
from agent.nodes.output    import output_node

# Build once at import time — graph structure never changes at runtime
_compiled_graph = None


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph for the agent pipeline.

    Nodes are registered in execution order. Each node:
        1. Receives the shared AgentState
        2. Performs its focused responsibility
        3. Returns the updated AgentState

    Returns:
        A compiled LangGraph runnable — call .invoke({"query": ...}) on it.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes in pipeline order ──────────────────────────────────
    graph.add_node("process",   process_node)    # parse raw query
    graph.add_node("reasoning", reasoning_node)  # expand tags from dataset
    graph.add_node("recommend", recommend_node)  # call pre-trained .pkl model
    graph.add_node("output",    output_node)     # format for frontend

    # ── Define execution edges ─────────────────────────────────────────────
    graph.set_entry_point("process")
    graph.add_edge("process",   "reasoning")
    graph.add_edge("reasoning", "recommend")
    graph.add_edge("recommend", "output")
    graph.add_edge("output",    END)

    return graph.compile()


def _get_graph():
    """Return the singleton compiled graph, building it once if needed."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(query: str) -> dict:
    """
    Run the full agent pipeline for a user query.

    Args:
        query: Raw user input string (e.g. "dark anime like Death Note").

    Returns:
        {"results": [ {"title": ..., "image": ..., "synopsis": ...,
                        "score": float, "genres": [...]} ]}

    Raises:
        ValueError        : If query is empty.
        FileNotFoundError : If the .pkl model file is missing.
        RuntimeError      : If any node in the pipeline fails.
    """
    if not query or not query.strip():
        raise ValueError("run_agent: query must be a non-empty string.")

    graph  = _get_graph()
    state  = initial_state(query)
    result = graph.invoke(state)

    return {"results": result.get("results", [])}
