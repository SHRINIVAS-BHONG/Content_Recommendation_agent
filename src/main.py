from langgraph.graph import StateGraph
from graph import AgentState

from nodes.process import process_query
from nodes.reasoning import reasoning_node
from nodes.recommend import recommendation_node
from nodes.output import output_node


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("process", process_query)
    builder.add_node("reason", reasoning_node)
    builder.add_node("recommend", recommendation_node)
    builder.add_node("output", output_node)

    builder.set_entry_point("process")

    builder.add_edge("process", "reason")
    builder.add_edge("reason", "recommend")
    builder.add_edge("recommend", "output")

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()

    query = input("Enter your query: ")
    graph.invoke({"query": query})