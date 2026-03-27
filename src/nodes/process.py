def process_query(state):
    query = state["query"].lower()
    return {"query": query}