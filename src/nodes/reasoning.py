def reasoning_node(state):
    query = state["query"]

    if "study" in query or "learn" in query:
        category = "educational"
    elif "fun" in query or "movie" in query:
        category = "entertainment"
    else:
        category = "general"

    return {"category": category}