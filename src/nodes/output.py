def output_node(state):
    print("\nRecommended Content:")
    for item in state["result"]:
        print("-", item)
    return state