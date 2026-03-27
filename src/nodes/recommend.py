import json

def recommendation_node(state):
    with open("data/content_db.json") as f:
        db = json.load(f)

    category = state["category"]
    return {"result": db.get(category, [])}