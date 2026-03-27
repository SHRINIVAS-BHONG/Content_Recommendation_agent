from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    category: str
    result: List[str]