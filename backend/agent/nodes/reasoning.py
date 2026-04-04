"""
nodes/reasoning.py — Node 2: Enrich and expand query tags using real dataset signal.

Responsibilities:
    - Expand base tags via dataset co-occurrence (tag_mapper.expand_tags)
    - If a reference title exists → fetch its actual tags from the dataset
    - Merge everything and deduplicate

Reads  : state["tags"], state["type"], state["reference"]
Updates: state["tags"]

Uses: services/tag_mapper.py
"""

from agent.state import AgentState
from agent.services.tag_mapper import expand_tags, get_reference_tags
from agent.utils.helpers import deduplicate


def reasoning_node(state: AgentState) -> AgentState:
    """
    Enrich the tag list using co-occurrence data from the real dataset.

    Flow:
        state["tags"]      → expand_tags(tags, media_type)  → dataset-grounded tags
        state["reference"] → get_reference_tags(ref, type)  → tags of the ref title
        combined           → deduplicate                     → state["tags"]

    The media_type is forwarded to tag_mapper so it queries the correct
    dataset index (anime vs manga).

    Args:
        state: AgentState with tags, type, and reference already populated
               by the process node.

    Returns:
        Updated AgentState with state["tags"] enriched and de-duplicated.
    """
    current_tags: list = state.get("tags", [])
    media_type: str    = state.get("type", "anime")
    reference: str     = state.get("reference", "")

    # ── Step 1: Expand base tags using dataset co-occurrence ───────────────
    # tag_mapper reads the real JSON dataset and finds tags that frequently
    # appear together — no static dictionary involved.
    expanded = expand_tags(current_tags, media_type=media_type)

    # ── Step 2: Enrich from reference title (if one was extracted) ─────────
    # Looks up the actual entry in the dataset and pulls its tags/genres.
    # Example: "like Fullmetal Alchemist" → ["drama", "action", "military", ...]
    if reference:
        ref_tags = get_reference_tags(reference, media_type=media_type)
        expanded.extend(ref_tags)

    # ── Step 3: Deduplicate while preserving order ─────────────────────────
    state["tags"] = deduplicate(expanded)

    return state
