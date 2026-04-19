"""
nodes/recommend.py — Node 3: Load the pre-trained model and get recommendations.

Responsibilities:
    - Load the correct .pkl model (anime or manga) exactly once at startup
    - Build a clean input payload from the enriched state
    - Call model.recommend(payload) and store raw results in state

This node does NOT contain any ML logic. It is a thin, typed adapter between
the agent state and whatever interface your pre-trained model exposes.

Reads  : state["tags"], state["type"], state["reference"]
Updates: state["results"]

Model contract
--------------
Your .pkl file must be a joblib/pickle-serialised object that exposes:

    model.recommend(input_data: dict) -> list[dict]

Where input_data has the shape:
    {
        "tags"      : list[str],   # enriched tag list from reasoning node
        "type"      : str,         # "anime" or "manga"
        "reference" : str          # reference title or ""
    }

And each returned dict should contain at minimum:
    { "title": str, "image": str, "synopsis": str, "score": float, "genres": list }

Model paths
-----------
Place your model files at:
    data/anime_recommender.pkl
    data/manga_recommender.pkl

Or override via environment variables:
    ANIME_MODEL_PATH=path/to/anime_model.pkl
    MANGA_MODEL_PATH=path/to/manga_model.pkl
"""

import os
import joblib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from backend.agent.state import AgentState

# ── Model file paths ──────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[3]   # project root

_ANIME_MODEL_PATH = Path(
    os.environ.get("ANIME_MODEL_PATH", _BASE_DIR / "data" / "anime_recommender.pkl")
)
_MANGA_MODEL_PATH = Path(
    os.environ.get("MANGA_MODEL_PATH", _BASE_DIR / "data" / "manga_recommender.pkl")
)


# ── Model loader ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=2)   # load each model at most once per process lifetime
def _load_model(model_path: str) -> Any:
    """
    Load a joblib/pickle model from disk. Cached so the file is read only once.

    Args:
        model_path: Absolute path to the .pkl file (str, not Path, for hashability).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError : If the .pkl file does not exist at the given path.
        RuntimeError      : If joblib cannot deserialise the file.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"recommend_node: Model file not found at '{path}'.\n"
            f"Expected a joblib/pickle file. Place your trained model there or "
            f"set the ANIME_MODEL_PATH / MANGA_MODEL_PATH environment variables."
        )

    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"recommend_node: Failed to load model from '{path}'. "
            f"Ensure the file is a valid joblib/pickle object.\nError: {exc}"
        ) from exc


def _get_model(media_type: str) -> Any:
    """Return the correct pre-trained model for anime or manga."""
    if media_type == "manga":
        return _load_model(str(_MANGA_MODEL_PATH))
    return _load_model(str(_ANIME_MODEL_PATH))


# ── Node ──────────────────────────────────────────────────────────────────────

def recommend_node(state: AgentState) -> AgentState:
    """
    Call the pre-trained recommender model with the enriched state data.

    Flow:
        state → choose payload (enriched or legacy) → load model (cached)
        → model.recommend(payload) → apply personalization weights
        → deduplicate against user history → state["results"]

    When state["model_input"] is non-empty, the full enriched payload built
    by the reasoning node is passed directly to model.recommend().  When it
    is empty (e.g. the node was called without a reasoning step), the node
    falls back to the legacy {tags, type, reference} format for backwards
    compatibility.

    For authenticated users, personalization weights are applied to boost
    or suppress recommendations based on user preferences, and deduplication
    is performed against recent interaction history.

    Args:
        state: AgentState after reasoning node has enriched the tags.

    Returns:
        Updated AgentState with state["results"] containing personalized
        and deduplicated result dicts.

    Raises:
        FileNotFoundError : Model .pkl file is missing.
        RuntimeError      : Model file is corrupt or recommend() call fails.
        AttributeError    : Loaded object does not have a .recommend() method.
    """
    media_type: str = state.get("type", "anime")
    tags: List[str] = state.get("tags", [])
    reference: str  = state.get("reference", "")
    model_input: Dict[str, Any] = state.get("model_input", {})
    
    # User context fields
    is_authenticated: bool = state.get("is_authenticated", False)
    personalization_weights: Dict[str, float] = state.get("personalization_weights") or {}
    conversation_context: Dict[str, Any] = state.get("conversation_context") or {}

    # ── Choose the input payload ───────────────────────────────────────────
    # Use the enriched model_input built by the reasoning node when available;
    # fall back to the legacy flat payload for backwards compatibility.
    if model_input:
        input_data = model_input
        payload_format = "enriched"
    else:
        input_data = {
            "tags":      tags,
            "type":      media_type,
            "reference": reference,
        }
        payload_format = "legacy"

    # ── Append reasoning trace entry ───────────────────────────────────────
    reasoning_trace: List[str] = state.get("reasoning_trace", [])
    reasoning_trace.append(
        f"recommend_node: using {payload_format} payload format "
        f"({'model_input was populated' if payload_format == 'enriched' else 'model_input was empty — falling back to legacy {tags, type, reference} format'})."
    )

    # ── Load model (first call reads disk; subsequent calls use lru_cache) ─
    model = _get_model(media_type)

    # ── Validate the model exposes the expected interface ──────────────────
    if not hasattr(model, "recommend"):
        raise AttributeError(
            f"recommend_node: Loaded model does not have a .recommend() method. "
            f"Got type: {type(model)}. Ensure your pkl exports an object with "
            f"a recommend(input_data: dict) -> list method."
        )

    # ── Call the model ─────────────────────────────────────────────────────
    try:
        raw_results = model.recommend(input_data)
    except Exception as exc:
        raise RuntimeError(
            f"recommend_node: model.recommend() raised an error.\n"
            f"Input was: {input_data}\nError: {exc}"
        ) from exc

    # ── Validate results ───────────────────────────────────────────────────
    if not isinstance(raw_results, list):
        raise RuntimeError(
            f"recommend_node: model.recommend() must return a list, "
            f"got {type(raw_results)} instead."
        )

    # ── Apply personalization for authenticated users ──────────────────────
    personalized_results = raw_results
    if is_authenticated and personalization_weights:
        personalized_results = _apply_personalization_weights(
            raw_results, personalization_weights
        )
        reasoning_trace.append(
            f"recommend_node: Applied personalization weights to {len(raw_results)} results "
            f"using {len(personalization_weights)} genre weights."
        )

    # ── Deduplicate against user history ───────────────────────────────────
    final_results = personalized_results
    if is_authenticated and conversation_context:
        recent_results = conversation_context.get("recent_results", [])
        if recent_results:
            final_results = _deduplicate_against_history(
                personalized_results, recent_results
            )
            deduped_count = len(personalized_results) - len(final_results)
            reasoning_trace.append(
                f"recommend_node: Deduplicated {deduped_count} results against "
                f"{len(recent_results)} items in user history."
            )

    state["results"] = final_results
    state["reasoning_trace"] = reasoning_trace
    return state


def _apply_personalization_weights(
    results: List[Dict[str, Any]], 
    weights: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Apply personalization weights to recommendation results.
    
    Boosts or suppresses results based on genre preferences.
    
    Args:
        results: List of recommendation dicts with 'genres' and 'score' fields
        weights: Dict mapping genre names to weight multipliers
    
    Returns:
        List of results with adjusted scores, sorted by new score
    """
    weighted_results = []
    
    for result in results:
        # Make a copy to avoid modifying the original
        weighted_result = result.copy()
        original_score = float(result.get("score", 0.0))
        
        # Calculate weighted score based on genres
        genres = result.get("genres", [])
        if isinstance(genres, str):
            genres = [genres]
        
        # Find the maximum weight for any genre in this result
        max_weight = 1.0  # Default weight
        for genre in genres:
            if genre in weights:
                max_weight = max(max_weight, weights[genre])
        
        # Apply the weight
        weighted_score = original_score * max_weight
        weighted_result["score"] = weighted_score
        weighted_result["original_score"] = original_score
        weighted_result["personalization_weight"] = max_weight
        
        weighted_results.append(weighted_result)
    
    # Sort by weighted score (descending)
    weighted_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    return weighted_results


def _deduplicate_against_history(
    results: List[Dict[str, Any]], 
    recent_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove results that were recently recommended to the user.
    
    Args:
        results: Current recommendation results
        recent_results: List of recently recommended items
    
    Returns:
        Filtered results with duplicates removed
    """
    # Extract titles from recent results for comparison
    recent_titles = set()
    for item in recent_results:
        title = item.get("title", "")
        if title:
            recent_titles.add(title.lower().strip())
    
    # Filter out results that match recent titles
    deduplicated = []
    for result in results:
        title = result.get("title", "")
        if title and title.lower().strip() not in recent_titles:
            deduplicated.append(result)
    
    return deduplicated
