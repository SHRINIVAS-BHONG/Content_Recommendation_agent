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

from agent.state import AgentState

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
        → model.recommend(payload) → raw results list → state["results"]

    When state["model_input"] is non-empty, the full enriched payload built
    by the reasoning node is passed directly to model.recommend().  When it
    is empty (e.g. the node was called without a reasoning step), the node
    falls back to the legacy {tags, type, reference} format for backwards
    compatibility.

    Args:
        state: AgentState after reasoning node has enriched the tags.

    Returns:
        Updated AgentState with state["results"] containing raw result dicts.

    Raises:
        FileNotFoundError : Model .pkl file is missing.
        RuntimeError      : Model file is corrupt or recommend() call fails.
        AttributeError    : Loaded object does not have a .recommend() method.
    """
    media_type: str = state.get("type", "anime")
    tags: List[str] = state.get("tags", [])
    reference: str  = state.get("reference", "")
    model_input: Dict[str, Any] = state.get("model_input", {})

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
    state["reasoning_trace"] = reasoning_trace

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

    # ── Validate and store results ─────────────────────────────────────────
    if not isinstance(raw_results, list):
        raise RuntimeError(
            f"recommend_node: model.recommend() must return a list, "
            f"got {type(raw_results)} instead."
        )

    state["results"] = raw_results
    return state
