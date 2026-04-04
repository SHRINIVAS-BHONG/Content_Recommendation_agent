"""
main.py — FastAPI entry point for the AI recommendation agent.

Run with:
    cd project_root
    uvicorn backend.main:app --reload

The agent pipeline:
    POST /recommend  →  process → reasoning → recommend → output
                                                 ↑
                                    loads anime_recommender.pkl
                                    or  manga_recommender.pkl
                                    (from data/ directory, once at startup)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from agent.graph import run_agent

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Anime / Manga Recommendation Agent",
    description=(
        "LangGraph-powered agent that processes a user query through "
        "process → reasoning → recommend → output nodes and returns "
        "ranked recommendations from a pre-trained ML model."
    ),
    version="2.0.0",
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "dark anime like Attack on Titan"},
                {"query": "romance manga like Fruits Basket"},
            ]
        }
    }


class RecommendationItem(BaseModel):
    title:    str
    image:    str
    synopsis: str
    score:    float
    genres:   List[str]


class RecommendationResponse(BaseModel):
    results: List[RecommendationItem]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(body: QueryRequest):
    """
    Run the full agent pipeline on a user query.

    The pipeline:
      1. process   — parse query into tags, type, reference
      2. reasoning — expand tags using dataset co-occurrence
      3. recommend — call pre-trained .pkl model with enriched data
      4. output    — normalise results to frontend schema

    Request:
        { "query": "dark anime like Death Note" }

    Response:
        { "results": [ { "title": "...", "image": "...", "synopsis": "...",
                          "score": 9.1, "genres": ["..."] } ] }

    Errors:
        400 — empty query
        404 — model .pkl file not found at expected path
        500 — model failed to produce results
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        output = run_agent(query=body.query)
        return output

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health_check():
    """Liveness probe — returns 200 if the service is up."""
    return {"status": "ok", "service": "recommendation-agent"}
