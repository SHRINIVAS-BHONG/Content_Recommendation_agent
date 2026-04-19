"""
main.py — FastAPI entry point for the AI recommendation agent.

Run with:
    cd project_root
    uvicorn backend.main:app --reload

The agent pipeline:
    POST /recommend  →  process → reasoning → recommend → evaluate → (refine →)* output
                                                 ↑
                                    loads anime_recommender.pkl
                                    or  manga_recommender.pkl
                                    (from data/ directory, once at startup)

The response includes:
  - results          : ranked recommendation items (with optional similarity_score / match_reason)
  - reasoning_trace  : list of human-readable strings explaining each decision made by the agent
  - refinement_count : number of quality-evaluation refinement cycles that occurred (0–2)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from backend.agent.graph import run_agent

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Anime / Manga Recommendation Agent",
    description=(
        "LangGraph-powered agent that processes a user query through a dynamic "
        "pipeline: process → (simple|deep)_reasoning → recommend → evaluate → "
        "(refine →)* output. Returns ranked recommendations from a pre-trained "
        "ML model together with a reasoning trace and refinement count."
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
    title:            str
    image:            str
    synopsis:         str
    score:            float
    genres:           List[str]
    similarity_score: Optional[float] = None
    match_reason:     Optional[str]   = None


class RecommendationResponse(BaseModel):
    results:          List[RecommendationItem]
    reasoning_trace:  List[str]
    refinement_count: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint - Welcome message and API information."""
    return {
        "message": "🎌 Anime & Manga Recommendation API",
        "description": "LangGraph-powered recommendation system with AI agent pipeline",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "recommend": "POST /recommend"
        },
        "example_usage": {
            "endpoint": "POST /recommend",
            "body": {"query": "dark anime like Death Note"},
            "description": "Get personalized anime/manga recommendations"
        },
        "features": [
            "7-node AI agent pipeline",
            "26,000+ anime & manga database", 
            "Reasoning trace explanations",
            "Automatic query refinement",
            "Content-based filtering"
        ]
    }

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(body: QueryRequest):
    """
    Run the full agent pipeline on a user query.

    The pipeline:
      1. process        — parse query into tags, type, reference, intent,
                          complexity_score, and semantic_hints
      2. routing        — route to simple_reasoning_node (complexity < 0.5)
                          or deep_reasoning_node (complexity >= 0.5)
      3. reasoning      — expand tags, build enriched model_input payload
      4. recommend      — call pre-trained .pkl model with enriched data
      5. evaluator      — assess coverage, diversity, and avg_score
      6. refine (0–2×)  — adjust model_input and retry if quality is low
      7. output         — normalise results to frontend schema

    Request:
        { "query": "dark anime like Death Note" }

    Response:
        {
          "results": [
            {
              "title": "...", "image": "...", "synopsis": "...",
              "score": 9.1, "genres": ["..."],
              "similarity_score": 0.87,   // optional, present when model provides it
              "match_reason": "..."        // optional, present when model provides it
            }
          ],
          "reasoning_trace":  ["Step 1: ...", "Step 2: ..."],
          "refinement_count": 0
        }

    Errors:
        400 — empty query
        404 — model .pkl file not found at expected path
        500 — model failed to produce results
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        output = run_agent(query=body.query)
        return RecommendationResponse(
            results=output.get("results", []),
            reasoning_trace=output.get("reasoning_trace", []),
            refinement_count=output.get("refinement_count", 0),
        )

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
