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
  - page             : the page number returned (0-indexed)
  - has_more         : whether more results are available on the next page
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    page: int = 0   # 0-indexed page number for Load More

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "dark anime like Attack on Titan", "page": 0},
                {"query": "romance manga like Fruits Basket", "page": 1},
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
    page:             int
    has_more:         bool


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
            "Content-based filtering",
            "Paginated Load More support",
        ]
    }


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(body: QueryRequest):
    """
    Run the full agent pipeline on a user query.

    Supports pagination via the `page` field (0-indexed). Page 0 runs the
    full agent pipeline; subsequent pages re-run with the page offset so
    the model returns the next batch of 10 deduplicated results.

    Request:
        { "query": "dark anime like Death Note", "page": 0 }

    Response:
        {
          "results": [...],           // 10 items for this page
          "reasoning_trace": [...],
          "refinement_count": 0,
          "page": 0,
          "has_more": true            // false when no more results exist
        }

    Errors:
        400 — empty query
        404 — model .pkl file not found at expected path
        500 — model failed to produce results
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        output = run_agent(query=body.query, page=body.page)
        results = output.get("results", [])
        return RecommendationResponse(
            results=results,
            reasoning_trace=output.get("reasoning_trace", []),
            refinement_count=output.get("refinement_count", 0),
            page=body.page,
            has_more=len(results) == 10,  # if we got a full page, assume more exist
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
