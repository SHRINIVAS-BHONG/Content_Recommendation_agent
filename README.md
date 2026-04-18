# Anime & Manga Recommendation Agent

> **An intelligent LangGraph-powered recommendation system that transforms natural language queries into ranked anime/manga recommendations through a dynamic 7-node pipeline with self-refinement and full explainability.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0+-orange.svg)](https://github.com/langchain-ai/langgraph)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Query Understanding](#-query-understanding)
- [Quality Evaluation & Refinement](#-quality-evaluation--refinement)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

This recommendation agent processes complex natural language queries like:

> **"dark anime like Attack on Titan with strong character development"**

Through a sophisticated pipeline that:

- ✅ **Understands Intent**: Classifies queries into 5 categories (`find_similar`, `mood_search`, `character_search`, `complex_search`, `genre_search`)
- ✅ **Routes Dynamically**: Complexity-based routing (threshold: 0.5) between simple and deep reasoning paths
- ✅ **Self-Refines**: Automatic quality evaluation with up to 2 refinement cycles
- ✅ **Explains Everything**: Complete reasoning trace for every decision
- ✅ **Grounds in Data**: Tag expansion uses co-occurrence patterns from actual datasets

### Key Features

| Feature | Description |
|---------|-------------|
| **7-Node Pipeline** | `process → (simple\|deep)_reasoning → recommend → evaluator → (refine →)* output` |
| **Conditional Routing** | 2 decision points: complexity-based + quality-based |
| **Bounded Refinement** | Max 2 cycles with 3 strategies (coverage, score, broadening) |
| **Dataset-Aware** | Co-occurrence index built from `refined_anime_dataset.json` & `refined_manga_dataset.json` |
| **Model Agnostic** | Works with any `.pkl` model exposing `recommend(dict) -> list[dict]` |

---

## 🏗️ Architecture

### Pipeline Flow Diagram

```
                           ┌─────────────┐
                           │   START     │
                           └──────┬──────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │ process_node   │
                         └────────┬───────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  route_query     │
                         │ complexity>=0.5? │
                         └────┬──────┬──────┘
                              │      │
                         No   │      │   Yes
                              │      │
                 ┌────────────┘      └────────────┐
                 │                                │
                 ▼                                ▼
          ┌──────────────┐                  ┌──────────────┐
          │   simple     │                  │    deep      │
          │  reasoning   │                  │  reasoning   │
          └──────┬───────┘                  └──────┬───────┘
                 │                                 │
                 └────────────┬────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   recommend     │◄──────────┐
                    │     (model)     │           │
                    └─────────┬───────┘           │
                              │                   │
                              ▼                   │
                    ┌─────────────────┐           │
                    │   evaluator     │           │
                    └─────────┬───────┘           │
                              │                   │
                              ▼                   │
                    ┌─────────────────┐           │
                    │route_evaluation │           │
                    │  needs_refine & │           │
                    │    count < 2?   │           │
                    └────┬──────┬─────┘           │
                         │      │                 │
                    Yes  │      │  No             │
                         │      │                 │
                         ▼      ▼                 │
                  ┌──────────┐ ┌──────────┐       │
                  │  refine  │ │  output  │       │
                  │ count++  │ └────┬─────┘       │
                  └────┬─────┘      │             │
                       │            ▼             │
                       │         ┌─────┐          │
                       │         │ END │          │
                       │         └─────┘          │
                       │                          │
                       └──────────────────────────┘
                         LOOP BACK (max 2×)
```

### Architecture Overview

**7-Node Pipeline**: `process` → `(simple|deep)_reasoning` → `recommend` → `evaluator` → `(refine)` → `output`

**2 Conditional Routes**:
- **Route 1**: `route_query()` - Complexity-based (threshold: 0.5)
- **Route 2**: `route_evaluation()` - Quality-based (verdict + count < 2)

**1 Bounded Refinement Loop**: `refine` → `recommend` (max 2 iterations)

**Shared State**: 15-field `AgentState` TypedDict passed through all nodes

### Node Responsibilities

<table>
<thead>
<tr>
<th>Node</th>
<th>Reads From State</th>
<th>Writes To State</th>
<th>Core Logic</th>
<th>Service/Module Used</th>
</tr>
</thead>
<tbody>

<tr>
<td><strong>process_node</strong></td>
<td>
• <code>query</code>
</td>
<td>
• <code>tags</code><br>
• <code>type</code><br>
• <code>reference</code><br>
• <code>intent</code><br>
• <code>complexity_score</code><br>
• <code>semantic_hints</code>
</td>
<td>
Parses raw query into 6 structured fields using rule-based NLP
</td>
<td>
<code>services/query_parser.py</code><br>
→ 6 extraction functions
</td>
</tr>

<tr>
<td><strong>simple_reasoning_node</strong></td>
<td>
• <code>tags</code><br>
• <code>type</code><br>
• <code>reference</code><br>
• <code>intent</code><br>
• <code>semantic_hints</code><br>
• <code>reference_synopsis</code><br>
• <code>reasoning_trace</code>
</td>
<td>
• <code>tags</code> (expanded)<br>
• <code>search_strategy</code> = <code>"tag_only"</code><br>
• <code>model_input</code><br>
• <code>reasoning_trace</code> (appended)
</td>
<td>
<strong>Fast path for simple queries:</strong><br>
1. Expand tags via co-occurrence<br>
2. Merge reference tags if present<br>
3. Deduplicate<br>
4. Build model_input with 8 keys
</td>
<td>
<code>services/tag_mapper.py</code><br>
→ <code>expand_tags()</code><br>
→ <code>get_reference_tags()</code><br>
<code>utils/helpers.py</code><br>
→ <code>deduplicate()</code>
</td>
</tr>

<tr>
<td><strong>deep_reasoning_node</strong></td>
<td>
• <code>tags</code><br>
• <code>type</code><br>
• <code>reference</code><br>
• <code>intent</code><br>
• <code>semantic_hints</code><br>
• <code>reference_synopsis</code><br>
• <code>reasoning_trace</code>
</td>
<td>
• <code>tags</code> (expanded)<br>
• <code>search_strategy</code> (intent-mapped)<br>
• <code>reference_synopsis</code> (looked up)<br>
• <code>model_input</code><br>
• <code>reasoning_trace</code> (appended 4×)
</td>
<td>
<strong>Full analysis for complex queries:</strong><br>
1. Expand tags via co-occurrence<br>
2. Translate semantic hints → tags<br>
3. Lookup reference synopsis from dataset<br>
4. Map intent → search_strategy<br>
5. Deduplicate<br>
6. Build model_input with 8 keys
</td>
<td>
<code>services/tag_mapper.py</code><br>
→ <code>expand_tags()</code><br>
→ <code>_load_dataset()</code><br>
<code>utils/helpers.py</code><br>
→ <code>deduplicate()</code><br>
<strong>Hint mapping:</strong><br>
→ <code>_HINT_TO_TAGS</code> dict<br>
<strong>Strategy mapping:</strong><br>
→ <code>_INTENT_TO_STRATEGY</code> dict
</td>
</tr>

<tr>
<td><strong>recommend_node</strong></td>
<td>
• <code>model_input</code><br>
• <code>tags</code> (fallback)<br>
• <code>type</code> (fallback)<br>
• <code>reference</code> (fallback)<br>
• <code>reasoning_trace</code>
</td>
<td>
• <code>results</code><br>
• <code>reasoning_trace</code> (appended)
</td>
<td>
<strong>Model invocation:</strong><br>
1. Choose payload (enriched or legacy)<br>
2. Load .pkl model (cached via <code>@lru_cache</code>)<br>
3. Call <code>model.recommend(payload)</code><br>
4. Validate return type is <code>list</code><br>
5. Store raw results
</td>
<td>
<code>joblib</code> (model loading)<br>
<code>functools.lru_cache</code> (caching)<br>
<strong>Model paths:</strong><br>
→ <code>data/anime_recommender.pkl</code><br>
→ <code>data/manga_recommender.pkl</code>
</td>
</tr>

<tr>
<td><strong>evaluator_node</strong></td>
<td>
• <code>results</code><br>
• <code>tags</code><br>
• <code>refinement_count</code><br>
• <code>reasoning_trace</code>
</td>
<td>
• <code>quality_report</code><br>
• <code>reasoning_trace</code> (appended)
</td>
<td>
<strong>Quality assessment:</strong><br>
Computes 6 metrics:<br>
1. <code>coverage</code> (0.0-1.0)<br>
2. <code>diversity</code> (int)<br>
3. <code>avg_score</code> (float)<br>
4. <code>result_count</code> (int)<br>
5. <code>verdict</code> (str)<br>
6. <code>uncovered_tags</code> (list)
</td>
<td>
<code>services/quality_evaluator.py</code><br>
→ <code>evaluate_results()</code><br>
<strong>Verdict logic:</strong><br>
→ <code>result_count < 3</code><br>
→ <code>coverage < 0.4 & count < 2</code><br>
→ <code>avg_score < 6.0 & count < 2</code>
</td>
</tr>

<tr>
<td><strong>refine_node</strong></td>
<td>
• <code>quality_report</code><br>
• <code>tags</code><br>
• <code>search_strategy</code><br>
• <code>refinement_count</code><br>
• <code>model_input</code><br>
• <code>reasoning_trace</code>
</td>
<td>
• <code>tags</code> (adjusted)<br>
• <code>search_strategy</code> (adjusted)<br>
• <code>model_input</code> (rebuilt)<br>
• <code>refinement_count</code> (+1)<br>
• <code>reasoning_trace</code> (appended)
</td>
<td>
<strong>Applies 3 refinement strategies:</strong><br>
<strong>Strategy 1 (Low Coverage):</strong><br>
→ Add <code>uncovered_tags</code> to <code>tags</code><br>
<strong>Strategy 2 (Low Score):</strong><br>
→ Switch <code>tag_only</code> → <code>semantic</code><br>
<strong>Strategy 3 (2nd Cycle):</strong><br>
→ Remove last tag<br>
→ Set <code>strictness="low"</code><br>
Then rebuilds <code>model_input</code> and increments count
</td>
<td>
<code>utils/helpers.py</code><br>
→ <code>deduplicate()</code><br>
<strong>Triggers:</strong><br>
→ <code>uncovered_tags</code> non-empty<br>
→ <code>avg_score < 6.0</code><br>
→ <code>refinement_count == 1</code>
</td>
</tr>

<tr>
<td><strong>output_node</strong></td>
<td>
• <code>results</code><br>
• <code>reasoning_trace</code><br>
• <code>refinement_count</code>
</td>
<td>
• <code>results</code> (normalized)
</td>
<td>
<strong>Result normalization:</strong><br>
1. Filter out non-dict entries<br>
2. Apply <code>normalize_result()</code> to each<br>
3. Preserve <code>reasoning_trace</code> & <code>refinement_count</code>
</td>
<td>
<code>utils/helpers.py</code><br>
→ <code>normalize_result()</code><br>
→ <code>safe_get()</code><br>
→ <code>format_score()</code><br>
<strong>Ensures all required fields present with safe defaults</strong>
</td>
</tr>

</tbody>
</table>

### AgentState Structure (15 Fields)

```python
class AgentState(TypedDict):
    # Core fields
    query: str                    # Raw user input (never modified)
    tags: List[str]               # Extracted/expanded genre/mood tags
    type: str                     # "anime" | "manga"
    reference: str                # "like X" reference title
    results: List[Any]            # Final recommendation list
    
    # Query understanding (NEW)
    intent: str                   # "find_similar" | "genre_search" | "mood_search" | "character_search" | "complex_search"
    complexity_score: float       # 0.0-1.0 (drives routing)
    semantic_hints: List[str]     # Nuanced preference phrases
    
    # Model input enrichment (NEW)
    search_strategy: str          # "tag_only" | "semantic" | "hybrid" | "reference"
    reference_synopsis: str       # Full synopsis from dataset
    model_input: Dict             # Enriched payload for model.recommend()
    
    # Refinement loop (NEW)
    refinement_count: int         # 0, 1, or 2 (loop guard)
    quality_report: Dict          # {coverage, diversity, avg_score, verdict, uncovered_tags}
    
    # Explainability (NEW)
    reasoning_trace: List[str]    # Human-readable decision log
```

### Conditional Routing Logic

**Route 1: `route_query` (after process_node)**
```python
def route_query(state: AgentState) -> str:
    if state["complexity_score"] >= 0.5:
        return "deep_reasoning_node"
    return "simple_reasoning_node"
```

**Route 2: `route_evaluation` (after evaluator_node)**
```python
def route_evaluation(state: AgentState) -> str:
    verdict = state["quality_report"].get("verdict", "quality_ok")
    if verdict == "needs_refinement" and state["refinement_count"] < 2:
        return "refine_node"
    return "output_node"
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.14)
- **Pre-trained models**: `anime_recommender.pkl` and `manga_recommender.pkl` in `data/` directory
- **Datasets**: `refined_anime_dataset.json` and `refined_manga_dataset.json` in `data/` (for co-occurrence indexing)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd <project-directory>

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (requirements.txt)

```
langgraph>=0.2.0
langchain-core>=0.3.0
fastapi>=0.111.0
uvicorn>=0.30.0
pydantic>=2.0.0
```

### Running the Server

```bash
# From project root
uvicorn backend.main:app --reload
```

**Server URLs:**
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

---

## 📡 API Reference

### `POST /recommend`

Submit a natural language query to get ranked recommendations.

#### Request

```json
{
  "query": "dark anime like Attack on Titan with strong character development"
}
```

#### Response

```json
{
  "results": [
    {
      "title": "Death Note",
      "image": "https://cdn.myanimelist.net/images/anime/9/9453.jpg",
      "synopsis": "A high school student discovers a supernatural notebook...",
      "score": 9.0,
      "genres": ["Mystery", "Psychological", "Thriller", "Supernatural"],
      "similarity_score": 0.87,
      "match_reason": "Dark themes and psychological depth"
    }
  ],
  "reasoning_trace": [
    "[simple_reasoning_node] Expanded 1 base tag(s) to 7 tag(s) via co-occurrence. search_strategy='tag_only'. reference='Attack On Titan'.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='quality_ok' | result_count=10 | coverage=0.78 | avg_score=8.50 | refinement_count=0"
  ],
  "refinement_count": 0
}
```

#### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[RecommendationItem]` | Ranked recommendations |
| `results[].title` | `str` | Anime/manga title |
| `results[].image` | `str` | Cover image URL |
| `results[].synopsis` | `str` | Plot summary |
| `results[].score` | `float` | Rating (0.0-10.0) |
| `results[].genres` | `List[str]` | Genre tags |
| `results[].similarity_score` | `float \| None` | Optional: model-provided similarity |
| `results[].match_reason` | `str \| None` | Optional: model-provided explanation |
| `reasoning_trace` | `List[str]` | Step-by-step decision log |
| `refinement_count` | `int` | Number of refinement cycles (0-2) |

#### Status Codes

| Code | Description |
|------|-------------|
| `200` | Success — recommendations returned |
| `400` | Bad Request — empty/whitespace-only query |
| `404` | Not Found — model `.pkl` file missing at expected path |
| `500` | Internal Server Error — pipeline failure |

### `GET /health`

Health check endpoint for monitoring/load balancers.

#### Response

```json
{
  "status": "ok",
  "service": "recommendation-agent"
}
```

---

## 🧠 Query Understanding

### Intent Classification (Priority Order)

The `extract_intent()` function classifies queries using rule-based heuristics:

| Priority | Intent | Trigger | Example Query |
|----------|--------|---------|---------------|
| 1 | `find_similar` | Contains `"like <Title>"` pattern | `"anime like Death Note"` |
| 2 | `mood_search` | Primary mood words (`dark`, `sad`, `happy`, `melancholy`, `uplifting`, `depressing`, `wholesome`) | `"sad and melancholy anime"` |
| 3 | `character_search` | Character trait phrases (`strong female lead`, `anti-hero`, `protagonist`, `villain`, `character development`) | `"strong female lead fantasy"` |
| 4 | `complex_search` | Multiple tags (≥2) + conjunctions (`and`, `with`, `but not`) | `"action and romance with comedy"` |
| 5 | `genre_search` | Default fallback | `"fantasy adventure anime"` |

### Complexity Scoring (0.0-1.0)

The `compute_complexity_score()` function uses this heuristic:

| Factor | Contribution | Condition |
|--------|-------------|-----------|
| Word count | +0.3 | `len(query.split()) > 8` |
| Reference present | +0.2 | `"like <Title>"` pattern found |
| Semantic hints | +0.2 | Any hint phrase detected |
| Multiple tags + conjunctions | +0.3 | `tag_count >= 2` AND conjunction present |

**Result**: Clamped to `[0.0, 1.0]`  
**Routing**: `complexity_score >= 0.5` → `deep_reasoning_node`

### Semantic Hints Extraction

The `extract_semantic_hints()` function scans for 14 nuanced preference phrases:

| Hint Phrase | Translated Tags |
|-------------|-----------------|
| `"character development"` | `["coming of age", "growth"]` |
| `"slow burn"` | `["romance", "drama"]` |
| `"plot twist"` | `["mystery", "thriller"]` |
| `"dark themes"` | `["psychological", "horror"]` |
| `"found family"` | `["drama", "slice of life"]` |
| `"redemption arc"` | `["drama", "action"]` |

**Full list**: `character development`, `plot twist`, `slow burn`, `coming of age`, `strong female lead`, `anti-hero`, `redemption arc`, `found family`, `dark themes`, `psychological`, `slice of life`, `isekai`, `romance`, `comedy`, `tragedy`

### Tag Extraction

The `extract_tags()` function uses **whole-word matching** (regex: `\b<tag>\b`) for 20 keywords:

```python
TAG_KEYWORDS = [
    "dark", "romance", "action", "comedy", "horror", "thriller",
    "mystery", "fantasy", "sci-fi", "adventure", "psychological",
    "slice of life", "drama", "supernatural", "mecha", "sports",
    "historical", "military", "magic", "school", "isekai"
]
```

### Intent → Search Strategy Mapping

Used by `deep_reasoning_node`:

| Intent | Search Strategy |
|--------|-----------------|
| `find_similar` | `"reference"` |
| `mood_search` | `"semantic"` |
| `complex_search` | `"hybrid"` |
| Others | `"tag_only"` |

---

## 🔄 Quality Evaluation & Refinement

### Quality Metrics

The `evaluate_results()` function computes 4 metrics:

| Metric | Formula | Type |
|--------|---------|------|
| **coverage** | `covered_tags / total_tags` | `float` (0.0-1.0) |
| **diversity** | `len(distinct_genres)` | `int` |
| **avg_score** | `mean(result["score"])` | `float` |
| **result_count** | `len(results)` | `int` |

### Verdict Logic

```python
if (
    result_count < 3
    or (coverage < 0.4 and refinement_count < 2)
    or (avg_score < 6.0 and refinement_count < 2)
):
    verdict = "needs_refinement"
else:
    verdict = "quality_ok"
```

### Refinement Strategies

The `refine_node` applies up to 3 strategies:

| Strategy | Trigger | Action |
|----------|---------|--------|
| **Strategy 1: Low Coverage** | `uncovered_tags` is non-empty | Add `uncovered_tags` to `state["tags"]` (deduplicated) |
| **Strategy 2: Low Score** | `avg_score < 6.0` AND `search_strategy == "tag_only"` | Switch `search_strategy` to `"semantic"` |
| **Strategy 3: Second Cycle** | `refinement_count == 1` | Remove last tag + set `model_input["strictness"] = "low"` |

**Loop Guard**: Maximum 2 refinement cycles (`refinement_count < 2`)

---

## 📂 Project Structure

```
.
├── backend/
│   ├── main.py                          # FastAPI entry point (2 endpoints)
│   │
│   ├── agent/
│   │   ├── graph.py                     # LangGraph pipeline orchestration
│   │   ├── state.py                     # AgentState TypedDict (15 fields)
│   │   └── nodes/
│   │       ├── process.py               # Query parsing
│   │       ├── reasoning.py             # Tag enrichment (simple + deep)
│   │       ├── recommend.py             # Model invocation
│   │       ├── evaluator.py             # Quality assessment
│   │       ├── refine.py                # Refinement strategies
│   │       └── output.py                # Result normalization
│   │
│   ├── services/
│   │   ├── query_parser.py              # 6 extraction functions
│   │   ├── tag_mapper.py                # Co-occurrence expansion
│   │   └── quality_evaluator.py         # Metrics computation
│   │
│   └── utils/
│       └── helpers.py                   # Shared utilities
│
├── .gitignore
├── requirements.txt                     # 5 dependencies
└── README.md
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `backend/agent/` | LangGraph pipeline core (graph, state, 7 nodes) |
| `backend/services/` | Business logic (parsing, expansion, evaluation) |
| `backend/utils/` | Shared helper functions |
| `data/` | Pre-trained models & datasets (gitignored) |

---

## 🔧 Configuration

### Environment Variables

Override default model paths:

```bash
export ANIME_MODEL_PATH=/path/to/custom_anime_model.pkl
export MANGA_MODEL_PATH=/path/to/custom_manga_model.pkl
```

**Default paths** (if not set):
- Anime: `<project_root>/data/anime_recommender.pkl`
- Manga: `<project_root>/data/manga_recommender.pkl`

### Model Contract

Your `.pkl` files must be joblib/pickle-serialized objects exposing:

```python
class RecommenderModel:
    def recommend(self, input_data: dict) -> list[dict]:
        """
        Args:
            input_data: {
                # Required fields
                "tags": list[str],              # Expanded tag list
                "type": str,                    # "anime" | "manga"
                "reference": str,               # Reference title or ""
                
                # Optional enrichment fields (from reasoning nodes)
                "intent": str,                  # Intent classification
                "semantic_hints": list[str],    # Nuanced preferences
                "search_strategy": str,         # "tag_only" | "semantic" | "hybrid" | "reference"
                "reference_synopsis": str,      # Full synopsis from dataset
                "complexity": str,              # "simple" | "complex"
                
                # Optional refinement fields (from refine_node)
                "strictness": str,              # "low" (added in 2nd refinement cycle)
            }
        
        Returns:
            list[dict]: Each dict must contain:
                {
                    # Required fields
                    "title": str,
                    "image": str,               # URL
                    "synopsis": str,
                    "score": float,
                    "genres": list[str],
                    
                    # Optional fields (passed through to frontend)
                    "similarity_score": float,  # Optional
                    "match_reason": str,        # Optional
                }
        """
```

### Dataset Format

`refined_anime_dataset.json` and `refined_manga_dataset.json` must be JSON arrays:

```json
[
  {
    "title": "Attack on Titan",
    "title_english": "Attack on Titan",
    "title_japanese": "進撃の巨人",
    "title_synonyms": ["Shingeki no Kyojin", "AoT"],
    "synopsis": "Centuries ago, mankind was slaughtered...",
    "tags": ["action", "drama", "military"],
    "genres": ["Action", "Drama", "Fantasy", "Military"]
  }
]
```

**Used for**:
- Co-occurrence indexing (`tag_mapper.py`)
- Reference tag lookup (`get_reference_tags`)
- Reference synopsis lookup (`deep_reasoning_node`)

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: `FileNotFoundError: Model file not found`

**Cause**: `.pkl` files missing from `data/` directory

**Solution**:
```bash
# Place your trained models in data/
cp /path/to/anime_recommender.pkl data/
cp /path/to/manga_recommender.pkl data/
```

#### Issue: `AttributeError: 'NoneType' object has no attribute 'recommend'`

**Cause**: `.pkl` file is not a valid joblib/pickle object or doesn't expose `recommend()` method

**Solution**:
```python
# Verify your model structure
import joblib
model = joblib.load("data/anime_recommender.pkl")
print(type(model))
print(hasattr(model, "recommend"))
```

#### Issue: `RuntimeError: model.recommend() must return a list`

**Cause**: Model's `recommend()` method returns wrong type (dict, str, etc.)

**Solution**: Ensure `recommend()` returns `list[dict]`, not a single dict

#### Issue: Tests fail with `hypothesis` errors

**Cause**: Missing test dependencies

**Solution**:
```bash
pip install pytest hypothesis
```

#### Issue: `WARNING: Dataset file not found or empty`

**Cause**: Missing `refined_anime_dataset.json` or `refined_manga_dataset.json`

**Solution**:
```bash
# Place datasets in data/
cp /path/to/refined_anime_dataset.json data/
cp /path/to/refined_manga_dataset.json data/
```

**Impact**: Tag expansion will fall back to semantic map only (no co-occurrence data)

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Cold start** | ~2-3 seconds | Model loading + dataset indexing (first request only) |
| **Warm requests** | ~100-300ms | Cached model + index (subsequent requests) |
| **Memory footprint** | ~200-500MB | Depends on dataset size and model complexity |
| **Refinement overhead** | +100-200ms per cycle | Max 2 cycles = +200-400ms worst case |
| **Test suite** | ~20 seconds | 35 tests with hypothesis property-based testing |

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest backend/tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 🙏 Acknowledgments

- **LangGraph**: State machine framework for agent pipelines
- **FastAPI**: High-performance async API framework
- **Hypothesis**: Property-based testing library
- **Pydantic**: Data validation and settings management

---

**Built with ❤️ using LangGraph, FastAPI, and Python**

**Version**: 2.0.0 (as specified in `backend/main.py`)
