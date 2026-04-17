# Anime & Manga Recommendation Agent

A LangGraph-powered recommendation agent that wraps a pre-trained `.pkl` model with intelligent query understanding, complexity-based routing, a self-evaluating quality loop, and a human-readable reasoning trace.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Graph Structure](#graph-structure)
- [Project Structure](#project-structure)
- [What Is Not in the Repository](#what-is-not-in-the-repository)
- [Agent State](#agent-state)
- [Nodes](#nodes)
- [Services](#services)
- [API Reference](#api-reference)
- [Setup and Installation](#setup-and-installation)
- [Running the Server](#running-the-server)
- [Running Tests](#running-tests)
- [Model Contract](#model-contract)
- [Environment Variables](#environment-variables)

---

## Overview

The agent takes a natural-language query like `"dark psychological anime like Death Note"` and returns ranked recommendations from a pre-trained ML model. It does not replace the model — it wraps it with:

- **Deep query understanding** — intent classification, complexity scoring, semantic hint extraction
- **Complexity-based routing** — simple queries take a fast path; complex queries get full multi-factor analysis
- **Quality evaluation** — after the model responds, the agent checks coverage, diversity, and average score
- **Bounded refinement loop** — if quality is low, the agent adjusts the input and retries up to 2 times
- **Reasoning trace** — every decision is logged in plain English and returned alongside results

---

## Architecture

```
User Query
    |
    v
+---------------+
| process_node  |  <- parse tags, type, reference, intent,
+-------+-------+     complexity_score, semantic_hints
        |
        v  route_query
        |  complexity_score < 0.5  --> simple_reasoning_node
        |  complexity_score >= 0.5 --> deep_reasoning_node
        |
   +----+-------------------------+
   |                              |
   v                              v
+--------------------+   +--------------------+
| simple_reasoning   |   | deep_reasoning     |
| _node              |   | _node              |
|                    |   |                    |
| - expand tags      |   | - expand tags      |
| - ref tag lookup   |   | - hint -> tags     |
| - strategy=        |   | - synopsis lookup  |
|   "tag_only"       |   | - intent-based     |
| - build            |   |   strategy         |
|   model_input      |   | - build            |
+--------+-----------+   |   model_input      |
         |               +--------+-----------+
         +---------------+        |
                          |       |
                          v       v
                   +------+-------+
                   | recommend_node|  <- model.recommend(model_input)
                   +-------+-------+
                           |
                           v
                   +-------+-------+
                   | evaluator_node|  <- coverage, diversity,
                   +-------+-------+     avg_score -> verdict
                           |
              +------------+------------+
              |                         |
   verdict=needs_refinement    verdict=quality_ok
   AND refinement_count < 2    OR count >= 2
              |                         |
              v                         v
      +-------+------+         +--------+-----+
      |  refine_node |-------->| output_node  |--> END
      |              |         |              |
      | - add tags   |         | - normalise  |
      | - switch     |         | - attach     |
      |   strategy   |         |   trace      |
      | - broaden    |         +--------------+
      | - count + 1  |
      +--------------+
        (loops back to recommend_node, max 2 times)
```

---

## Graph Structure

The graph is built with [LangGraph](https://github.com/langchain-ai/langgraph) and compiled once at import time in `backend/agent/graph.py`.

### Nodes (7 total)

| Node | File | Role |
|------|------|------|
| `process_node` | `nodes/process.py` | Parse raw query into structured state fields |
| `simple_reasoning_node` | `nodes/reasoning.py` | Fast-path tag expansion for low-complexity queries |
| `deep_reasoning_node` | `nodes/reasoning.py` | Full multi-factor analysis for complex queries |
| `recommend_node` | `nodes/recommend.py` | Call the pre-trained `.pkl` model |
| `evaluator_node` | `nodes/evaluator.py` | Compute quality metrics and issue a verdict |
| `refine_node` | `nodes/refine.py` | Adjust model input and loop back for a retry |
| `output_node` | `nodes/output.py` | Normalise results and attach reasoning trace |

### Conditional Edges (2 total)

**`route_query`** — fires after `process_node`:

| Condition | Destination |
|-----------|-------------|
| `complexity_score >= 0.5` | `deep_reasoning_node` |
| `complexity_score < 0.5` | `simple_reasoning_node` |

**`route_evaluation`** — fires after `evaluator_node`:

| Condition | Destination |
|-----------|-------------|
| `verdict == "needs_refinement"` AND `refinement_count < 2` | `refine_node` |
| all other cases | `output_node` |

### Fixed Edges

```
simple_reasoning_node --> recommend_node
deep_reasoning_node   --> recommend_node
recommend_node        --> evaluator_node
refine_node           --> recommend_node   (refinement loop, max 2 iterations)
output_node           --> END
```

---

## Project Structure

The following shows only files that are **tracked by git**. Files and directories listed in `.gitignore` (datasets, model files, test outputs, notebooks, virtual environment) are excluded — see [What Is Not in the Repository](#what-is-not-in-the-repository).

```
.
├── backend/
│   ├── main.py                      # FastAPI app, Pydantic schemas, /recommend endpoint
│   ├── agent/
│   │   ├── graph.py                 # LangGraph graph, route_query, route_evaluation,
│   │   │                            #   run_agent entry point
│   │   ├── state.py                 # AgentState TypedDict + initial_state()
│   │   └── nodes/
│   │       ├── process.py           # Node 1: query parsing
│   │       ├── reasoning.py         # Nodes 2a/2b: simple_reasoning_node,
│   │       │                        #   deep_reasoning_node
│   │       ├── recommend.py         # Node 3: model invocation with enriched payload
│   │       ├── evaluator.py         # Node 4: quality evaluation
│   │       ├── refine.py            # Node 5: three refinement strategies
│   │       └── output.py            # Node 6: result normalisation
│   ├── services/
│   │   ├── query_parser.py          # extract_tags, detect_type, extract_reference,
│   │   │                            #   extract_intent, compute_complexity_score,
│   │   │                            #   extract_semantic_hints
│   │   ├── tag_mapper.py            # Dataset co-occurrence index, expand_tags,
│   │   │                            #   get_reference_tags
│   │   └── quality_evaluator.py     # evaluate_results -> coverage, diversity,
│   │                                #   avg_score, verdict, uncovered_tags
│   └── utils/
│       └── helpers.py               # deduplicate, safe_get, format_score,
│                                    #   normalize_result
├── requirements.txt
└── README.md
```

> **Note:** `backend/tests/` is gitignored and will not be present after a fresh clone.
> Run `python -m pytest backend/tests/ -v` only if you have the test files locally.

---

## What Is Not in the Repository

The following are excluded by `.gitignore` and must be provided manually:

### Dataset files (gitignored: `data/`, `*.json`)

The tag mapper and deep reasoning node read these at runtime:

```
data/
├── refined_anime_dataset.json
└── refined_manga_dataset.json
```

Each entry in the JSON should contain at minimum:

```json
{
  "title": "...",
  "title_english": "...",
  "title_japanese": "...",
  "title_synonyms": ["..."],
  "synopsis": "...",
  "genres": ["action", "drama"],
  "tags": ["shounen", "military"]
}
```

If the files are absent, `tag_mapper` returns empty expansions and `deep_reasoning_node` skips the synopsis lookup — the pipeline still runs but with reduced quality.

### Model files (gitignored: `data/`)

Place your trained joblib/pickle models here:

```
data/
├── anime_recommender.pkl
└── manga_recommender.pkl
```

See [Model Contract](#model-contract) for the required interface. Paths can be overridden with environment variables — see [Environment Variables](#environment-variables).

### Test files (gitignored: `tests`)

The `backend/tests/` directory is gitignored. The 35 property-based tests exist locally but are not committed. To run them you need the test files present in your working directory.

### Virtual environment (gitignored: `venv`)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### Notebooks (gitignored: `refining_anime.ipynb`, `refining_manga.ipynb`)

The data-refinement notebooks used to produce the dataset JSON files are not committed.

---

## Agent State

All nodes share a single `AgentState` TypedDict defined in `backend/agent/state.py`. Fields are never deleted — only updated.

```python
class AgentState(TypedDict):
    # Set by process_node
    query:              str        # raw user input, never modified after entry
    tags:               List[str]  # extracted + expanded genre/mood tags
    type:               str        # "anime" | "manga"
    reference:          str        # title from "like X" pattern, or ""
    intent:             str        # "find_similar" | "genre_search" | "mood_search"
                                   # | "character_search" | "complex_search"
    complexity_score:   float      # 0.0-1.0; drives routing decision
    semantic_hints:     List[str]  # nuanced phrases e.g. ["slow burn", "dark themes"]

    # Set by reasoning nodes
    search_strategy:    str        # "tag_only" | "semantic" | "hybrid" | "reference"
    reference_synopsis: str        # synopsis of reference title from dataset, or ""
    model_input:        Dict       # full enriched payload sent to model.recommend()

    # Set by recommend_node
    results:            List[Any]  # raw results from model.recommend()

    # Set by evaluator_node
    quality_report:     Dict       # coverage, diversity, avg_score, result_count,
                                   # verdict, uncovered_tags

    # Updated by refine_node
    refinement_count:   int        # 0, 1, or 2 — hard cap prevents infinite loops

    # Appended by every node
    reasoning_trace:    List[str]  # human-readable log of every decision made
```

**Default values** set by `initial_state(query)`:

| Field | Default |
|-------|---------|
| `tags` | `[]` |
| `type` | `"anime"` |
| `reference` | `""` |
| `intent` | `"genre_search"` |
| `complexity_score` | `0.0` |
| `semantic_hints` | `[]` |
| `search_strategy` | `"hybrid"` |
| `reference_synopsis` | `""` |
| `model_input` | `{}` |
| `results` | `[]` |
| `refinement_count` | `0` |
| `quality_report` | `{}` |
| `reasoning_trace` | `[]` |

---

## Nodes

### `process_node`

Parses the raw query into all structured state fields. Raises `ValueError` for empty or whitespace-only queries.

- Extracts genre/mood tags using whole-word regex matching (e.g. `"action"` inside `"inaction"` is not matched)
- Detects media type (`"anime"` or `"manga"`), defaulting to `"anime"`
- Extracts a reference title from `"like <Title>"` patterns (title-cased)
- Classifies intent in priority order: `find_similar` > `mood_search` > `character_search` > `complex_search` > `genre_search`
- Computes complexity score: base `0.0`, `+0.3` for >8 words, `+0.2` for reference present, `+0.2` for semantic hints, `+0.3` for multi-signal conjunctions; clamped to `[0.0, 1.0]`

### `simple_reasoning_node`

Fast path for queries with `complexity_score < 0.5`.

- Expands tags via dataset co-occurrence index (`tag_mapper.expand_tags`)
- Merges reference title tags if a reference was extracted
- Sets `search_strategy = "tag_only"`
- Builds `model_input` with all 8 required keys: `tags`, `type`, `reference`, `intent`, `semantic_hints`, `search_strategy`, `reference_synopsis`, `complexity`
- Deduplicates tags preserving insertion order

### `deep_reasoning_node`

Full analysis for queries with `complexity_score >= 0.5`.

- Expands tags via co-occurrence
- Translates `semantic_hints` into additional tags (e.g. `"slow burn"` adds `"romance"`, `"drama"`)
- Looks up the reference title's synopsis from the dataset when `intent == "find_similar"`; handles missing title and missing dataset file gracefully with trace warnings
- Sets `search_strategy` based on intent:

  | Intent | search_strategy |
  |--------|----------------|
  | `find_similar` | `"reference"` |
  | `mood_search` | `"semantic"` |
  | `complex_search` | `"hybrid"` |
  | all others | `"tag_only"` |

- Builds the full enriched `model_input` payload with `complexity = "complex"`
- Appends multiple trace entries documenting each step

### `recommend_node`

Thin adapter between the agent state and the pre-trained model.

- Uses `state["model_input"]` (enriched payload) when non-empty
- Falls back to legacy `{"tags", "type", "reference"}` format when `model_input` is `{}`
- Loads the `.pkl` model once per process lifetime using `lru_cache`
- Raises `FileNotFoundError` if the model file is missing
- Raises `AttributeError` if the loaded object has no `.recommend()` method
- Raises `RuntimeError` if `model.recommend()` raises or returns a non-list value

### `evaluator_node`

Self-reflection step that decides whether results are good enough.

Delegates all computation to `quality_evaluator.evaluate_results` and stores the full report in `state["quality_report"]`. The `verdict` field in that report drives the `route_evaluation` conditional edge.

### `refine_node`

Adjusts the model input payload before a retry. Applies three strategies in this order:

1. **Strategy 3** (second cycle, applied first): when `refinement_count == 1`, removes the last tag from `state["tags"]` and adds `"strictness": "low"` to `model_input`
2. **Strategy 1** (low coverage): adds `uncovered_tags` from the quality report to `state["tags"]`
3. **Strategy 2** (low score): switches `search_strategy` from `"tag_only"` to `"semantic"` when `avg_score < 6.0`

Always increments `refinement_count` by exactly 1 and appends a trace entry describing which strategies fired.

### `output_node`

Normalises raw model results to the frontend schema. Preserves `reasoning_trace` and `refinement_count` in state unchanged.

Each result is normalised to:

```python
{
    "title":            str,          # default: "Unknown"
    "image":            str,          # default: ""
    "synopsis":         str,          # default: "No synopsis available."
    "score":            float,        # default: 0.0
    "genres":           List[str],    # default: []
    "similarity_score": float | None, # passed through if model provides it
    "match_reason":     str | None,   # passed through if model provides it
}
```

Non-dict entries in `state["results"]` are silently skipped without raising an error.

---

## Services

### `query_parser.py`

| Function | Returns | Description |
|----------|---------|-------------|
| `extract_tags(query)` | `List[str]` | Whole-word regex scan for genre/mood keywords |
| `detect_type(query)` | `str` | `"anime"` or `"manga"` |
| `extract_reference(query)` | `str` | Title from `"like <Title>"` pattern, or `""` |
| `extract_intent(query)` | `str` | One of 5 valid intent values |
| `compute_complexity_score(query)` | `float` | Value in `[0.0, 1.0]` |
| `extract_semantic_hints(query)` | `List[str]` | Matched nuanced preference phrases |

### `tag_mapper.py`

Builds a co-occurrence index from the actual JSON datasets at startup (cached with `lru_cache`). Falls back to a curated semantic map for mood words absent from the dataset (e.g. `"dark"`, `"sad"`, `"epic"`).

| Function | Returns | Description |
|----------|---------|-------------|
| `expand_tags(tags, media_type)` | `List[str]` | Expanded tag list; always a superset of input |
| `get_reference_tags(reference, media_type)` | `List[str]` | Tags from a matched dataset entry, or `[]` |

If the dataset files are absent (gitignored — see above), `expand_tags` returns the original tags unchanged and `get_reference_tags` returns `[]`.

### `quality_evaluator.py`

| Function | Returns | Description |
|----------|---------|-------------|
| `evaluate_results(results, tags, refinement_count)` | `Dict` | Quality report with 6 keys |

**Report keys:**

| Key | Type | Description |
|-----|------|-------------|
| `coverage` | `float` | Fraction of query tags found in any result's genres |
| `diversity` | `int` | Count of distinct genres across all results |
| `avg_score` | `float` | Mean of `result["score"]` values |
| `result_count` | `int` | Number of results returned |
| `verdict` | `str` | `"quality_ok"` or `"needs_refinement"` |
| `uncovered_tags` | `List[str]` | Query tags not represented in any result |

**Verdict logic:**

```
"needs_refinement"  if  result_count < 3
                    OR  (coverage < 0.4  AND  refinement_count < 2)
                    OR  (avg_score < 6.0 AND  refinement_count < 2)
"quality_ok"        otherwise
```

---

## API Reference

### `POST /recommend`

Run the full agent pipeline on a user query.

**Request body:**

```json
{ "query": "dark psychological anime like Death Note" }
```

**Response:**

```json
{
  "results": [
    {
      "title": "Monster",
      "image": "https://cdn.myanimelist.net/...",
      "synopsis": "A brilliant surgeon saves a boy's life...",
      "score": 9.1,
      "genres": ["mystery", "psychological", "thriller"],
      "similarity_score": 0.87,
      "match_reason": "High overlap in psychological thriller tags"
    }
  ],
  "reasoning_trace": [
    "[deep_reasoning_node] Step 1: Co-occurrence expansion — 2 base tag(s) -> 9 expanded tag(s).",
    "[deep_reasoning_node] Step 2: Semantic hint translation — hints=[] -> added tags=[].",
    "[deep_reasoning_node] Step 3: Found reference title 'Death Note' in dataset. Synopsis length: 312 chars.",
    "[deep_reasoning_node] Step 4: search_strategy='reference' (intent='find_similar'). Final tag count: 9.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='quality_ok' | result_count=10 | coverage=0.78 | avg_score=8.20 | refinement_count=0"
  ],
  "refinement_count": 0
}
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| `400` | Empty or whitespace-only query |
| `404` | `.pkl` model file not found at expected path |
| `500` | Model raised an exception or returned unexpected data |

### `GET /health`

Liveness probe.

```json
{ "status": "ok", "service": "recommendation-agent" }
```

---

## Setup and Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-directory>

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Provide the required files that are not in the repository:**

```
data/
├── anime_recommender.pkl       <- your trained anime model
├── manga_recommender.pkl       <- your trained manga model
├── refined_anime_dataset.json  <- anime dataset for tag co-occurrence
└── refined_manga_dataset.json  <- manga dataset for tag co-occurrence
```

The `data/` directory and all `.json` files are gitignored. The pipeline degrades gracefully if they are absent but will not return useful recommendations without the model files.

---

## Running the Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive Swagger docs: `http://localhost:8000/docs`

---

## Running Tests

> The `backend/tests/` directory is gitignored and will not be present after a fresh clone.

If you have the test files locally:

```bash
python -m pytest backend/tests/ -v
```

The suite contains **35 property-based tests** using [Hypothesis](https://hypothesis.readthedocs.io/), covering all 24 correctness properties defined in the spec.

| Test file | Properties covered |
|-----------|-------------------|
| `test_state.py` | P24: initial_state defaults |
| `test_query_parser.py` | P5: complexity score range, P6: intent valid values |
| `test_process.py` | P1: whole-word tag extraction, P2: detect_type, P3: reference round-trip, P4: whitespace rejection |
| `test_reasoning.py` | P8: tag expansion superset, P9: deduplication, P10: model_input keys, P11: trace growth, P12: search_strategy mapping |
| `test_recommend.py` | P13: results stored unchanged, P14: non-list raises RuntimeError |
| `test_quality_evaluator.py` | P15: report keys, P16: verdict logic |
| `test_evaluator.py` | P11: trace growth (evaluator node) |
| `test_refine.py` | P11: trace growth (refine node), P18: count increments by 1, P19: uncovered tags added |
| `test_output.py` | P20: normalisation completeness, P21: non-dict exclusion |
| `test_graph.py` | P7: route_query determinism, P17: route_evaluation correctness |

---

## Model Contract

Your `.pkl` file must be a joblib/pickle-serialised object that exposes:

```python
model.recommend(input_data: dict) -> list[dict]
```

The agent sends an enriched payload:

```python
{
    # Core fields (always present)
    "tags":               List[str],  # expanded tag list
    "type":               str,        # "anime" | "manga"
    "reference":          str,        # reference title or ""

    # Enrichment fields (model can use or ignore)
    "intent":             str,        # "find_similar" | "genre_search" | ...
    "semantic_hints":     List[str],  # ["slow burn", "dark themes", ...]
    "search_strategy":    str,        # "tag_only" | "semantic" | "hybrid" | "reference"
    "reference_synopsis": str,        # full synopsis of reference title, or ""
    "complexity":         str,        # "simple" | "complex"
}
```

The model must return a list of dicts. Each dict should contain at minimum:

```python
{
    "title":    str,
    "image":    str,
    "synopsis": str,
    "score":    float,
    "genres":   List[str],
    # optional — passed through to the API response if present:
    "similarity_score": float,
    "match_reason":     str,
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANIME_MODEL_PATH` | `data/anime_recommender.pkl` | Path to the anime recommender model |
| `MANGA_MODEL_PATH` | `data/manga_recommender.pkl` | Path to the manga recommender model |

Example override:

```bash
ANIME_MODEL_PATH=/models/my_anime_model.pkl uvicorn backend.main:app --reload
```
