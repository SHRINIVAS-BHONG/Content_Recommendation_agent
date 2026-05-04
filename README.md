# Anime & Manga Recommendation Agent

## Project Description

An intelligent recommendation system that uses AI-powered agents to provide personalized anime and manga recommendations. The system employs a 7-node LangGraph pipeline that processes natural language queries, understands user intent, and delivers relevant recommendations through a hybrid scoring approach combining semantic similarity, tag-based filtering, and popularity metrics.

### Key Features

- **AI-Powered Agent Pipeline**: 7-node LangGraph workflow with intelligent query routing based on complexity
- **Smart Query Understanding**: Automatically extracts intent, complexity, tags, and semantic hints from natural language
- **Self-Refining System**: Evaluates result quality and automatically refines recommendations (up to 2 cycles)
- **Hybrid Recommendation Engine**: Combines Sentence-BERT embeddings, Jaccard similarity, and popularity scoring
- **Large Dataset**: 26,000+ anime and manga titles with comprehensive metadata
- **Explainable AI**: Provides detailed reasoning traces showing the decision-making process
- **Fast Performance**: Cached models with LRU caching for instant responses (~50-200ms per query)

### Technology Stack

**Backend:**
- **Framework**: FastAPI (Python 3.9+)
- **AI Framework**: LangGraph for agent orchestration
- **ML Models**: Sentence-BERT (all-MiniLM-L6-v2) for semantic embeddings
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Testing**: Pytest, Hypothesis (property-based testing)

**Frontend:**
- **Framework**: React 18.3.1
- **Build Tool**: Vite 5.4.10
- **Styling**: CSS3 with custom animations
- **HTTP Client**: Native Fetch API
- **Linting**: ESLint with React plugins

### System Architecture

The system follows a 7-node agent pipeline:

1. **Process Node**: Extracts tags, type, reference, intent, complexity score, and semantic hints
2. **Routing**: Routes to Simple Reasoning (complexity < 0.5) or Deep Reasoning (complexity ≥ 0.5)
3. **Reasoning Node**: Expands tags via co-occurrence analysis and translates semantic hints
4. **Recommend Node**: Loads ML model and computes hybrid scores for top 10 results
5. **Evaluator Node**: Assesses result quality (coverage, diversity, average score)
6. **Refine Node**: Applies refinement strategies if quality is insufficient (max 2 cycles)
7. **Output Node**: Normalizes and formats the final response

## Installation Steps

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git
- 4GB+ RAM (for model training)
- 500MB+ disk space (for models and datasets)

### Step-by-Step Installation

**1. Clone the Repository**
```bash
git clone <repository-url>
cd Content_Recommendation_agent
```

**2. Create Virtual Environment**

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- langgraph==1.0.10
- langchain-core==1.2.18
- fastapi==0.136.0
- uvicorn==0.44.0
- pandas==2.2.3
- numpy==2.2.3
- scikit-learn==1.6.1
- sentence-transformers==4.1.0
- And other required packages

**4. Prepare Dataset**

Ensure the following files exist in the `data/` directory:
- `refined_anime_dataset.json`
- `refined_manga_dataset.json` 

**5. Train ML Models** (First Time Only)

Run the training script to generate the recommendation models:
```bash
python scripts/train_model.py
```

This process will:
- Load anime and manga datasets
- Generate Sentence-BERT embeddings for all entries
- Create and save two model files:
  - `data/anime_recommender.pkl` (~150 MB)
  - `data/manga_recommender.pkl` (~150 MB)
- Training time: ~5-10 minutes depending on your hardware

**Note**: The `.pkl` files are gitignored and must be generated locally.

## How to Run the Project

### Starting the Backend Server

**1. Activate Virtual Environment** (if not already activated)

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

**2. Start the FastAPI Server**
```bash
uvicorn backend.main:app --reload
```

Expected output:
```
INFO:     Will watch for changes in these directories: ['C:\\...\\Content_Recommendation_agent']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using StatReload
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**3. Access the Backend API**

- **API Base URL**: http://127.0.0.1:8000
- **Interactive API Docs**: http://127.0.0.1:8000/docs (Swagger UI)
- **Alternative Docs**: http://127.0.0.1:8000/redoc (ReDoc)

### Starting the Frontend Application

**1. Navigate to Frontend Directory**
```bash
cd frontend
```

**2. Install Node Dependencies** (first time only)
```bash
npm install
```

This will install:
- React 18.3.1
- Vite 5.4.10
- ESLint and related plugins

**3. Start the Development Server**
```bash
npm run dev
```

Expected output:
```
  VITE v5.4.10  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

**4. Access the Frontend Application**

Open your browser and visit: **http://localhost:5173**

The frontend provides:
- Interactive search interface
- Real-time recommendation results
- Reasoning trace visualization
- Agent pipeline explanation
- Memory system showcase

### Running Both Backend and Frontend

For full functionality, run both servers simultaneously:

**Terminal 1 (Backend):**
```bash
# From project root
venv\Scripts\activate  # Windows
uvicorn backend.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
# From project root
cd frontend
npm run dev
```

Then visit http://localhost:5173 to use the complete application.

### Building Frontend for Production

```bash
cd frontend
npm run build
```

This creates an optimized production build in `frontend/dist/`.

To preview the production build:
```bash
npm run preview
```

### Testing the API

**Option 1: Using the Frontend UI** (Recommended)
1. Start both backend and frontend servers
2. Open http://localhost:5173 in your browser
3. Enter a query in the search box (e.g., "dark anime like Death Note")
4. Click "Get Recommendations"
5. View results and click "Why this result?" to see reasoning trace

**Option 2: Using Swagger UI**
1. Open http://127.0.0.1:8000/docs in your browser
2. Click on `POST /recommend`
3. Click "Try it out"
4. Enter your query in the request body
5. Click "Execute"

**Option 3: Using cURL**
```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "dark anime like Death Note"}'
```

**Option 4: Using Python requests**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/recommend",
    json={"query": "dark anime like Death Note"}
)

print(response.json())
```

### Stopping the Server

Press `CTRL+C` in the terminal where the server is running.

## Example Input/Output

### Example 1: Simple Genre Query

**Input:**
```json
{
  "query": "action anime"
}
```

**Output:**
```json
{
  "results": [
    {
      "title": "Fullmetal Alchemist: Brotherhood",
      "image": "https://cdn.myanimelist.net/images/anime/1223/96541.jpg",
      "synopsis": "After a horrific alchemy experiment goes wrong in the Elric household, brothers Edward and Alphonse are left in a catastrophic new reality...",
      "score": 9.09,
      "genres": ["action", "adventure", "drama", "fantasy"],
      "similarity_score": 0.7234,
      "match_reason": "Strong thematic match (72%). High genre overlap (80%)"
    },
    {
      "title": "Attack on Titan",
      "image": "https://cdn.myanimelist.net/images/anime/10/47347.jpg",
      "synopsis": "Centuries ago, mankind was slaughtered to near extinction by monstrous humanoid creatures called titans...",
      "score": 8.54,
      "genres": ["action", "drama", "fantasy", "shounen"],
      "similarity_score": 0.6891,
      "match_reason": "Strong thematic match (69%). High genre overlap (75%)"
    }
  ],
  "reasoning_trace": [
    "[simple_reasoning_node] Expanded 1 base tag(s) to 7 tag(s) via co-occurrence. search_strategy='tag_only'. reference=''. authenticated=False.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='quality_ok' | result_count=10 | coverage=0.71 | avg_score=8.23 | refinement_count=0"
  ],
  "refinement_count": 0
}
```

### Example 2: Reference-Based Query

**Input:**
```json
{
  "query": "anime like Death Note"
}
```

**Output:**
```json
{
  "results": [
    {
      "title": "Code Geass: Hangyaku no Lelouch",
      "image": "https://cdn.myanimelist.net/images/anime/5/50331.jpg",
      "synopsis": "The Holy Empire of Britannia is establishing itself as a dominant military nation...",
      "score": 8.7,
      "genres": ["action", "drama", "mecha", "military", "school", "sci-fi", "super power"],
      "similarity_score": 0.8456,
      "match_reason": "Strong thematic match (85%). Similar to reference title"
    },
    {
      "title": "Monster",
      "image": "https://cdn.myanimelist.net/images/anime/10/18793.jpg",
      "synopsis": "Dr. Kenzo Tenma, an elite neurosurgeon recently engaged to his hospital director's daughter...",
      "score": 8.87,
      "genres": ["drama", "horror", "mystery", "police", "psychological", "seinen", "thriller"],
      "similarity_score": 0.8123,
      "match_reason": "Strong thematic match (81%). High genre overlap (70%)"
    }
  ],
  "reasoning_trace": [
    "[deep_reasoning_node] Step 1: Co-occurrence expansion — 0 base tag(s) → 6 expanded tag(s).",
    "[deep_reasoning_node] Step 2: Semantic hint translation — hints=[] → added tags=[].",
    "[deep_reasoning_node] Step 4: Found reference title 'Death Note' in dataset. Synopsis length: 342 chars.",
    "[deep_reasoning_node] Step 5: search_strategy='reference' (intent='find_similar'). Final tag count: 6. Personalization applied: False.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='quality_ok' | result_count=10 | coverage=0.83 | avg_score=8.65 | refinement_count=0"
  ],
  "refinement_count": 0
}
```

### Example 3: Complex Query with Semantic Hints

**Input:**
```json
{
  "query": "dark psychological thriller with plot twists and character development"
}
```

**Output:**
```json
{
  "results": [
    {
      "title": "Steins;Gate",
      "image": "https://cdn.myanimelist.net/images/anime/5/73199.jpg",
      "synopsis": "The self-proclaimed mad scientist Rintarou Okabe rents out a room in a rickety old building in Akihabara...",
      "score": 9.07,
      "genres": ["drama", "sci-fi", "steins;gate", "thriller"],
      "similarity_score": 0.8734,
      "match_reason": "Strong thematic match (87%). High genre overlap (85%)"
    },
    {
      "title": "Psycho-Pass",
      "image": "https://cdn.myanimelist.net/images/anime/5/43399.jpg",
      "synopsis": "Justice, and the enforcement of it, has changed. In the 22nd century, Japan enforces the Sibyl System...",
      "score": 8.32,
      "genres": ["action", "police", "psychological", "sci-fi"],
      "similarity_score": 0.8521,
      "match_reason": "Strong thematic match (85%). High genre overlap (80%)"
    }
  ],
  "reasoning_trace": [
    "[deep_reasoning_node] Step 1: Co-occurrence expansion — 3 base tag(s) → 9 expanded tag(s).",
    "[deep_reasoning_node] Step 2: Semantic hint translation — hints=['plot twist', 'character development'] → added tags=['mystery', 'suspense', 'coming of age', 'drama', 'slice of life'].",
    "[deep_reasoning_node] Step 5: search_strategy='semantic' (intent='mood_search'). Final tag count: 14. Personalization applied: False.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='quality_ok' | result_count=10 | coverage=0.79 | avg_score=8.45 | refinement_count=0"
  ],
  "refinement_count": 0
}
```

### Example 4: Query with Refinement

**Input:**
```json
{
  "query": "obscure mecha anime"
}
```

**Output:**
```json
{
  "results": [
    {
      "title": "Gurren Lagann",
      "image": "https://cdn.myanimelist.net/images/anime/4/5123.jpg",
      "synopsis": "Simon and Kamina were born and raised in a deep, underground village, hidden from the fabled surface...",
      "score": 8.63,
      "genres": ["action", "adventure", "comedy", "mecha", "sci-fi"],
      "similarity_score": 0.7234,
      "match_reason": "Strong thematic match (72%). High genre overlap (60%)"
    }
  ],
  "reasoning_trace": [
    "[simple_reasoning_node] Expanded 2 base tag(s) to 8 tag(s) via co-occurrence. search_strategy='tag_only'. reference=''. authenticated=False.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='needs_refinement' | result_count=2 | coverage=0.25 | avg_score=7.89 | refinement_count=0",
    "[refine_node] Strategy 1 (low coverage): added uncovered tags ['obscure'] → tags now 9 item(s). | Strategy 2 (low score): avg_score=7.89 < 6.0 and search_strategy was 'tag_only' → switched to 'semantic'.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='quality_ok' | result_count=10 | coverage=0.67 | avg_score=8.12 | refinement_count=1"
  ],
  "refinement_count": 1
}
```

### Example 5: Manga Query

**Input:**
```json
{
  "query": "romance manga with strong female lead"
}
```

**Output:**
```json
{
  "results": [
    {
      "title": "Akatsuki no Yona",
      "image": "https://cdn.myanimelist.net/images/manga/1/140555.jpg",
      "synopsis": "Princess Yona lives a life of luxury and ease, completely sheltered from the problems of the seemingly peaceful Kingdom of Kouka...",
      "score": 8.55,
      "genres": ["action", "adventure", "comedy", "fantasy", "romance", "shoujo"],
      "similarity_score": 0.8345,
      "match_reason": "Strong thematic match (83%). High genre overlap (75%)"
    },
    {
      "title": "Fruits Basket",
      "image": "https://cdn.myanimelist.net/images/manga/2/134339.jpg",
      "synopsis": "Tohru Honda is a compassionate girl who is down on her luck...",
      "score": 8.42,
      "genres": ["drama", "romance", "shoujo", "slice of life", "supernatural"],
      "similarity_score": 0.8123,
      "match_reason": "Strong thematic match (81%). High genre overlap (70%)"
    }
  ],
  "reasoning_trace": [
    "[deep_reasoning_node] Step 1: Co-occurrence expansion — 1 base tag(s) → 7 expanded tag(s).",
    "[deep_reasoning_node] Step 2: Semantic hint translation — hints=['strong female lead'] → added tags=['action', 'adventure', 'shoujo', 'mahou shoujo'].",
    "[deep_reasoning_node] Step 5: search_strategy='semantic' (intent='character_search'). Final tag count: 11. Personalization applied: False.",
    "recommend_node: using enriched payload format (model_input was populated).",
    "[evaluator_node] verdict='quality_ok' | result_count=10 | coverage=0.82 | avg_score=8.34 | refinement_count=0"
  ],
  "refinement_count": 0
}
```

### Understanding the Output

**Response Fields:**
- `results`: Array of up to 10 recommendation objects
  - `title`: Anime/manga title
  - `image`: Cover image URL
  - `synopsis`: Brief description (truncated to 250 characters)
  - `score`: Community rating (0-10 scale)
  - `genres`: List of genre tags
  - `similarity_score`: Computed relevance score (0-1 scale)
  - `match_reason`: Human-readable explanation of why this was recommended

- `reasoning_trace`: Array of strings showing the agent's decision-making process
  - Shows which nodes were executed
  - Displays tag expansion and strategy selection
  - Reveals quality evaluation metrics
  - Indicates if refinement occurred

- `refinement_count`: Number of quality refinement cycles (0-2)
  - 0: High quality results on first attempt
  - 1-2: System refined the query to improve results

## Additional Information

### API Endpoints

- `POST /recommend` - Get recommendations (main endpoint)
- `GET /health` - Health check
- `GET /` - API information and welcome message
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - Alternative ReDoc documentation

### Supported Query Types

1. **Simple Genre Queries**: "action anime", "romance manga"
2. **Reference-Based**: "anime like Death Note", "manga similar to One Piece"
3. **Complex Queries**: "dark psychological thriller with plot twists"
4. **Mood-Based**: "sad emotional anime", "feel-good comedy"
5. **Character-Focused**: "strong female lead", "anti-hero protagonist"

### How the System Works

**Query Processing Pipeline:**
1. Extract tags, media type, reference title, intent, and complexity
2. Route to appropriate reasoning node based on complexity score
3. Expand tags using dataset co-occurrence analysis
4. Translate semantic hints into additional tags
5. Compute hybrid scores (semantic + tag overlap + popularity)
6. Evaluate result quality (coverage, diversity, average score)
7. Refine if needed (max 2 cycles) or return results

**Scoring Formula:**
```
final_score = w_semantic × semantic_similarity 
            + w_jaccard × tag_overlap 
            + w_popularity × normalized_rating
```

Weights vary by search strategy:
- `reference`: (0.7, 0.2, 0.1) - Emphasizes semantic similarity
- `tag_only`: (0.3, 0.6, 0.1) - Emphasizes tag overlap
- `semantic`: (0.8, 0.1, 0.1) - Almost pure semantic matching
- `hybrid`: (0.6, 0.3, 0.1) - Balanced approach (default)

### Project Structure

```
Content_Recommendation_agent/
├── backend/                       # FastAPI Backend
│   ├── main.py                    # FastAPI application entry point
│   ├── agent/
│   │   ├── graph.py               # LangGraph pipeline definition
│   │   ├── state.py               # Shared state structure (19 fields)
│   │   └── nodes/
│   │       ├── process.py         # Query parsing and extraction
│   │       ├── reasoning.py       # Simple & deep reasoning nodes
│   │       ├── recommend.py       # ML model invocation
│   │       ├── evaluator.py       # Quality assessment
│   │       ├── refine.py          # Refinement strategies
│   │       └── output.py          # Response normalization
│   ├── models/
│   │   └── recommender.py         # ZestyRecommender ML engine
│   ├── services/
│   │   ├── authentication.py      # OAuth 2.0 flow (future)
│   │   ├── session_manager.py     # JWT session management (future)
│   │   ├── memory_system.py       # User memory & preferences (future)
│   │   ├── privacy_manager.py     # Data privacy controls (future)
│   │   ├── query_parser.py        # Tag/intent/complexity extraction
│   │   ├── tag_mapper.py          # Co-occurrence tag expansion
│   │   └── quality_evaluator.py   # Metric computation
│   ├── config/
│   │   └── auth_example.py        # Authentication config template
│   └── utils/
│       └── helpers.py             # Utility functions
├── frontend/                      # React + Vite Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Hero/              # Landing hero section
│   │   │   ├── PipelineStory/     # Agent pipeline visualization
│   │   │   ├── RecommendationEngine/  # Main search interface
│   │   │   │   ├── RecommendationEngine.jsx
│   │   │   │   ├── ResultCard.jsx     # Individual result card
│   │   │   │   └── ThinkingAnimation.jsx  # Loading animation
│   │   │   ├── WhyThisResult/     # Reasoning trace drawer
│   │   │   ├── MemorySection/     # Memory system showcase
│   │   │   ├── AuthCard/          # Authentication info
│   │   │   └── CTAFooter/         # Call-to-action footer
│   │   ├── assets/                # Images and static files
│   │   ├── App.jsx                # Main React component
│   │   ├── api.js                 # Backend API client
│   │   ├── main.jsx               # React entry point
│   │   └── App.css                # Global styles
│   ├── public/                    # Static assets
│   ├── index.html                 # HTML template
│   ├── package.json               # Node dependencies
│   └── vite.config.js             # Vite configuration
├── data/
│   ├── refined_anime_dataset.json # 13,000+ anime entries
│   ├── refined_manga_dataset.json # 13,000+ manga entries
│   ├── anime_recommender.pkl      # Trained anime model (gitignored)
│   └── manga_recommender.pkl      # Trained manga model (gitignored)
├── scripts/
│   └── train_model.py             # Model training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Troubleshooting

**Issue: "uvicorn: command not found"**
- Solution: Ensure virtual environment is activated and dependencies are installed

**Issue: "Model file not found"**
- Solution: Run `python scripts/train_model.py` to generate model files

**Issue: "Dataset file not found"**
- Solution: Ensure `refined_anime_dataset.json` and `refined_manga_dataset.json` exist in `data/` directory

**Issue: "Out of memory during training"**
- Solution: Reduce batch size in `train_model.py` or use a machine with more RAM

**Issue: "Slow response times"**
- Solution: Models are cached after first load. Subsequent requests should be fast (~50-200ms)

### Testing

Run the test suite:
```bash
pytest backend/tests/ -v
```

Run with coverage report:
```bash
pytest backend/tests/ --cov=backend --cov-report=html
```

### Performance Metrics

- **Model Loading**: ~2-3 seconds (first request only, then cached)
- **Query Processing**: ~50-200ms per request
- **Refinement Cycles**: +100-150ms per cycle (maximum 2 cycles)
- **Dataset Size**: 26,000+ entries
- **Model Size**: ~150 MB per model file

### Quality Metrics

| Metric | Threshold | Action if Below |
|--------|-----------|-----------------|
| Result Count | ≥ 3 | Trigger refinement |
| Coverage | ≥ 0.4 | Add uncovered tags |
| Average Score | ≥ 6.0 | Switch to semantic search |

**Typical Refinement Distribution:**
- 0 cycles: ~70% of queries (high quality on first attempt)

## Contributors

This project was developed as part of an AI and Knowledge Representation course project.

## License

This project is for educational purposes.

## Acknowledgments

- **MyAnimeList** for providing anime and manga data
- **Sentence-BERT** for semantic embedding models
- **LangGraph** for agent orchestration framework
- **FastAPI** for the web framework

---
