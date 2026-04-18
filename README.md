# Anime & Manga Recommendation Agent

A LangGraph-powered recommendation system with Google OAuth 2.0 authentication, per-user memory, and a self-refining 7-node pipeline.

**Stack:** FastAPI · LangGraph · Google OAuth 2.0 · JWT (RS256) · Hypothesis (property-based testing)

---

## Agent Pipeline

```mermaid
flowchart TD
    A([User Query]) --> B[process_node\nextract tags · type · reference\nintent · complexity_score · semantic_hints]

    B -->|complexity < 0.5| C[simple_reasoning_node\nexpand tags via co-occurrence\nset strategy = tag_only\nbuild model_input]
    B -->|complexity >= 0.5| D[deep_reasoning_node\nexpand tags + translate semantic_hints\nlookup reference synopsis\nmap intent to strategy\nbuild model_input]

    C --> E
    D --> E

    E[recommend_node\nload .pkl model via joblib lru_cache\ncall model.recommend\napply personalization weights\ndeduplicate vs user history]

    E --> F[evaluator_node\ncompute coverage · diversity · avg_score\nverdict = needs_refinement if\n  result_count < 3\n  OR coverage < 0.4\n  OR avg_score < 6.0]

    F -->|needs_refinement AND count < 2| G[refine_node\nS1: add uncovered_tags\nS2: switch tag_only to semantic\nS3: drop last tag + strictness=low\nrefinement_count += 1]

    G -->|loop back max 2x| E

    F -->|quality_ok OR count >= 2| H[output_node\nnormalize result dicts\nguarantee required fields]

    H --> I([API Response\nresults · reasoning_trace · refinement_count])
```

**Routing rules:**
- `route_query` — `complexity_score >= 0.5` → deep reasoning, else simple
- `route_evaluation` — `verdict == "needs_refinement" AND refinement_count < 2` → refine loop, else output

**Refinement strategies** (applied in order inside `refine_node`):
| Strategy | Trigger | Action |
|---|---|---|
| S1 — low coverage | `uncovered_tags` non-empty | add uncovered tags back |
| S2 — low score | `avg_score < 6.0` AND strategy is `tag_only` | switch to `semantic` |
| S3 — second cycle | `refinement_count == 1` | drop last tag, set `strictness = "low"` |

---

## Authentication Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant F as FastAPI
    participant A as AuthService
    participant G as Google OAuth

    C->>F: POST /auth/login
    F->>A: initiate_oauth()
    A-->>F: redirect_url · state · code_verifier
    F-->>C: { redirect_url, state, code_verifier }

    C->>G: visit redirect_url (PKCE + state)
    G-->>C: authorization_code

    C->>F: POST /auth/callback { code, state, code_verifier }
    F->>A: handle_oauth_callback()
    A->>G: POST /token { code, code_verifier, client_secret }
    G-->>A: { access_token, refresh_token }
    A->>G: GET /userinfo (Bearer access_token)
    G-->>A: { id, email, name, picture }
    A->>A: create_or_update_user_profile()
    A->>A: SessionManager.create_session()<br/>JWT RS256 · 24h · refresh SHA-256 · 30d
    A-->>F: { access_token, refresh_token, user_profile }
    F-->>C: { access_token, refresh_token, user_profile }
```

### JWT Token Lifecycle

**Create Session**
```mermaid
flowchart TD
    A([create_session]) --> B[build JWT payload\nsub · iat · exp now+24h · session_id · email · jti]
    B --> C[jwt.encode RSA private key RS256\n→ access_token]
    C --> D[secrets.token_urlsafe 32\nstore as SHA-256 hash · expires now+30d\n→ refresh_token]
    D --> E([cache SessionContext in _active_sessions])
```

**Validate Session**
```mermaid
flowchart TD
    A([validate_session token]) --> B[jwt.decode RSA public key\nverify signature · expiry · iss · aud]
    B -->|valid| C{session_id in\n_active_sessions?}
    C -->|yes · not expired| D([return SessionContext])
    C -->|no or expired| E([return None])
    B -->|invalid / expired| E
```

**Refresh & Invalidate**
```mermaid
flowchart TD
    A([should_refresh_token]) --> B{expires_at - now\n<= 2h?}
    B -->|yes| C[hash token → lookup _refresh_tokens\ncheck not expired\nissue new access_token same session_id]
    C --> D([return new JWTToken])
    B -->|no| E([no action])

    F([invalidate_session]) --> G[remove from _active_sessions]
    G --> H([remove all refresh tokens\nfor this session_id])
```

---

## Authenticated Request Flow

```mermaid
flowchart TD
    A([POST /recommend\nAuthorization: Bearer JWT]) --> B[Auth Middleware\nvalidate_session token]
    B -->|invalid / expired| C([HTTP 401])
    B -->|valid| D[load user context from MemorySystem\nget_user_preferences\nget_conversation_context recent=5]
    D --> E[build personalization_weights\npreferred genre → 1.5 boost\navoided genre  → 0.3 suppress]
    E --> F[run_agent\nquery · user_id · user_preferences\nconversation_context · feedback_history]
    F --> G[LangGraph Pipeline\nsee Agent Pipeline above]
    G --> H[store_interaction\nquery · results · reasoning_trace\nsession_id · processing_time_ms]
    H --> I([return results · reasoning_trace\nrefinement_count · is_authenticated])
```

---

## Memory & Preference Learning

**Store Interaction**
```mermaid
flowchart TD
    A([store_interaction]) --> B{privacy_manager\nshould_collect_data?}
    B -->|no| C([return dummy — nothing stored])
    B -->|yes| D[registry.store_interaction\nquery · results · feedback\ntimestamp · session_id · reasoning_trace]
    D --> E([log_data_modification audit])
```

**Get Conversation History**
```mermaid
flowchart TD
    A([get_conversation_history\nlimit=50 offset=0]) --> B[validate_access\nrequesting_user == resource_user]
    B -->|denied| C([PermissionError])
    B -->|allowed| D([registry.get_user_interactions\nList sorted reverse-chronological])
```

**Learn Preferences**
```mermaid
flowchart TD
    A([learn_preferences\nmin_interactions=5]) --> B{enough\ninteractions?}
    B -->|no| C([return learned=false])
    B -->|yes| D[for each interaction\nfeedback_weight:\nrating 5/4 → +1.0\nrating 3   → +0.7\nrating 2   → +0.3\nrating 1/0 → -0.5\nnone       → +0.5]
    D --> E[accumulate genre_scores\nnormalize to 0–1\nstore patterns confidence >= 0.3]
    E --> F([return learned=true · patterns_stored])
```

**Update Preference Confidence**
```mermaid
flowchart TD
    A([update_preference_confidence]) --> B{pattern exists?}
    B -->|yes| C[EMA: new = 0.7 × old + 0.3 × weight\nclamped to -1.0 – 1.0]
    B -->|no| D[new = weight × 0.5\nconservative start]
    C --> E([store updated pattern])
    D --> E
```

---

## Project Structure

```
backend/
├── main.py                    # FastAPI app + all endpoints
├── agent/
│   ├── graph.py               # LangGraph pipeline, route_query(), route_evaluation()
│   ├── state.py               # AgentState TypedDict (19 fields) + initial_state()
│   └── nodes/
│       ├── process.py         # extract tags, type, reference, intent, complexity, hints
│       ├── reasoning.py       # simple + deep reasoning, tag expansion, hint translation
│       ├── recommend.py       # load .pkl model, personalization weights, deduplication
│       ├── evaluator.py       # coverage, diversity, avg_score, verdict
│       ├── refine.py          # 3 refinement strategies, rebuild model_input
│       └── output.py          # normalize results to response schema
├── services/
│   ├── authentication.py      # OAuth 2.0 flow, PKCE, user profile management
│   ├── session_manager.py     # JWT RS256 create/validate/refresh/invalidate
│   ├── memory_system.py       # conversation history, preference learning, EMA updates
│   ├── privacy_manager.py     # retention policies, audit logs, GDPR access control
│   ├── user_memory_store.py   # in-memory user registry (UserMemoryRegistry)
│   ├── query_parser.py        # 6 extraction functions for process_node
│   ├── tag_mapper.py          # co-occurrence tag expansion + _HINT_TO_TAGS
│   └── quality_evaluator.py   # evaluate_results() → coverage/diversity/verdict
├── utils/helpers.py           # deduplicate(), normalize_result(), safe_get()
└── tests/                     # unit + integration + property-based (Hypothesis)
data/
├── anime_recommender.pkl      # gitignored — joblib model
├── manga_recommender.pkl
├── refined_anime_dataset.json # used for co-occurrence index + synopsis lookup
└── refined_manga_dataset.json
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/recommend` | optional | Run agent pipeline, returns results + reasoning_trace |
| `GET` | `/health` | — | Liveness check |
| `POST` | `/auth/login` | — | Start OAuth, returns redirect_url + state + code_verifier |
| `POST` | `/auth/callback` | — | Exchange code for JWT + refresh token |
| `POST` | `/auth/logout` | ✅ | Invalidate session |
| `GET` | `/user/profile` | ✅ | Get user profile |
| `PUT` | `/user/preferences` | ✅ | Update preferences |
| `GET` | `/user/history` | ✅ | Paginated conversation history (50/page) |
| `DELETE` | `/user/account` | ✅ | Schedule full data deletion |

---

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in values below
uvicorn backend.main:app --reload
# Docs → http://localhost:8000/docs
```

Required `.env`:

```
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
REDIRECT_URI=http://localhost:8000/auth/callback
JWT_SECRET_KEY=your-secret-key
```

```bash
pytest backend/tests/ -v   # run full test suite
```

---

## Model Contract

Models must be joblib-serialized with `recommend(dict) -> list[dict]`.

Input keys: `tags`, `type`, `reference`, `intent`, `semantic_hints`, `search_strategy`, `reference_synopsis`, `complexity`, `is_authenticated`, `user_preferences`, `conversation_context` (+ optional `strictness` on 2nd refinement cycle).

Each result dict must have: `title`, `image`, `synopsis`, `score`, `genres`. Optional: `similarity_score`, `match_reason`.
