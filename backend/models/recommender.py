"""
recommender.py — Hybrid Recommendation Engine for Zesty.

Combines three signals for scoring:
    1. Semantic Similarity  — Sentence-BERT embeddings + cosine similarity
    2. Tag Overlap          — Jaccard similarity on genre/tag sets
    3. Popularity Bias      — Normalized community rating

Scoring formula (default "hybrid" strategy):
    final = 0.5 × semantic + 0.3 × tag_jaccard + 0.2 × popularity

The weights shift depending on search_strategy:
    "reference"  → heavier on semantic (0.65)
    "tag_only"   → heavier on Jaccard  (0.55)
    "semantic"   → almost all semantic  (0.70)
    "hybrid"     → balanced default     (0.50 / 0.30 / 0.20)

Popularity weight is 0.20 (up from 0.10) so well-known, highly-rated titles
surface more reliably alongside thematic matches.

Contract (required by backend/agent/nodes/recommend.py):
    model.recommend(input_data: dict) -> list[dict]
"""

import re
import ast
import numpy as np
import pandas as pd


class ZestyRecommender:
    """
    Hybrid Recommendation Engine.

    Training-time (in __init__):
        - Cleans the raw dataset tags
        - Encodes all titles+tags+synopses into Sentence-BERT embeddings
        - Pre-normalises embeddings for fast cosine similarity at query time
        - Pre-computes normalised popularity scores

    Query-time (in recommend()):
        - Computes semantic, Jaccard, and popularity scores
        - Blends them with strategy-dependent weights
        - Deduplicates by franchise (strips season/OVA/Specials suffixes)
        - Returns top-10 results with match reasons
    """

    def __init__(self, data_list, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._encoder = None  # NOT stored in .pkl — lazy-loaded at query time

        # ── Build DataFrame ────────────────────────────────────────────────
        self.df = pd.DataFrame(data_list)

        # ── Clean tags ─────────────────────────────────────────────────────
        self.df["clean_tags"] = self.df["tags"].apply(self._clean_tags)
        self.df["tag_set"] = self.df["clean_tags"].apply(
            lambda x: set(x.lower().split()) if x else set()
        )

        # ── Build combined metadata text for embedding ─────────────────────
        self.df["metadata"] = (
            self.df["title"].fillna("")
            + ". "
            + self.df["clean_tags"].fillna("")
            + ". "
            + self.df["synopsis"].fillna("")
        )

        # ── Compute Sentence-BERT embeddings ───────────────────────────────
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name)
        texts = self.df["metadata"].tolist()
        self.embeddings = encoder.encode(
            texts, show_progress_bar=True, batch_size=64
        )

        # Pre-normalise so dot-product == cosine similarity later
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms

        # ── Pre-compute normalised popularity ──────────────────────────────
        scores = self.df["score"].astype(float).fillna(0)
        max_score = scores.max() if scores.max() > 0 else 1
        self.popularity = (scores / max_score).values

    # ── Tag cleaning ──────────────────────────────────────────────────────

    @staticmethod
    def _clean_tags(tags):
        """Standardise messy tag data into a clean space-separated string."""
        if not tags:
            return ""
        if isinstance(tags, list):
            cleaned = []
            for t in tags:
                t_str = str(t).strip().strip("'\"[]")
                if t_str:
                    cleaned.append(t_str)
            return " ".join(cleaned)
        try:
            parsed = ast.literal_eval(str(tags))
            if isinstance(parsed, list):
                return " ".join(str(t).strip() for t in parsed)
        except (ValueError, SyntaxError):
            pass
        return (
            str(tags)
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace(",", " ")
        )

    # ── Lazy encoder loader ───────────────────────────────────────────────

    def _get_encoder(self):
        """Lazy-load SentenceTransformer at query time (not stored in .pkl)."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    # ── Jaccard helper ────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(set_a, set_b):
        """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    # ── Base-title deduplication helper ──────────────────────────────────

    # Compiled once at class level for efficiency
    _SUFFIX_RE = re.compile(
        r"\s*[\(\[]?("
        r"season\s*\d*|s\d+|part\s*\d+|cour\s*\d+|"
        r"\d+(?:st|nd|rd|th)\s+season|"
        r"\bii\b|\biii\b|\biv\b|\bv\b|\bvi\b|\bvii\b|\bviii\b|"
        r"specials?|ova|ona|"
        r"movie\s*\d*|film\s*\d*|the\s+movie|"
        r"final\s+(?:season|arc|chapter)|"
        r"rewrite|re-cut|recap s?|recaps?"
        r")[\)\]]?$",
        re.IGNORECASE,
    )

    @classmethod
    def _base_title(cls, title: str) -> str:
        """
        Reduce a title to its franchise root for deduplication.

        Strategy (applied in order):
          1. Strip trailing season/part/OVA/movie/specials markers.
          2. Strip ": subtitle" colon-subtitles (word-colon-space pattern).
             Titles where the colon is mid-word (Re:Zero, Steins;Gate) are
             unaffected because they have no space after the colon's word.
          3. Strip trailing markers again — second pass catches "Movie 1" that
             was hidden behind a colon subtitle in step 1.
          4. Lowercase.

        Examples:
            'Trinity Seven Movie 1: Eternity Library'  -> 'trinity seven'
            'Sword Art Online Movie: Ordinal Scale'    -> 'sword art online'
            'One Piece Film: Red'                      -> 'one piece'
            'Shingeki no Kyojin: The Final Season'     -> 'shingeki no kyojin'
            'Re:Zero kara Hajimeru Isekai Seikatsu'    -> 're:zero kara hajimeru isekai seikatsu'
            'Steins;Gate'                              -> 'steins;gate'
            'Death Note Rewrite'                       -> 'death note'
        """
        t = title.strip()

        # Pass 1 — strip trailing suffix markers
        prev = None
        while prev != t:
            prev = t
            t = cls._SUFFIX_RE.sub("", t).strip()

        # Strip ": subtitle" — only when colon has a space before it
        # (preserves "Re:Zero", "Steins;Gate" where colon is mid-word)
        t = re.sub(r":\s+\S.*$", "", t).strip()

        # Pass 2 — strip again after colon removal (catches "Movie 1" etc.)
        prev = None
        while prev != t:
            prev = t
            t = cls._SUFFIX_RE.sub("", t).strip()

        return t.lower()

    # ── Main recommendation method ────────────────────────────────────────

    def recommend(self, input_data):
        """
        Produce ranked recommendations from a structured input payload.

        Args:
            input_data (dict): Payload built by the reasoning node.
                Required keys: tags (list[str])
                Optional keys: reference (str), search_strategy (str),
                               semantic_hints (list[str]), page (int, 0-indexed)

        Returns:
            list[dict]: Up to 10 recommendation dicts per page, each containing
                title, image, synopsis, score, genres, similarity_score,
                and match_reason.
        """
        tags = input_data.get("tags", [])
        reference = input_data.get("reference", "")
        strategy = input_data.get("search_strategy", "hybrid")
        semantic_hints = input_data.get("semantic_hints", [])
        media_type = input_data.get("type", "anime")
        page = int(input_data.get("page", 0))
        page_size = 10
        candidate_scan_limit = 500  # scan enough to fill multiple pages

        query_tag_set = set(t.lower() for t in tags)

        # ── 1. Semantic similarity ─────────────────────────────────────────
        semantic_scores = np.zeros(len(self.df))

        if reference:
            mask = self.df["title"].str.contains(reference, case=False, na=False)
            ref_indices = self.df[mask].index
            if not ref_indices.empty:
                ref_vec = self.embeddings[ref_indices[0]]
                semantic_scores = self.embeddings @ ref_vec
            else:
                query_text = f"{reference} {' '.join(tags)} {' '.join(semantic_hints)}"
                encoder = self._get_encoder()
                q_vec = encoder.encode([query_text])[0]
                q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
                semantic_scores = self.embeddings @ q_vec
        else:
            query_text = " ".join(tags) + " " + " ".join(semantic_hints)
            if query_text.strip():
                encoder = self._get_encoder()
                q_vec = encoder.encode([query_text])[0]
                q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
                semantic_scores = self.embeddings @ q_vec

        semantic_scores = np.maximum(semantic_scores, 0)

        # ── 2. Jaccard tag similarity ──────────────────────────────────────
        jaccard_scores = np.array(
            [self._jaccard(query_tag_set, ts) for ts in self.df["tag_set"]]
        )

        # ── 3. Blend scores ────────────────────────────────────────────────
        # Popularity weight 0.25 ensures well-known, highly-rated titles
        # surface reliably. Semantic + Jaccard cover thematic relevance.
        weights = {
            "reference": (0.55, 0.20, 0.25),
            "tag_only":  (0.20, 0.55, 0.25),
            "semantic":  (0.60, 0.15, 0.25),
            "hybrid":    (0.45, 0.30, 0.25),
        }
        w_sem, w_jac, w_pop = weights.get(strategy, weights["hybrid"])
        final_scores = (
            w_sem * semantic_scores
            + w_jac * jaccard_scores
            + w_pop * self.popularity
        )

        # ── 4. Build full deduplicated ranked list ─────────────────────────
        top_indices = final_scores.argsort()[::-1]

        all_results = []
        seen_base_titles = set()
        SCORE_THRESHOLD = 7.0

        def _collect(indices, min_score, target):
            for rank, idx in enumerate(indices):
                if len(all_results) >= target:
                    break
                if rank >= candidate_scan_limit:
                    break

                row = self.df.iloc[idx]
                title = str(row.get("title", "Unknown"))
                item_score = float(row.get("score", 0.0))

                if reference and reference.lower() in title.lower():
                    continue
                if item_score < min_score:
                    continue

                base = self._base_title(title)
                if base in seen_base_titles:
                    continue
                seen_base_titles.add(base)

                synopsis_raw = str(row.get("synopsis", "No synopsis available."))
                synopsis = (
                    synopsis_raw[:500] + "..."
                    if len(synopsis_raw) > 500
                    else synopsis_raw
                )

                all_results.append({
                    "title": title,
                    "image": str(row.get("image", "")),
                    "synopsis": synopsis,
                    "score": item_score,
                    "genres": (
                        row["clean_tags"].split() if row["clean_tags"] else []
                    ),
                    "similarity_score": round(float(final_scores[idx]), 4),
                    "match_reason": self._build_match_reason(
                        float(semantic_scores[idx]),
                        float(jaccard_scores[idx]),
                        strategy,
                    ),
                })

        # First pass: well-rated titles only (score >= 7.0)
        _collect(top_indices, SCORE_THRESHOLD, (page + 1) * page_size + 10)
        # Second pass: fill with anything if not enough results
        if len(all_results) < (page + 1) * page_size:
            _collect(top_indices, 0.0, (page + 1) * page_size + 10)

        # ── 5. Paginate ────────────────────────────────────────────────────
        start = page * page_size
        end = start + page_size
        return all_results[start:end]

    # ── Match reason builder ──────────────────────────────────────────────

    @staticmethod
    def _build_match_reason(semantic, jaccard, strategy):
        """Generate a human-readable explanation for why this item matched."""
        parts = []
        if semantic > 0.5:
            parts.append(f"Strong thematic match ({semantic:.0%})")
        elif semantic > 0.3:
            parts.append(f"Moderate thematic match ({semantic:.0%})")
        elif semantic > 0.15:
            parts.append(f"Slight thematic match ({semantic:.0%})")

        if jaccard > 0.3:
            parts.append(f"High genre overlap ({jaccard:.0%})")
        elif jaccard > 0.1:
            parts.append(f"Some genre overlap ({jaccard:.0%})")

        if strategy == "reference":
            parts.append("Similar to reference title")

        return ". ".join(parts) if parts else "General recommendation"
