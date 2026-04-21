"""
recommender.py — Hybrid Recommendation Engine for Zesty.

Combines three signals for scoring:
    1. Semantic Similarity  — Sentence-BERT embeddings + cosine similarity
    2. Tag Overlap          — Jaccard similarity on genre/tag sets
    3. Popularity Bias      — Normalized community rating

Scoring formula (default "hybrid" strategy):
    final = 0.6 × semantic + 0.3 × tag_jaccard + 0.1 × popularity

The weights shift depending on search_strategy:
    "reference"  → heavier on semantic (0.7)
    "tag_only"   → heavier on Jaccard  (0.6)
    "semantic"   → almost all semantic  (0.8)
    "hybrid"     → balanced default     (0.6 / 0.3 / 0.1)

Contract (required by backend/agent/nodes/recommend.py):
    model.recommend(input_data: dict) -> list[dict]
"""

import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
        - Returns top-N results with match reasons
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

    # ── Main recommendation method ────────────────────────────────────────

    def recommend(self, input_data):
        """
        Produce ranked recommendations from a structured input payload.

        Args:
            input_data (dict): Payload built by the reasoning node.
                Required keys: tags (list[str])
                Optional keys: reference (str), search_strategy (str),
                               semantic_hints (list[str])

        Returns:
            list[dict]: Up to 10 recommendation dicts, each containing
                title, image, synopsis, score, genres, similarity_score,
                and match_reason.
        """
        tags = input_data.get("tags", [])
        reference = input_data.get("reference", "")
        strategy = input_data.get("search_strategy", "hybrid")
        semantic_hints = input_data.get("semantic_hints", [])
        limit = 10

        query_tag_set = set(t.lower() for t in tags)

        # ── 1. Semantic similarity ─────────────────────────────────────────
        semantic_scores = np.zeros(len(self.df))

        if reference:
            # Try to find the reference title in the dataset
            mask = self.df["title"].str.contains(
                reference, case=False, na=False
            )
            ref_indices = self.df[mask].index

            if not ref_indices.empty:
                # Use pre-computed embedding — no model load needed
                ref_vec = self.embeddings[ref_indices[0]]
                semantic_scores = self.embeddings @ ref_vec
            else:
                # Reference not in dataset — encode the query instead
                query_text = f"{reference} {' '.join(tags)} {' '.join(semantic_hints)}"
                encoder = self._get_encoder()
                q_vec = encoder.encode([query_text])[0]
                q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
                semantic_scores = self.embeddings @ q_vec
        else:
            # No reference — encode tags + hints as the query
            query_text = " ".join(tags) + " " + " ".join(semantic_hints)
            if query_text.strip():
                encoder = self._get_encoder()
                q_vec = encoder.encode([query_text])[0]
                q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
                semantic_scores = self.embeddings @ q_vec

        semantic_scores = np.maximum(semantic_scores, 0)  # clamp negatives

        # ── 2. Jaccard tag similarity ──────────────────────────────────────
        jaccard_scores = np.array(
            [self._jaccard(query_tag_set, ts) for ts in self.df["tag_set"]]
        )

        # ── 3. Blend with strategy-dependent weights ───────────────────────
        weights = {
            "reference": (0.7, 0.2, 0.1),
            "tag_only":  (0.3, 0.6, 0.1),
            "semantic":  (0.8, 0.1, 0.1),
            "hybrid":    (0.6, 0.3, 0.1),
        }
        w_sem, w_jac, w_pop = weights.get(strategy, weights["hybrid"])
        final_scores = (
            w_sem * semantic_scores
            + w_jac * jaccard_scores
            + w_pop * self.popularity
        )

        # ── 4. Rank and collect results ────────────────────────────────────
        top_indices = final_scores.argsort()[::-1]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]

            # Skip the reference title itself
            if reference and reference.lower() in str(row["title"]).lower():
                continue

            synopsis_raw = str(row.get("synopsis", "No synopsis available."))
            synopsis = (
                synopsis_raw[:250] + "..."
                if len(synopsis_raw) > 250
                else synopsis_raw
            )

            results.append(
                {
                    "title": str(row.get("title", "Unknown")),
                    "image": str(row.get("image", "")),
                    "synopsis": synopsis,
                    "score": float(row.get("score", 0.0)),
                    "genres": (
                        row["clean_tags"].split() if row["clean_tags"] else []
                    ),
                    "similarity_score": round(float(final_scores[idx]), 4),
                    "match_reason": self._build_match_reason(
                        float(semantic_scores[idx]),
                        float(jaccard_scores[idx]),
                        strategy,
                    ),
                }
            )

            if len(results) >= limit:
                break

        return results

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
