"""
quality_evaluator.py — Quality metric computation for the evaluator node.

Encapsulates all metric logic so that evaluator_node stays thin.
"""

from typing import List, Dict


def evaluate_results(
    results: List[dict],
    tags: List[str],
    refinement_count: int,
) -> Dict:
    """
    Compute quality metrics for a list of recommendation results and issue a verdict.

    Args:
        results:          List of result dicts, each expected to have "score" and "genres" keys.
        tags:             The query tags that the results should cover.
        refinement_count: Number of refinement cycles already completed (0–2).

    Returns:
        A dict with keys:
            coverage       (float)     — fraction of tags represented in any result's genres
            diversity      (int)       — count of distinct genres across all results
            avg_score      (float)     — mean of result["score"] values
            result_count   (int)       — number of results
            verdict        (str)       — "quality_ok" or "needs_refinement"
            uncovered_tags (List[str]) — tags not represented in any result's genres
    """
    result_count = len(results)

    # ── Collect all genres (case-insensitive) ────────────────────────────
    all_genres_lower: set = set()
    for result in results:
        for genre in result.get("genres", []):
            if isinstance(genre, str):
                all_genres_lower.add(genre.lower())

    # ── Coverage ─────────────────────────────────────────────────────────
    if not tags:
        coverage = 0.0
        uncovered_tags: List[str] = []
    else:
        covered = [t for t in tags if t.lower() in all_genres_lower]
        uncovered_tags = [t for t in tags if t.lower() not in all_genres_lower]
        coverage = len(covered) / len(tags)

    # ── Diversity ─────────────────────────────────────────────────────────
    diversity = len(all_genres_lower)

    # ── Average score ─────────────────────────────────────────────────────
    if not results:
        avg_score = 0.0
    else:
        scores = [result.get("score", 0.0) for result in results]
        avg_score = sum(scores) / len(scores)

    # ── Verdict logic ─────────────────────────────────────────────────────
    if (
        result_count < 3
        or (coverage < 0.4 and refinement_count < 2)
        or (avg_score < 6.0 and refinement_count < 2)
    ):
        verdict = "needs_refinement"
    else:
        verdict = "quality_ok"

    return {
        "coverage": coverage,
        "diversity": diversity,
        "avg_score": avg_score,
        "result_count": result_count,
        "verdict": verdict,
        "uncovered_tags": uncovered_tags,
    }
