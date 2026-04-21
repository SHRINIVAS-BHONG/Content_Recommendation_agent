"""
train_model.py — Build and serialise the ZestyRecommender .pkl models.

Usage (from the project root):
    python scripts/train_model.py

Outputs:
    data/anime_recommender.pkl
    data/manga_recommender.pkl
"""

import json
import sys
import time
from pathlib import Path

import joblib

# Add project root to sys.path so we can import backend.models
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.models.recommender import ZestyRecommender


def train():
    data_dir = Path(__file__).resolve().parents[1] / "data"

    # ── Anime ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Training Anime Recommender")
    print("=" * 60)
    anime_path = data_dir / "refined_anime_dataset.json"
    with open(anime_path, "r", encoding="utf-8") as f:
        anime_data = json.load(f)
    print(f"  Loaded {len(anime_data):,} anime entries from {anime_path.name}")

    t0 = time.time()
    anime_model = ZestyRecommender(anime_data)
    print(f"  Encoding complete in {time.time() - t0:.1f}s")

    out_anime = data_dir / "anime_recommender.pkl"
    joblib.dump(anime_model, out_anime)
    print(f"  ✓ Saved → {out_anime}  ({out_anime.stat().st_size / 1e6:.1f} MB)")

    # ── Manga ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Training Manga Recommender")
    print("=" * 60)
    manga_path = data_dir / "refined_manga_dataset.json"
    with open(manga_path, "r", encoding="utf-8") as f:
        manga_data = json.load(f)
    print(f"  Loaded {len(manga_data):,} manga entries from {manga_path.name}")

    t0 = time.time()
    manga_model = ZestyRecommender(manga_data)
    print(f"  Encoding complete in {time.time() - t0:.1f}s")

    out_manga = data_dir / "manga_recommender.pkl"
    joblib.dump(manga_model, out_manga)
    print(f"  ✓ Saved → {out_manga}  ({out_manga.stat().st_size / 1e6:.1f} MB)")

    print()
    print("Done! Both models saved to data/")


if __name__ == "__main__":
    train()
