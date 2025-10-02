#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content-based Movie Recommender (genres + tags surrogate for plot)
- Downloads MovieLens (ml-latest-small)
- Builds a TF-IDF model over a "soup" of genres + top tags
- Recommends by cosine similarity
- Fuzzy title search
Usage:
    python content_based.py --title "Toy Story (1995)" --topn 10
"""

import argparse
import io
import os
import zipfile
import certifi
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-latest-small")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EXTRACT_DIR):
        print("📥 Downloading MovieLens (ml-latest-small)...")
        r = requests.get(DATA_URL, timeout=60, verify=certifi.where())
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            f.write(r.content)
        print("📦 Extracting...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        print("✅ Data ready at:", EXTRACT_DIR)
    else:
        print("✅ Data already present at:", EXTRACT_DIR)


def load_movielens():
    movies = pd.read_csv(os.path.join(EXTRACT_DIR, "movies.csv"))
    # movies: movieId, title, genres (pipe-separated)
    # tags: userId, movieId, tag, timestamp (sparse but useful as "plot-ish" signals)
    tags = pd.read_csv(os.path.join(EXTRACT_DIR, "tags.csv"))
    links = pd.read_csv(os.path.join(EXTRACT_DIR, "links.csv"))  # may be handy later (tmdbId for posters/plots)

    # Normalize genres
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["genre_tokens"] = (
        movies["genres"]
        .str.replace("-", " ", regex=False)
        .str.replace("|", " ", regex=False)
        .str.lower()
    )

    # Aggregate tags per movie (most common first)
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(list)
        .apply(lambda lst: [t for t in lst if t and t != "nan"])
        .reset_index()
    )
    # Keep top-K tags by frequency per movie
    TOP_K_TAGS = 25
    top_tags_per_movie = {}
    for _, row in tag_agg.iterrows():
        counts = Counter(row["tag"])
        top = [t for t, _ in counts.most_common(TOP_K_TAGS)]
        top_tags_per_movie[row["movieId"]] = " ".join(top)

    movies["tags_text"] = movies["movieId"].map(lambda mid: top_tags_per_movie.get(mid, ""))

    # Build the soup = genres + tags (tags act as a plot/keywords surrogate)
    movies["soup"] = (movies["genre_tokens"].fillna("") + " " + movies["tags_text"].fillna("")).str.strip()

    # Attach tmdbId for later extras (posters, real overviews)
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    return movies


def build_model(movies: pd.DataFrame):
    """Build TF-IDF vectorizer and cosine-similarity matrix over the soup."""
    # If tags are very sparse, fall back to just genres
    use_soup = movies["soup"].copy()
    missing_mask = use_soup.str.len() < 3
    use_soup.loc[missing_mask] = movies.loc[missing_mask, "genre_tokens"].fillna("")

    # TF-IDF with unigrams+bigrams, English stop words, min_df=2 to reduce noise
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2)
    tfidf = vectorizer.fit_transform(use_soup)  # shape: (n_movies, n_features)

    # Cosine similarity via linear_kernel (since TF-IDF is already L2-normalized)
    cosine_sim = linear_kernel(tfidf, tfidf)  # (n_movies x n_movies)

    # Mapping: title -> indices (there can be duplicates with same title different year)
    title_to_indices = defaultdict(list)
    for idx, title in enumerate(movies["title"].tolist()):
        title_to_indices[title].append(idx)

    return vectorizer, tfidf, cosine_sim, title_to_indices


def _best_title_match(query: str, titles: list, limit: int = 5):
    """Fuzzy match query to known titles; return (title, score)."""
    # Use token_set_ratio to be lenient with years, punctuation
    matches = process.extract(query, titles, scorer=fuzz.token_set_ratio, limit=limit)
    return matches  # [(best_title, score, idx_in_titles) ...]


def recommend_by_title(query_title: str, movies: pd.DataFrame, cosine_sim: np.ndarray, topn: int = 10, min_score: int = 70):
    titles = movies["title"].tolist()
    matches = _best_title_match(query_title, titles, limit=5)
    if not matches:
        raise ValueError(f"No close matches for title: {query_title!r}")

    # Pick the best match above a threshold; otherwise show suggestions
    best_title, score, best_idx_in_titles = matches[0]
    if score < min_score:
        suggestions = [m[0] for m in matches]
        raise ValueError(
            f"Couldn't confidently match {query_title!r} (best score {score}). "
            f"Did you mean: {', '.join(suggestions)} ?"
        )

    # There can be duplicates of the same title; gather all their indices
    candidate_indices = [i for i, t in enumerate(titles) if t == best_title]

    # Average the similarity vectors if multiple same-titled entries exist
    sim_vec = np.mean(cosine_sim[candidate_indices, :], axis=0)

    # Rank all movies by similarity
    # Exclude the candidate_indices themselves from recommendations
    all_indices = np.arange(sim_vec.shape[0])
    exclude = set(candidate_indices)
    mask = np.array([i not in exclude for i in all_indices], dtype=bool)
    ranked = np.argsort(sim_vec[mask])[::-1]  # descending

    # Map back to actual indices
    final_indices = all_indices[mask][ranked][:topn]
    results = movies.iloc[final_indices][["movieId", "title", "genres", "tmdbId"]].copy()
    results["similarity"] = sim_vec[final_indices]
    results.reset_index(drop=True, inplace=True)

    return best_title, results


def main():
    parser = argparse.ArgumentParser(description="Content-based movie recommender (genres + tags).")
    parser.add_argument("--title", type=str, required=True, help='Movie title, e.g. "Toy Story (1995)"')
    parser.add_argument("--topn", type=int, default=10, help="Number of recommendations")
    args = parser.parse_args()

    ensure_data()
    print("📚 Loading MovieLens...")
    movies = load_movielens()

    print("🧠 Building TF-IDF model...")
    _, _, cosine_sim, _ = build_model(movies)

    print(f"🔎 Finding movies similar to: {args.title!r}")
    try:
        matched_title, recs = recommend_by_title(args.title, movies, cosine_sim, topn=args.topn)
    except ValueError as e:
        print("❌", e)
        return

    print(f"✅ Matched: {matched_title}")
    print(f"\n🎬 Top {args.topn} recommendations:")
    for i, row in recs.iterrows():
        print(f"{i+1:>2}. {row['title']}  |  genres: {row['genres']}  |  score: {row['similarity']:.4f}")

    print("\nTip: If title matching seems off, include the year, e.g. 'Heat (1995)'.")
    print("Next steps: We'll add collaborative filtering (Surprise) and a Streamlit app.")


if __name__ == "__main__":
    main()

