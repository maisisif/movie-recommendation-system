import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import process, fuzz

# --- SETTINGS ---
MOVIELENS_DIR = "data/ml-latest-small"
KAGGLE_MOVIES_PATH = "data/kaggle-movies/movies_metadata.csv"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200"  # poster size
PLACEHOLDER_POSTER = "https://via.placeholder.com/200x300?text=No+Image"

# --- Load data ---
@st.cache_data
def load_data():
    # MovieLens
    movies = pd.read_csv(os.path.join(MOVIELENS_DIR, "movies.csv"))
    tags = pd.read_csv(os.path.join(MOVIELENS_DIR, "tags.csv"))
    links = pd.read_csv(os.path.join(MOVIELENS_DIR, "links.csv"))

    # Kaggle TMDB metadata
    kaggle_meta = pd.read_csv(KAGGLE_MOVIES_PATH, low_memory=False)

    # --- Preprocess MovieLens ---
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["genre_tokens"] = (
        movies["genres"].str.replace("-", " ", regex=False)
        .str.replace("|", " ", regex=False)
        .str.lower()
    )

    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tag_agg = tags.groupby("movieId")["tag"].apply(list).reset_index()
    top_tags = {row.movieId: " ".join(row.tag[:20]) for _, row in tag_agg.iterrows()}
    movies["tags_text"] = movies["movieId"].map(lambda mid: top_tags.get(mid, ""))

    movies["soup"] = (
        movies["genre_tokens"].fillna("") + " " + movies["tags_text"].fillna("")
    ).str.strip()

    # --- Merge with tmdbId ---
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    # Convert Kaggle `id` column to numeric for join
    kaggle_meta["id"] = pd.to_numeric(kaggle_meta["id"], errors="coerce")

    # Merge poster_path
    movies = movies.merge(
        kaggle_meta[["id", "poster_path"]],
        left_on="tmdbId",
        right_on="id",
        how="left",
    )

    return movies


@st.cache_resource
def build_model(movies):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", min_df=2
    )
    tfidf = vectorizer.fit_transform(movies["soup"])
    cosine_sim = linear_kernel(tfidf, tfidf)
    return cosine_sim


def recommend(movie_title, movies, cosine_sim, topn=10):
    titles = movies["title"].tolist()
    matches = process.extract(
        movie_title, titles, scorer=fuzz.token_set_ratio, limit=5
    )
    if not matches:
        return None, []
    best_title, score, _ = matches[0]
    if score < 70:
        return None, [m[0] for m in matches]

    idx = movies[movies["title"] == best_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : topn + 1]
    indices = [i for i, _ in sim_scores]
    recs = movies.iloc[indices][["title", "genres", "poster_path"]].copy()
    recs["score"] = [s for _, s in sim_scores]
    return best_title, recs


# --- Streamlit UI ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender (Content-Based + Posters)")
st.write("Type a movie title and get visual recommendations based on genres & tags.")

movies = load_data()
cosine_sim = build_model(movies)

user_input = st.text_input("Enter a movie title:", "Toy Story (1995)")
topn = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    best_title, results = recommend(user_input, movies, cosine_sim, topn=topn)

    if best_title is None:
        st.error(f"Couldn't find an exact match. Did you mean: {', '.join(results)}?")
    else:
        st.success(f"Recommendations based on **{best_title}**:")

        # Display posters in a grid
        cols = st.columns(5)
        for i, row in results.iterrows():
            col = cols[i % 5]
            if pd.notna(row["poster_path"]):
                poster_url = TMDB_IMG_BASE + str(row["poster_path"])
                try:
                    col.image(poster_url, caption=row["title"], use_container_width=True)
                except:
                    col.image(PLACEHOLDER_POSTER, caption=row["title"], use_container_width=True)
            else:
                col.image(PLACEHOLDER_POSTER, caption=row["title"], use_container_width=True)

