import streamlit as st
import pandas as pd
import os
import requests
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import process, fuzz

# =====================
# SETTINGS
# =====================
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data"
ML_DIR = os.path.join(DATA_DIR, "ml-latest-small")
ML_ZIP_PATH = os.path.join(DATA_DIR, "ml.zip")

TMDB_KEY = os.getenv("TMDB_KEY")
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200"
PLACEHOLDER_POSTER = "https://via.placeholder.com/200x300?text=No+Image"


# =====================
# AUTO-DOWNLOAD MOVIELENS
# =====================
def ensure_movielens():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ML_DIR):
        return

    st.warning("MovieLens dataset not found. Downloading...")

    r = requests.get(MOVIELENS_URL)
    with open(ML_ZIP_PATH, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(ML_ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)

    st.success("MovieLens dataset downloaded and extracted.")


# =====================
# TMDB FETCH (ON-DEMAND, CACHED)
# =====================
@st.cache_data
def tmdb_get_metadata(tmdb_id):
    """Fetch TMDB poster + overview only when needed."""
    if pd.isna(tmdb_id) or tmdb_id == 0:
        return None, ""

    try:
        url = TMDB_MOVIE_URL.format(int(tmdb_id))
        data = requests.get(url, params={"api_key": TMDB_KEY}).json()
        poster = data.get("poster_path")
        overview = data.get("overview", "")
        return poster, overview
    except:
        return None, ""


# =====================
# LOAD MOVIELENS DATA
# =====================
@st.cache_data
def load_data():
    ensure_movielens()

    movies = pd.read_csv(os.path.join(ML_DIR, "movies.csv"))
    tags = pd.read_csv(os.path.join(ML_DIR, "tags.csv"))
    links = pd.read_csv(os.path.join(ML_DIR, "links.csv"))

    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["genre_tokens"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.lower()
    )

    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tag_map = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x[:20]))
    movies["tags_text"] = movies["movieId"].map(lambda mid: tag_map.get(mid, ""))

    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    # Soup WITHOUT TMDB metadata (fast!)
    movies["soup"] = (
        movies["genre_tokens"] + " " +
        movies["tags_text"]
    )

    return movies


# =====================
# BUILD MODEL
# =====================
@st.cache_resource
def build_model(movies):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    tfidf = vectorizer.fit_transform(movies["soup"])
    cosine_sim = linear_kernel(tfidf, tfidf)
    return cosine_sim


# =====================
# RECOMMENDER
# =====================
def recommend(title, movies, cosine_sim, topn=10):
    titles = movies["title"].tolist()
    matches = process.extract(title, titles, scorer=fuzz.token_set_ratio, limit=5)

    best_title, score, _ = matches[0]
    if score < 70:
        return None, [m[0] for m in matches]

    idx = movies[movies["title"] == best_title].index[0]

    sim_scores = sorted(
        list(enumerate(cosine_sim[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:topn+1]

    indices = [i for i, _ in sim_scores]
    recs = movies.iloc[indices].copy()

    return best_title, recs


# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender (Fast TMDB-On-Demand)")

movies = load_data()
cosine_sim = build_model(movies)

title = st.text_input("Enter a movie title:", "Toy Story (1995)")
topn = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    best_title, recs = recommend(title, movies, cosine_sim, topn)

    if best_title is None:
        st.error(f"Did you mean: {', '.join(recs)}?")
    else:
        st.success(f"Recommendations based on **{best_title}**:")

        cols = st.columns(5)

        for i, (_, row) in enumerate(recs.iterrows()):
            col = cols[i % 5]

            poster, overview = tmdb_get_metadata(row["tmdbId"])

            img = TMDB_IMG_BASE + poster if isinstance(poster, str) else PLACEHOLDER_POSTER

            col.image(img, caption=row["title"], use_container_width=True)
            col.caption(row["genres"])

