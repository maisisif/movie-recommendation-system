# Movie Recommendation System

A content-based movie recommendation web application built with Python and Streamlit.

The system generates recommendations using TF-IDF vectorization and cosine similarity on movie genres and user tags from the MovieLens (ml-latest-small) dataset. Movie posters and short descriptions are fetched on demand from The Movie Database (TMDB).

---

## How it works

- MovieLens data is downloaded automatically on first run if not present
- Genres and user tags are combined into a text representation
- TF-IDF and cosine similarity are used to compute movie similarity
- TMDB is queried only to retrieve posters and overviews
- The interface is implemented using Streamlit

---

## Technologies used

- Python  
- Streamlit  
- Pandas  
- scikit-learn  
- RapidFuzz  
- TMDB API  

---

## Project structure

```text
.
├── app.py
├── requirements.txt
└── data/        # created automatically (MovieLens is downloaded here)
```

## Setup

Follow these steps to get the project running locally.

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3.Set TMDB API Key
You will need an API key to fetch movie posters and metadata.
1. Create an API key at themoviedb.org/settings/api.
2. Set it as an environment variable in your terminal:
```bash
export TMDB_KEY=YOUR_TMDB_API_KEY
```

Note: Do not commit API keys to the repository. For production, consider using a .env file added to .gitignore.

### 4. Run the application
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser to view the app.



## Notes
MovieLens is used for the core recommendation logic.

TMDB is used only for metadata (posters and overviews).

The dataset is downloaded automatically if missing from the data/ folder.










