import pandas as pd
from fastapi import FastAPI
import joblib
from thefuzz import process

infos_movies = pd.read_pickle('data/infos_movies.p')
data_encoded = pd.read_pickle('data/data_weight.p')
model = joblib.load('NearestNeighbors.pkl')


def scoring(originalTitle: str, limit: int) -> list:
    scores = process.extract(originalTitle, infos_movies.originalTitle, limit=limit)
    return scores


def movie(neigh_ind):
    return {**infos_movies.iloc[neigh_ind[0]].fillna("NaN").to_dict(),
        **{"recommandations": infos_movies.iloc[neigh_ind[1:]].fillna("NaN").to_dict(orient='records')}}


app = FastAPI()


@app.get("/")
def root():
    return {"Documentations": "/docs", "movie": ["/title", "/tconst"], "search": "/search"}


@app.get("/title/{originalTitle}")
def title(originalTitle: str, score_min: int = 93, limit: int = 10):
    scores = scoring(originalTitle, limit=limit)
    if scores[0][1] >= score_min:
        originalTitle = scores[0][0]
        neigh_ind = model.kneighbors(
            data_encoded[infos_movies.originalTitle == originalTitle],
            return_distance=False
        )[0]
        return movie(neigh_ind)
    else:
        return {"Error: Title don't match, you can use": "/search", "Matching probabilities": scores}


@app.get("/search/{title_search}")
def search(title_search: str, limit: int = 10):
    scores = scoring(title_search, limit=limit)
    result = [
        {
            'score': score[1],
            'movie': infos_movies[infos_movies.index == score[2]].fillna("NaN").to_dict(orient='records')[0],
        } for score in scores
    ]
    return result


@app.get("/tconst/{tconst_movie}")
def tconst(tconst_movie: str):
    print(tconst_movie in infos_movies.index)
    if tconst_movie in infos_movies.index:
        neigh_ind = model.kneighbors(data_encoded[data_encoded.index == tconst_movie], return_distance=False)[0]
        return movie(neigh_ind)
    else:
        return {"Error : tconst not found, you can use": "/search"}
