import spacy
import re
import requests
import pandas as pd


class NlpSpacy:
    def __init__(self, nlp=spacy.load('en_core_web_sm')):
        self.nlp = nlp

    def token(self, sentence: str) -> list:
        sentence = re.sub(r'\s+', ' ', sentence)
        token = []
        for w in self.nlp(sentence).ents:
            token.append(w.text.replace(" ", "_"))
            sentence = sentence.replace(w.text, '')
        for w in self.nlp(sentence):
            if not w.is_punct and not w.is_stop and w.text != " ":
                token.append(w.lemma_)
        return list(set(token))


class MovieRecommendation:
    def __init__(self, url: str = "http://127.0.0.1:8000"):
        self.url = url

    def tconst(self, tconst: str, show_movie: bool = True):
        response = requests.get(f"{self.url}/tconst/{tconst}")
        if response.status_code == 200:
            if show_movie:
                return pd.json_normalize(response.json())
            else:
                return pd.json_normalize(response.json()['recommandations'])
        else:
            print("404 Not Found")

    def title(self, title: str, show_movie: bool = True):
        response = requests.get(f"{self.url}/title/{title}")
        if response.status_code == 200:
            if show_movie:
                return pd.json_normalize(response.json())
            else:
                return pd.json_normalize(response.json()['recommandations'])
        else:
            print("404 Not Found")

    def search(self, title: str, limit: int = 10):
        response = requests.get(f"{self.url}/search/{title}&{limit}")
        if response.status_code == 200:
            return pd.json_normalize(response.json())
        else:
            print("404 Not Found")
