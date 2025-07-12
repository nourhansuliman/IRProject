from flask import Flask
import joblib
from nltk.corpus import stopwords
import string
import requests
from Database.db_connector import insert_expanded_test_query, fetch_test_corpus

app = Flask(__name__)

# ---- Caches ----
test_corpus = {}
fasttext_model = {}

def preload_data():
    datasets = ['A', 'Q']
    for dataset in datasets:
        print(f"Preloading data for dataset: {dataset}")
        test_corpus[dataset] = fetch_test_corpus(dataset)
        fasttext_model[dataset] = joblib.load(f"Offline Documents/{dataset}/fasttext_model.pkl")
    print("Preloading complete.")


def preprocess_via_api(text):
    url = "http://127.0.0.1:5001/preprocess"
    try:
        response = requests.post(url, json={"text": text})
        response.raise_for_status()
        return response.json()["preprocessed_text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling preprocessing API: {e}")
        return text


stop_words = set(stopwords.words('english'))

common_noise_words = {"thing", "something", "anything", "everything", "stuff"}

def expand_queries_with_filtering(dataset_name: str, topn: int = 5, max_final_terms: int = 6
):
    queries = test_corpus[dataset_name]
    model = fasttext_model[dataset_name]


    for qid, query in queries.items():
        preprocessed = preprocess_via_api(query).split()
        expanded_terms = []

        max_expansion_per_term = max(1, int(len(preprocessed) * 0.5))
        score_threshold = 0.65 if len(preprocessed) < 4 else 0.75

        for token in preprocessed:
            if token in model.wv:
                try:
                    similar = model.wv.most_similar_cosmul(token, topn=topn)
                except KeyError:
                    continue

                filtered = [
                    w for w, score in similar
                    if score >= score_threshold
                    and len(w) > 3
                    and w.isalpha()
                    and w not in stop_words
                    and w not in preprocessed
                    and w not in common_noise_words
                    and not any(c in string.punctuation for c in w)
                    and w in model.wv.key_to_index
                ][:max_expansion_per_term]

                expanded_terms.extend(filtered)

        expanded_terms = list(dict.fromkeys(expanded_terms))[:max_final_terms]
        final_query = query + " " + " ".join(expanded_terms) if expanded_terms else query

        insert_expanded_test_query(qid, query, final_query, dataset_name)

    print(f"Saved expanded queries to database for dataset {dataset_name}")


if __name__ == '__main__':
    preload_data()
    expand_queries_with_filtering("A")
    expand_queries_with_filtering("Q")
    app.run(port=5002)