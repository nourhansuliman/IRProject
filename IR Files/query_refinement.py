import json
import joblib
import string
from flask import Flask, request, jsonify, Blueprint
import os
from difflib import get_close_matches
from nltk.corpus import stopwords
import requests
from Database.db_connector import fetch_training_words

app = Flask(__name__)

# ---- Caches ----
training_words = {}
fasttext_model = {}

def preload_data():
    datasets = ['A', 'Q']
    for dataset in datasets:
        print(f"Preloading data for dataset: {dataset}")
        training_words[dataset] = fetch_training_words(dataset)
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


stop_words = set(stopwords.words("english"))
common_noise_words = {"thing", "something", "anything", "everything", "stuff"}

def suggest_autocomplete(prefix, dataset_name):
    words = training_words[dataset_name]

    #words = load_training_words(dataset_name)
    return [w for w in words if w.startswith(prefix.lower())][:10]


def log_user_query(dataset_name, query):
    if dataset_name == "A":
        log_path = 'Offline Documents/A/query_logs.json'
    elif dataset_name == "Q":
        log_path = 'Offline Documents/Q/query_logs.json'
    else:
        return

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    if query not in data:
        data.append(query)

    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)


def suggest_from_logs(dataset_name, query):
    if dataset_name == "A":
        log_path = 'Offline Documents/A/query_logs.json'
    elif dataset_name == "Q":
        log_path = 'Offline Documents/Q/query_logs.json'
    else:
        return []

    if not os.path.exists(log_path):
        return []

    with open(log_path, "r") as f:
        data = json.load(f)

    return get_close_matches(query, data, n=5, cutoff=0.5)

@app.route("/query-refinement/refine-query", methods=["POST"])
def refine_query():
    data = request.get_json()
    query = data.get("query", "")
    prefix = data.get("prefix", "")
    dataset_name = data.get("dataset", "")
    topn = data.get("topn", 5)

    if not dataset_name:
        return jsonify({"error": "Missing dataset name"}), 400

    response = {
        "original_query": query,
        "prefix": prefix,
        "autocomplete": [],
        "expansion": {},
        "logs_suggestions": []
    }

    # تنفيذ Autocomplete
    if prefix:
        response["autocomplete"] = suggest_autocomplete(prefix, dataset_name)

    # تنفيذ Query Expansion
    if query:
        model = fasttext_model[dataset_name]

        if model:
            cleaned_query = preprocess_via_api(query)
            tokens = cleaned_query.split()
            expanded_terms = []

            max_expansion_per_term = max(1, int(len(tokens) * 0.5))
            score_threshold = 0.65 if len(tokens) < 4 else 0.75

            for token in tokens:
                if token in model.wv:
                    similar_words = model.wv.most_similar_cosmul(token, topn=topn)
                    filtered = [
                        w for w, score in similar_words
                        if score >= score_threshold
                            and len(w) > 3
                            and w.isalpha()
                            and w not in tokens
                            and w not in stop_words
                            and w not in common_noise_words
                            and not any(c in string.punctuation for c in w)
                            and w in model.wv.key_to_index
                        ][:max_expansion_per_term]

                    expanded_terms.extend(filtered)

            expanded_terms = list(dict.fromkeys(expanded_terms))[:6]
            refined_query = query + " " + " ".join(expanded_terms) if expanded_terms else query
            response["expansion"] = refined_query

        log_user_query(dataset_name, query)

        response["logs_suggestions"] = suggest_from_logs(dataset_name, query)

    return jsonify(response)

if __name__ == '__main__':
    preload_data()
    app.run(port=5007)