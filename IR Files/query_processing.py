import math

import numpy as np
import requests
from nltk import word_tokenize

from Database.db_connector import fetch_corpus

def preprocess_via_api(text):
    url = "http://127.0.0.1:5001/preprocess"
    try:
        response = requests.post(url, json={"text": text})
        response.raise_for_status()
        return response.json()["preprocessed_text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling preprocessing API: {e}")
        return text


def retrieve_documents_via_api(query, dataset_name, k=10, model_type="we"):
    url = "http://127.0.0.1:5002/vsm_search"
    payload = {
        "query": query,
        "dataset_name": dataset_name,
        "k": k,
        "model_type": model_type
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling retrieval API: {e}")
        return []


def calculate_query_tfidf(query, inverted_index, name, corpus):
    tfidf_vector = {}
    terms = word_tokenize(query)
    #corpus = fetch_corpus(name)
    for term in set(terms):
        tf = terms.count(term) / len(terms)
        idf = 0
        if term in inverted_index:
            idf = math.log(len(corpus) / len(inverted_index[term]))
        tfidf_vector[term] = tf * idf

    return tfidf_vector


def query_word_embedding(word_embedding_model, query):
    query_embedding = np.zeros(word_embedding_model.vector_size)
    query_tokens = word_tokenize(query)
    num_tokens = 0
    for word in query_tokens:
        if word in word_embedding_model.wv:
            query_embedding += word_embedding_model.wv[word]
            num_tokens += 1
    if num_tokens > 0:
        query_embedding /= num_tokens
    return query_embedding


def query_expansion(query, name, k, model_type="we"):
    ppq = preprocess_via_api(query)
    print("Preprocessed Query: ", ppq)
    final_results = retrieve_documents_via_api(ppq, name, k, model_type=model_type)
    return final_results
