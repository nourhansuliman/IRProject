import math
import os
import joblib
from collections import defaultdict
import mysql.connector
import ir_datasets
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
import requests
from flask import Flask, request, jsonify
from gensim.models import FastText

from Database.db_connector import (
    insert_corpus,
    insert_training_corpus,
    insert_inverted_index,
    insert_tfidf_vector,
    insert_doc_embedding,
    fetch_corpus,
    fetch_inverted_index,
    insert_test_corpus,
    insert_test_qrel,
    fetch_training_corpus,
)

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir_system"
    )

app = Flask(__name__)

def preprocess_via_api(text):
    url = "http://127.0.0.1:5001/preprocess"
    try:
        response = requests.post(url, json={"text": text})
        response.raise_for_status()
        return response.json()["preprocessed_text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling preprocessing API: {e}")
        return text


def create_corpus(docs_iter, dataset_name):
    for i, doc in enumerate(docs_iter):
        processed_text = preprocess_via_api(doc.text)
        insert_corpus(doc.doc_id, processed_text, dataset_name)
        print(f"Inserted doc {i}: {doc.doc_id} into dataset {dataset_name}")


def create_training_corpus(docs_iter, dataset_name):
    for doc in docs_iter:
        processed_text = preprocess_via_api(doc.text)
        insert_training_corpus(doc.doc_id, processed_text, dataset_name)


def create_inverted_index(dataset_name):
    corpus = fetch_corpus(dataset_name)
    inverted_index = defaultdict(list)
    for doc_id, doc_content in corpus.items():
        terms = word_tokenize(doc_content)
        for term in terms:
            inverted_index[term].append(doc_id)

    conn = get_connection()
    cursor = conn.cursor()

    for term, doc_ids in inverted_index.items():
        insert_inverted_index(term, doc_ids, dataset_name, conn, cursor)

    conn.commit()
    cursor.close()
    conn.close()


def calculate_doc_tfidf_vectors(dataset_name):
    corpus = fetch_corpus(dataset_name)
    inverted_index = fetch_inverted_index(dataset_name)

    conn = get_connection()
    cursor = conn.cursor()

    for doc_id, doc_text in corpus.items():
        tfidf_vector = {}
        terms = word_tokenize(doc_text)
        for term in terms:
            tf = terms.count(term) / len(terms)
            if term in inverted_index:
                idf = math.log(len(corpus) / len(inverted_index[term]))
                tfidf_vector[term] = tf * idf
        insert_tfidf_vector(doc_id, tfidf_vector, dataset_name, conn, cursor)

    conn.commit()
    cursor.close()
    conn.close()


def train_model(dataset_name):
    corpus = fetch_corpus(dataset_name)
    tokenized_corpus = [text.split() for text in corpus.values()]

    folder_path = os.path.join("Offline Documents", dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    model_path = os.path.join(folder_path, "word_embedding_model.pkl")
    try:
        word_embedding_model = joblib.load(model_path)
    except FileNotFoundError:
        word_embedding_model = Word2Vec(
            tokenized_corpus,
            vector_size=300,
            window=8,
            min_count=7,
            sg=1
        )
        joblib.dump(word_embedding_model, model_path)

    conn = get_connection()
    cursor = conn.cursor()

    for faiss_idx, (doc_id, doc_text) in enumerate(corpus.items()):
        tokens = doc_text.split()
        doc_embedding = np.zeros(word_embedding_model.vector_size)
        num_tokens = 0
        for token in tokens:
            if token in word_embedding_model.wv:
                doc_embedding += word_embedding_model.wv[token]
                num_tokens += 1
        if num_tokens > 0:
            doc_embedding /= num_tokens
            insert_doc_embedding(doc_id, doc_embedding, dataset_name, faiss_idx, conn, cursor)

    conn.commit()
    cursor.close()
    conn.close()







def create_test_corpus_and_qrels(test_dataset, dataset_name):
    conn = get_connection()
    cursor = conn.cursor()

    for query in test_dataset.queries_iter():
        cleaned_query = preprocess_via_api(query.text)
        insert_test_corpus(query.query_id, cleaned_query, dataset_name, conn, cursor)

    conn.commit()
    cursor.close()
    conn.close()

    conn = get_connection()
    cursor = conn.cursor()

    print("test_corpus Done")

    for qrel in test_dataset.qrels_iter():
        insert_test_qrel(qrel.query_id, qrel.doc_id, int(qrel.relevance), dataset_name, conn, cursor)

    conn.commit()
    cursor.close()
    conn.close()


def train_fasttext_model(dataset_name):
    folder_path = os.path.join("Offline Documents", dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    model_path = os.path.join(folder_path, 'fasttext_model.pkl')

    loaded_corpus = fetch_corpus(dataset_name)
    tokenized_corpus = [text.split() for text in loaded_corpus.values() if text.strip()]

    if os.path.exists(model_path):
        print("FastText model already exists")
        return

    fasttext_model = FastText(
        tokenized_corpus,
        vector_size=200,
        window=3,
        min_count=1,
        sg=1
    )
    joblib.dump(fasttext_model, model_path)


def update_antique():
    name = "A"
    antique_dataset = ir_datasets.load('antique')
    antique_train_dataset = ir_datasets.load('antique/train')
    test_dataset = ir_datasets.load("antique/test")
    dataset_docs = antique_dataset.docs_iter()
    dataset_train = antique_train_dataset.docs_iter()
    create_corpus(dataset_docs, name)
    print('create_corpus Done for Antique')
    create_training_corpus(dataset_train, name)
    print('create_training_corpus Done for Antique')
    create_inverted_index(name)
    print('create_inverted_index Done for Antique')
    calculate_doc_tfidf_vectors(name)
    print('calculate_doc_tfidf_vectors Done for Antique')
    train_model(name)
    print('train_model Done for Antique')

    create_test_corpus_and_qrels(test_dataset, name)
    print('create_test_corpus_and_qrels Done for Antique')
    train_fasttext_model(name)
    print('train_fasttext_model Done for Antique')

def update_quora():
    name = "Q"
    quora_dataset = ir_datasets.load("beir/quora")
    test_dataset = ir_datasets.load("beir/quora/test")
    dataset_docs = quora_dataset.docs_iter()
    create_corpus(dataset_docs, name)
    print('create_corpus Done for Quora')
    create_training_corpus(dataset_docs, name)
    print('create_training_corpus Done for Quora')
    create_inverted_index(name)
    print('create_inverted_index Done for Quora')
    calculate_doc_tfidf_vectors(name)
    print('calculate_doc_tfidf_vectors Done for Quora')
    train_model(name)
    print('train_model Done for Quora')

    create_test_corpus_and_qrels(test_dataset, name)
    print('create_test_corpus_and_qrels Done for Quora')
    train_fasttext_model(name)
    print('train_fasttext_model Done for Quora')


#update_quora()

if __name__ == '__main__':
    print("Starting server...")
    #update_antique()
    update_quora()
    print("Ending server...")
    app.run()