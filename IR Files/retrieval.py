import faiss
import numpy as np
import math
import joblib
from gensim.models import Word2Vec
from flask import Flask, request, jsonify
from nltk import word_tokenize
from Database.db_connector import fetch_inverted_index, fetch_tfidf_vectors, fetch_document_embeddings, fetch_document_embeddings_with_ids, fetch_corpus
from query_processing import calculate_query_tfidf, query_word_embedding

app = Flask(__name__)

# ---- Caches ----
inverted_index_cache = {}
tfidf_vectors_cache = {}
document_embeddings_cache = {}
word_embedding_models = {}
document_embeddings_ids_cache = {}
faiss_index = {}
corpus_data = {}

def preload_data():
    datasets = ['A', 'Q']
    for dataset in datasets:
        print(f"Preloading data for dataset: {dataset}")
        inverted_index_cache[dataset] = fetch_inverted_index(dataset)
        tfidf_vectors_cache[dataset] = fetch_tfidf_vectors(dataset)
        document_embeddings_cache[dataset] = fetch_document_embeddings(dataset)
        word_embedding_models[dataset] = joblib.load(f"Offline Documents/{dataset}/word_embedding_model.pkl")
        _, document_embeddings_ids_cache[dataset] = fetch_document_embeddings_with_ids(dataset)
        faiss_index[dataset] = faiss.read_index(f'Offline Documents/{dataset}/faiss_index.index')
        corpus_data[dataset] = fetch_corpus(dataset)
    print("Preloading complete.")


def manual_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    similarity = dot_product / (norm1 * norm2)
    return 0 if np.isnan(similarity) else similarity


def vector_space_model_tfidf(query, dataset_name, k):
    inverted_index = inverted_index_cache[dataset_name]
    tfidf_vectors = tfidf_vectors_cache[dataset_name]
    corpus = corpus_data[dataset_name]

    query_terms = word_tokenize(query)
    query_vector = calculate_query_tfidf(query, inverted_index, dataset_name, corpus)
    N = len(corpus)

    similarity_scores = {}
    relevant_docs = set()

    for term in query_terms:
        if term in inverted_index:
            relevant_docs.update(inverted_index[term])

    query_magnitude = math.sqrt(sum(value ** 2 for value in query_vector.values()))

    for doc_id in relevant_docs:
        doc_vector = tfidf_vectors.get(doc_id, {})
        common_terms = set(query_vector.keys()) & set(doc_vector.keys())

        if not common_terms:
            continue

        dot_product = sum(query_vector[t] * doc_vector[t] for t in common_terms)
        doc_magnitude = math.sqrt(sum(w ** 2 for w in doc_vector.values()))
        if query_magnitude == 0 or doc_magnitude == 0:
            continue

        similarity = dot_product / (query_magnitude * doc_magnitude)

        coverage = len(common_terms) / len(query_vector)
        similarity *= (1 + 7.8 * coverage)

        idf_boost = 0
        for t in common_terms:
            df = len(inverted_index.get(t, []))
            idf = math.log((N + 1) / (1 + df)) + 1
            if idf > 5:
                idf_boost += 1.5
        similarity *= (1 + idf_boost)

        similarity_scores[doc_id] = similarity

    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    sorted_ids = [doc_id for doc_id, _ in sorted_scores]

    return sorted_ids, similarity_scores


def vector_space_model_we(query, dataset_name, k):
    inverted_index = inverted_index_cache[dataset_name]
    document_embeddings = document_embeddings_cache[dataset_name]
    word_embedding = word_embedding_models[dataset_name]

    query_embedding = query_word_embedding(word_embedding, query)
    query_tokens = word_tokenize(query)
    query_token_set = set(query_tokens)
    corpus = corpus_data[dataset_name]
    N = len(corpus)

    similarity_scores = {}

    for term in query_tokens:
        if term in inverted_index:
            for doc_id in inverted_index[term]:

                if doc_id in similarity_scores:
                    continue

                doc_embedding = document_embeddings.get(doc_id)
                if doc_embedding is None or np.linalg.norm(doc_embedding) == 0:
                    continue

                base_similarity = manual_cosine_similarity(query_embedding, doc_embedding)

                doc_text = corpus.get(doc_id, "")
                doc_tokens_seq = word_tokenize(doc_text)[:100]
                doc_tokens_set = set(doc_tokens_seq)

                common_terms = query_token_set & doc_tokens_set
                coverage = len(common_terms) / len(query_token_set) if query_token_set else 0

                positions = [i for i, token in enumerate(doc_tokens_seq) if token in query_token_set]
                if len(positions) >= 2:
                    min_distance = min(abs(positions[i] - positions[i + 1]) for i in range(len(positions) - 1))
                    proximity = 1 / (1 + min_distance)
                else:
                    proximity = 0

                final_score = (
                    base_similarity * 0.6 +
                    coverage * 0.25 +
                    proximity * 0.1
                )

                if not np.isnan(final_score):
                    similarity_scores[doc_id] = final_score

    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    sorted_ids = [doc_id for doc_id, _ in sorted_scores]

    return sorted_ids, similarity_scores



def get_weighted_query_embedding(query_tokens, model, inverted_index, N):
    weighted_vectors = []
    weights = []

    for token in query_tokens:
        if token in model.wv:
            df = len(inverted_index.get(token, []))
            idf = np.log((N + 1) / (1 + df)) + 1  # IDF style weighting
            weighted_vectors.append(model.wv[token] * idf)
            weights.append(idf)

    if not weighted_vectors:
        return None

    weighted_avg = np.sum(weighted_vectors, axis=0) / (np.sum(weights) + 1e-10)
    return weighted_avg.astype(np.float32)

def compute_manual_features(query_tokens, doc_tokens):
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)

    common = query_set & doc_set
    coverage = len(common) / len(query_set) if query_set else 0

    positions = [i for i, token in enumerate(doc_tokens) if token in query_set]
    if len(positions) >= 2:
        min_distance = min([abs(positions[i] - positions[i + 1]) for i in range(len(positions) - 1)])
        proximity = 1 / (1 + min_distance)
    else:
        proximity = 0

    return coverage, proximity


def vector_space_model_vs(query, dataset_name, k):
    model = word_embedding_models[dataset_name]
    document_ids = document_embeddings_ids_cache[dataset_name]
    index = faiss_index[dataset_name]
    inverted_index = inverted_index_cache[dataset_name]
    corpus = corpus_data[dataset_name]

    tokens = word_tokenize(query)
    N = len(document_ids)

    query_vector = get_weighted_query_embedding(tokens, model, inverted_index, N)
    if query_vector is None:
        return [], {}

    query_vector = query_vector.astype(np.float32)
    faiss.normalize_L2(query_vector.reshape(1, -1))
    query_vector = np.expand_dims(query_vector, axis=0)

    index.nprobe = 30
    k_initial = max(k, 50)
    distances, indices = index.search(query_vector, k_initial)

    if len(indices[0]) == 0 or indices[0][0] == -1:
        return [], {}

    print("FAISS returned indices:", indices[0])

    candidate_ids = []
    candidate_scores = []

    print(f"document_ids length: {len(document_ids)}")
    print(f"indices returned: {indices}")
    for i, idx in enumerate(indices[0][:5]):
        if idx < len(document_ids):
            doc_id = document_ids[idx]
            print(f"→ Index {idx} maps to doc_id: {doc_id}")
            print("→ Found in corpus?", doc_id in corpus)

    for i, idx in enumerate(indices[0]):
        if idx >= len(document_ids):
            continue

        doc_id = document_ids[idx]
        base_score = float(distances[0][i])  # Inner product ≈ cosine similarity

        doc_text = corpus.get(doc_id, "")

        doc_tokens = word_tokenize(doc_text)[:100]

        coverage, proximity = compute_manual_features(tokens, doc_tokens)

        final_score = (
            base_score * 0.6 +
            coverage * 0.25 +
            proximity * 0.15
        )

        candidate_ids.append(doc_id)
        candidate_scores.append(final_score)

    sorted_items = sorted(zip(candidate_ids, candidate_scores), key=lambda x: x[1], reverse=True)[:k]
    sorted_ids = [doc_id for doc_id, _ in sorted_items]
    similarity_scores = {doc_id: score for doc_id, score in sorted_items}

    return sorted_ids, similarity_scores


def normalize_scores(scores):
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    range_score = max_score - min_score + 1e-8
    return {doc_id: (score - min_score) / range_score for doc_id, score in scores.items()}

def vector_space_model_hybrid(query, name, k, alpha=0.5):
    _, tfidf_scores = vector_space_model_tfidf(query, name, k * 5)
    _, we_scores = vector_space_model_we(query, name, k * 5)

    tfidf_scores = normalize_scores(tfidf_scores)
    we_scores = normalize_scores(we_scores)

    all_doc_ids = set(tfidf_scores.keys()).union(we_scores.keys())
    hybrid_scores = {}

    for doc_id in all_doc_ids:
        tfidf_score = tfidf_scores.get(doc_id, 0.0)
        we_score = we_scores.get(doc_id, 0.0)
        hybrid_scores[doc_id] = alpha * tfidf_score + (1 - alpha) * we_score

    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    sorted_ids = [doc_id for doc_id, _ in sorted_hybrid]
    return sorted_ids


def vector_space_model_vs_hybrid(query, name, k, alpha=0.7):
    _, tfidf_scores = vector_space_model_tfidf(query, name, k * 5)
    _, vs_scores = vector_space_model_vs(query, name, k * 5)

    tfidf_scores = normalize_scores(tfidf_scores)
    vs_scores = normalize_scores(vs_scores)

    all_doc_ids = set(tfidf_scores.keys()).union(vs_scores.keys())
    hybrid_scores = {}

    for doc_id in all_doc_ids:
        tfidf_score = tfidf_scores.get(doc_id, 0.0)
        we_score = vs_scores.get(doc_id, 0.0)
        hybrid_scores[doc_id] = alpha * tfidf_score + (1 - alpha) * we_score

    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    sorted_ids = [doc_id for doc_id, _ in sorted_hybrid]
    return sorted_ids


@app.route('/vsm_search', methods=['POST'])
def vsm_search():
    data = request.get_json()
    query = data.get("query")
    dataset_name = data.get("dataset_name")  # "A" or "Q"
    model_type = data.get("model_type")  
    k = data.get("k", 10)

    if not query or not dataset_name or not model_type:
        return jsonify({"error": "Missing parameters"}), 400

    if model_type == "tfidf":
        result, _ = vector_space_model_tfidf(query, dataset_name, k)
    elif model_type == "we":
        result,_ = vector_space_model_we(query, dataset_name, k)
    elif model_type == "hybrid":
        result = vector_space_model_hybrid(query, dataset_name, k)
    elif model_type == "vs":
        result, _ = vector_space_model_vs(query, dataset_name, k)
    elif model_type == "hybrid_vs":
        result = vector_space_model_vs_hybrid(query, dataset_name, k)
    else:
        return jsonify({"error": "Invalid model_type"}), 400

    return jsonify(result)


if __name__ == '__main__':
    preload_data()
    app.run(port=5002)
