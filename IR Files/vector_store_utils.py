import faiss
import os
import numpy as np
from flask import Flask
from Database.db_connector import fetch_document_embeddings_with_ids

app = Flask(__name__)

# ---- Caches ----
document_embeddings_cache = {}
document_ids_cache = {}
faiss_index = {}


def preload_data():
    datasets = ['A', 'Q']
    for dataset in datasets:
        print(f"Preloading data for dataset: {dataset}")
        vectors, doc_ids = fetch_document_embeddings_with_ids(dataset)
        document_embeddings_cache[dataset] = vectors
        document_ids_cache[dataset] = doc_ids
    print("Preloading complete.")


def build_vector_index(dataset_name, nlist=100):
    index_path = f'Offline Documents/{dataset_name}/faiss_index.index'

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        faiss_index[dataset_name] = index
        print(f"FAISS index already exists and loaded from: {index_path}")
        return

    document_vectors = document_embeddings_cache[dataset_name]
    if not document_vectors:
        print(f"No document vectors found for dataset {dataset_name}")
        return

    document_vectors = np.array(document_vectors, dtype=np.float32)

    faiss.normalize_L2(document_vectors)

    dim = document_vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(document_vectors)
    index.add(document_vectors)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    faiss_index[dataset_name] = index
    print(f"FAISS index built and saved to: {index_path}")


if __name__ == '__main__':
    preload_data()
    build_vector_index("A")
    build_vector_index("Q")
    app.run(port=5002)
