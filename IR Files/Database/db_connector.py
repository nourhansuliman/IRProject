import mysql.connector
import joblib
import io
import json
import faiss
import zlib


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ir_system"
    )


def insert_corpus(doc_id, content, dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "REPLACE INTO corpus (doc_id, dataset_name, text) VALUES (%s, %s, %s)",
        (doc_id, dataset_name, content)
    )
    conn.commit()
    cursor.close()
    conn.close()


def insert_training_corpus(doc_id, content, dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "REPLACE INTO training_corpus (doc_id, dataset_name, text) VALUES (%s, %s, %s)",
        (doc_id, dataset_name, content)
    )
    conn.commit()
    cursor.close()
    conn.close()


def insert_inverted_index(term, doc_ids, dataset_name, conn=None, cursor=None):
    local_connection = False

    if conn is None or cursor is None:
        conn = get_connection()
        cursor = conn.cursor()
        local_connection = True

    doc_ids_json = json.dumps(doc_ids)
    cursor.execute(
        "REPLACE INTO inverted_index (term, dataset_name, doc_ids) VALUES (%s, %s, %s)",
        (term, dataset_name, doc_ids_json)
    )

    if local_connection:
        conn.commit()
        cursor.close()
        conn.close()



def insert_tfidf_vector(doc_id, vector_dict, dataset_name, conn=None, cursor=None):
    local_connection = False

    if conn is None or cursor is None:
        conn = get_connection()
        cursor = conn.cursor()
        local_connection = True

    vector_json = json.dumps(vector_dict)
    cursor.execute(
        "REPLACE INTO tfidf_vectors (doc_id, dataset_name, vector) VALUES (%s, %s, %s)",
        (doc_id, dataset_name, vector_json)
    )
    if local_connection:
        conn.commit()
        cursor.close()
        conn.close()


def insert_doc_embedding(doc_id, vector, dataset_name, faiss_idx, conn=None, cursor=None):
    local_connection = False
    if conn is None or cursor is None:
        conn = get_connection()
        cursor = conn.cursor()
        local_connection = True

    buffer = io.BytesIO()
    joblib.dump(vector, buffer, compress=True)
    buffer.seek(0)
    vector_blob = buffer.read()

    cursor.execute(
        """
        REPLACE INTO document_embeddings (doc_id, dataset_name, embedding, faiss_idx)
        VALUES (%s, %s, %s, %s)
        """,
        (doc_id, dataset_name, vector_blob, faiss_idx)
    )

    if local_connection:
        conn.commit()
        cursor.close()
        conn.close()



def fetch_corpus(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_id, text FROM corpus WHERE dataset_name = %s",
        (dataset_name,)
    )
    data = {doc_id: text for doc_id, text in cursor.fetchall()}
    cursor.close()
    conn.close()
    return data


def fetch_training_corpus(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_id, text FROM training_corpus WHERE dataset_name = %s",
        (dataset_name,)
    )
    data = {doc_id: text for doc_id, text in cursor.fetchall()}
    cursor.close()
    conn.close()
    return data


def fetch_inverted_index(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT term, doc_ids FROM inverted_index WHERE dataset_name = %s",
        (dataset_name,)
    )
    data = {term: json.loads(doc_ids) for term, doc_ids in cursor.fetchall()}
    cursor.close()
    conn.close()
    return data


def fetch_tfidf_vectors(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_id, vector FROM tfidf_vectors WHERE dataset_name = %s",
        (dataset_name,)
    )
    data = {}
    for doc_id, vector_json in cursor.fetchall():
        data[doc_id] = json.loads(vector_json)
    cursor.close()
    conn.close()
    return data


def fetch_document_embeddings(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_id, embedding FROM document_embeddings WHERE dataset_name = %s",
        (dataset_name,)
    )
    data = {}
    for doc_id, vector_blob in cursor.fetchall():
        buffer = io.BytesIO(vector_blob)
        vector = joblib.load(buffer)
        data[doc_id] = vector
    cursor.close()
    conn.close()
    return data






def insert_test_corpus(query_id, content, dataset_name, conn=None, cursor=None):
    local_connection = False
    if conn is None or cursor is None:
        conn = get_connection()
        cursor = conn.cursor()
        local_connection = True

    cursor.execute(
        "REPLACE INTO test_corpus (query_id, dataset_name, text) VALUES (%s, %s, %s)",
        (query_id, dataset_name, content)
    )
    if local_connection:
        conn.commit()
        cursor.close()
        conn.close()


def insert_test_qrel(query_id, doc_id, relevance, dataset_name, conn=None, cursor=None):
    local_connection = False
    if conn is None or cursor is None:
        conn = get_connection()
        cursor = conn.cursor()
        local_connection = True

    cursor.execute(
        "REPLACE INTO test_qrels (query_id, dataset_name, doc_id, relevance) VALUES (%s, %s, %s, %s)",
        (query_id, dataset_name, doc_id, relevance)
    )
    if local_connection:
        conn.commit()
        cursor.close()
        conn.close()


def insert_faiss_index(dataset_name, index):
    conn = get_connection()
    cursor = conn.cursor()

    # كتابة الـ FAISS index في buffer
    buffer = io.BytesIO()
    faiss.write_index(index, faiss.PyCallbackIOWriter(buffer.write))
    index_blob = buffer.getvalue()

    # ✅ ضغط البيانات
    compressed_blob = zlib.compress(index_blob)

    cursor.execute(
        "REPLACE INTO faiss_indexes (dataset_name, index_blob) VALUES (%s, %s)",
        (dataset_name, compressed_blob)
    )
    conn.commit()
    cursor.close()
    conn.close()


def insert_expanded_test_query(query_id, original_text, expanded_text, dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "REPLACE INTO expanded_test_queries (query_id, dataset_name, original_text, expanded_text) VALUES (%s, %s, %s, %s)",
        (query_id, dataset_name, original_text, expanded_text)
    )
    conn.commit()
    cursor.close()
    conn.close()


def fetch_test_corpus(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT query_id, text FROM test_corpus WHERE dataset_name = %s",
        (dataset_name,)
    )
    data = {query_id: text for query_id, text in cursor.fetchall()}
    cursor.close()
    conn.close()
    return data


def fetch_test_qrels(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT query_id, doc_id, relevance FROM test_qrels WHERE dataset_name = %s",
        (dataset_name,)
    )
    qrels = {}
    for query_id, doc_id, relevance in cursor.fetchall():
        qrels.setdefault(query_id, {})[doc_id] = relevance
    cursor.close()
    conn.close()
    return qrels


def fetch_document_embeddings_with_ids(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT doc_id, embedding FROM document_embeddings WHERE dataset_name = %s ORDER BY faiss_idx ASC",
        (dataset_name,)
    )
    doc_ids = []
    embeddings = []
    for doc_id, vector_blob in cursor.fetchall():
        buffer = io.BytesIO(vector_blob)
        vector = joblib.load(buffer)
        doc_ids.append(doc_id)
        embeddings.append(vector)
    cursor.close()
    conn.close()
    return embeddings, doc_ids



def fetch_training_words(dataset_name):
    corpus = fetch_corpus(dataset_name)
    word_set = set()
    for doc in corpus.values():
        words = doc.lower().split()
        word_set.update(words)
    return sorted(list(word_set))

def fetch_expanded_test_queries(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT query_id, expanded_text FROM expanded_test_queries WHERE dataset_name = %s",
        (dataset_name,)
    )
    data = {query_id: text for query_id, text in cursor.fetchall()}
    cursor.close()
    conn.close()
    return data


def fetch_faiss_index(dataset_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT index_blob FROM faiss_indexes WHERE dataset_name = %s",
        (dataset_name,)
    )
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"No FAISS index found for dataset {dataset_name}")

    # ✅ فك الضغط
    compressed_blob = result[0]
    index_blob = zlib.decompress(compressed_blob)
    print(f"🔍 Compressed FAISS index size: {len(compressed_blob) / 1024 / 1024:.2f} MB")

    buffer = io.BytesIO(index_blob)
    index = faiss.read_index(faiss.PyCallbackIOReader(buffer.read))
    cursor.close()
    conn.close()
    return index
