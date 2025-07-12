CREATE TABLE IF NOT EXISTS corpus (
    doc_id VARCHAR(255),
    dataset_name VARCHAR(10),
    text TEXT,
    PRIMARY KEY (doc_id, dataset_name)
);

CREATE TABLE IF NOT EXISTS training_corpus (
    doc_id VARCHAR(255),
    dataset_name VARCHAR(10),
    text TEXT,
    PRIMARY KEY (doc_id, dataset_name)
);

CREATE TABLE IF NOT EXISTS inverted_index (
    term VARCHAR(255),
    dataset_name VARCHAR(10),
    doc_ids JSON,
    PRIMARY KEY (term, dataset_name)
);

CREATE TABLE IF NOT EXISTS tfidf_vectors (
    doc_id VARCHAR(255),
    dataset_name VARCHAR(10),
    vector JSON,
    PRIMARY KEY (doc_id, dataset_name)
);

CREATE TABLE IF NOT EXISTS document_embeddings (
    doc_id VARCHAR(255),
    dataset_name VARCHAR(10),
    embedding BLOB,
    PRIMARY KEY (doc_id, dataset_name)
);





CREATE TABLE IF NOT EXISTS test_corpus (
    query_id VARCHAR(255),
    dataset_name VARCHAR(10),
    text TEXT,
    PRIMARY KEY (query_id, dataset_name)
);


CREATE TABLE IF NOT EXISTS test_qrels (
    query_id VARCHAR(255),
    dataset_name VARCHAR(10),
    doc_id VARCHAR(255),
    relevance INT,
    PRIMARY KEY (query_id, dataset_name, doc_id)
);


CREATE TABLE IF NOT EXISTS faiss_indexes (
    dataset_name VARCHAR(10) PRIMARY KEY,
    index_blob LONGBLOB
);

CREATE TABLE IF NOT EXISTS expanded_test_queries (
    query_id VARCHAR(255),
    dataset_name VARCHAR(10),
    original_text TEXT,
    expanded_text TEXT,
    PRIMARY KEY (query_id, dataset_name)
);
