# Information-Retrieval-System
This is an Information Retrieval System that supports two major datasets and uses TF-IDF and Word Embedding models to retrieve the most relevant documents. The system is built around a modular architecture using Flask, Gensim, NLTK, and a MYSQL database for fast and efficient processing.


1) Antique:

* Contains 404K documents
* Includes 200 queries
* Consists of 6.6K qrels (relevance judgments)

2) Beir/quora:

* Contains 523K documents
* Includes 10K queries
* Consists of 16K qrels (relevance judgments)
* Both datasets include both testing and Qrel(Relevance Judgment).

# Database Schema (Database/schema.sql)
Tables:

* corpus: Stores preprocessed documents
* training_corpus: Stores training subset
* inverted_index: Term → list of doc_ids
* tfidf_vectors: TF-IDF values per document
* document_embeddings: Word2Vec document-level embeddings
* test_corpus: To store the preprocessed test documents
* test_qrels: To store the Qrels data
* expanded_test_queries: To store the expanded test queries results
All tables include a dataset_name field to support multiple datasets simultaneously.

# Database Integration (Database/db_connector)
Stores the following components inside MYSQL:

* Preprocessed corpus and training data.
* Inverted index.
* TF-IDF vectors.
* Document embeddings (Word2Vec).
* Qrels and Preprocessed test corpus.
* Expanded test queries.

# Preprocessing Service
Handles all text normalization steps via a Flask API.

* Replace possessive forms using Contractions.
* Correct spelling mistakes using the autocorrect library.
* Tokenize the text into individual words or special characters using the word_tokenize function.
* Convert to lowercase.
* Remove punctuation marks.
* Remove stop words: Remove predefined set of stop words (e.g., a, an, the, over, etc.).
* Stemming: Reduce words to their word stem or root form.
* POS: Part Of Speech to identify the type of each word. Perform lemmatization using WordNetLemmatizer to obtain the base or dictionary form based on POS.
* Remove auxiliary verbs and delete single-character words.

# Offline Operations
We have represented files in several formats for the purpose of their use

functions:
* create_corpus: Preprocess and insert documents into DB.
* create_training_corpus: Preprocess training subset.
* create_inverted_index: Build inverted index and insert to DB.
* calculate_doc_tfidf_vectors: Compute and insert TF-IDF vectors.
* train_model: Train Word2Vec and store document embeddings.
* create_test_corpus_and_qrels: Insert test data and Qrels data
* train_fasttext_model: Train the FastText model

# Query Processing
Handles preprocessing and vector generation for queries

functions:
* preprocess_via_api(text): Calls preprocessing API.
* calculate_query_tfidf(query, inverted_index, name): Calculates the TF-IDF vector for a given query based on the inverted index and corpus.
* query_word_embedding(word_embedding_model, query): Generates the word embedding representation of a query using a trained word embedding model.
* retrieve_documents_via_api(query, name, k, model_type): Api call to retrieve the top-k documents relevant to the given query using the vector space model.
* query_expansion(query, name, k, model_type): Performs query expansion by combining multiple techniques and retrieves relevant documents based on the preprocessed query.

# Retrieval Service (Vector Space Model)
Runs a Flask server which is responsible for matching the results and returning the closest documents.

functions:
* manual_cosine_similarity(vec1, vec2): Calculates the cosine similarity between two vectors using manual computation. It computes the dot product of the vectors and normalizes them based on their magnitudes. Returns the similarity score.
* vector_space_model_tfidf(query, name, k): Performs vector space model retrieval using TF-IDF weighting. Takes a query, dataset name, and the number of documents to retrieve (k) as input. Calculates the TF-IDF vector for the query, computes the similarity between the query and each document using TF-IDF vectors, and returns the top-k most relevant document.
* vector_space_model(query, name, k): Performs vector space model retrieval using word embeddings. Takes a query, dataset name, and the number of documents to retrieve (k) as input. Converts the query to a word embedding vector, calculates the similarity between the query and each document using document embeddings, and returns the top-k most relevant document.
* vsm_search(): Handles a POST request and returns the top-k relevant documents for a query using either TF-IDF or word embedding similarity based on the selected dataset and model type
* get_weighted_query_embedding: Represent the query using Weighted Embedding
* compute_manual_features: Enhance similarity score using coverage and word proximity
* vector_space_model_vs: Perform retrieval using Vector Store
* normalize_scores: Normalize values in the dictionary to the range [0, 1]
* vector_space_model_hybrid: Perform retrieval using Hybrid model
* vector_space_model_vs_hybrid: Perform retrieval using Hybrid with Vector Store

# Flask Server
This service handles user search queries from the GUI, applies retrieval logic, and returns the most relevant documents for display.

# Evaluation
* evaluate_model: Perform evaluation to compute:
    Precision@k: The proportion of relevant documents among the top-k results
    Recall: The proportion of relevant documents that were retrieved
    MAP (Mean Average Precision): The average of cumulative precision over all queries
    MRR (Mean Reciprocal Rank): The average reciprocal rank of the first relevant document
* log_evaluate: Save evaluation results to a JSON file

#Evaluate refinement
* evaluate_refinement_model: Perform evaluation on the expanded queries

#Query refinement:
* suggest_autocomplete: Suggest words starting with the entered characters from the training vocabulary
* log_user_query: Log the user’s query
* suggest_from_logs: Suggest similar queries from the query log
* refine_query: API endpoint that combines all features:
    Autocomplete: From training vocabulary
    Query Expansion: Generate similar words for each term using FastText
    Log Suggestions: Suggestions from previous query logs

# Queries test refinement
* expand_queries_with_filtering: Used to expand test queries

# Vector store utils
* build_vector_index: Builds the FAISS index to be used in the Vector Store feature

# How to communicate between services
1. Frontend → Main API
Sends query input to Flask server.
2. Main API → Preprocessing Service
Forwards the raw query for cleaning.
3. Main API → Retrieval Logic
Passes cleaned query + dataset name to VSM logic.
4. Retrieval Logic → MySQL DB
Fetches:
Inverted Index
TF-IDF vectors
Document embeddings
5. Retrieval Logic → Main API
Returns list of relevant document IDs (ranked).
6. Main API → Docs Store
Retrieves full text/snippets for the returned doc IDs from the Docs Store.
7. Main API → Frontend
Sends back final search results to GUI.