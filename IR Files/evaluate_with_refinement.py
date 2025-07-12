import json
import os
from flask import Flask, request, jsonify
import requests
import numpy as np
from Database.db_connector import fetch_expanded_test_queries, fetch_test_qrels

app = Flask(__name__)

# ---- Caches ----
expand_test_corpus = {}
test_qrels = {}

def preload_data():
    datasets = ['A', 'Q']
    for dataset in datasets:
        print(f"Preloading data for dataset: {dataset}")
        expand_test_corpus[dataset] = fetch_expanded_test_queries(dataset)
        test_qrels[dataset] = fetch_test_qrels(dataset)
    print("Preloading complete.")


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


def log_exp_evaluate(dataset_name, evaluate):
    if dataset_name == "A":
        log_path = 'Offline Documents/A/evaluate_exp_logs.json'
    elif dataset_name == "Q":
        log_path = 'Offline Documents/Q/evaluate_exp_logs.json'
    else:
        return

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)
    else:
        data = []


    data.append(evaluate)

    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)

def evaluate_refinement_model(name, model_type="hybrid", k=10, min_rel_val=1):
    queries_dict = expand_test_corpus[name]
    qrels_dict = test_qrels[name]

    total_queries = 0
    all_precisions = []
    all_recalls = []
    all_ap = []
    all_rr = []

    for query_id, query_entry in queries_dict.items():
        if query_id not in qrels_dict:
            continue

        query_text = query_entry
        relevance_dict = qrels_dict[query_id]
        ground_truth = {doc_id for doc_id, rel in relevance_dict.items() if rel >= min_rel_val}
        if not ground_truth:
            continue

        total_queries += 1
        retrieved = retrieve_documents_via_api(query_text, name, k, model_type)

        # Precision@k
        retrieved_set = set(retrieved[:k])
        relevant_retrieved = retrieved_set.intersection(ground_truth)
        precision_at_k = len(relevant_retrieved) / k
        all_precisions.append(precision_at_k)

        # Recall
        recall = len(relevant_retrieved) / len(ground_truth)
        all_recalls.append(recall)

        # Average Precision (AP)
        num_relevant = 0
        precision_sum = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in ground_truth:
                num_relevant += 1
                precision_sum += num_relevant / (i + 1)
        ap = precision_sum / len(ground_truth) if len(ground_truth) > 0 else 0
        all_ap.append(ap)

        # Reciprocal Rank (RR)
        rr = 0
        for i, doc_id in enumerate(retrieved):
            if doc_id in ground_truth:
                rr = 1 / (i + 1)
                break
        all_rr.append(rr)

    print(f"Evaluation Results on Dataset: {name}, Model: {model_type}, min_rel_val ≥ {min_rel_val}")
    print("-----------------------------------------------------")
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Mean Average Precision (MAP):     {np.mean(all_ap):.4f}")
    print(f"Precision@{k}:                    {np.mean(all_precisions):.4f}")
    print(f"Recall:                           {np.mean(all_recalls):.4f}")
    print(f"Mean Reciprocal Rank (MRR):       {np.mean(all_rr):.4f}")
    print("-----------------------------------------------------\n")

    log_exp_evaluate(name, f"Query Refinement Evaluation Results on Dataset: {name}, Model: {model_type}, min_rel_val ≥ {min_rel_val}")
    log_exp_evaluate(name, "-----------------------------------------------------")
    log_exp_evaluate(name, f"Total Queries Evaluated: {total_queries}")
    log_exp_evaluate(name, f"Mean Average Precision (MAP):     {np.mean(all_ap):.4f}")
    log_exp_evaluate(name, f"Precision@{k}:                     {np.mean(all_precisions):.4f}")
    log_exp_evaluate(name, f"Recall:                           {np.mean(all_recalls):.4f}")
    log_exp_evaluate(name, f"Mean Reciprocal Rank (MRR):       {np.mean(all_rr):.4f}")


@app.route('/evaluate_refinement', methods=['POST'])
def evaluate_api():
    data = request.get_json()

    dataset = data.get('dataset')  # "A" أو "Q"
    model_type = data.get('model_type', 'hybrid')
    k = int(data.get('k', 10))
    min_rel_val = int(data.get('min_rel_val', 1))

    if dataset not in ['A', 'Q']:
        return jsonify({'error': 'Invalid dataset name. Use "A" or "Q".'}), 400

    try:
        evaluate_refinement_model(dataset, model_type=model_type, k=k, min_rel_val=min_rel_val)
        return jsonify({'status': 'Evaluation completed successfully.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    preload_data()
    app.run(port=5005)