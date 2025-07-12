import ir_datasets
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from query_processing import query_expansion

app = Flask(__name__)
CORS(app, origins='http://127.0.0.1:5500')
CORS(app, origins='*')



def get_docs(relevant_docs, docstore):
    result = {}
    for doc_id in relevant_docs:
        doc = docstore.get(doc_id)
        text = doc.text
        result[doc_id] = text
    return result


@app.route('/query', methods=['POST'])
def post_query():
    query_data = request.get_json()
    query_text = query_data['query']
    # "we" | "tfidf" | "hybrid"
    model_type = query_data['model_type'] #"we"
    dataset_name = query_data['dataset_name']
    print(query_text, dataset_name)

    if dataset_name == "A":
        antique_dataset = ir_datasets.load("antique")
        antique_docstore = antique_dataset.docs_store()
        expanded_results = query_expansion(query_text, dataset_name, 10, model_type)
        res = get_docs(expanded_results, antique_docstore)
        print("Antique", res)
    if dataset_name == "Q":
        quora_dataset = ir_datasets.load("beir/quora")
        quora_docstore = quora_dataset.docs_store()
        expanded_results = query_expansion(query_text, dataset_name, 10, model_type)
        res = get_docs(expanded_results, quora_docstore)
        print("Quora", res)
    ordered_docs = [{'doc_id': doc_id, 'text': text} for doc_id, text in res.items()]
    response = {'relevant_docs': ordered_docs}
    return jsonify(response)


def refine_query_via_api():
    url = "http://127.0.0.1:5007/query-refinement/refine-query"
    try:
        response = requests.post(url)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")


def evaluate_via_api(dataset, model_type="hybrid", k=10, min_rel_val=1):
    url = "http://127.0.0.1:5003/evaluate"
    try:
        payload = {
            "dataset": dataset,
            "model_type": model_type,
            "k": k,
            "min_rel_val": min_rel_val
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Evaluation API response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error calling evaluation API: {e}")


def evaluate_refinement_via_api(dataset, model_type="hybrid", k=10, min_rel_val=1):
    url = "http://127.0.0.1:5005/evaluate_refinement"
    try:
        payload = {
            "dataset": dataset,
            "model_type": model_type,
            "k": k,
            "min_rel_val": min_rel_val
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Evaluation API response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error calling evaluation API: {e}")

if __name__ == '__main__':
    print("Starting Main server...")
    #evaluate_via_api("A", model_type="tfidf")
    #evaluate_via_api("A", model_type="we")
    #evaluate_via_api("A", model_type="hybrid")
    #evaluate_via_api("A", model_type="vs")
    #evaluate_via_api("A", model_type="hybrid_vs")

    #evaluate_via_api("Q", model_type="tfidf")
    #evaluate_via_api("Q", model_type="we")
    #evaluate_via_api("Q", model_type="hybrid")
    #evaluate_via_api("Q", model_type="vs")
    #evaluate_via_api("Q", model_type="hybrid_vs")

    #evaluate_refinement_via_api("A", model_type="hybrid", k=10)

    #evaluate_refinement_via_api("Q", model_type="hybrid", k=10)

    #test_queries = ir_datasets.load("antique/test").queries

    #for query in test_queries:
    #  if query.query_id == "1964316":
    #     print(query.text)
    app.run()
