from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Dataset


def load_data():
    df = pd.read_csv("sbert.csv")
    # Convert stored embeddings from string to list
    df["Embeddings"] = df["Embeddings"].apply(eval)
    return df


df = load_data()

# Load SBERT Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to Generate Embeddings


def get_embedding(text):
    # Convert NumPy array to list for JSON compatibility
    return model.encode(text).tolist()

# Find Similar Assessments


def find_similar_assessments(query, df, top_n=5):
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)

    # Ensure embeddings match dimensions
    df["Similarity"] = df["Embeddings"].apply(
        lambda x: cosine_similarity([query_embedding[0]], x)[0][0]
    )

    results = df.sort_values(by="Similarity", ascending=False).head(top_n)
    return results[["Assessment Name", "Description", "Remote_Testing", "adaptive", "URL", "Assessment Length (min)", "Job Levels", "Similarity"]]

# API Endpoint


@app.route("/recommend", methods=["GET"])
def recommend():
    query = request.args.get("query")

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = find_similar_assessments(query, df)

    # Convert results to JSON
    return jsonify(results.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
