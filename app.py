import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV


@st.cache_data
def load_data():
    df = pd.read_csv("sbert.csv")
    df["Embeddings"] = df["Embeddings"].apply(eval)  # Convert string to list
    return df


df = load_data()

# Load Hugging Face model and tokenizer


@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


tokenizer, model = load_model()

# Function to generate embeddings using Hugging Face model


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to find similar assessments


def find_similar_assessments(query, df, top_n=5):
    query_embedding = get_embedding(query)

    df["Similarity"] = df["Embeddings"].apply(
        lambda x: cosine_similarity([query_embedding], [x[:384]])[0][0]
    )

    results = df.sort_values(by="Similarity", ascending=False).head(top_n)

    # Include "URL" in the selection
    return results[["Assessment Name", "Description", "Remote_Testing", "adaptive", "URL", "Similarity"]]


# Streamlit UI
st.title("🔍 SHL Assessment Recommendation Engine")

# User input
query = st.text_input("Enter job description or query:")

if query:
    st.subheader("Top Matching Assessments:")
    results = find_similar_assessments(query, df)

    # Format Table with Clickable URL
    def make_clickable(url, name):
        return f'<a href="{url}" target="_blank">{name}</a>'

    results["Assessment Name"] = results.apply(
        lambda row: make_clickable(row["URL"], row["Assessment Name"]), axis=1)
    results = results.drop(columns=["URL"])  # Remove raw URL column

    # Display Table with HTML Rendering
    st.write(
        results.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
