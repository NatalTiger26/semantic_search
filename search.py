import json
from pathlib import Path

import numpy as np

from embeddings import get_embedding

CACHE_FILE = Path("embeddings_cache.json")


def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_embedding_matrix(documents):

    if CACHE_FILE.exists():

        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            embeddings = json.load(f)

        # Simple cache validation
        if len(embeddings) == len(documents):
            print("Loaded embeddings from cache.")
            return np.array(embeddings)

    embeddings = []

    print("Generating embeddings...")

    for doc in documents:

        embedding = get_embedding(
            doc["text"],
            input_type="passage"
        )

        embeddings.append(embedding)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    return np.array(embeddings)


def search(query, embedding_matrix, documents, top_k=3):

    query_embedding = get_embedding(
        query,
        input_type="query"
    )

    scores = []

    for embedding in embedding_matrix:
        score = cosine_similarity(query_embedding, embedding)
        scores.append(score)

    scores = np.array(scores)

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in indices:
        results.append(
            {
                "score": float(scores[idx]),
                "document": documents[idx]
            }
        )

    return results