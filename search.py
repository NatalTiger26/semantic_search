"""Search and retrieval algorithms for semantic search.

Provides functionality to:
    - Load and validate corpus documents from JSON.
    - Compute cosine similarities (scalar and vectorized).
    - Construct, cache, and reload dense embedding matrices.
    - Query the embedding space and rank documents by cosine similarity.
    - Reduce dimensionality with PCA (via SVD) and visualize clusters by topic.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from embeddings import OFFLINE_DIM, get_embedding

CACHE_FILE: Path = Path("embeddings_cache.json")
OFFLINE_CACHE_FILE: Path = Path("embeddings_cache_offline.json")
logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_FILE",
    "DocumentFormatError",
    "build_embedding_matrix",
    "compute_explained_variance",
    "cosine_similarity",
    "load_documents",
    "pca_via_svd",
    "plot_embedding_space",
    "search",
]


class DocumentFormatError(ValueError, TypeError):
    """Raised when the document corpus file format or content is invalid."""


def load_documents(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate documents from a JSON file.

    Expected document structure:
        [
            {"id": 0, "topic": "Physics", "text": "Passage text..."},
            ...
        ]

    Args:
        path: Path to the JSON corpus file.

    Returns:
        A list of document dictionaries.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        DocumentFormatError: If JSON is not a list or document entries lack required fields.
    """
    file_path: Path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise DocumentFormatError(
            f"Expected a list of documents in '{file_path}', but found {type(data).__name__}."
        )

    for i, doc in enumerate(data):
        if not isinstance(doc, dict):
            raise DocumentFormatError(f"Document at index {i} is not a dictionary.")
        if "text" not in doc:
            raise DocumentFormatError(f"Document at index {i} is missing required 'text' field.")

    return data


def cosine_similarity(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Compute cosine similarity between two vectors, or between a vector and matrix.

    Supports:
        - 1D vector vs 1D vector -> returns scalar float.
        - 1D vector vs 2D matrix (or 2D matrix vs 1D vector) -> returns 1D array of floats.
        - 2D matrix vs 2D matrix -> returns 1D array of row-wise pairwise similarities.

    Gracefully handles zero-norm vectors by returning 0.0 without divide-by-zero warnings.

    Args:
        a: First vector or matrix.
        b: Second vector or matrix.

    Returns:
        Scalar float or NumPy array representing cosine similarities in [-1.0, 1.0].

    Raises:
        ValueError: If array dimensions are incompatible.
    """
    a_arr: np.ndarray = np.asarray(a, dtype=np.float64)
    b_arr: np.ndarray = np.asarray(b, dtype=np.float64)

    if a_arr.ndim == 1 and b_arr.ndim == 1:
        norm_a: float = float(np.linalg.norm(a_arr))
        norm_b: float = float(np.linalg.norm(b_arr))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    if a_arr.ndim == 1 and b_arr.ndim == 2:
        norm_a_val: float = float(np.linalg.norm(a_arr))
        norm_b_vals: np.ndarray = np.linalg.norm(b_arr, axis=1)
        denom: np.ndarray = norm_a_val * norm_b_vals
        zero_mask: np.ndarray = denom == 0.0
        denom_safe: np.ndarray = np.where(zero_mask, 1.0, denom)
        sim: np.ndarray = np.dot(b_arr, a_arr) / denom_safe
        sim[zero_mask] = 0.0
        return sim

    if a_arr.ndim == 2 and b_arr.ndim == 1:
        return cosine_similarity(b_arr, a_arr)

    if a_arr.ndim == 2 and b_arr.ndim == 2:
        if a_arr.shape != b_arr.shape:
            raise ValueError(
                f"Shape mismatch for 2D pairwise cosine similarity: {a_arr.shape} vs {b_arr.shape}"
            )
        norm_a_rows: np.ndarray = np.linalg.norm(a_arr, axis=1)
        norm_b_rows: np.ndarray = np.linalg.norm(b_arr, axis=1)
        denom_2d: np.ndarray = norm_a_rows * norm_b_rows
        zero_mask_2d: np.ndarray = denom_2d == 0.0
        denom_safe_2d: np.ndarray = np.where(zero_mask_2d, 1.0, denom_2d)
        sim_2d: np.ndarray = np.sum(a_arr * b_arr, axis=1) / denom_safe_2d
        sim_2d[zero_mask_2d] = 0.0
        return sim_2d

    raise ValueError(
        f"Unsupported array dimensions for cosine similarity: {a_arr.ndim} and {b_arr.ndim}"
    )


def build_embedding_matrix(
    documents: list[dict[str, Any]],
    cache_file: str | Path | None = CACHE_FILE,
    force_refresh: bool = False,
) -> np.ndarray:
    """Build or load an embedding matrix for the supplied corpus.

    If a valid cache exists at `cache_file` and `force_refresh` is False, embeddings
    are loaded directly from the cache. Validates document count and dimensionality
    to ensure the cache matches the active embedding mode.

    Args:
        documents: List of corpus documents.
        cache_file: Path to cache file, or None to bypass caching. Defaults to CACHE_FILE.
        force_refresh: If True, regenerates embeddings even if cache exists.

    Returns:
        NumPy array of shape (N, D) where N is number of documents and D is embedding dimension.
    """
    if not documents:
        return np.empty((0, 0), dtype=np.float32)

    cache_path: Path | None = None
    active_mode: str = os.environ.get("LAB3_EMBEDDING_MODE", "offline").lower().strip()
    expected_dim: int = OFFLINE_DIM if active_mode == "offline" else 1024

    if cache_file is not None:
        target_path: Path = Path(cache_file)
        if target_path == CACHE_FILE and active_mode == "offline":
            # Protect pre-computed high-dimensional API embeddings from being overwritten
            # when running in offline fallback mode
            if target_path.exists():
                try:
                    with open(target_path, encoding="utf-8") as f:
                        probe_data: Any = json.load(f)
                    if (
                        isinstance(probe_data, list)
                        and probe_data
                        and len(probe_data[0]) != OFFLINE_DIM
                    ):
                        cache_path = OFFLINE_CACHE_FILE
                    else:
                        cache_path = target_path
                except Exception:
                    cache_path = OFFLINE_CACHE_FILE
            else:
                cache_path = target_path
        else:
            cache_path = target_path

    if cache_path is not None and cache_path.exists() and not force_refresh:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached_embeddings: Any = json.load(f)

            if isinstance(cached_embeddings, list) and len(cached_embeddings) == len(documents):
                cached_dim: int = len(cached_embeddings[0]) if cached_embeddings else 0
                if cached_dim == expected_dim or active_mode not in ("offline", "api"):
                    print("Loaded embeddings from cache.")
                    return np.asarray(cached_embeddings, dtype=np.float32)
                print(
                    f"Cache dimension ({cached_dim}) differs from mode '{active_mode}' "
                    f"expected dimension ({expected_dim}). Rebuilding embeddings..."
                )
        except (json.JSONDecodeError, OSError, IndexError) as err:
            logger.warning("Failed to read cache file '%s': %s. Rebuilding...", cache_path, err)

    embeddings: list[list[float]] = []
    print("Generating embeddings...")

    for doc in documents:
        embedding: list[float] = get_embedding(
            doc["text"],
            input_type="passage",
        )
        embeddings.append(embedding)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f)
        except OSError as err:
            logger.warning("Failed to write embeddings cache to '%s': %s", cache_path, err)

    return np.asarray(embeddings, dtype=np.float32)


def search(
    query: str,
    embedding_matrix: np.ndarray | Sequence[Sequence[float]],
    documents: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search for relevant documents in the corpus using cosine similarity ranking.

    Args:
        query: Search query text.
        embedding_matrix: 2D array of shape (N, D) containing document embeddings.
        documents: List of corpus documents corresponding row-by-row to the matrix.
        top_k: Number of top results to return. Defaults to 3.

    Returns:
        List of dictionaries with keys "score" (float) and "document" (dict),
        ordered from highest to lowest similarity.

    Raises:
        ValueError: If matrix rows do not match document count, or if query
            embedding dimensionality does not match the embedding matrix.
    """
    if not documents or top_k <= 0:
        return []

    mat: np.ndarray = np.asarray(embedding_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"embedding_matrix must be 2-dimensional, got shape {mat.shape}.")

    if len(documents) != mat.shape[0]:
        raise ValueError(
            f"Document count ({len(documents)}) does not match matrix rows ({mat.shape[0]})."
        )

    query_embedding: list[float] = get_embedding(
        query,
        input_type="query",
    )
    query_vec: np.ndarray = np.asarray(query_embedding, dtype=np.float64)

    if query_vec.shape[0] != mat.shape[1]:
        raise ValueError(
            f"Embedding dimensionality mismatch: query dimension ({query_vec.shape[0]}) "
            f"does not match embedding matrix dimension ({mat.shape[1]}). "
            "Ensure both were generated with the same embedding model and mode."
        )

    # Compute vectorized cosine similarity across all rows
    scores: np.ndarray = cast(np.ndarray, cosine_similarity(query_vec, mat))

    # Retrieve indices of top_k highest scores
    k: int = min(top_k, len(scores))
    indices: np.ndarray = np.argsort(scores)[::-1][:k]

    results: list[dict[str, Any]] = [
        {
            "score": float(scores[idx]),
            "document": documents[idx],
        }
        for idx in indices
    ]

    return results


def pca_via_svd(
    data: np.ndarray | Sequence[Sequence[float]],
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Principal Component Analysis via Singular Value Decomposition (SVD).

    Centers the input data matrix, computes the compact SVD, and projects the data
    onto the first `n_components` right singular vectors (principal directions).

    Args:
        data: Data matrix of shape (N, D).
        n_components: Number of principal components to retain. Defaults to 2.

    Returns:
        A tuple of:
            - projection: Reduced data matrix of shape (N, n_components).
            - components: Principal axes in feature space of shape (n_components, D).

    Raises:
        ValueError: If n_components is less than 1 or greater than feature dimension.
    """
    data_arr: np.ndarray = np.asarray(data, dtype=np.float64)
    if data_arr.ndim != 2:
        raise ValueError(f"Data matrix must be 2D, got shape {data_arr.shape}.")

    n_samples, n_features = data_arr.shape
    if n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components={n_components} must be between 1 and min(n_samples, n_features)="
            f"{min(n_samples, n_features)}."
        )

    mean: np.ndarray = np.mean(data_arr, axis=0)
    centered_data: np.ndarray = data_arr - mean

    _, _, vt = np.linalg.svd(centered_data, full_matrices=False)
    components: np.ndarray = vt[:n_components]
    projection: np.ndarray = centered_data @ components.T

    return projection, components


def compute_explained_variance(
    data: np.ndarray | Sequence[Sequence[float]],
    n_components: int = 2,
) -> np.ndarray:
    """Compute the explained variance ratio for the top principal components.

    Args:
        data: Input data matrix of shape (N, D).
        n_components: Number of components. Defaults to 2.

    Returns:
        1D NumPy array of explained variance ratios for the requested components.
    """
    data_arr: np.ndarray = np.asarray(data, dtype=np.float64)
    if data_arr.ndim != 2 or data_arr.shape[0] < 2:
        return np.zeros(n_components, dtype=np.float64)

    mean: np.ndarray = np.mean(data_arr, axis=0)
    centered_data: np.ndarray = data_arr - mean

    _, s, _ = np.linalg.svd(centered_data, full_matrices=False)
    eigenvalues: np.ndarray = (s**2) / (data_arr.shape[0] - 1)
    total_var: float = float(np.sum(eigenvalues))

    if total_var <= 0.0:
        return np.zeros(n_components, dtype=np.float64)

    ratios: np.ndarray = (eigenvalues / total_var)[:n_components]
    return ratios


def plot_embedding_space(
    embedding_matrix: np.ndarray | Sequence[Sequence[float]],
    documents: list[dict[str, Any]],
    n_components: int = 2,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Project embeddings to 2D with PCA and create a topic-colored scatter plot.

    Args:
        embedding_matrix: Matrix of document embeddings of shape (N, D).
        documents: Corpus documents corresponding to the embedding matrix rows.
        n_components: Number of principal components (typically 2). Defaults to 2.
        output_path: Optional path to save the generated figure.
        show: Whether to call plt.show() interactively. Defaults to True.

    Returns:
        The matplotlib Figure object.
    """
    projection, _ = pca_via_svd(embedding_matrix, n_components=n_components)

    topics: list[str] = [doc.get("topic", "Unknown") for doc in documents]
    unique_topics: list[str] = sorted(set(topics))

    fig, ax = plt.subplots(figsize=(8, 6))

    for topic in unique_topics:
        idx: list[int] = [i for i, t in enumerate(topics) if t == topic]
        ax.scatter(
            projection[idx, 0],
            projection[idx, 1],
            label=topic,
            alpha=0.8,
            edgecolors="none",
            s=40,
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Semantic Search Embedding Space (PCA Projection)")
    ax.legend(title="Topic", loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if output_path is not None:
        save_path: Path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
