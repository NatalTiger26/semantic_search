"""Unit tests for the search and indexing module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from search import (
    DocumentFormatError,
    build_embedding_matrix,
    cosine_similarity,
    load_documents,
    search,
)


def test_load_documents_success(tmp_path: Path) -> None:
    """Verify load_documents loads a valid JSON document list."""
    sample_docs: list[dict[str, Any]] = [
        {"id": 0, "topic": "Tech", "text": "Python is a versatile programming language."},
        {"id": 1, "topic": "Cooking", "text": "Sourdough bread requires wild yeast."},
    ]
    file_path: Path = tmp_path / "test_docs.json"
    file_path.write_text(json.dumps(sample_docs), encoding="utf-8")

    loaded: list[dict[str, Any]] = load_documents(file_path)
    assert loaded == sample_docs
    assert len(loaded) == 2


def test_load_documents_file_not_found() -> None:
    """Verify load_documents raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        load_documents("non_existent_corpus_file_12345.json")


def test_load_documents_invalid_json(tmp_path: Path) -> None:
    """Verify load_documents raises DocumentFormatError if JSON is not a list or missing fields."""
    not_a_list_path: Path = tmp_path / "dict.json"
    not_a_list_path.write_text(json.dumps({"text": "single"}), encoding="utf-8")
    with pytest.raises(DocumentFormatError, match="Expected a list of documents"):
        load_documents(not_a_list_path)

    missing_field_path: Path = tmp_path / "missing.json"
    missing_field_path.write_text(json.dumps([{"id": 1}]), encoding="utf-8")
    with pytest.raises(DocumentFormatError, match="missing required 'text' field"):
        load_documents(missing_field_path)


def test_cosine_similarity_edge_cases() -> None:
    """Verify cosine_similarity with identical, orthogonal, opposite, and zero vectors."""
    v1: np.ndarray = np.array([1.0, 0.0, 0.0])
    v2: np.ndarray = np.array([0.0, 1.0, 0.0])
    v_opp: np.ndarray = np.array([-1.0, 0.0, 0.0])
    v_zero: np.ndarray = np.array([0.0, 0.0, 0.0])

    # Identical
    assert cosine_similarity(v1, v1) == pytest.approx(1.0)
    # Orthogonal
    assert cosine_similarity(v1, v2) == pytest.approx(0.0)
    # Opposite
    assert cosine_similarity(v1, v_opp) == pytest.approx(-1.0)
    # Zero vector returns 0.0 without nan
    assert cosine_similarity(v1, v_zero) == pytest.approx(0.0)
    assert cosine_similarity(v_zero, v_zero) == pytest.approx(0.0)


def test_cosine_similarity_vectorized() -> None:
    """Verify 1D vs 2D cosine similarity behaves identically to pairwise calculations."""
    query: np.ndarray = np.array([1.0, 0.0, 0.0])
    matrix: np.ndarray = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    sims: np.ndarray = cosine_similarity(query, matrix)  # type: ignore[assignment]
    assert isinstance(sims, np.ndarray)
    assert sims.shape == (4,)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(0.0)
    assert sims[2] == pytest.approx(-1.0)
    assert sims[3] == pytest.approx(0.0)


def test_build_embedding_matrix_and_caching(tmp_path: Path) -> None:
    """Verify build_embedding_matrix writes and reloads cache properly."""
    docs: list[dict[str, Any]] = [
        {"id": 0, "topic": "Tech", "text": "Doc alpha"},
        {"id": 1, "topic": "Tech", "text": "Doc beta"},
    ]
    cache_path: Path = tmp_path / "test_cache.json"

    # First generation
    mat1: np.ndarray = build_embedding_matrix(docs, cache_file=cache_path)
    assert mat1.shape == (2, 64)
    assert cache_path.exists()

    # Second generation loads from cache
    mat2: np.ndarray = build_embedding_matrix(docs, cache_file=cache_path)
    assert np.array_equal(mat1, mat2)


def test_build_embedding_matrix_empty() -> None:
    """Verify build_embedding_matrix returns empty array for empty document list."""
    mat: np.ndarray = build_embedding_matrix([])
    assert mat.shape == (0, 0)


def test_search_ranking() -> None:
    """Verify search returns top_k highest similarity results in descending order."""
    docs: list[dict[str, Any]] = [
        {"id": 0, "topic": "Physics", "text": "Quantum mechanics and wavefunctions."},
        {"id": 1, "topic": "Culinary", "text": "Baking baguettes with sourdough starter."},
        {"id": 2, "topic": "History", "text": "The fall of the Roman Republic in 27 BC."},
    ]
    mat: np.ndarray = build_embedding_matrix(docs, cache_file=None)

    results: list[dict[str, Any]] = search(
        "quantum mechanics",
        mat,
        docs,
        top_k=3,
    )

    assert len(results) == 3
    # Scores must be in descending order
    scores: list[float] = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # The Physics document should rank first
    assert results[0]["document"]["topic"] == "Physics"


def test_search_empty_docs_or_top_k() -> None:
    """Verify search returns empty list when given no docs or top_k <= 0."""
    docs: list[dict[str, Any]] = [
        {"id": 0, "topic": "Physics", "text": "Sample text."},
    ]
    mat: np.ndarray = np.zeros((1, 64), dtype=np.float32)

    assert search("query", mat, [], top_k=3) == []
    assert search("query", mat, docs, top_k=0) == []
    assert search("query", mat, docs, top_k=-1) == []


def test_search_dimension_mismatch() -> None:
    """Verify search raises ValueError when query dimension differs from matrix."""
    docs: list[dict[str, Any]] = [
        {"id": 0, "topic": "Physics", "text": "Sample."},
    ]
    fake_matrix: np.ndarray = np.zeros((1, 128), dtype=np.float32)

    with pytest.raises(ValueError, match="dimensionality mismatch"):
        search("test query", fake_matrix, docs, top_k=1)
