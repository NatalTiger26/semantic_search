"""Unit tests for dimensionality reduction and visualization functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from search import (
    compute_explained_variance,
    pca_via_svd,
    plot_embedding_space,
)


def test_pca_via_svd_shapes_and_orthogonality() -> None:
    """Verify projection dimensions and principal component orthogonality."""
    np.random.seed(42)
    data: np.ndarray = np.random.randn(20, 10)
    n_components: int = 2

    projection, components = pca_via_svd(data, n_components=n_components)

    assert projection.shape == (20, n_components)
    assert components.shape == (n_components, 10)

    # Principal components must be orthonormal
    dot_prod: float = float(np.dot(components[0], components[1]))
    assert dot_prod == pytest.approx(0.0, abs=1e-7)

    norm_c0: float = float(np.linalg.norm(components[0]))
    norm_c1: float = float(np.linalg.norm(components[1]))
    assert norm_c0 == pytest.approx(1.0, rel=1e-5)
    assert norm_c1 == pytest.approx(1.0, rel=1e-5)


def test_pca_via_svd_invalid_components() -> None:
    """Verify pca_via_svd raises ValueError for invalid n_components."""
    data: np.ndarray = np.ones((5, 3))
    with pytest.raises(ValueError, match="n_components"):
        pca_via_svd(data, n_components=0)
    with pytest.raises(ValueError, match="n_components"):
        pca_via_svd(data, n_components=10)


def test_compute_explained_variance() -> None:
    """Verify variance ratios are bounded between 0 and 1 and descending."""
    np.random.seed(42)
    data: np.ndarray = np.random.randn(30, 8)
    ratios: np.ndarray = compute_explained_variance(data, n_components=3)

    assert len(ratios) == 3
    assert all(0.0 <= r <= 1.0 for r in ratios)
    # Variance ratios must be non-increasing
    assert ratios[0] >= ratios[1] >= ratios[2]
    # Sum of variance ratios cannot exceed 1.0
    assert float(np.sum(ratios)) <= 1.0 + 1e-6


def test_plot_embedding_space_file_output(tmp_path: Path) -> None:
    """Verify plot_embedding_space executes and writes an image file to disk."""
    np.random.seed(42)
    matrix: np.ndarray = np.random.randn(10, 8)
    docs: list[dict[str, Any]] = [
        {"id": i, "topic": "Physics" if i < 5 else "Biology", "text": f"Doc {i}"}
        for i in range(10)
    ]
    output_img: Path = tmp_path / "test_pca.png"

    plot_embedding_space(
        embedding_matrix=matrix,
        documents=docs,
        n_components=2,
        output_path=output_img,
        show=False,
    )

    assert output_img.exists()
    assert output_img.stat().st_size > 0
