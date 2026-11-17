"""Unit tests for the embeddings module."""

from __future__ import annotations

import math
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    OFFLINE_DIM,
    _get_embedding_api,
    _get_embedding_offline,
    get_embedding,
)


def test_offline_embedding_shape_and_norm() -> None:
    """Verify offline embedding returns a list with correct dimension and unit L2 norm."""
    text: str = "The quick brown fox jumps over the lazy dog."
    vec: list[float] = _get_embedding_offline(text, dim=OFFLINE_DIM)

    assert isinstance(vec, list)
    assert len(vec) == OFFLINE_DIM
    assert all(isinstance(x, float) for x in vec)

    norm: float = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, rel=1e-5)


def test_offline_embedding_deterministic() -> None:
    """Verify offline embedding produces identical outputs for the same text."""
    text: str = "Machine learning and artificial intelligence"
    vec1: list[float] = _get_embedding_offline(text)
    vec2: list[float] = _get_embedding_offline(text)

    assert vec1 == vec2


def test_offline_embedding_distinct_texts() -> None:
    """Verify distinct semantic passages produce different vectors."""
    text1: str = "Quantum mechanics and black holes"
    text2: str = "Baking sourdough bread and yeast fermentation"

    vec1: list[float] = _get_embedding_offline(text1)
    vec2: list[float] = _get_embedding_offline(text2)

    assert vec1 != vec2


def test_offline_embedding_empty_string() -> None:
    """Verify empty string returns a zero vector without error."""
    vec: list[float] = _get_embedding_offline("")

    assert len(vec) == OFFLINE_DIM
    assert all(x == 0.0 for x in vec)


def test_api_embedding_missing_key() -> None:
    """Verify _get_embedding_api raises ValueError when API key is missing."""
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="NVIDIA_API_KEY not found"),
    ):
        _get_embedding_api("test query", api_key=None)


def test_api_embedding_invalid_input_type() -> None:
    """Verify _get_embedding_api raises ValueError for unsupported input_type."""
    with pytest.raises(ValueError, match="Invalid input_type"):
        _get_embedding_api("test", input_type="invalid_type", api_key="dummy_key")


def test_api_embedding_mocked_success() -> None:
    """Verify _get_embedding_api correctly formats request and parses response."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    expected_embedding: list[float] = [0.1, 0.2, 0.3, 0.4]
    mock_response.json.return_value = {
        "data": [
            {
                "embedding": expected_embedding,
                "index": 0,
            }
        ],
        "model": DEFAULT_EMBEDDING_MODEL,
        "usage": {"total_tokens": 10},
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        result: list[float] = _get_embedding_api(
            "Quantum computing",
            input_type="query",
            api_key="test_api_key",
        )

        assert result == expected_embedding
        mock_post.assert_called_once()
        kwargs = mock_post.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer test_api_key"
        assert kwargs["json"]["model"] == DEFAULT_EMBEDDING_MODEL
        assert kwargs["json"]["input"] == ["Quantum computing"]
        assert kwargs["json"]["input_type"] == "query"


def test_api_embedding_malformed_response() -> None:
    """Verify _get_embedding_api raises ValueError if response data is missing."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}  # missing "data"

    with (
        patch("requests.post", return_value=mock_response),
        pytest.raises(ValueError, match="Malformed response"),
    ):
        _get_embedding_api("test", api_key="test_key")


def test_api_embedding_http_error() -> None:
    """Verify HTTP error raises requests.HTTPError."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")

    with patch("requests.post", return_value=mock_response), pytest.raises(requests.HTTPError):
        _get_embedding_api("test", api_key="invalid_key")


def test_get_embedding_mode_dispatch() -> None:
    """Verify get_embedding routes to correct backend based on mode argument."""
    with (
        patch("embeddings._get_embedding_offline", return_value=[0.1] * 64) as mock_off,
        patch("embeddings._get_embedding_api", return_value=[0.2] * 1024) as mock_api,
    ):
        res_off = get_embedding("dummy text", mode="offline")
        assert res_off == [0.1] * 64
        mock_off.assert_called_once_with("dummy text")
        mock_api.assert_not_called()

        mock_off.reset_mock()
        res_api = get_embedding("dummy text", input_type="passage", mode="api")
        assert res_api == [0.2] * 1024
        mock_api.assert_called_once_with("dummy text", input_type="passage")
        mock_off.assert_not_called()


def test_get_embedding_invalid_mode() -> None:
    """Verify get_embedding raises ValueError when given an unknown mode."""
    with pytest.raises(ValueError, match="Unknown embedding mode"):
        get_embedding("test", mode="unsupported_mode")
