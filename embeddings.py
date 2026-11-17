"""Embedding generation module for semantic search.

Supports two operational modes:
    1. "offline" (default): A deterministic hashed bag-of-words representation
       requiring no external API keys or network connection. Ideal for local
       development, integration testing, and CI environments.
    2. "api": Integrates with the NVIDIA NIM embeddings microservice
       (model: nvidia/nv-embedqa-e5-v5) for high-dimensional semantic embeddings.

Both modes return a flat Python list of floats, providing a uniform interface
for downstream similarity calculations and matrix operations.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

# Load configuration from .env file if available
load_dotenv()

# Configuration and constants
DEFAULT_EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_MODEL: str = DEFAULT_EMBEDDING_MODEL
EMBEDDING_URL: str = "https://integrate.api.nvidia.com/v1/embeddings"
OFFLINE_DIM: int = 64

EMBEDDING_MODE: str = os.environ.get("LAB3_EMBEDDING_MODE", "offline")
API_KEY: str | None = os.environ.get("NVIDIA_API_KEY")

_WORD_RE: re.Pattern[str] = re.compile(r"[a-z']+")

__all__ = [
    "API_KEY",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_MODE",
    "EMBEDDING_MODEL",
    "EMBEDDING_URL",
    "OFFLINE_DIM",
    "_get_embedding_api",
    "_get_embedding_offline",
    "get_embedding",
]


def _get_embedding_offline(text: str, dim: int = OFFLINE_DIM) -> list[float]:
    """Compute a deterministic hashed bag-of-words vector locally.

    Each word in the lower-cased input is mapped to a bucket in [0, dim) via MD5
    hash modulo, accumulating token frequencies. The resultant vector is
    L2-normalized to ensure unit length, allowing cosine similarity to behave
    consistently.

    Args:
        text: Input string to embed.
        dim: Dimensionality of the output embedding vector. Defaults to OFFLINE_DIM (64).

    Returns:
        A list of floats of length `dim`, representing the L2-normalized vector.
        Returns a zero-vector if no words are found.
    """
    vec: list[float] = [0.0] * dim
    words: list[str] = _WORD_RE.findall(text.lower())

    for word in words:
        bucket: int = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0

    norm: float = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]

    return vec


def _get_embedding_api(
    text: str,
    input_type: str = "passage",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[float]:
    """Fetch semantic embeddings from the NVIDIA NIM API.

    Calls the NVIDIA NIM embeddings endpoint (`nvidia/nv-embedqa-e5-v5`).
    Requires an active NVIDIA API key and network connectivity.

    Args:
        text: Input text string to embed.
        input_type: Model input type. Must be either "passage" (for document
            indexing) or "query" (for search queries). Defaults to "passage".
        api_key: Optional explicit API key. If omitted, falls back to the
            NVIDIA_API_KEY environment variable.
        timeout: Network request timeout in seconds. Defaults to 30.0.

    Returns:
        Dense float embedding vector returned by the model.

    Raises:
        ValueError: If NVIDIA_API_KEY is unset or empty, or if `input_type`
            is invalid.
        requests.RequestException: If the network request fails or returns an
            HTTP error status.
    """
    resolved_key: str | None = api_key or os.environ.get("NVIDIA_API_KEY") or API_KEY
    if not resolved_key or not resolved_key.strip():
        raise ValueError(
            "NVIDIA_API_KEY not found. Please set the NVIDIA_API_KEY environment "
            "variable or supply it in a .env file."
        )

    if input_type not in ("passage", "query"):
        raise ValueError(f"Invalid input_type: {input_type!r}. Must be 'passage' or 'query'.")

    headers: dict[str, str] = {
        "Authorization": f"Bearer {resolved_key.strip()}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": EMBEDDING_MODEL,
        "input": [text],
        "input_type": input_type,
    }

    response: requests.Response = requests.post(
        EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    try:
        embedding: list[float] = data["data"][0]["embedding"]
        return embedding
    except (KeyError, IndexError, TypeError) as err:
        raise ValueError(
            f"Malformed response received from NVIDIA NIM embeddings endpoint: {data}"
        ) from err


def get_embedding(
    text: str,
    input_type: str = "passage",
    mode: str | None = None,
) -> list[float]:
    """Retrieve text embedding using either offline or API mode.

    The active mode is resolved from the `mode` argument, the `LAB3_EMBEDDING_MODE`
    environment variable, or the module default (defaults to "offline").

    Args:
        text: Text to be embedded.
        input_type: Role of text; "passage" for indexed documents or "query" for
            search queries. Defaults to "passage".
        mode: Optional override for embedding mode ("offline" or "api").

    Returns:
        List of floats containing the embedding representation.

    Raises:
        ValueError: If an unknown mode is provided or API key is missing in API mode.
    """
    resolved_mode: str = (
        (mode or os.environ.get("LAB3_EMBEDDING_MODE") or EMBEDDING_MODE).lower().strip()
    )

    if resolved_mode == "api":
        return _get_embedding_api(text, input_type=input_type)
    if resolved_mode == "offline":
        return _get_embedding_offline(text)

    raise ValueError(f"Unknown embedding mode: {resolved_mode!r}. Expected 'offline' or 'api'.")
