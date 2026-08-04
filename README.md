# Semantic Search Engine

A small semantic search system: it turns a corpus of short text documents into
embeddings, searches them by meaning (not just keyword overlap) using cosine
similarity, and visualizes the resulting embedding space with PCA.

## What it does

1. Loads a corpus of documents from `documents.json`.
2. Fetches an embedding vector for every document (either from the NVIDIA
   NIM API or from a local offline fallback) and caches the results.
3. Given a text query, embeds the query and ranks all documents by cosine
   similarity to return the top-k most relevant matches.
4. Projects the embedding matrix down to 2 dimensions with PCA (via SVD) and
   plots it as a scatter plot colored by topic, so semantically similar
   documents visibly cluster together.

## File structure

```
semantic_search/
├── documents.json              # the corpus (topic, text, id per entry)
├── embeddings.py                # fetches/computes embeddings (API or offline mode)
├── search.py                    # loading, caching, cosine similarity, search()
├── semantic_search_starter.ipynb  # notebook that runs the full pipeline end to end
├── embeddings_cache.json        # auto-generated cache of computed embeddings
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

If you want to use the real API mode, create a `.env` file in this folder
with your NVIDIA NIM key:

```
NVIDIA_API_KEY=your-key-here
```

`embeddings.py` loads this automatically via `python-dotenv` — no manual
`export` needed.

## Running it

Open `semantic_search_starter.ipynb` and run all cells top to bottom. It will:

1. Load `documents.json` and build the embedding matrix.
2. Run a couple of example search queries and print the top matches.
3. Show a PCA scatter plot of the embedding space, colored by topic.

You can also import and use the pieces directly in a script:

```python
from search import load_documents, build_embedding_matrix, search

documents = load_documents("documents.json")
embedding_matrix = build_embedding_matrix(documents)

results = search("quantum mechanics", embedding_matrix, documents, top_k=3)
for r in results:
    print(r["score"], r["document"]["topic"], r["document"]["text"])
```

## Offline mode vs. API mode

Embedding generation is controlled by the `LAB3_EMBEDDING_MODE` environment
variable, read in `embeddings.py`:

| Mode | How to enable | What it does |
|---|---|---|
| `offline` (default) | do nothing, or `LAB3_EMBEDDING_MODE=offline` | Computes a deterministic hashed bag-of-words vector locally. No API key, no network call. Good for building and testing the pipeline, but it matches on shared vocabulary rather than meaning, so search quality and cluster separation are weaker. |
| `api` | `LAB3_EMBEDDING_MODE=api` (with `NVIDIA_API_KEY` set in `.env`) | Calls the NVIDIA NIM embeddings endpoint (`nvidia/nv-embedqa-e5-v5`) for real semantic embeddings. Produces noticeably better search relevance and cleaner PCA clusters. |

Switching modes is a one-line environment variable change — no code edits
needed. Example:

```bash
export LAB3_EMBEDDING_MODE=api
```

or set it in your `.env` file alongside `NVIDIA_API_KEY`.

If API mode is selected but no key is found, `_get_embedding_api` raises a
`ValueError` immediately instead of failing silently.

## Embedding cache

The first time embeddings are built for a corpus, `build_embedding_matrix`
writes them to `embeddings_cache.json`. On later runs it loads straight from
that cache instead of re-calling the API, as long as the cached embedding
count still matches the number of documents in `documents.json`. If you
change the size of your corpus, the cache is treated as stale and
embeddings are refetched. Delete `embeddings_cache.json` at any time to force
a full rebuild (for example, after switching from offline to API mode, since
the two modes produce embeddings of different dimensionality).

## Notes

- Cosine similarity and the PCA-via-SVD projection are the two core linear
  algebra techniques underlying retrieval and visualization here.
- In offline mode, cluster separation in the PCA plot is expected to be
  looser than in API mode, since the hashed bag-of-words vectors only
  capture shared vocabulary, not underlying meaning.