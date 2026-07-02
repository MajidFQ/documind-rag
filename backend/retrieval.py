"""
retrieval.py — Semantic search and cross-encoder reranking for DocuMind.

Two-stage retrieval pipeline:

  Stage 1 — Embedding retrieval (fast, approximate):
    The query is embedded into a vector, then ChromaDB finds the top_k chunks
    whose vectors are closest by cosine distance. This is fast (milliseconds)
    but approximate — embedding similarity captures general topic overlap,
    not fine-grained relevance.

  Stage 2 — Cross-encoder reranking (slower, precise):
    The top_k candidates from stage 1 are re-scored by a cross-encoder model
    that reads the query and each chunk *together* in one pass. Because it sees
    both texts at once it can pick up on exact phrasing, negation, and nuance
    that embedding comparison misses. The candidates are then reordered by this
    more accurate score.

  Combined entry point:
    retrieve_and_rerank() chains both stages. main.py and generation.py should
    call this function — it is the single public interface for retrieval.

  Distance metric note:
    The Chroma collection is created with hnsw:space="cosine", so ChromaDB
    returns cosine *distance* (0 = identical, 2 = opposite). We convert to a
    0–1 relevance score with:  relevance = 1 - (distance / 2)
    This makes scores intuitive: 1.0 means perfect match, 0.0 means no match.
"""

import pathlib
from typing import List, Dict, Optional

from sentence_transformers import CrossEncoder

try:
    from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
except OSError:
    CHROMA_PERSIST_DIR = "./chroma_store"
    CHROMA_COLLECTION_NAME = "documind"

from embeddings import EmbeddingModel, VectorStore


# Cross-encoder model used for reranking.
# ms-marco-MiniLM-L-6-v2 is trained specifically on passage relevance for
# search tasks, is only ~80 MB, and runs fast on CPU.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Distance → relevance conversion
# ---------------------------------------------------------------------------

def cosine_distance_to_relevance(distance: float) -> float:
    """
    Convert a ChromaDB cosine distance value to a 0–1 relevance score.

    ChromaDB cosine distance ranges from 0 (vectors are identical) to 2
    (vectors point in opposite directions). Dividing by 2 normalises to
    [0, 1], then subtracting from 1 flips it so higher = more relevant.

    Formula:  relevance = 1 - (distance / 2)

    Args:
        distance: Raw cosine distance returned by ChromaDB (0–2 range).

    Returns:
        Relevance score in [0, 1] where 1.0 is a perfect match.
    """
    # Clamp to [0, 2] defensively — floating point can occasionally drift
    clamped = max(0.0, min(2.0, distance))
    return round(1.0 - (clamped / 2.0), 4)


# ---------------------------------------------------------------------------
# Stage 1 — Embedding retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    query: str,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = 5,
    embedding_model: Optional[EmbeddingModel] = None,
    vector_store: Optional[VectorStore] = None,
) -> List[Dict]:
    """
    Embed a query and return the top_k most similar chunks from ChromaDB.

    Each returned dict contains everything downstream needs for answer
    generation and citations:
      - text:         the raw chunk text
      - relevance:    0–1 similarity score (1 = most similar)
      - source:       original filename
      - page_number:  page in the source document (-1 means not available)
      - chunk_id:     position of the chunk within the document
      - method:       chunking strategy used ('recursive' or 'fixed_size')

    Args:
        query:            User's question as a plain string.
        collection_name:  ChromaDB collection to search.
        top_k:            Maximum number of chunks to return.
        embedding_model:  Optional pre-loaded EmbeddingModel (avoids re-loading).
        vector_store:     Optional pre-connected VectorStore (avoids re-connecting).

    Returns:
        List of result dicts sorted by relevance descending.

    Raises:
        ValueError:   If the query is empty or the collection has no data.
        RuntimeError: If ChromaDB query fails.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    # Accept injected instances so the caller can reuse already-loaded models
    model = embedding_model or EmbeddingModel()
    store = vector_store or VectorStore()

    collection = store.create_collection(collection_name)
    collection_size = collection.count()

    if collection_size == 0:
        raise ValueError(
            f"Collection '{collection_name}' is empty. "
            "Ingest at least one document before running queries."
        )

    # Cap top_k at the actual collection size to avoid ChromaDB errors
    effective_top_k = min(top_k, collection_size)

    query_vector = model.embed_query(query)

    try:
        raw_results = collection.query(
            query_embeddings=[query_vector],
            n_results=effective_top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as error:
        raise RuntimeError(
            f"ChromaDB query failed for collection '{collection_name}'. "
            f"Original error: {error}"
        )

    # ChromaDB wraps results in a list-of-lists because it supports batch
    # queries; we always send one query at a time so we unwrap index [0].
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    results = []
    for text, meta, distance in zip(documents, metadatas, distances):
        results.append({
            "text": text,
            "relevance": cosine_distance_to_relevance(distance),
            "source": meta.get("source", "unknown"),
            "page_number": meta.get("page_number", -1),
            "chunk_id": meta.get("chunk_id", -1),
            "method": meta.get("method", "unknown"),
        })

    # Already sorted by ChromaDB, but make explicit for clarity
    results.sort(key=lambda r: r["relevance"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Stage 2 — Cross-encoder reranking
# ---------------------------------------------------------------------------

# WHY reranking helps:
#
#   Embedding similarity (stage 1) encodes a query and a chunk independently
#   into separate vectors, then measures their distance. This is fast and
#   scales to millions of chunks, but it is approximate: the model never sees
#   the query and chunk *together*, so it can miss subtle relevance signals
#   like exact phrasing matches, negations, or domain-specific nuance.
#
#   A cross-encoder reads the query and the candidate chunk concatenated into
#   one input, producing a single relevance score. Because both texts are
#   processed together, the model can attend to the relationship between them
#   directly. This produces significantly better relevance rankings.
#
#   The cost is speed: a cross-encoder must run a full forward pass per
#   candidate, making it too slow to rank all chunks in a large corpus.
#   The practical solution (used here) is to use embeddings to narrow the
#   candidate pool quickly, then use the cross-encoder to reorder only those
#   top_k candidates precisely. Best of both worlds.

class CrossEncoderReranker:
    """
    Reranks a list of candidate chunks using a cross-encoder relevance model.

    The cross-encoder sees the query and each chunk together (not as separate
    vectors), giving it much better accuracy than embedding similarity alone.
    It is only applied to the small candidate set returned by stage 1, keeping
    the overall latency acceptable.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        """
        Load the cross-encoder model from local cache or download on first use.

        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
        """
        try:
            print(f"Loading cross-encoder reranker '{model_name}' ...")
            self.model = CrossEncoder(model_name)
            print("Cross-encoder loaded.")
        except Exception as error:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}'. "
                f"Make sure sentence-transformers is installed. "
                f"Original error: {error}"
            )

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Reorder candidates by cross-encoder relevance score.

        Scores are added to each candidate dict under the 'rerank_score' key
        so callers can see both the original embedding relevance and the
        cross-encoder score. The list is returned sorted best-first.

        Args:
            query:      The user's question.
            candidates: List of result dicts from retrieve_top_k().

        Returns:
            The same list, reordered by cross-encoder score descending.
            Each dict gains a 'rerank_score' key (raw cross-encoder logit).
        """
        if not candidates:
            return []

        if not query or not query.strip():
            raise ValueError("Query must not be empty for reranking.")

        # Build (query, chunk_text) pairs — that is what cross-encoders expect
        pairs = [(query, candidate["text"]) for candidate in candidates]

        try:
            scores = self.model.predict(pairs)
        except Exception as error:
            raise RuntimeError(
                f"Cross-encoder scoring failed. Original error: {error}"
            )

        # Attach the score to each candidate in place
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = round(float(score), 4)

        # Sort by cross-encoder score, highest first
        reranked = sorted(candidates, key=lambda r: r["rerank_score"], reverse=True)
        return reranked


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def retrieve_and_rerank(
    query: str,
    collection_name: str = CHROMA_COLLECTION_NAME,
    top_k: int = 5,
    rerank: bool = True,
    embedding_model: Optional[EmbeddingModel] = None,
    vector_store: Optional[VectorStore] = None,
    reranker: Optional[CrossEncoderReranker] = None,
) -> List[Dict]:
    """
    Run the full two-stage retrieval pipeline and return ranked results.

    This is the function main.py and generation.py should call. It handles
    both stages and supports injecting pre-loaded model instances to avoid
    reloading models on every call (important for interactive use in main.py).

    Stage 1: Embed the query and retrieve top_k candidates from ChromaDB.
    Stage 2: Optionally rerank candidates with a cross-encoder for precision.

    Args:
        query:            The user's question.
        collection_name:  ChromaDB collection to search.
        top_k:            Number of candidates to retrieve in stage 1.
                          If reranking is on, all top_k are reranked and the
                          same top_k are returned (reordered, not trimmed).
        rerank:           If True (default), apply cross-encoder reranking.
                          Set to False for faster but less accurate results.
        embedding_model:  Optional pre-loaded EmbeddingModel.
        vector_store:     Optional pre-connected VectorStore.
        reranker:         Optional pre-loaded CrossEncoderReranker.

    Returns:
        List of result dicts sorted by relevance. When reranking is on, order
        is determined by cross-encoder score; otherwise by embedding relevance.
    """
    # Stage 1
    candidates = retrieve_top_k(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    if not rerank:
        return candidates

    # Stage 2
    _reranker = reranker or CrossEncoderReranker()
    return _reranker.rerank(query, candidates)


# ---------------------------------------------------------------------------
# Test — run with: python retrieval.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from chunking import recursive_chunking, attach_metadata

    # ── Build a richer test corpus so top_k > 1 is meaningful ────────────
    CORPUS = [
        {
            "text": (
                "Retrieval-Augmented Generation (RAG) combines a retrieval system "
                "with a language model. The retrieval step fetches relevant passages "
                "from a document store, which are then passed as context to the model."
            ),
            "label": "RAG definition",
        },
        {
            "text": (
                "Chunking is the process of splitting a document into smaller pieces "
                "before embedding. If chunks are too large the embedding loses focus; "
                "if they are too small a single chunk may not contain enough context."
            ),
            "label": "Chunking strategy",
        },
        {
            "text": (
                "Cosine similarity measures the angle between two vectors. It is "
                "preferred over Euclidean distance for text embeddings because it is "
                "insensitive to vector magnitude, focusing only on direction."
            ),
            "label": "Cosine similarity",
        },
        {
            "text": (
                "Cross-encoders read the query and the candidate passage together in "
                "one forward pass. This joint processing allows the model to score "
                "relevance more accurately than comparing independent embeddings."
            ),
            "label": "Cross-encoder explanation",
        },
        {
            "text": (
                "Overlap between consecutive chunks ensures that a sentence split "
                "across a chunk boundary appears in full in at least one chunk. "
                "Without overlap, retrieval may miss information at boundaries."
            ),
            "label": "Overlap explanation",
        },
        {
            "text": (
                "ChromaDB is an open-source vector database designed for AI "
                "applications. It stores embeddings alongside metadata and supports "
                "fast approximate nearest-neighbour search using HNSW indexing."
            ),
            "label": "ChromaDB description",
        },
        {
            "text": (
                "Hallucination in language models refers to the generation of "
                "confident-sounding but factually incorrect statements. RAG reduces "
                "hallucination by grounding answers in retrieved source documents."
            ),
            "label": "Hallucination and RAG",
        },
    ]

    TEST_COLLECTION = "test_retrieval_pipeline"
    QUERY = "How does a cross-encoder improve search accuracy compared to embeddings?"

    print("=" * 65)
    print("DocuMind retrieval.py — two-stage retrieval test")
    print("=" * 65)

    # ── Ingest test corpus ────────────────────────────────────────────────
    print("\nIngesting test corpus ...")
    model = EmbeddingModel()
    store = VectorStore()

    try:
        store.delete_collection(TEST_COLLECTION)
    except Exception:
        pass

    chunks = []
    for i, item in enumerate(CORPUS):
        chunks.append({
            "text": item["text"],
            "chunk_id": i,
            "source": "test_corpus.txt",
            "page_number": None,
            "method": "recursive",
        })
    chunks = attach_metadata(chunks, source_filename="test_corpus.txt")

    texts = [c["text"] for c in chunks]
    vectors = model.embed_texts(texts)
    stored = store.add_chunks(chunks, vectors, TEST_COLLECTION)
    print(f"Stored {stored} chunks in '{TEST_COLLECTION}'.")

    # ── Stage 1: Embedding retrieval only ─────────────────────────────────
    print(f"\nQuery: \"{QUERY}\"")
    print("\n" + "-" * 65)
    print("STAGE 1 — Embedding retrieval only (no reranking)")
    print("-" * 65)

    embedding_results = retrieve_and_rerank(
        query=QUERY,
        collection_name=TEST_COLLECTION,
        top_k=4,
        rerank=False,
        embedding_model=model,
        vector_store=store,
    )

    for i, r in enumerate(embedding_results):
        label = next(
            (item["label"] for item in CORPUS if item["text"] == r["text"]),
            "unknown"
        )
        print(f"  [{i+1}] relevance={r['relevance']:.4f}  [{label}]")
        print(f"       {r['text'][:90]}...")

    # ── Stage 2: With cross-encoder reranking ─────────────────────────────
    print("\n" + "-" * 65)
    print("STAGE 2 — After cross-encoder reranking")
    print("-" * 65)

    reranker = CrossEncoderReranker()
    reranked_results = retrieve_and_rerank(
        query=QUERY,
        collection_name=TEST_COLLECTION,
        top_k=4,
        rerank=True,
        embedding_model=model,
        vector_store=store,
        reranker=reranker,
    )

    for i, r in enumerate(reranked_results):
        label = next(
            (item["label"] for item in CORPUS if item["text"] == r["text"]),
            "unknown"
        )
        print(
            f"  [{i+1}] relevance={r['relevance']:.4f}  "
            f"rerank_score={r['rerank_score']:.4f}  [{label}]"
        )
        print(f"       {r['text'][:90]}...")

    # ── Side-by-side comparison ───────────────────────────────────────────
    print("\n" + "-" * 65)
    print("SIDE-BY-SIDE: rank shift after reranking")
    print("-" * 65)

    embed_order = [
        next(item["label"] for item in CORPUS if item["text"] == r["text"])
        for r in embedding_results
    ]
    rerank_order = [
        next(item["label"] for item in CORPUS if item["text"] == r["text"])
        for r in reranked_results
    ]

    print(f"  {'Rank':<6} {'Embedding only':<30} {'After reranking':<30}")
    print(f"  {'-'*5:<6} {'-'*28:<30} {'-'*28:<30}")
    for rank, (e, r) in enumerate(zip(embed_order, rerank_order), start=1):
        marker = "  " if e == r else "← moved"
        print(f"  {rank:<6} {e:<30} {r:<30} {marker}")

    # ── Clean up ──────────────────────────────────────────────────────────
    store.delete_collection(TEST_COLLECTION)
    print("\n✓ Test collection cleaned up.")

    print("\n" + "=" * 65)
    print(f"✓ PASS — retrieval and reranking completed without errors.")
    print("=" * 65)
