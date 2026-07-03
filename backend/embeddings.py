"""
embeddings.py — Text embedding and persistent vector storage for DocuMind.

Here is how this file fits into the pipeline:

  1. EmbeddingModel wraps sentence-transformers and turns text chunks into
     dense vectors (lists of floats). The model is loaded once at startup
     and reused for every call — loading it fresh each time would waste 2–3
     seconds per request.

  2. VectorStore wraps ChromaDB and handles persisting those vectors to disk
     so the data survives between runs. It also stores the chunk text and
     metadata (source, page number, chunk ID, chunking method) alongside each
     vector so that retrieval later returns everything needed for citations.

  3. process_and_store() is the single top-level function that main.py calls
     when a user uploads a file. It chains together:
       extract_text → chunk → attach_metadata → embed → store
"""

import os
import pathlib
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Import only the constants we need from config.
# We guard against the GROQ_API_KEY check crashing the import by reading the
# embedding/chroma constants ourselves if config fails to load.
try:
    from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
except OSError:
    # config.py raises OSError when GROQ_API_KEY is absent.  embeddings.py
    # doesn't need that key, so we fall back to the same defaults here.
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR = "./chroma_store"
    CHROMA_COLLECTION_NAME = "documind"
from ingestion import extract_text
from chunking import fixed_size_chunking, recursive_chunking, attach_metadata


# ---------------------------------------------------------------------------
# WHY a local embedding model instead of an API-based one?
#
#   - Cost: every chunk sent to an API costs money; a 300-page PDF can
#     produce thousands of chunks, making API-based embedding expensive at scale.
#   - Speed: a local model avoids a network round-trip for each batch,
#     which matters during ingestion of large documents.
#   - No external dependency: the app works offline and is not affected by
#     API outages, rate limits, or key rotation.
#
# all-MiniLM-L6-v2 is a well-regarded 80 MB model that produces 384-dimensional
# vectors. It punches well above its weight for semantic similarity tasks and
# runs comfortably on CPU.
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers for generating text embeddings.

    The model is loaded once when the instance is created and reused for every
    subsequent call. Loading it fresh on each call would add 2–3 seconds of
    latency and waste memory.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """
        Load the embedding model from local cache or download it on first use.

        The first run downloads ~80 MB from HuggingFace and caches it locally.
        Subsequent runs load from cache in under a second.
        """
        try:
            print(f"Loading embedding model '{model_name}' ...")
            self.model = SentenceTransformer(model_name)
            print("Embedding model loaded.")
        except Exception as error:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                f"Make sure sentence-transformers is installed and the model name is correct. "
                f"Original error: {error}"
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings and return a list of float vectors.

        Used during ingestion to embed all chunks from an uploaded document.
        Processes the whole list in one batch, which is faster than calling
        embed_query() in a loop.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of vectors, one per input string. Each vector is a list of floats.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            # convert numpy arrays to plain Python lists for ChromaDB compatibility
            return [vector.tolist() for vector in embeddings]
        except Exception as error:
            raise RuntimeError(
                f"Failed to generate embeddings for {len(texts)} texts. "
                f"Original error: {error}"
            )

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string and return one float vector.

        Used at question-answering time, not during ingestion. Kept as a
        separate method to make the call site in retrieval.py read clearly.

        Args:
            query: The user's search question.

        Returns:
            A single embedding vector as a list of floats.
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty.")

        try:
            vector = self.model.encode(query, show_progress_bar=False)
            return vector.tolist()
        except Exception as error:
            raise RuntimeError(
                f"Failed to embed query. Original error: {error}"
            )


# ---------------------------------------------------------------------------
# Vector storage
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Persistent ChromaDB wrapper for storing and managing chunk embeddings.

    Uses a persistent client so data written in one process survives to the
    next. All collections live under CHROMA_PERSIST_DIR on disk.
    """

    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR) -> None:
        """
        Create or reconnect to a persistent ChromaDB instance.

        Args:
            persist_directory: Local folder path where ChromaDB stores its data.
        """
        try:
            # Resolve to an absolute path so ChromaDB always writes to the same place
            # regardless of where the script is invoked from.
            abs_path = str(pathlib.Path(persist_directory).resolve())
            self.client = chromadb.PersistentClient(path=abs_path)
            print(f"ChromaDB connected. Storage directory: {abs_path}")
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at '{persist_directory}'. "
                f"Original error: {error}"
            )

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get an existing collection or create it if it does not exist yet.

        Using get_or_create_collection means this is always safe to call —
        no need to check whether the collection already exists.

        Args:
            collection_name: Name of the ChromaDB collection.

        Returns:
            A ChromaDB Collection object ready for reads and writes.
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                # cosine distance works well for sentence-transformer embeddings
                metadata={"hnsw:space": "cosine"},
            )
            return collection
        except Exception as error:
            raise RuntimeError(
                f"Failed to create or retrieve collection '{collection_name}'. "
                f"Original error: {error}"
            )

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        collection_name: str,
    ) -> int:
        """
        Store chunk text, embeddings, and metadata in ChromaDB.

        Duplicate detection: chunk IDs are unique per source file. If a
        chunk_id already exists in the collection it is overwritten via
        upsert so that re-uploading a file never creates duplicates.

        Metadata stored per chunk:
          - source: original file path (for citations)
          - page_number: page in source document, or -1 when unavailable
          - chunk_id: position of this chunk within the document
          - method: 'fixed_size' or 'recursive' (for evaluation comparisons)

        Args:
            chunks:          List of chunk dicts from chunking + attach_metadata.
            embeddings:      Parallel list of embedding vectors.
            collection_name: Name of the collection to write to.

        Returns:
            Number of chunks successfully stored.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same length."
            )

        if not chunks:
            return 0

        collection = self.create_collection(collection_name)

        # Build unique IDs that encode both the source file and position so that
        # chunks from different documents never collide in the same collection.
        source = chunks[0].get("source", "unknown")
        # Sanitise the source name: replace characters that confuse ChromaDB IDs
        safe_source = pathlib.Path(source).name.replace(" ", "_").replace(".", "_")

        ids = [f"{safe_source}_chunk_{chunk['chunk_id']}" for chunk in chunks]

        metadatas = [
            {
                "source": chunk.get("source", "unknown"),
                # ChromaDB metadata values must be str/int/float/bool — store
                # None as -1 so the field is always present and queryable.
                "page_number": chunk.get("page_number") if chunk.get("page_number") is not None else -1,
                "chunk_id": chunk.get("chunk_id", 0),
                "method": chunk.get("method", "unknown"),
            }
            for chunk in chunks
        ]

        documents = [chunk["text"] for chunk in chunks]

        try:
            # upsert = insert if new, overwrite if the ID already exists
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            return len(chunks)
        except Exception as error:
            raise RuntimeError(
                f"Failed to store chunks in collection '{collection_name}'. "
                f"Original error: {error}"
            )

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its data from ChromaDB.

        Useful for resetting state during testing or when a user wants to
        re-ingest a document from scratch. The collection will be recreated
        empty on the next call to create_collection().

        Args:
            collection_name: Name of the collection to delete.
        """
        try:
            self.client.delete_collection(name=collection_name)
            print(f"Collection '{collection_name}' deleted.")
        except Exception as error:
            raise RuntimeError(
                f"Failed to delete collection '{collection_name}'. "
                "It may not exist yet. Original error: {error}"
            )


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------

def process_and_store(
    file_path: str,
    chunking_method: str = "recursive",
    collection_name: str = CHROMA_COLLECTION_NAME,
    original_filename: Optional[str] = None,
) -> int:
    """
    Run the full ingestion pipeline for one document and persist it to ChromaDB.

    This is the single function main.py calls when a user uploads a file.
    It chains together every step:
      1. extract_text    — pull raw text (+ page numbers) from PDF or DOCX
      2. chunk           — split into overlapping segments
      3. attach_metadata — tag each chunk with source, page, chunk_id, method
      4. embed           — turn text into vectors via EmbeddingModel
      5. store           — upsert vectors + metadata into ChromaDB

    Args:
        file_path:          Absolute or relative path to the uploaded file.
        chunking_method:    'recursive' (default) or 'fixed_size'.
        collection_name:    ChromaDB collection to write into.
        original_filename:  Display name for citations (e.g. the user's
                            original upload name). Defaults to the basename
                            of file_path when not provided. Pass this from
                            main.py so temp file paths don't appear in
                            citations.

    Returns:
        Total number of chunks stored.

    Raises:
        ValueError:  If the file type is unsupported or chunking_method is unknown.
        RuntimeError: If any pipeline step fails (wrapped with context).
    """
    print(f"\n--- Starting ingestion for '{file_path}' ---")

    # ── Step 1: Extract text ──────────────────────────────────────────────
    print("Step 1/4: Extracting text ...")
    pages = extract_text(file_path)
    print(f"  Extracted {len(pages)} page(s) / section(s).")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────
    print(f"Step 2/4: Chunking with method='{chunking_method}' ...")

    # Build a word-position → page_number map so attach_metadata can tag
    # each chunk with its approximate source page.
    all_chunks: List[Dict] = []
    global_chunk_id = 0

    for page in pages:
        if chunking_method == "recursive":
            page_chunks = recursive_chunking(page["text"])
        elif chunking_method == "fixed_size":
            page_chunks = fixed_size_chunking(page["text"])
        else:
            raise ValueError(
                f"Unknown chunking method '{chunking_method}'. "
                "Choose 'recursive' or 'fixed_size'."
            )

        # Re-number chunk_ids globally (across pages) so IDs are unique per doc
        for chunk in page_chunks:
            chunk["chunk_id"] = global_chunk_id
            chunk["page_number"] = page["page_number"]  # carry page forward
            chunk["source"] = page["source"]
            global_chunk_id += 1

        all_chunks.extend(page_chunks)

    print(f"  Produced {len(all_chunks)} chunks.")

    if not all_chunks:
        print("  Warning: no chunks produced — file may be empty. Aborting.")
        return 0

    # ── Step 3: Attach metadata ───────────────────────────────────────────
    # Page number and source are already attached per-chunk above, so
    # attach_metadata here mostly acts as a consistency pass.
    print("Step 3/4: Attaching metadata ...")
    source_filename = original_filename or pathlib.Path(file_path).name
    # Build page_map from the chunk_id values we just assigned
    page_map = {
        chunk["chunk_id"]: chunk.get("page_number")
        for chunk in all_chunks
    }
    all_chunks = attach_metadata(all_chunks, source_filename, page_map)

    # ── Step 4: Embed ─────────────────────────────────────────────────────
    print("Step 4/4: Embedding chunks ...")
    embedding_model = EmbeddingModel()
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_model.embed_texts(texts)
    print(f"  Generated {len(embeddings)} embedding vectors.")

    # ── Step 5: Store ─────────────────────────────────────────────────────
    print("Storing in ChromaDB ...")
    vector_store = VectorStore()
    stored_count = vector_store.add_chunks(all_chunks, embeddings, collection_name)
    print(f"  Stored {stored_count} chunks in collection '{collection_name}'.")
    print(f"--- Ingestion complete for '{file_path}' ---\n")

    return stored_count


# ---------------------------------------------------------------------------
# End-to-end test — run with: python embeddings.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    SAMPLE_TEXT = """
    Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval
    system with a language model. Instead of relying solely on the model's training
    data, RAG first fetches relevant passages from a document store and passes them
    to the model as context.

    This approach has several advantages. The model can answer questions about
    documents it has never seen during training. Answers are grounded in retrieved
    text, which reduces hallucination. Citations can point directly to the source
    passages, making it easy for users to verify claims.

    The quality of a RAG system depends heavily on how documents are chunked.
    If chunks are too large, the embedding loses focus. If they are too small,
    a single chunk may not contain enough context to answer a question. The ideal
    chunk size balances specificity with completeness.

    Overlap between chunks ensures that sentences split across a boundary appear
    in full in at least one chunk. Without overlap, a question about a concept
    described across two consecutive chunks might match neither chunk well.
    """

    # Write sample text to a temporary .txt file so process_and_store() can
    # extract it via the normal ingestion path.
    # Because ingestion.py only handles .pdf and .docx, we test the components
    # directly here to keep the test self-contained.
    print("=" * 60)
    print("DocuMind embeddings.py — end-to-end pipeline test")
    print("=" * 60)

    TEST_COLLECTION = "test_embeddings_pipeline"

    # ── Chunk the sample text directly ───────────────────────────────────
    from chunking import recursive_chunking, attach_metadata

    chunks = recursive_chunking(SAMPLE_TEXT.strip())
    chunks = attach_metadata(chunks, source_filename="sample_test.txt")
    print(f"\nChunks produced: {len(chunks)}")

    # ── Embed ─────────────────────────────────────────────────────────────
    model = EmbeddingModel()
    texts = [c["text"] for c in chunks]
    vectors = model.embed_texts(texts)
    print(f"Vectors generated: {len(vectors)}, dimension: {len(vectors[0])}")

    # ── Store ─────────────────────────────────────────────────────────────
    store = VectorStore()

    # Reset the test collection before each run so the test is repeatable
    try:
        store.delete_collection(TEST_COLLECTION)
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    stored = store.add_chunks(chunks, vectors, TEST_COLLECTION)
    print(f"Chunks stored in ChromaDB: {stored}")

    # ── Verify via a direct ChromaDB query ───────────────────────────────
    collection = store.create_collection(TEST_COLLECTION)
    actual_count = collection.count()
    print(f"ChromaDB reports {actual_count} entries in '{TEST_COLLECTION}'")

    # ── Sanity-check embed_query ──────────────────────────────────────────
    query_vector = model.embed_query("What is retrieval-augmented generation?")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    print("\nTop-2 retrieved chunks for query 'What is retrieval-augmented generation?':")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n  [Result {i+1}] distance={dist:.4f}, source={meta['source']}")
        print(f"  {doc[:120]}...")

    # ── Clean up test collection ──────────────────────────────────────────
    store.delete_collection(TEST_COLLECTION)
    print("\n✓ Test collection cleaned up.")

    print("\n" + "=" * 60)
    if stored == len(chunks) == actual_count:
        print(f"✓ PASS — {stored} chunks stored and confirmed in ChromaDB.")
    else:
        print(f"✗ FAIL — mismatch: produced={len(chunks)}, stored={stored}, confirmed={actual_count}")
    print("=" * 60)
