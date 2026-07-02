"""
chunking.py — Text chunking strategies for DocuMind.

Before text can be embedded and stored, it needs to be split into smaller
pieces (chunks). This module provides two strategies so we can compare
retrieval quality between them:

  1. fixed_size_chunking  — simple, predictable, fast. Good baseline.
  2. recursive_chunking   — respects natural language boundaries. Better
                            for preserving meaning in retrieved passages.

Both return the same list-of-dicts format so the rest of the pipeline
(embeddings, retrieval) doesn't need to care which strategy was used.
"""

import re
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Strategy 1 — Fixed-size chunking
# ---------------------------------------------------------------------------

def split_words(text: str) -> List[str]:
    """
    Split a string into a list of individual words.

    Splitting by whitespace is intentionally simple here — the fixed-size
    strategy works on word counts, not character counts, so we just need
    the word boundaries.
    """
    return text.split()


def build_fixed_chunk(words: List[str], start_index: int, chunk_size: int) -> str:
    """
    Slice a window of words from the full word list and return it as a string.

    Kept separate so the slicing logic is easy to read and test on its own.
    """
    end_index = start_index + chunk_size
    return " ".join(words[start_index:end_index])


def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split text into fixed-size word windows with overlapping boundaries.

    WHY overlap matters: if a chunk ends mid-thought, the next chunk starts
    'overlap' words back so that context crossing the boundary is captured
    in both chunks. Without overlap, a sentence split across two chunks
    would be partially missing from each, hurting retrieval accuracy.

    WHY this strategy exists: it's simple, predictable, and fast. Every chunk
    is roughly the same size, which keeps embedding comparisons fair. It's a
    good baseline to measure the recursive strategy against.

    Returns a list of dicts with keys: 'text', 'chunk_id', 'method'.
    """
    if not text or not text.strip():
        return []

    words = split_words(text.strip())

    # If the entire text is shorter than one chunk, return it as-is
    if len(words) <= chunk_size:
        return [{
            "text": " ".join(words),
            "chunk_id": 0,
            "method": "fixed_size",
        }]

    chunks = []
    chunk_id = 0
    position = 0

    # Step forward by (chunk_size - overlap) each iteration so consecutive
    # chunks share 'overlap' words at their boundary
    step = chunk_size - overlap

    while position < len(words):
        chunk_text = build_fixed_chunk(words, position, chunk_size)

        chunks.append({
            "text": chunk_text,
            "chunk_id": chunk_id,
            "method": "fixed_size",
        })

        chunk_id += 1
        position += step

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — Recursive chunking
# ---------------------------------------------------------------------------

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text on double newlines (blank lines between paragraphs).

    Double newlines are the most reliable signal of a paragraph break in
    plain text and in text extracted from PDFs/DOCX files.
    Returns only non-empty paragraphs.
    """
    raw_paragraphs = re.split(r"\n{2,}", text)
    return [paragraph.strip() for paragraph in raw_paragraphs if paragraph.strip()]


def split_into_sentences(text: str) -> List[str]:
    """
    Split a paragraph into individual sentences using punctuation markers.

    We look for '.', '!', or '?' followed by whitespace and an uppercase letter,
    which covers the vast majority of English sentence endings without needing
    a full NLP library. This avoids splitting on abbreviations like 'Dr.' or
    decimal numbers like '3.14'.
    Returns only non-empty sentences.
    """
    # The lookahead (?=[A-Z]) keeps the capital letter in the next sentence
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [sentence.strip() for sentence in raw_sentences if sentence.strip()]


def count_words(text: str) -> int:
    """
    Return the number of whitespace-separated words in a string.

    Used throughout recursive chunking to decide whether a piece of text
    needs further splitting.
    """
    return len(text.split())


def chunk_oversized_sentence(
    sentence: str,
    max_chunk_size: int,
    starting_chunk_id: int,
) -> List[Dict]:
    """
    Fall back to fixed-size splitting for a single sentence that exceeds max_chunk_size.

    This is the last resort in the recursive strategy. It only triggers when
    one sentence is longer than the chunk size, which is rare but possible in
    documents with very long run-on sentences or code blocks embedded in text.
    The overlap is set to 0 here because splitting a single sentence further
    makes overlap semantically meaningless.
    """
    words = sentence.split()
    chunks = []
    chunk_id = starting_chunk_id
    position = 0

    while position < len(words):
        chunk_text = " ".join(words[position : position + max_chunk_size])
        chunks.append({
            "text": chunk_text,
            "chunk_id": chunk_id,
            "method": "recursive",
        })
        chunk_id += 1
        position += max_chunk_size

    return chunks


def merge_sentences_into_chunks(
    sentences: List[str],
    max_chunk_size: int,
    starting_chunk_id: int,
) -> List[Dict]:
    """
    Greedily pack sentences into chunks without exceeding max_chunk_size words.

    WHY greedy packing: we want to keep as many related sentences together as
    possible. We keep adding sentences to the current chunk until the next one
    would push it over the limit, then we seal the chunk and start a new one.
    This keeps semantically related sentences in the same chunk more often than
    splitting by word count alone would.

    Falls back to chunk_oversized_sentence() for any single sentence that is
    already longer than max_chunk_size on its own.
    """
    chunks = []
    chunk_id = starting_chunk_id
    current_sentences: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence)

        # A single sentence that is already too long gets its own split pass
        if sentence_word_count > max_chunk_size:
            # Flush whatever we have accumulated so far
            if current_sentences:
                chunks.append({
                    "text": " ".join(current_sentences),
                    "chunk_id": chunk_id,
                    "method": "recursive",
                })
                chunk_id += 1
                current_sentences = []
                current_word_count = 0

            oversized_chunks = chunk_oversized_sentence(sentence, max_chunk_size, chunk_id)
            chunks.extend(oversized_chunks)
            chunk_id += len(oversized_chunks)
            continue

        # If adding this sentence would exceed the limit, seal the current chunk first
        if current_word_count + sentence_word_count > max_chunk_size and current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "chunk_id": chunk_id,
                "method": "recursive",
            })
            chunk_id += 1
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    # Don't forget the last accumulated group of sentences
    if current_sentences:
        chunks.append({
            "text": " ".join(current_sentences),
            "chunk_id": chunk_id,
            "method": "recursive",
        })

    return chunks


def recursive_chunking(text: str, max_chunk_size: int = 500) -> List[Dict]:
    """
    Split text by respecting natural language boundaries in order of preference.

    WHY this strategy exists: fixed-size chunking doesn't care about sentence
    or paragraph boundaries — it will cheerfully cut a sentence in half. That
    hurts retrieval because neither chunk contains a complete thought. This
    strategy tries to keep meaning intact by splitting at the largest natural
    boundary that still fits within max_chunk_size.

    The hierarchy is:
      1. Split on paragraph breaks (double newlines) — keeps topics together.
      2. If a paragraph is still too big, split on sentence boundaries.
      3. If a single sentence is too big, fall back to word-count splitting.

    Returns a list of dicts with keys: 'text', 'chunk_id', 'method'.
    """
    if not text or not text.strip():
        return []

    # If the whole text fits in one chunk, no splitting needed
    if count_words(text.strip()) <= max_chunk_size:
        return [{
            "text": text.strip(),
            "chunk_id": 0,
            "method": "recursive",
        }]

    paragraphs = split_into_paragraphs(text)

    # If there are no paragraph breaks, treat the whole text as one paragraph
    if not paragraphs:
        paragraphs = [text.strip()]

    all_chunks: List[Dict] = []
    chunk_id = 0

    for paragraph in paragraphs:
        paragraph_word_count = count_words(paragraph)

        if paragraph_word_count <= max_chunk_size:
            # Paragraph fits in one chunk — keep it whole, don't split mid-thought
            all_chunks.append({
                "text": paragraph,
                "chunk_id": chunk_id,
                "method": "recursive",
            })
            chunk_id += 1
        else:
            # Paragraph is too big — split on sentence boundaries
            sentences = split_into_sentences(paragraph)

            # If sentence splitting found nothing useful, treat it as one block
            if not sentences:
                sentences = [paragraph]

            sentence_chunks = merge_sentences_into_chunks(sentences, max_chunk_size, chunk_id)
            all_chunks.extend(sentence_chunks)
            chunk_id += len(sentence_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Metadata tagging
# ---------------------------------------------------------------------------

def attach_metadata(
    chunks: List[Dict],
    source_filename: str,
    page_map: Optional[Dict[int, int]] = None,
) -> List[Dict]:
    """
    Add 'source' and 'page_number' fields to every chunk dict in place.

    WHY this is a separate step: chunking produces text chunks from a flat
    string, but it doesn't know which file the text came from or what page
    it was on. Those fields are attached here, after chunking, using the
    page_map produced during ingestion.

    page_map is an optional dict mapping chunk_id → page_number. When
    available (PDF documents), each chunk gets the page it most likely came
    from. When not available (DOCX documents), page_number is set to None.

    Modifies chunks in place AND returns them so the caller can chain calls.
    """
    if page_map is None:
        page_map = {}

    for chunk in chunks:
        chunk["source"] = source_filename
        # Fall back to None cleanly if no page mapping exists for this chunk
        chunk["page_number"] = page_map.get(chunk["chunk_id"], None)

    return chunks


# ---------------------------------------------------------------------------
# Quick visual comparison — run with: python chunking.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_TEXT = """
    Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval
    system with a language model. Instead of relying solely on the model's training
    data, RAG first fetches relevant passages from a document store, then passes
    those passages to the model as context.

    This approach has several advantages. The model can answer questions about
    documents it has never seen during training. Answers are grounded in retrieved
    text, which reduces hallucination. Citations can point directly to the source
    passages, making it easy for users to verify claims.

    The quality of a RAG system depends heavily on how documents are chunked before
    embedding. If chunks are too large, the embedding loses focus. If they are too
    small, a single chunk may not contain enough context to answer a question. The
    ideal chunk size balances specificity with completeness.

    Overlap between chunks ensures that sentences split across a boundary appear
    in full in at least one chunk. Without overlap, a question about a concept
    described across two consecutive chunks might match neither chunk well.
    """

    print("=" * 70)
    print("STRATEGY 1 — Fixed-size chunking (chunk_size=40, overlap=8 words)")
    print("=" * 70)
    fixed_chunks = fixed_size_chunking(SAMPLE_TEXT, chunk_size=40, overlap=8)
    for chunk in fixed_chunks:
        print(f"\n[Chunk {chunk['chunk_id']}]")
        print(chunk["text"])

    print("\n" + "=" * 70)
    print("STRATEGY 2 — Recursive chunking (max_chunk_size=40 words)")
    print("=" * 70)
    recursive_chunks = recursive_chunking(SAMPLE_TEXT, max_chunk_size=40)
    for chunk in recursive_chunks:
        print(f"\n[Chunk {chunk['chunk_id']}]")
        print(chunk["text"])

    print("\n" + "=" * 70)
    print(f"Fixed-size produced {len(fixed_chunks)} chunks.")
    print(f"Recursive produced  {len(recursive_chunks)} chunks.")
    print("=" * 70)
