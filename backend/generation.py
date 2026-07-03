"""
generation.py — LLM answer generation and faithfulness checking for DocuMind.

This file is the final stage of the RAG pipeline. It takes the retrieved
chunks from retrieval.py, builds a grounded prompt, calls the Groq LLM, and
then runs a separate faithfulness check to flag potential hallucinations.

Flow:

  1. generate_answer(query, retrieved_chunks, groq_client)
       Builds a prompt with retrieved chunks as labeled context, instructs
       the LLM to answer only from that context and cite sources, then
       returns the answer text and a list of which sources were cited.

  2. check_faithfulness(query, answer, retrieved_chunks, groq_client)
       Makes a SEPARATE Groq call with a fresh context that contains only
       the context passages and the generated answer. Asks the LLM to judge
       whether the answer is fully supported by the context, returning a
       score from 1–10 and a one-sentence explanation.

       WHY a separate call:
         Asking the same LLM call that produced the answer to also grade
         its own answer is unreliable — the model is biased toward justifying
         what it just said, and its grading is anchored to the same reasoning
         chain. A fresh, isolated call with a neutral framing produces more
         honest faithfulness judgments. Think of it as getting a second
         opinion from a colleague who hasn't seen your draft yet.

  3. answer_query(query, collection_name, groq_client, ...)
       The single public entry point. Chains retrieval → generation →
       faithfulness check and returns one clean result dict that main.py
       and the frontend can consume directly.

  Model instances (embedding, vector store, reranker) are loaded ONCE by the
  caller and passed in. Never load them inside these functions — each load
  takes 2–4 seconds and would make every query painfully slow.
"""

import re
import time
from typing import List, Dict, Optional

from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

try:
    from config import GROQ_API_KEY, LLM_MODEL, CHROMA_COLLECTION_NAME, TOP_K
except OSError:
    # Fallback if config can't load (e.g. GROQ_API_KEY not yet set)
    GROQ_API_KEY = None
    LLM_MODEL = "llama-3.1-8b-instant"
    CHROMA_COLLECTION_NAME = "documind"
    TOP_K = 5

from retrieval import retrieve_and_rerank, CrossEncoderReranker
from embeddings import EmbeddingModel, VectorStore


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_context_block(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a clearly labeled context block for the prompt.

    Each chunk gets a header showing its source file, page number, and its
    position in the ranked list. Clear labeling makes it easy for the LLM to
    cite which source each claim comes from, and makes citations verifiable
    by the user.

    Args:
        chunks: List of result dicts from retrieve_and_rerank().

    Returns:
        A formatted multi-line string ready to embed in a prompt.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page_number", -1)
        page_label = f"page {page}" if page and page != -1 else "page unknown"
        lines.append(f"[Source {i}: {source}, {page_label}]")
        lines.append(chunk["text"].strip())
        lines.append("")  # blank line between chunks for readability
    return "\n".join(lines).strip()


def build_answer_prompt(query: str, context_block: str) -> str:
    """
    Build the system + user prompt for the answer-generation call.

    The prompt is designed around three constraints:
      1. Answer ONLY from the provided context — never from prior training.
      2. Cite the source label (e.g. [Source 2]) for every factual claim.
      3. Explicitly admit when the context is insufficient, rather than guessing.

    These constraints reduce hallucination and make answers auditable.

    Args:
        query:         The user's question.
        context_block: Formatted context string from build_context_block().

    Returns:
        A tuple (system_prompt, user_prompt) ready for the Groq chat API.
    """
    system_prompt = (
        "You are a precise document assistant. "
        "Answer questions strictly using the provided context passages. "
        "Do not use any knowledge from your training data. "
        "For every factual claim you make, cite the source label in brackets, "
        "for example: [Source 1] or [Source 3]. "
        "If the provided context does not contain enough information to answer "
        "the question, respond with exactly: "
        "'I don't have enough information in the provided documents to answer this.'"
    )

    user_prompt = (
        f"Context passages:\n\n"
        f"{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite sources for each claim):"
    )

    return system_prompt, user_prompt


def build_faithfulness_prompt(
    query: str, answer: str, context_block: str
) -> tuple:
    """
    Build the prompt for the separate faithfulness-check call.

    The prompt frames the LLM as an impartial judge who only reads the context
    and the answer — it has no memory of the generation call. This isolation
    is what makes the judgment more reliable than self-grading.

    The LLM is asked to respond in a strict format so the score is easy to
    parse reliably:
        Score: <integer 1-10>
        Explanation: <one sentence>

    Args:
        query:         The original user question.
        answer:        The answer produced by generate_answer().
        context_block: The same context that was given to the answer call.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are an impartial judge evaluating whether an AI answer is "
        "fully supported by the provided context passages. "
        "Score faithfulness from 1 (completely unsupported or fabricated) "
        "to 10 (every claim is directly traceable to the context). "
        "Respond in EXACTLY this format — nothing else:\n"
        "Score: <integer>\n"
        "Explanation: <one sentence>"
    )

    user_prompt = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f"Rate the faithfulness of this answer to the context passages."
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Source citation parsing
# ---------------------------------------------------------------------------

def extract_cited_sources(answer: str, chunks: List[Dict]) -> List[Dict]:
    """
    Parse [Source N] citations from the answer and return the cited chunks.

    This gives downstream consumers (main.py, the frontend) a clean list of
    exactly which documents were cited, without having to parse the answer
    text themselves.

    Args:
        answer: The LLM's answer text.
        chunks: The retrieved chunks passed to the LLM (1-indexed in the prompt).

    Returns:
        List of chunk dicts that were cited. Deduplicated and ordered by
        first appearance in the answer. Returns all chunks if the LLM cited
        none explicitly (defensive fallback).
    """
    # Match [Source 1], [Source 12], etc. — case-insensitive
    cited_indices = re.findall(r'\[source\s+(\d+)\]', answer, re.IGNORECASE)

    seen = set()
    cited_chunks = []
    for idx_str in cited_indices:
        idx = int(idx_str) - 1  # convert 1-based citation to 0-based list index
        if 0 <= idx < len(chunks) and idx not in seen:
            cited_chunks.append(chunks[idx])
            seen.add(idx)

    # If the LLM answered but cited nothing, return all chunks as a fallback
    # so the caller always has source info to display
    if not cited_chunks and chunks:
        return chunks

    return cited_chunks


# ---------------------------------------------------------------------------
# Groq API call helper
# ---------------------------------------------------------------------------

def call_groq(
    groq_client: Groq,
    system_prompt: str,
    user_prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    purpose: str = "request",
) -> str:
    """
    Make one Groq chat completion call and return the response text.

    Kept as a separate helper so both generate_answer() and check_faithfulness()
    share the same error handling and retry logic without duplicating it.

    Temperature is set low (0.1) by default — we want deterministic, factual
    answers, not creative variation. The faithfulness call uses 0.0 for even
    more consistent scoring.

    Handles:
      - RateLimitError: waits 20 seconds and retries once before raising.
      - APIStatusError: wraps with context about which call failed.
      - APIConnectionError: wraps with a network troubleshooting hint.

    Args:
        groq_client:   Initialised Groq() client instance.
        system_prompt: System role message.
        user_prompt:   User role message.
        model:         Groq model identifier.
        temperature:   Sampling temperature (lower = more deterministic).
        max_tokens:    Maximum tokens in the response.
        purpose:       Human-readable label for error messages ('answer' or
                       'faithfulness check').

    Returns:
        The raw response text string from the LLM.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    for attempt in (1, 2):  # one retry on rate-limit
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            if attempt == 1:
                print(f"  Rate limit hit for {purpose}. Waiting 20 seconds ...")
                time.sleep(20)
                continue
            raise RuntimeError(
                f"Groq rate limit exceeded for {purpose} after retry. "
                "Wait a minute and try again, or reduce query frequency."
            )

        except APIConnectionError as error:
            raise RuntimeError(
                f"Could not reach the Groq API for {purpose}. "
                "Check your internet connection. "
                f"Original error: {error}"
            )

        except APIStatusError as error:
            raise RuntimeError(
                f"Groq API returned an error for {purpose} "
                f"(HTTP {error.status_code}). "
                f"Original error: {error.message}"
            )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    retrieved_chunks: List[Dict],
    groq_client: Groq,
    model: str = LLM_MODEL,
) -> Dict:
    """
    Build a grounded prompt from retrieved chunks and call the LLM for an answer.

    The LLM is constrained to answer only from the provided context and must
    cite which source each claim comes from. If the context is insufficient,
    the LLM is instructed to say so rather than guess.

    Args:
        query:            The user's question.
        retrieved_chunks: List of chunk dicts from retrieve_and_rerank().
        groq_client:      Initialised Groq client (loaded once by the caller).
        model:            Groq model to use for generation.

    Returns:
        Dict with:
          - answer:        The LLM's response text.
          - sources_cited: List of chunk dicts that were cited in the answer.
    """
    if not retrieved_chunks:
        return {
            "answer": "I don't have enough information in the provided documents to answer this.",
            "sources_cited": [],
        }

    context_block = build_context_block(retrieved_chunks)
    system_prompt, user_prompt = build_answer_prompt(query, context_block)

    answer_text = call_groq(
        groq_client=groq_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.1,
        max_tokens=1024,
        purpose="answer generation",
    )

    cited_chunks = extract_cited_sources(answer_text, retrieved_chunks)

    return {
        "answer": answer_text,
        "sources_cited": cited_chunks,
    }


def check_faithfulness(
    query: str,
    answer: str,
    retrieved_chunks: List[Dict],
    groq_client: Groq,
    model: str = LLM_MODEL,
) -> Dict:
    """
    Score how well the generated answer is supported by the retrieved context.

    Makes a SEPARATE Groq call with only the context and the answer — no
    memory of the generation call. This isolation is important:

    WHY a separate call rather than self-grading:
      If you ask the same model, in the same context window, to grade its own
      answer, it tends to justify what it just said — anchoring bias makes it
      score its own output higher than an impartial judge would. A fresh call
      with a neutral framing ("you are an impartial judge") produces more
      honest and calibrated faithfulness scores.

    The response is parsed for a "Score: N" line. If parsing fails, the score
    defaults to -1 and the raw text is returned as the explanation, so the
    caller is never left with a hard crash over a formatting hiccup.

    Args:
        query:            The original user question.
        answer:           The answer produced by generate_answer().
        retrieved_chunks: The chunks that were given as context.
        groq_client:      Initialised Groq client.
        model:            Groq model to use for the faithfulness check.

    Returns:
        Dict with:
          - faithfulness_score: int from 1 (hallucinated) to 10 (fully grounded).
                                -1 if the response could not be parsed.
          - faithfulness_explanation: one-sentence rationale from the LLM.
    """
    if not retrieved_chunks or not answer:
        return {
            "faithfulness_score": -1,
            "faithfulness_explanation": "Could not check faithfulness — missing answer or context.",
        }

    context_block = build_context_block(retrieved_chunks)
    system_prompt, user_prompt = build_faithfulness_prompt(query, answer, context_block)

    raw_response = call_groq(
        groq_client=groq_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=0.0,    # zero temperature for consistent, reproducible scoring
        max_tokens=128,     # the response is short by design — score + one sentence
        purpose="faithfulness check",
    )

    return parse_faithfulness_response(raw_response)


def parse_faithfulness_response(raw_response: str) -> Dict:
    """
    Parse the LLM's faithfulness-check response into a structured dict.

    Expected format:
        Score: 8
        Explanation: The answer accurately reflects the retrieved context.

    If parsing fails, the score is set to -1 and the full raw text is stored
    as the explanation so the caller can debug what the model returned.

    Args:
        raw_response: Raw text from the faithfulness-check Groq call.

    Returns:
        Dict with 'faithfulness_score' (int) and 'faithfulness_explanation' (str).
    """
    score = -1
    explanation = raw_response.strip()

    score_match = re.search(r'score[:\s]+(\d+)', raw_response, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        # Clamp to valid range in case the LLM drifts outside 1–10
        score = max(1, min(10, score))

    explanation_match = re.search(
        r'explanation[:\s]+(.+)', raw_response, re.IGNORECASE | re.DOTALL
    )
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return {
        "faithfulness_score": score,
        "faithfulness_explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Combined pipeline entry point
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    collection_name: str = CHROMA_COLLECTION_NAME,
    groq_client: Optional[Groq] = None,
    top_k: int = TOP_K,
    rerank: bool = True,
    embedding_model: Optional[EmbeddingModel] = None,
    vector_store: Optional[VectorStore] = None,
    reranker: Optional[CrossEncoderReranker] = None,
    model: str = LLM_MODEL,
) -> Dict:
    """
    Run the full RAG pipeline end-to-end and return a structured result.

    This is the single function main.py calls for each user question.
    It chains: retrieve_and_rerank → generate_answer → check_faithfulness.

    All model instances should be passed in pre-loaded. If omitted, they are
    created fresh here — convenient for testing, but slow for production use
    where the same models serve many queries in a session.

    Args:
        query:            The user's question.
        collection_name:  ChromaDB collection to retrieve from.
        groq_client:      Initialised Groq client. Created from GROQ_API_KEY
                          env variable if not provided.
        top_k:            Number of chunks to retrieve and rerank.
        rerank:           Whether to apply cross-encoder reranking.
        embedding_model:  Pre-loaded EmbeddingModel (pass in for performance).
        vector_store:     Pre-connected VectorStore (pass in for performance).
        reranker:         Pre-loaded CrossEncoderReranker (pass in for performance).
        model:            Groq model for both answer and faithfulness calls.

    Returns:
        Dict with:
          - answer:                    The LLM's grounded answer text.
          - sources_cited:             Chunk dicts that were cited in the answer.
          - faithfulness_score:        1–10 score from the faithfulness check.
          - faithfulness_explanation:  One-sentence rationale from the judge call.
          - retrieved_chunks:          All retrieved chunks (for transparency/debug).
    """
    # Initialise Groq client if not injected
    _client = groq_client or Groq(api_key=GROQ_API_KEY)

    # Stage 1+2: Retrieve and rerank
    retrieved_chunks = retrieve_and_rerank(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        rerank=rerank,
        embedding_model=embedding_model,
        vector_store=vector_store,
        reranker=reranker,
    )

    # Stage 3: Generate grounded answer
    generation_result = generate_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
        groq_client=_client,
        model=model,
    )

    # Stage 4: Check faithfulness (separate call — see check_faithfulness docstring)
    faithfulness_result = check_faithfulness(
        query=query,
        answer=generation_result["answer"],
        retrieved_chunks=retrieved_chunks,
        groq_client=_client,
        model=model,
    )

    return {
        "answer": generation_result["answer"],
        "sources_cited": generation_result["sources_cited"],
        "faithfulness_score": faithfulness_result["faithfulness_score"],
        "faithfulness_explanation": faithfulness_result["faithfulness_explanation"],
        "retrieved_chunks": retrieved_chunks,
    }


# ---------------------------------------------------------------------------
# End-to-end test — run with: python generation.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pathlib
    from dotenv import load_dotenv
    from chunking import attach_metadata

    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file and re-run.")
        raise SystemExit(1)

    # ── Build the same test corpus used in retrieval.py ───────────────────
    CORPUS = [
        {
            "text": (
                "Retrieval-Augmented Generation (RAG) combines a retrieval system "
                "with a language model. The retrieval step fetches relevant passages "
                "from a document store, which are then passed as context to the model."
            ),
        },
        {
            "text": (
                "Chunking is the process of splitting a document into smaller pieces "
                "before embedding. If chunks are too large the embedding loses focus; "
                "if they are too small a single chunk may not contain enough context."
            ),
        },
        {
            "text": (
                "Cross-encoders read the query and the candidate passage together in "
                "one forward pass. This joint processing allows the model to score "
                "relevance more accurately than comparing independent embeddings."
            ),
        },
        {
            "text": (
                "Hallucination in language models refers to the generation of "
                "confident-sounding but factually incorrect statements. RAG reduces "
                "hallucination by grounding answers in retrieved source documents."
            ),
        },
        {
            "text": (
                "ChromaDB is an open-source vector database designed for AI "
                "applications. It stores embeddings alongside metadata and supports "
                "fast approximate nearest-neighbour search using HNSW indexing."
            ),
        },
        {
            "text": (
                "Cosine similarity measures the angle between two vectors. It is "
                "preferred over Euclidean distance for text embeddings because it is "
                "insensitive to vector magnitude, focusing only on direction."
            ),
        },
        {
            "text": (
                "Overlap between consecutive chunks ensures that a sentence split "
                "across a chunk boundary appears in full in at least one chunk. "
                "Without overlap, retrieval may miss information at boundaries."
            ),
        },
    ]

    TEST_COLLECTION = "test_generation_pipeline"
    QUERY = "What is RAG and how does it reduce hallucination?"

    print("=" * 65)
    print("DocuMind generation.py — full pipeline test")
    print("=" * 65)

    # ── Load models once ──────────────────────────────────────────────────
    print("\nLoading models (loaded once, reused for all calls) ...")
    emb_model = EmbeddingModel()
    vec_store = VectorStore()
    reranker  = CrossEncoderReranker()
    groq_client = Groq(api_key=api_key)
    print("All models loaded.")

    # ── Ingest corpus ─────────────────────────────────────────────────────
    print(f"\nIngesting {len(CORPUS)} test chunks ...")
    try:
        vec_store.delete_collection(TEST_COLLECTION)
    except Exception:
        pass

    chunks = []
    for i, item in enumerate(CORPUS):
        chunks.append({
            "text": item["text"],
            "chunk_id": i,
            "source": "rag_overview.txt",
            "page_number": i + 1,
            "method": "recursive",
        })
    chunks = attach_metadata(chunks, source_filename="rag_overview.txt")
    texts   = [c["text"] for c in chunks]
    vectors = emb_model.embed_texts(texts)
    stored  = vec_store.add_chunks(chunks, vectors, TEST_COLLECTION)
    print(f"Stored {stored} chunks.")

    # ── Run full pipeline ─────────────────────────────────────────────────
    print(f"\nRunning full pipeline for query:")
    print(f"  \"{QUERY}\"")
    print()

    result = answer_query(
        query=QUERY,
        collection_name=TEST_COLLECTION,
        groq_client=groq_client,
        top_k=4,
        rerank=True,
        embedding_model=emb_model,
        vector_store=vec_store,
        reranker=reranker,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print("=" * 65)
    print("ANSWER")
    print("=" * 65)
    print(result["answer"])

    print("\n" + "=" * 65)
    print("SOURCES CITED")
    print("=" * 65)
    if result["sources_cited"]:
        for i, src in enumerate(result["sources_cited"], start=1):
            page = src.get("page_number", -1)
            page_label = f"p.{page}" if page and page != -1 else "page unknown"
            print(f"  [{i}] {src['source']} — {page_label}")
            print(f"       {src['text'][:100]}...")
    else:
        print("  (none cited explicitly)")

    print("\n" + "=" * 65)
    print("FAITHFULNESS CHECK")
    print("=" * 65)
    score = result["faithfulness_score"]
    print(f"  Score:       {score}/10")
    print(f"  Explanation: {result['faithfulness_explanation']}")

    score_label = (
        "✓ High confidence" if score >= 8
        else "⚠ Moderate — review sources" if score >= 5
        else "✗ Low — likely hallucination"
    ) if score != -1 else "? Parse error"
    print(f"  Assessment:  {score_label}")

    print("\n" + "=" * 65)
    print("RETRIEVED CHUNKS (debug)")
    print("=" * 65)
    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        relevance = chunk.get("relevance", "?")
        rerank_score = chunk.get("rerank_score", "n/a")
        print(f"  [Chunk {i}] relevance={relevance}  rerank={rerank_score}")
        print(f"            {chunk['text'][:85]}...")

    # ── Clean up ──────────────────────────────────────────────────────────
    vec_store.delete_collection(TEST_COLLECTION)
    print("\n✓ Test collection cleaned up.")
    print("\n✓ PASS — full generation pipeline completed.")
