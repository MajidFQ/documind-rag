# DocuMind

DocuMind is a Retrieval-Augmented Generation (RAG) system for document question-answering. Upload a PDF or DOCX, ask questions in plain English, and get answers grounded in the document's actual content — with source citations and a separate hallucination-check score on every response. It runs entirely on local infrastructure except for the LLM call, which goes to the Groq API.

---

## Architecture

```
                        INGESTION PIPELINE
                        ──────────────────
  Document (PDF/DOCX)
        │
        ▼
  Ingestion          extract_text()        pypdf / python-docx
  (ingestion.py)     → pages with text
        │               + page numbers
        ▼
  Chunking           fixed_size_chunking() or recursive_chunking()
  (chunking.py)      → list of text chunks
        │               with chunk_id, method
        ▼
  Metadata tagging   attach_metadata()
                     → source filename, page number added to each chunk
        │
        ▼
  Embedding          EmbeddingModel.embed_texts()    all-MiniLM-L6-v2
  (embeddings.py)    → 384-dim vectors, one per chunk
        │
        ▼
  Vector Store       VectorStore.add_chunks()        ChromaDB (cosine)
  (embeddings.py)    → persisted to ./chroma_store
        │
        ▼
     [on disk — survives between runs]


                        QUERY PIPELINE
                        ──────────────
  User question (string)
        │
        ▼
  Embed query        EmbeddingModel.embed_query()
        │
        ▼
  Retrieval          VectorStore → ChromaDB ANN search
  (retrieval.py)     → top-k chunks by cosine similarity
        │
        ▼
  Reranking          CrossEncoderReranker.rerank()   ms-marco-MiniLM-L-6-v2
  (retrieval.py)     → chunks reordered by joint query+chunk score
        │
        ▼
  Generation         generate_answer()               Groq / llama-3.1-8b-instant
  (generation.py)    → answer grounded in context, with [Source N] citations
        │
        ▼
  Faithfulness check check_faithfulness()            Groq (separate call)
  (generation.py)    → score 1–10 + one-sentence explanation
        │
        ▼
  Response           { answer, sources_cited, faithfulness_score,
                       faithfulness_explanation, retrieved_chunks }
```

---

## Key design decisions

### Two chunking strategies

Two strategies are included so retrieval quality can be compared:

- **Fixed-size** (`fixed_size_chunking`) splits on word count with configurable overlap. Every chunk is roughly the same size, which keeps embedding comparisons fair, but it will cut sentences in half at boundaries. Simple, predictable, and a good baseline.

- **Recursive** (`recursive_chunking`) splits first on paragraph breaks, then on sentence boundaries, then falls back to word-count splitting only for oversized single sentences. It preserves natural language structure, which produces more coherent chunks for embedding. It is the default for this reason.

The tradeoff is simplicity vs. coherence. For short, well-structured documents the difference is small. For long, densely-paragraphed documents (e.g. research papers, legal contracts), recursive chunking noticeably improves retrieval quality.

### Cosine similarity for retrieval

Sentence-transformer embeddings encode meaning as direction, not magnitude — a long document and a short sentence can mean the same thing, so their vectors should be close regardless of length. Cosine similarity measures the angle between vectors, making it insensitive to vector magnitude. L2 (Euclidean) distance would penalise long documents unfairly because their embeddings tend to have larger norms. ChromaDB's `hnsw:space=cosine` setting is set explicitly rather than relying on defaults so the behaviour is predictable if the collection is ever migrated.

### Cross-encoder reranking

Embedding retrieval encodes the query and each chunk independently, then compares them. This is fast and scales to millions of chunks, but the model never sees the query and chunk together — it can miss exact phrase matches, negations, or subtle relevance signals.

A cross-encoder reads the query and candidate chunk concatenated into one input and outputs a single relevance score. Because both texts are processed jointly, it catches what embedding comparison misses. The cost is speed: it requires one forward pass per candidate, making it too slow to rank an entire corpus. The solution used here is the standard two-stage approach: embeddings narrow the candidate pool quickly (top-k), then the cross-encoder reorders only those k candidates precisely. The UI exposes a toggle to disable reranking when latency matters more than precision.

### Faithfulness check as a separate LLM call

After the answer is generated, a second independent Groq call asks a neutral judge prompt: *"Given this context and this answer, is the answer fully supported by the context? Score 1–10."*

Asking the same model, in the same context window, to grade its own answer is unreliable. The model is anchored to the reasoning chain it just produced and is biased toward justifying its output. A fresh call with only the context passages and the answer — no memory of the generation call — produces more honest, calibrated scores. The judge prompt uses `temperature=0` for consistent, reproducible scoring.

### Local sentence-transformers instead of an API embedding model

- **Cost**: a 300-page document produces thousands of chunks. At API pricing, embedding a large corpus gets expensive fast. `all-MiniLM-L6-v2` runs for free after the one-time 90 MB download.
- **Speed**: no network round-trip per batch. Local inference on CPU is slower per token than a GPU API, but the absence of latency per request matters more for interactive use.
- **No external dependency**: the ingestion pipeline works offline and is unaffected by API outages, rate limits, or key rotation.

The model chosen (`all-MiniLM-L6-v2`) is 384-dimensional, ~90 MB, and performs well above its size on semantic similarity benchmarks. It is a standard choice for RAG systems that need to run on CPU.

---

## Setup

### Prerequisites

- Python 3.10+
- A Groq API key ([console.groq.com](https://console.groq.com) — free tier works)

### Install

```bash
git clone https://github.com/your-username/DocuMind.git
cd DocuMind

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### Run

Start the FastAPI backend (from the project root):

```bash
venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The first startup takes 10–15 seconds while `all-MiniLM-L6-v2` and the cross-encoder load from the HuggingFace cache (downloaded on first run, ~180 MB total).

In a separate terminal, start the Streamlit frontend:

```bash
venv/bin/streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### API docs

FastAPI's auto-generated interactive docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) while the backend is running.

---

## Project structure

```
DocuMind/
├── backend/
│   ├── __init__.py       # Makes backend a proper Python package
│   ├── config.py         # Central config: loads .env, defines constants
│   ├── ingestion.py      # PDF and DOCX text extraction
│   ├── chunking.py       # Fixed-size and recursive chunking strategies
│   ├── embeddings.py     # EmbeddingModel, VectorStore, process_and_store()
│   ├── retrieval.py      # retrieve_top_k(), CrossEncoderReranker, retrieve_and_rerank()
│   ├── generation.py     # generate_answer(), check_faithfulness(), answer_query()
│   ├── eval.py           # (placeholder for RAGAS-based evaluation)
│   └── main.py           # FastAPI app: /upload, /query, /collections
├── frontend/
│   └── app.py            # Streamlit UI
├── .env.example          # Template for environment variables
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Known limitations and what I'd improve with more time

**No automated evaluation suite.** There is an `eval.py` stub but no working evaluation pipeline. The next step would be integrating [RAGAS](https://github.com/explodinggradients/ragas) to measure context precision, context recall, answer faithfulness, and answer relevance against a golden Q&A dataset. Right now the only quality signal per query is the faithfulness score, which is a proxy, not a ground-truth metric.

**Chunking parameters are not tuned per document type.** The default `max_chunk_size=500` words works reasonably well for general prose, but it's not optimal for dense technical documents, contracts with numbered clauses, or documents that are mostly tables. A smarter system would inspect the document type and adjust chunk size accordingly, or let the user configure it per upload.

**Single-user, single-process, no auth.** The backend holds all model instances in `app.state` with no request isolation. Concurrent uploads to the same collection name would race. There is no authentication, so anyone who can reach port 8000 can upload and query. For a production deployment this would need proper auth (OAuth2/JWT), per-user collections, and async ingestion via a task queue.

**Ingestion blocks the HTTP request.** `POST /upload` runs the entire ingestion pipeline synchronously. For large PDFs this can take 30–60 seconds, which will time out browser clients. The fix is to return a job ID immediately and run ingestion as a background task (Celery, FastAPI `BackgroundTasks`, or similar), with a `GET /jobs/{id}` endpoint for status polling.

**No OCR support.** Scanned PDFs (images of pages) produce no extractable text and are rejected with an error. Adding [Tesseract](https://github.com/tesseract-ocr/tesseract) or a cloud OCR step would make the system useful for a much wider range of real-world documents.

**ChromaDB is local and single-node.** This is fine for local development and small document sets, but doesn't scale horizontally. Moving to a managed vector database (Pinecone, Weaviate, or Qdrant Cloud) would be the natural next step for a production deployment.

**The faithfulness score is a heuristic.** The 1–10 score from the separate LLM judge call is a useful signal but not a rigorous measure. It can be gamed by a model that confidently but incorrectly claims its answer is grounded. A proper hallucination detection system would use NLI (Natural Language Inference) models specifically trained on entailment tasks, not a general-purpose LLM grading itself through a different prompt.
