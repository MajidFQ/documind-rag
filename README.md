# DocuMind

A RAG (Retrieval-Augmented Generation) system that lets you chat with your PDF and DOCX documents. Answers come with source citations and a hallucination-check layer so you can trust what you read.

## Project Structure

```
documind/
├── backend/
│   ├── main.py          # FastAPI app entrypoint
│   ├── ingestion.py     # PDF/DOCX text extraction
│   ├── chunking.py      # Two chunking strategies
│   ├── embeddings.py    # Embedding generation + ChromaDB storage
│   ├── retrieval.py     # Similarity search + reranking
│   ├── generation.py    # LLM answer generation via Groq
│   ├── eval.py          # Hallucination/faithfulness check
│   └── config.py        # API keys and constants
├── frontend/
│   └── app.py           # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone the repo and create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

4. **Run the backend:**
   ```bash
   uvicorn backend.main:app --reload
   ```

5. **Run the frontend:**
   ```bash
   streamlit run frontend/app.py
   ```

## Key Design Decisions

- **pypdf** for PDF extraction — page-level text with page number tracking for citations
- **python-docx** for DOCX extraction — paragraph-level text, no page boundaries available
- **ChromaDB** for local vector storage — no external service needed
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings — fast, runs locally
- **Groq** for LLM inference — fast API with llama3
- **Hallucination check** — faithfulness scoring before answers reach the user
