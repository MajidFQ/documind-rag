"""
frontend/app.py — Streamlit UI for DocuMind.

This file is the complete user-facing interface. It talks to the FastAPI
backend over HTTP — it contains no ML code itself, only API calls and
display logic.

Page layout:

  Sidebar
    ├── Title + description
    ├── File uploader (PDF / DOCX)
    ├── Chunking method dropdown
    ├── Upload & Process button  →  POST /upload
    ├── Collection picker        →  GET /collections (refreshed after upload)
    └── Delete collection button →  DELETE /collections/{name}

  Main area
    ├── Empty-state message when no collection is selected
    ├── Question input + rerank toggle + Ask button  →  POST /query
    └── Chat history (session_state)
          Each Q&A entry shows:
            - The question
            - The answer
            - Sources (expandable)
            - Faithfulness score badge + explanation

All API calls go through the two helpers at the top (api_get / api_post /
api_delete). They handle connection errors and non-200 responses uniformly,
returning None on failure so callers can show st.error() without crashing.
"""

import streamlit as st
import requests

# ── Backend base URL ──────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# API helpers — all network calls live here, nowhere else
# ---------------------------------------------------------------------------

def api_get(path: str) -> dict | None:
    """
    Make a GET request to the backend and return the parsed JSON.

    Returns None and calls st.error() on any failure so callers don't need
    to handle exceptions themselves.

    Args:
        path: URL path, e.g. '/collections'.

    Returns:
        Parsed JSON dict, or None if the request failed.
    """
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the DocuMind backend. "
            "Make sure the FastAPI server is running on port 8000:\n\n"
            "```\nuvicorn main:app --host 127.0.0.1 --port 8000\n```"
        )
    except requests.exceptions.HTTPError as e:
        detail = _extract_detail(e.response)
        st.error(f"Backend error ({e.response.status_code}): {detail}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def api_post(path: str, **kwargs) -> dict | None:
    """
    Make a POST request to the backend and return the parsed JSON.

    Accepts the same keyword arguments as requests.post() — pass either
    `json=` for JSON bodies or `data=` + `files=` for multipart uploads.

    Returns None and calls st.error() on any failure.
    """
    try:
        response = requests.post(f"{API_BASE}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the DocuMind backend. "
            "Make sure the FastAPI server is running on port 8000."
        )
    except requests.exceptions.HTTPError as e:
        detail = _extract_detail(e.response)
        st.error(f"Backend error ({e.response.status_code}): {detail}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def api_delete(path: str) -> dict | None:
    """
    Make a DELETE request to the backend and return the parsed JSON.

    Returns None and calls st.error() on any failure.
    """
    try:
        response = requests.delete(f"{API_BASE}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the DocuMind backend.")
    except requests.exceptions.HTTPError as e:
        detail = _extract_detail(e.response)
        st.error(f"Backend error ({e.response.status_code}): {detail}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def _extract_detail(response: requests.Response) -> str:
    """
    Pull the 'detail' field from a FastAPI error response body, or fall back
    to the raw text. Keeps error messages readable rather than showing raw JSON.
    """
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def fetch_collections() -> list[dict]:
    """
    Fetch the list of collections from the backend.

    Returns an empty list if the backend is unreachable or returns an error,
    so the rest of the UI degrades gracefully rather than crashing.
    """
    data = api_get("/collections")
    if data is None:
        return []
    return data.get("collections", [])


def refresh_collections():
    """
    Re-fetch collections from the backend and store them in session_state.

    Called after every upload or delete so the sidebar dropdown stays in sync
    with actual backend state.
    """
    st.session_state.collections = fetch_collections()


# ---------------------------------------------------------------------------
# Faithfulness badge
# ---------------------------------------------------------------------------

def render_faithfulness_badge(score: int, explanation: str):
    """
    Render a colored metric badge for the faithfulness score.

    Color scale:
      Green  (8–10) — answer is well-grounded in the retrieved context.
      Yellow (5–7)  — partially grounded; worth checking the sources.
      Red    (1–4)  — low confidence; likely contains hallucinated content.
      Grey   (-1)   — score could not be parsed from the LLM response.

    Args:
        score:       Integer faithfulness score from the backend (1–10 or -1).
        explanation: One-sentence rationale from the faithfulness-check call.
    """
    if score == -1:
        color, label = "#888888", "? Unscored"
    elif score >= 8:
        color, label = "#2e7d32", f"✓ {score}/10 — High confidence"
    elif score >= 5:
        color, label = "#f57c00", f"~ {score}/10 — Review sources"
    else:
        color, label = "#c62828", f"✗ {score}/10 — Low confidence"

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background-color: {color};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 4px;
        ">{label}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Faithfulness: {explanation}")


# ---------------------------------------------------------------------------
# Chat history rendering
# ---------------------------------------------------------------------------

def render_chat_history():
    """
    Render all Q&A pairs stored in st.session_state.chat_history.

    Each entry is a dict produced by the /query response. Entries are shown
    newest-first so the most recent answer is always at the top.
    """
    for entry in reversed(st.session_state.chat_history):
        # Question
        with st.chat_message("user"):
            st.write(entry["query"])

        # Answer
        with st.chat_message("assistant"):
            st.write(entry["answer"])

            # Faithfulness badge
            render_faithfulness_badge(
                entry["faithfulness_score"],
                entry["faithfulness_explanation"],
            )

            # Sources — in an expander to keep the main view clean
            sources = entry.get("sources_cited", [])
            if sources:
                with st.expander(f"Sources ({len(sources)} cited)", expanded=False):
                    for src in sources:
                        name = src.get("source", "unknown")
                        page = src.get("page_number", -1)
                        page_label = f" — page {page}" if page and page != -1 else ""
                        chunk_id = src.get("chunk_id", "?")
                        st.markdown(f"**{name}**{page_label} *(chunk {chunk_id})*")
                        st.caption(src.get("text", "")[:300] + "…")
                        st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str | None:
    """
    Render the full sidebar and return the currently selected collection name,
    or None if no collection exists yet.

    Side effects:
      - Uploads a file and ingests it when the user clicks "Upload & Process".
      - Deletes a collection when the user confirms deletion.
      - Refreshes st.session_state.collections after both operations.

    Returns:
        The selected collection name string, or None.
    """
    with st.sidebar:
        st.title("DocuMind")
        st.caption("Upload a document, then ask questions about it.")
        st.divider()

        # ── File upload section ───────────────────────────────────────────
        st.subheader("Upload a Document")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx"],
            help="Supported formats: PDF and DOCX.",
            label_visibility="collapsed",
        )

        chunking_method = st.selectbox(
            "Chunking method",
            options=["recursive", "fixed_size"],
            index=0,
            help=(
                "**recursive** — splits on paragraph and sentence boundaries, "
                "preserving natural language structure. Better for most documents.\n\n"
                "**fixed_size** — splits by word count with overlap. "
                "Simple and predictable; good as a baseline."
            ),
        )

        collection_input = st.text_input(
            "Collection name",
            value="documind",
            help="Documents are stored in named collections. Use one name per topic or project.",
        )

        if st.button("Upload & Process", type="primary", disabled=uploaded_file is None):
            with st.spinner(f"Ingesting '{uploaded_file.name}' …"):
                result = api_post(
                    "/upload",
                    data={
                        "chunking_method": chunking_method,
                        "collection_name": collection_input,
                    },
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                )
            if result:
                st.success(
                    f"Stored **{result['chunks_stored']}** chunk(s) from "
                    f"**{result['filename']}** into collection "
                    f"**{result['collection_name']}**."
                )
                refresh_collections()
                # Auto-select the just-uploaded collection
                st.session_state.selected_collection = result["collection_name"]

        st.divider()

        # ── Collection picker ─────────────────────────────────────────────
        st.subheader("Active Collection")

        collections = st.session_state.get("collections", [])

        # Refresh on first load
        if not collections and "collections_loaded" not in st.session_state:
            st.session_state.collections_loaded = True
            refresh_collections()
            collections = st.session_state.get("collections", [])

        if not collections:
            st.info("No collections yet. Upload a document above to get started.")
            return None

        col_names = [c["name"] for c in collections]
        col_counts = {c["name"]: c["count"] for c in collections}

        # Preserve the previously selected collection across reruns
        default_idx = 0
        prev = st.session_state.get("selected_collection")
        if prev in col_names:
            default_idx = col_names.index(prev)

        selected = st.selectbox(
            "Query against:",
            options=col_names,
            index=default_idx,
            format_func=lambda n: f"{n}  ({col_counts[n]} chunks)",
        )
        st.session_state.selected_collection = selected

        # ── Delete collection ─────────────────────────────────────────────
        st.divider()

        # Two-step confirmation to prevent accidental deletes
        if st.session_state.get("confirm_delete") != selected:
            if st.button("🗑 Delete this collection", type="secondary"):
                st.session_state.confirm_delete = selected
                st.rerun()
        else:
            st.warning(f"Delete **{selected}**? This cannot be undone.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, delete", type="primary"):
                    result = api_delete(f"/collections/{selected}")
                    if result:
                        st.success(f"Deleted '{selected}'.")
                        st.session_state.confirm_delete = None
                        st.session_state.selected_collection = None
                        # Clear chat history — it belonged to this collection
                        st.session_state.chat_history = []
                        refresh_collections()
                        st.rerun()
            with col_no:
                if st.button("Cancel"):
                    st.session_state.confirm_delete = None
                    st.rerun()

    return selected


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

def render_main(selected_collection: str | None):
    """
    Render the query interface and chat history in the main content area.

    Shows a friendly empty-state message when no collection is selected yet,
    so the user sees clear guidance rather than a broken input form.

    Args:
        selected_collection: The collection name to query, or None.
    """
    st.header("Ask a Question")

    # ── Empty state ───────────────────────────────────────────────────────
    if not selected_collection:
        st.info(
            "👈 Upload a document using the sidebar to get started. "
            "Once ingested, select it from the collection dropdown and "
            "ask questions here."
        )
        return

    st.caption(f"Querying collection: **{selected_collection}**")

    # ── Query form ────────────────────────────────────────────────────────
    with st.form("query_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder="e.g. What are the main conclusions of this document?",
            height=80,
            label_visibility="collapsed",
        )

        col_btn, col_rerank, col_topk = st.columns([2, 2, 1])
        with col_btn:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)
        with col_rerank:
            rerank = st.toggle(
                "Cross-encoder reranking",
                value=True,
                help=(
                    "When on, a second model re-scores the retrieved chunks for "
                    "better accuracy. Adds ~1 second. Turn off for faster but "
                    "slightly less precise results."
                ),
            )
        with col_topk:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    if submitted:
        question = question.strip()
        if not question:
            st.warning("Please enter a question before clicking Ask.")
        else:
            with st.spinner("Retrieving and generating answer …"):
                result = api_post(
                    "/query",
                    json={
                        "query": question,
                        "collection_name": selected_collection,
                        "top_k": top_k,
                        "rerank": rerank,
                    },
                )
            if result:
                # Prepend to history so newest appears at top
                st.session_state.chat_history.append(result)
                st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        col_hist, col_clear = st.columns([5, 1])
        with col_hist:
            st.subheader("Conversation")
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        render_chat_history()
    else:
        st.caption("Your answers will appear here.")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state():
    """
    Set up all session_state keys on first load so every other function can
    read them without checking for KeyError.

    Called once at the top of main() before any widgets are rendered.
    """
    defaults = {
        "chat_history":         [],
        "collections":          [],
        "collections_loaded":   False,
        "selected_collection":  None,
        "confirm_delete":       None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Top-level entry point — sets page config, initialises state, and renders
    the sidebar and main area.
    """
    st.set_page_config(
        page_title="DocuMind",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    selected_collection = render_sidebar()
    render_main(selected_collection)


if __name__ == "__main__":
    main()
