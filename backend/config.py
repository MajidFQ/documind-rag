"""
config.py — Central configuration for DocuMind.

Loads secrets from the .env file and defines constants used across
the backend so that every module has a single place to look for settings.
"""

import os
from dotenv import load_dotenv

# Load variables from .env into the environment before anything reads them
load_dotenv()


def get_required_env(variable_name: str) -> str:
    """
    Read an environment variable and raise a clear error if it is missing.

    Failing early with a descriptive message is better than letting the app
    crash later with a cryptic AttributeError or API auth error.
    """
    value = os.getenv(variable_name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{variable_name}' is not set. "
            "Check your .env file and make sure it matches .env.example."
        )
    return value


# --- API Keys ---
GROQ_API_KEY: str = get_required_env("GROQ_API_KEY")

# --- Chunking ---
# How many characters each text chunk should contain
CHUNK_SIZE: int = 500

# How many characters overlap between consecutive chunks so context isn't lost
# at chunk boundaries
CHUNK_OVERLAP: int = 50

# --- Retrieval ---
# Number of chunks to fetch from the vector store per query
TOP_K: int = 5

# --- Embedding model ---
# Using a lightweight but capable model that runs locally via sentence-transformers
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# --- LLM ---
# Groq-hosted model used for answer generation
LLM_MODEL: str = "llama3-8b-8192"

# --- Chroma ---
# Local directory where ChromaDB persists its vector data
CHROMA_PERSIST_DIR: str = "./chroma_store"
CHROMA_COLLECTION_NAME: str = "documind"
