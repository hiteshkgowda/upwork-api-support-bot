"""Central configuration for the Upwork API Technical Support Bot.

Every tunable value and every secret is read from the environment so that
nothing sensitive (API keys, endpoints) is ever hard-coded into the source
(Assignment Rule #3). Local defaults are provided for the non-secret knobs so
the project runs out of the box once the .env is filled in.

See `.env.example` for the variables you are expected to set.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load key=value pairs from a local `.env` file into os.environ.
# `.env` is git-ignored; `.env.example` documents the expected keys.
load_dotenv()

# --- Project paths --------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Folder where the persisted FAISS index is written to / read from.
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Path to the source documentation (txt / md / pdf). Defaults to the Upwork
# API reference shipped in data/. Override with SOURCE_DOC_PATH in .env.
SOURCE_DOC_PATH = Path(
    os.getenv("SOURCE_DOC_PATH", str(DATA_DIR / "upwork_api_reference.pdf"))
)

# --- Chunking parameters (Assignment A2) ----------------------------------
# 500-character chunks with a 50-character overlap, exactly as specified.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Embedding model (Assignment A3) --------------------------------------
# A *local* sentence-transformers model: the documentation is embedded on
# this machine and never uploaded to a public service (Assignment Rule #1).
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Retrieval (Assignment B1) --------------------------------------------
# Number of chunks to fetch per query ("top 3 most relevant chunks").
TOP_K = int(os.getenv("TOP_K", "3"))

# --- LLM endpoint (Assignment B2) -----------------------------------------
# An OpenAI-compatible Chat Completions endpoint. This is intentionally
# generic so the same code works with the Hugging Face router, a local/remote
# Ollama server, vLLM, etc. All three values come from the environment.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
# Low temperature -> deterministic, fact-faithful answers (less hallucination).
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
