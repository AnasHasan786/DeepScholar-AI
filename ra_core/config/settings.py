import os
from pathlib import Path
from dotenv import load_dotenv

# Path logic: Go up two levels from ra_core/config/ to get to the project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"

# Load local .env file if it exists
load_dotenv(dotenv_path=ENV_PATH)

# ─── Database Configuration ───────────────────────────────────────────────────
# Priority 1: Environment Variable (Docker/Azure)
# Priority 2: Local Windows fallback
DEFAULT_LOCAL_DB = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
DB_URI = os.getenv("DB_URI", DEFAULT_LOCAL_DB)

# ─── LLM Configuration ────────────────────────────────────────────────────────
LLM_MODEL_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0  # CRITICAL: Keep at 0 for RAG consistency

# ─── API Keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── Other Constants ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
RETRIEVER_K = 6