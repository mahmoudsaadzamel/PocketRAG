"""
config.py  --  Every setting in the project lives here. Nothing else.

Why this file exists:
    When a setting is written in three different files, changing it means
    remembering all three. Sooner or later you change two of them, and the
    bug that follows is very hard to see.

    One place. Everything reads from here.
"""
import os

# ---------------------------------------------------------------- embedding
# ONE model for documents AND for queries. This is the rule from Session 1.
EMBED_MODEL = "all-MiniLM-L6-v2"      # 90 MB, CPU, 384 numbers per chunk

# ---------------------------------------------------------------- chunking
CHUNK_SIZE = 300      # characters per chunk
CHUNK_OVERLAP = 45    # characters repeated between neighbours (15% of size)

# ---------------------------------------------------------------- retrieval
TOP_K = 3             # how many chunks we hand to the model

# ---------------------------------------------------------------- generation
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Models get retired. We ask the API what exists instead of hardcoding one.
PREFERRED_MODELS = (
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)

# ---------------------------------------------------------------- storage
COLLECTION_NAME = "handbook"
DATA_FILE = "data/handbook.md"

# Where Chroma writes its files. Anything stored here survives the process
# exiting, which is the difference between a demo and something you can
# actually restart. docker-compose maps this folder out of the container.
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma")


def load_key(name: str = "GROQ_API_KEY"):
    """
    Find the API key wherever this code happens to be running.

    Colab keeps secrets in its own store, NOT in environment variables,
    which is why the plain os.environ lookup is not enough on its own.
    """
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return value.strip()
    except Exception:
        pass
    return os.environ.get(name)
