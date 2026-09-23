"""
embedder.py  --  Turns text into vectors. The ONLY place a model is loaded.

One job: text in, vectors out.

Why one file:
    The rule from Session 1 is that documents and queries must go through
    the SAME model. The easiest way to guarantee that is to make it
    impossible to do otherwise: there is exactly one model in this project,
    and it is loaded here.
"""
from sentence_transformers import SentenceTransformer

from app import config

# Loaded once, the first time this module is imported, then reused.
# Loading it per call would re-read 90 MB from disk every single time.
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_documents(texts):
    """Many texts in, many vectors out. Used when filling the store."""
    return get_model().encode(texts).tolist()


def embed_query(text):
    """One question in, one vector out. Used at search time."""
    return get_model().encode([text])[0].tolist()


def vector_size():
    """How many numbers describe one piece of text. 384 for MiniLM."""
    return get_model().get_sentence_embedding_dimension()
