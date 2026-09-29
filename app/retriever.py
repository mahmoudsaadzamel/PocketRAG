"""
retriever.py  --  Finds the chunks that answer a question.

One job: question in, ranked chunks out.

Two ways to search, and they fail at opposite things:

    dense   understands MEANING, is blind to exact rare words
    sparse  matches exact WORDS, is blind to meaning

Session 2 adds sparse search next to the dense search from Session 1.
"""
import re

from rank_bm25 import BM25Okapi

from app import config, embedder, store

_bm25 = None
_bm25_chunks = []
_bm25_metas = []


def tokenize(text):
    """
    Split text into words for BM25.

    Why not just text.lower().split()?
        Because that keeps punctuation glued to words. "refund." and
        "refund" become two different tokens, so a search for "refund"
        misses the sentence that contains it, and BM25 looks broken when
        the real fault is the tokenizer.

    This keeps letters and digits only, which also keeps product codes
    like NW4417 in one piece.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


# ------------------------------------------------------------------ dense
def search_dense(question, k=None):
    """
    Vector similarity. Finds chunks that MEAN the same thing, even when
    they share no words with the question.
    """
    k = k or config.TOP_K
    result = store.get().query(
        query_embeddings=[embedder.embed_query(question)],
        n_results=k,
    )
    return [
        {"text": doc, "meta": meta, "score": dist, "how": "dense"}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


# ----------------------------------------------------------------- sparse
def index_sparse(chunks, metadatas=None):
    """
    Build the BM25 index. Pure Python, no model, no network, instant.

    BM25 scores a chunk by how many of the query's words it contains,
    weighted so that rare words count for more than common ones.
    """
    global _bm25, _bm25_chunks, _bm25_metas
    _bm25_chunks = chunks
    _bm25_metas = metadatas or [{} for _ in chunks]
    _bm25 = BM25Okapi([tokenize(c) for c in chunks])


def search_sparse(question, k=None):
    """Exact word matching. Finds product codes and names dense search misses."""
    k = k or config.TOP_K
    if _bm25 is None:
        raise RuntimeError("Call index_sparse(chunks) first.")

    scores = _bm25.get_scores(tokenize(question))
    ranked = sorted(zip(_bm25_chunks, _bm25_metas, scores), key=lambda row: -row[2])
    return [
        {"text": text, "meta": meta, "score": score, "how": "sparse"}
        for text, meta, score in ranked[:k]
    ]
