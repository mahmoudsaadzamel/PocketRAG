"""
store.py  --  The knowledge store. Holds chunks, vectors and metadata.

One job: put chunks in, get matching chunks out.

It calls embedder.py for vectors and never loads a model itself, so there
is no way for a second embedding model to sneak into the collection.
"""
import chromadb

from app import config, embedder

# PersistentClient writes to disk. chromadb.Client() would keep everything
# in memory and lose it the moment the process ends, which also makes the
# volume mount in docker-compose.yml pointless.
_client = chromadb.PersistentClient(path=config.CHROMA_DIR)


def build(chunks, metadatas, name=None):
    """
    Create the collection from scratch and fill it.

    Deleting first makes this safe to run twice. Without it, running the
    same script again quietly doubles every chunk, and your results fill
    up with duplicates without a single error appearing.
    """
    name = name or config.COLLECTION_NAME

    if name in [c.name for c in _client.list_collections()]:
        _client.delete_collection(name)
    collection = _client.create_collection(name=name)

    collection.add(
        documents=chunks,
        embeddings=embedder.embed_documents(chunks),   # always explicit
        ids=[f"chunk{i}" for i in range(len(chunks))],
        metadatas=metadatas,
    )
    return collection


def get(name=None):
    return _client.get_collection(name or config.COLLECTION_NAME)


def count(name=None):
    return get(name).count()
