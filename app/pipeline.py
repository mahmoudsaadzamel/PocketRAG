"""
pipeline.py  --  Wires the components together. Contains no logic of its own.

One job: connect chunker -> store -> retriever -> generator.

Read this file top to bottom and you have read the whole system. Every
detail lives in the small file that owns it.

Run it:      python -m app.pipeline
"""
from pathlib import Path

from app import RRF, chunker, config, generator, retriever, store


def ingest(path=None):
    """Read a file, cut it up, and put the pieces in both indexes."""
    path = path or config.DATA_FILE
    text = Path(path).read_text(encoding="utf-8")

    chunks, metadatas = chunker.chunk_document(text, source=Path(path).name)

    store.build(chunks, metadatas)                # dense index
    retriever.index_sparse(chunks, metadatas)     # sparse index
    print(f"Chunks ingested and indexed: {len(chunks)}")
    return chunks


def ask(question, how="hybrid", k=None):
    """Retrieve, then generate. The two halves of RAG, in four lines."""
    search = {
        "dense": retriever.search_dense,
        "sparse": retriever.search_sparse,
        "hybrid": RRF.search_hybrid,          # SESSION 4: the default now
    }[how]
    hits = search(question, k or config.TOP_K)
    context = "\n\n".join(h["text"] for h in hits)
    return generator.answer(context, question), hits


def main():
    ingest()

    question = "How long does international shipping take?"
    answer, hits = ask(question)

    print("Q:", question)
    print("Retrieved:", [h["meta"].get("chunk_index") for h in hits])
    print("A:", answer)
    print()
    print("Embedding model (local):", config.EMBED_MODEL)
    print("Answer model (Groq):    ", generator.connect()[1])


if __name__ == "__main__":
    main()
