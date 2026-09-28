"""
pipeline.py  --  Wires the components together. Contains no logic of its own.

One job: connect chunker -> store -> retriever -> generator.

Read this file top to bottom and you have read the whole system. Every
detail lives in the small file that owns it.

Run it:      python -m app.pipeline
"""
from pathlib import Path

from app import chunker, config, generator, retriever, store


def ingest(path=None):
    """Read a file, cut it up, and put the pieces in both indexes."""
    path = path or config.DATA_FILE
    text = Path(path).read_text(encoding="utf-8")

    chunks, metadatas = chunker.chunk_document(text, source=Path(path).name)

    store.build(chunks, metadatas)     # dense index
    retriever.index_sparse(chunks)     # sparse index
    print("Chunks ingested and indexed.",chunks)
    return chunks


def ask(question, how="dense", k=None):
    """Retrieve, then generate. The two halves of RAG, in four lines."""
    search = retriever.search_dense if how == "dense" else retriever.search_sparse
    hits = search(question, k)
    context = "\n\n".join(h["text"] for h in hits)
    return generator.answer(context, question), hits


def main():
    chunks = ingest()
    print(f"Ingested {len(chunks)} chunks from {config.DATA_FILE}\n")

    question = "How long does international shipping take?"
    answer, hits = ask(question)

    print("Q:", question)
    print("Retrieved:", [h["meta"].get("chunk_index") for h in hits])
    print("A:", answer)


if __name__ == "__main__":
    main()
