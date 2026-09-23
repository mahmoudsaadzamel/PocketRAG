"""
chunker.py  --  NEW IN SESSION 2.  Cuts a long document into retrievable pieces.

One job: text in, list of chunks out.

It knows nothing about embeddings, databases or models. That is the point.
You can test it, change it, and break it without touching anything else.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def chunk_text(text, size=None, overlap=None):
    """
    Split one long string into overlapping chunks.

    'Recursive' means it tries separators in order and only falls back to a
    cruder one when a piece is still too big:

        paragraph break  ->  line break  ->  sentence  ->  word  ->  character

    So it cuts at a paragraph if it can, and only chops mid-word as a last
    resort. That ordering is the whole difference between readable chunks
    and shredded ones.
    """
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_document(text, source, size=None, overlap=None):
    """
    Same as chunk_text, but each chunk carries metadata saying where it
    came from and which piece it was.

    Without this you can never cite an answer, and you can never debug a
    wrong one, because you cannot tell which part of which file it used.
    """
    chunks = chunk_text(text, size, overlap)
    metadatas = [
        {"source": source, "chunk_index": i, "chunk_size": len(c)}
        for i, c in enumerate(chunks)
    ]
    return chunks, metadatas
