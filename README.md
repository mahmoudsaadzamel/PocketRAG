# rag-app

A small, real RAG application. One file per component.

This is the same pipeline you built in Session 1's notebook, moved into the
shape a real project has. It grows by one component each session.

---

## The folder

```
rag-app/
├── Dockerfile              the recipe for building a container
├── docker-compose.yml      one command to run the whole thing
├── requirements.txt        pinned versions, so everyone installs the same
├── .env.example            which secrets you need (never the secrets)
├── data/
│   └── handbook.md         the document we ingest
└── app/
    ├── config.py           every setting, in one place
    ├── chunker.py          NEW IN SESSION 2 — cuts documents into chunks
    ├── embedder.py         text -> vectors. The only place a model loads
    ├── store.py            holds chunks, vectors and metadata
    ├── retriever.py        finds matching chunks (dense + sparse)
    ├── generator.py        writes the answer (Groq)
    └── pipeline.py         wires them together. No logic of its own
```

## The rule this project is built on

**One file, one job.**

Every file above should be explainable in a single sentence. When you cannot
say what a file does without using the word "and" twice, it is doing too much
and should be split.

The payoff is concrete: when the final answer is wrong, you can open exactly
one file to find out why.

## How a question flows

```
question
   -> retriever.py   finds the best chunks
   -> pipeline.py    joins them into one context block
   -> generator.py   sends context + question to the model
   -> answer
```

And before any question can be asked:

```
data/handbook.md
   -> chunker.py     cuts it into overlapping pieces
   -> embedder.py    turns each piece into 384 numbers
   -> store.py       saves text + numbers + metadata together
```

---

## Running it

### On your own machine

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your real Groq key in .env
python -m app.pipeline
```

### With Docker

```bash
cp .env.example .env          # then put your real Groq key in .env
docker compose up
```

Docker's promise is that the second version works identically on your laptop,
a teammate's laptop, and a server, because the container carries Python, the
libraries and the embedding model with it.

### In Google Colab

Docker does not run in Colab. Upload this folder, then:

```python
import sys; sys.path.insert(0, "/content/rag-app")
from app import pipeline
pipeline.ingest()
```

Put your Groq key in Colab's key icon (left sidebar), named `GROQ_API_KEY`,
with *Notebook access* turned on.

---

## The one rule about secrets

Your API key goes in `.env`, which is listed in `.gitignore` and never leaves
your machine. `.env.example` is the file that gets committed: it shows a new
teammate which variables exist without ever showing them a real key.

A key pasted into a `.py` file will end up in your git history, and rewriting
history to remove it is much harder than doing this correctly the first time.
