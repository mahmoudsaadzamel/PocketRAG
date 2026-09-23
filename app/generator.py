"""
generator.py  --  Writes the answer. The only file that talks to Groq.

One job: context + question in, answer out.

It does not know where the context came from. Swap the whole retriever and
this file does not change by a single character. That is what "one job"
buys you.
"""
from openai import OpenAI

from app import config

_client = None
_model = None


def connect():
    """
    Open the connection and pick a model that actually exists today.

    Never hardcode a model name. Providers retire them, your code breaks in
    production on a day you changed nothing, and the error looks like a bug
    in your own code. Asking the API is two lines and never goes stale.
    """
    global _client, _model
    if _client is not None:
        return _client, _model

    key = config.load_key()
    if not key:
        raise RuntimeError(
            "No GROQ_API_KEY found. In Colab use the key icon in the sidebar; "
            "elsewhere put it in a .env file. Never paste it into the code."
        )

    _client = OpenAI(api_key=key, base_url=config.GROQ_BASE_URL)
    available = {m.id for m in _client.models.list()}
    _model = next(
        (m for m in config.PREFERRED_MODELS if m in available),
        sorted(available)[0],
    )
    return _client, _model


def answer(context, question, max_tokens=250):
    """
    Build the prompt and send it.

    The last sentence of the prompt is the most important line in this
    file. Without it the model blends retrieved text with its own memory,
    and you lose the ability to tell which part came from your documents.
    """
    client, model = connect()

    prompt = (
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n"
        "Answer using only the context above. "
        "If the context does not contain the answer, say so clearly."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content

    # An empty answer is not an error, so nothing would tell you. This does.
    if not text or not text.strip():
        reason = response.choices[0].finish_reason
        return f"[EMPTY ANSWER - finish_reason={reason}, try a larger max_tokens]"
    return text.strip()
