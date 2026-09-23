# A Dockerfile is a recipe. Each line is one step, and Docker follows them
# in order on a clean machine to build an image of your app.
#
# Build it:  docker build -t rag-app .
# Run it:    docker run --env-file .env rag-app

# 1. Start from a machine that already has Python 3.11.
#    "slim" is the small version: fewer tools, a much smaller image.
FROM python:3.11-slim

# 2. Work inside /app from here on, so paths stay short and predictable.
WORKDIR /app

# 3. Copy ONLY the requirements file first, then install.
#    Docker caches each step. Because your requirements change far less
#    often than your code, this ordering means editing a .py file does not
#    re-download every package. Copy everything first and it would.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download the embedding model INTO the image, at build time.
#    Without this the container downloads 90 MB on first run, every run,
#    and fails completely if the machine has no internet.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# 5. Now copy the actual code and data.
COPY app/ ./app/
COPY data/ ./data/

# 6. The command that runs when the container starts.
CMD ["python", "-m", "app.pipeline"]
