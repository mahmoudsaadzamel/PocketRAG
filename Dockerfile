
FROM python:3.11-slim

# 2. Work inside /app from here on, so paths stay short and predictable.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# 5. Now copy the actual code and data.
COPY app/ ./app/
COPY data/ ./data/

# 6. The command that runs when the container starts.
CMD ["python", "-m", "app.pipeline"]
