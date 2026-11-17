FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    OUTPUT_DIR=/tmp/outputs \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY storyforge ./storyforge
COPY data ./data

# Bake the embedding model into the image so the first request is fast.
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}')" || echo "model will download at runtime"

# Hugging Face Spaces runs containers as a non-root user.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/health')" || exit 1
CMD ["sh", "-c", "uvicorn storyforge.api:app --host 0.0.0.0 --port ${PORT}"]
