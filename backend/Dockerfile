FROM python:3.12-slim

WORKDIR /app

# System deps for Pillow JPEG/PNG support
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so they cache across code changes
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./

# Static dir is a volume mount in compose, but make it exist by default
RUN mkdir -p /app/static

EXPOSE 8000

# --timeout-keep-alive 1200 keeps idle connections open for 20 minutes,
# which matters because multi-scale ingest of one image can take 8+ minutes
# while we pace requests under the Jina rate limit. Default 5s keep-alive
# would drop the client connection mid-embed even though work is progressing.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "1200"]
