#!/bin/bash
# Diagnostic: see what env vars the backend container actually has
set -e
echo "=== docker compose config (what compose thinks) ==="
docker compose config | grep -A 30 "backend:" | head -50
echo
echo "=== Container env vars (what the process sees) ==="
# This requires the container to be running. If it's crashlooping, run a one-off.
docker compose run --rm backend env | grep -E "ES_URL|INFERENCE_ID|JINA|EIS_MODEL|INDEX_NAME" || true
