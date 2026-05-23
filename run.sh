#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Find Waldo — one-shot start + ingest
#
# Run from /Users/brianjackson/Downloads/findwaldo:
#     bash run.sh
#
# It will:
#   1. Verify backend/.env has a real API key
#   2. docker compose up --build -d   (starts backend on host:8100, frontend on host:8080)
#   3. Wait until /api/health returns ok
#   4. Run scripts/batch_ingest.py against /Users/brianjackson/waldoImages
#   5. Print the URL to open
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="/Users/brianjackson/Downloads/findwaldo"
IMAGES_DIR="/Users/brianjackson/waldoImages"
BACKEND_HOST_PORT=8200
FRONTEND_HOST_PORT=8280
HEALTH_URL="http://localhost:${BACKEND_HOST_PORT}/api/health"
APP_URL="http://localhost:${FRONTEND_HOST_PORT}"

c_green() { printf "\033[32m%s\033[0m\n" "$*"; }
c_red()   { printf "\033[31m%s\033[0m\n" "$*"; }
c_yel()   { printf "\033[33m%s\033[0m\n" "$*"; }
c_step()  { printf "\n\033[1;36m▸ %s\033[0m\n" "$*"; }
die()     { c_red "✗ $*"; exit 1; }

cd "$PROJECT_DIR"

# 1. Sanity ----------------------------------------------------------
c_step "Sanity checks"
[[ -f backend/.env ]] || die "backend/.env not found"
if grep -q "PASTE_YOUR_KEY_HERE" backend/.env; then
    die "Replace PASTE_YOUR_KEY_HERE in backend/.env with your real Elastic API key, then rerun."
fi
[[ -d "$IMAGES_DIR" ]] || die "Image folder not found: $IMAGES_DIR"
command -v docker  >/dev/null || die "docker not on PATH. Install Docker Desktop."
docker info >/dev/null 2>&1 || die "Docker daemon not running. Open Docker Desktop and retry."
command -v python3 >/dev/null || die "python3 required for ingest"
c_green "  ok"

# 2. Start containers ------------------------------------------------
c_step "Starting containers (docker compose up --build -d)"
echo "  backend  → host port ${BACKEND_HOST_PORT}"
echo "  frontend → host port ${FRONTEND_HOST_PORT}"
docker compose up --build -d

# 3. Wait for backend health ----------------------------------------
c_step "Waiting for backend health at ${HEALTH_URL}"
attempts=0
max_attempts=60   # 60 * 2s = 2 minutes
while (( attempts < max_attempts )); do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
        c_green "  backend healthy"
        break
    fi
    sleep 2
    attempts=$((attempts + 1))
    if (( attempts % 5 == 0 )); then
        echo "  still waiting... ($((attempts * 2))s elapsed)"
    fi
done
if (( attempts >= max_attempts )); then
    c_red "Backend did not become healthy in 2 minutes."
    echo ""
    echo "Last 40 lines of backend log:"
    docker compose logs --tail=40 backend
    exit 1
fi

# Show the cluster bootstrap output (model_id, dims, similarity)
echo ""
echo "Backend startup summary:"
docker compose logs backend 2>&1 | grep -E "Found inference|configured for|Creating index|already exists" | tail -5 || true

# 4. Ingest images ---------------------------------------------------
c_step "Ingesting images from ${IMAGES_DIR}"
python3 scripts/batch_ingest.py "$IMAGES_DIR" --backend "http://localhost:${BACKEND_HOST_PORT}"

# 5. Done ------------------------------------------------------------
c_step "Done"
c_green "Open the app: ${APP_URL}"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f backend     # live backend logs"
echo "  docker compose logs -f frontend    # live frontend logs"
echo "  docker compose down                # stop everything"
echo "  docker compose down -v             # stop + delete static images"
