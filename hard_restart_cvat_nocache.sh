#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILES="-f docker-compose.yml \
               -f docker-compose.dev.yml \
               -f components/serverless/docker-compose.serverless.yml"

echo "⚠️  Bringing CVAT down…"
docker compose $COMPOSE_FILES down -v --remove-orphans

echo "🛠  Rebuilding all services with NO cache…"
docker compose $COMPOSE_FILES build --no-cache

echo "🚀 Starting CVAT…"
docker compose $COMPOSE_FILES up -d

echo "✅ Done."
