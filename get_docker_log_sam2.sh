#!/usr/bin/env bash
set -euo pipefail

IMAGE="nuclio/processor-sam2-detector:latest"

# 1. get the full image ID (e.g. sha256:...)
# docker image inspect "$IMAGE" --format '{{.Id}}'
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')

echo "Image ID: $IMAGE_ID"

# 2. Print logs
docker logs "$(docker ps -aq --filter ancestor="$IMAGE_ID" --format '{{.ID}}')"
