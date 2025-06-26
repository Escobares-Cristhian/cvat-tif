#!/usr/bin/env bash
set -euo pipefail

IMAGE="nuclio/processor-sam2-detector:latest"

# 1. get the full image ID
IMAGE_ID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
echo "Image ID: $IMAGE_ID"

# 2. get the container ID of the running instance
CONTAINER_ID=$(docker ps -q --filter ancestor="$IMAGE_ID")
if [[ -z "$CONTAINER_ID" ]]; then
  echo "ERROR: no running container found for image $IMAGE" >&2
  exit 1
fi
echo "Container ID: $CONTAINER_ID"

# 3. print logs
docker logs "$CONTAINER_ID"

# 4. optional: copy all .png from container’s cwd into ./_sam2_images_tmp
if [[ "${1:-}" == "--copy-png" ]]; then
  echo "Copying .png files from container to ./_sam2_images_tmp…"
  mkdir -p _sam2_images_tmp
  # Use docker exec + tar so we can grab multiple files in one go
  docker exec "$CONTAINER_ID" sh -c 'tar cf - /opt/nuclio/sam2/*.png 2>/dev/null || true' \
    | tar xf - -C _sam2_images_tmp
  echo "Done. Look in ./_sam2_images_tmp/"
fi
