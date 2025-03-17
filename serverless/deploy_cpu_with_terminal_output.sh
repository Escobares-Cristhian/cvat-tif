#!/bin/bash
# Script to deploy Nuclio functions on CPU using the logging module,
# build and run the base image container with unbuffered Python output,
# and tail logs for both the base container and deployed functions

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}

export DOCKER_BUILDKIT=1

# Build the base Docker image
docker build -t cvat.openvino.base "$SCRIPT_DIR/openvino/base"

# Check if a container named "cvat_openvino" exists (running or stopped)
if docker ps -a --filter "name=cvat_openvino" --format "{{.ID}}" | grep -q .; then
    # Container exists: check if it's running.
    if ! docker ps --filter "name=cvat_openvino" --format "{{.ID}}" | grep -q .; then
        echo "Container cvat_openvino exists but is not running. Starting it..."
        docker start cvat_openvino
    else
        echo "Container cvat_openvino is already running."
    fi
else
    echo "Creating and starting cvat.openvino.base container with unbuffered Python output..."
    docker run --name cvat_openvino -d -e PYTHONUNBUFFERED=1 cvat.openvino.base
fi

# Create a Nuclio project
nuctl create project cvat --platform local

# Enable recursive globbing for finding function.yaml files
shopt -s globstar

# Loop through each Nuclio function configuration file
for func_config in "$FUNCTIONS_DIR"/**/function.yaml; do
    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$(dirname "$func_root")")"

    # Build a function-specific Docker image if a Dockerfile exists
    if [ -f "$func_root/Dockerfile" ]; then
        docker build -t "cvat.${func_rel_path//\//.}.base" "$func_root"
    fi

    echo "Deploying $func_rel_path function..."
    # Deploy the function with logging enabled and required environment variables
    nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --env USE_LOGGING=1 \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
done

# List deployed functions
nuctl get function --platform local

# --- Tail Logs Section ---
echo "Tailing logs for deployed functions and base container..."

# Ensure jq is installed.
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Please install jq to tail logs automatically."
    exit 1
fi

# Retrieve Nuclio function names from JSON output.
function_list=$(nuctl get function --platform local --output json | jq -r '.[].metadata.name')

# Tail logs for each deployed Nuclio function container.
for func in $function_list; do
    echo "Processing function: $func"
    container_id=$(docker ps --filter "name=nuclio-nuclio-${func}" --format "{{.ID}}")
    if [ -n "$container_id" ]; then
        echo "Tailing logs for function $func (container: $container_id)..."
        docker logs --follow "$container_id" &
    else
        echo "No running container found for function: $func"
    fi
done

# Tail logs for the base image container.
echo "Processing base image container: cvat.openvino.base"
base_container_id=$(docker ps --filter "name=cvat_openvino" --format "{{.ID}}")
if [ -n "$base_container_id" ]; then
    echo "Tailing logs for base container (ID: $base_container_id)..."
    docker logs --follow "$base_container_id" &
else
    echo "No running container found for cvat.openvino.base"
fi

# Wait for all background docker logs processes to finish (keeping the terminal open)
wait

