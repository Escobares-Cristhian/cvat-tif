#!/bin/bash
# Sample commands to deploy nuclio functions on CPU

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}

export DOCKER_BUILDKIT=1

docker build -t cvat.openvino.base "$SCRIPT_DIR/openvino/base"

nuctl create project cvat --platform local

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/**/function.yaml
do
    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$(dirname "$func_root")")"

    if [ -f "$func_root/Dockerfile" ]; then
        docker build -t "cvat.${func_rel_path//\//.}.base" "$func_root"
    fi

    echo "Deploying $func_rel_path function..."
    nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
done

nuctl get function --platform local

# --- Added Section: Tail Logs Using docker logs ---
echo "Tailing logs for deployed functions..."

# Ensure jq is installed.
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Please install jq to tail logs automatically."
    exit 1
fi

# Retrieve actual function names from the JSON output.
# This extracts the value of metadata.name from each function.
function_list=$(nuctl get function --platform local --output json | jq -r '.[].metadata.name')

# For each deployed function, find the container and tail its logs.
for func in $function_list; do
    echo "Processing function: $func"
    # Based on your docker ps output, Nuclio names the container as "nuclio-nuclio-<function_name>"
    container_id=$(docker ps --filter "name=nuclio-nuclio-${func}" --format "{{.ID}}")
    if [ -n "$container_id" ]; then
        echo "Tailing logs for function $func (container: $container_id)..."
        docker logs --follow "$container_id" &
    else
        echo "No running container found for function: $func"
    fi
done

# Wait for all background docker logs processes to finish (keeping the terminal open)
wait
# --- End of Added Section ---

