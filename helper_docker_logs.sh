#!/usr/bin/env bash
#
# get_cvat_logs.sh
# Fetch logs for all CVAT-related containers at once.

set -euo pipefail

containers=(
  cvat_server
  cvat_ui
  cvat_utils
  cvat_vector
  cvat_worker_analytics_reports
  cvat_worker_annotation
  cvat_worker_export
  cvat_worker_import
  cvat_worker_quality_reports
)

for c in "${containers[@]}"; do
  if docker inspect "$c" &>/dev/null; then
    echo
    echo "===== Logs for $c ====="
    docker logs --tail 50 "$c"
  else
    echo
    echo "!!! Container '$c' not found, skipping."
  fi
done
