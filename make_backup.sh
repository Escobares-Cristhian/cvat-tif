#!/usr/bin/env bash
set -euo pipefail

# 0. Start timer
SECONDS=0

# 0. Figure out the Compose project name (same as before)
project="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}"

# 1. Ensure backup folder is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <path_to_backup_folder>" >&2
  echo "Standard folder for Satellogic WS03: /Nas/cvat_data_storage/manual_backup"
  exit 1
fi

BACKUP_DIR="$1"

# 2. Validate that it exists and is a directory
if [ ! -d "$BACKUP_DIR" ]; then
  echo "Error: Backup folder not found or not a directory: $BACKUP_DIR" >&2
  echo "Standard folder for Satellogic WS03: /Nas/cvat_data_storage/manual_backup"
  exit 1
fi

# 3. Stop the Docker containers
echo "Stopping Docker containers..."
docker compose stop

# 4. Backup databases and data
echo "Backup databases and data from Docker containers..."

# 4.1 PostgreSQL data (correct)
docker run --rm --name temp_backup_db \
  -v "${project}_cvat_db":/var/lib/postgresql/data \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  tar -czvf /backup/cvat_db.tar.gz /var/lib/postgresql/data

# 4.2 CVAT application data (images + annotations)
#     — use cvat_data, not cvat_server
docker run --rm --name temp_backup_cvat_data \
  -v "${project}_cvat_data":/home/django/data \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  tar -czvf /backup/cvat_data.tar.gz /home/django/data

# 4.3 (Optional) CVAT keys
#     If you also want to back up any SSH keys or other secrets:
docker run --rm --name temp_backup_cvat_keys \
  -v "${project}_cvat_keys":/home/django/keys \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  tar -czvf /backup/cvat_keys.tar.gz /home/django/keys

# 4.4 (Optional) CVAT logs
docker run --rm --name temp_backup_cvat_logs \
  -v "${project}_cvat_logs":/home/django/logs \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  tar -czvf /backup/cvat_logs.tar.gz /home/django/logs

# 4.5 ClickHouse events database (correct)
docker run --rm --name temp_backup_clickhouse \
  -v "${project}_cvat_events_db":/var/lib/clickhouse/ \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  tar -czvf /backup/cvat_events_db.tar.gz /var/lib/clickhouse

# 5. Start the Docker containers
echo "Starting Docker containers..."
docker compose up -d

# 6. Print total elapsed time
echo " "
echo "Total execution time: $SECONDS seconds"
echo "Done!"
