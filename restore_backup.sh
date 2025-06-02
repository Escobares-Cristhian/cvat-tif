#!/usr/bin/env bash
set -euo pipefail

# 0. Start timer
SECONDS=0

# 0. Figure out the Compose project name.
#    - If you set COMPOSE_PROJECT_NAME in an .env, that’ll be used.
#    - Otherwise docker-compose will default to the cwd basename.
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

# 4. Restore databases and data
echo "Restoring databases and data from Docker containers..."

# 4.1 PostgreSQL data
docker run --rm --name temp_restore_db \
  -v "${project}_cvat_db":/var/lib/postgresql/data \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  bash -c "cd /var/lib/postgresql/data && tar -xvf /backup/cvat_db.tar.gz --strip 4"

# 4.2 CVAT application data (images + annotations)
docker run --rm --name temp_restore_cvat_data \
  -v "${project}_cvat_data":/home/django/data \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  bash -c "cd /home/django/data && tar -xvf /backup/cvat_data.tar.gz --strip 3"

# 4.3 CVAT keys
docker run --rm --name temp_restore_cvat_keys \
  -v "${project}_cvat_keys":/home/django/keys \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  bash -c "cd /home/django/keys && tar -xvf /backup/cvat_keys.tar.gz --strip 3"

# 4.4 CVAT logs
docker run --rm --name temp_restore_cvat_logs \
  -v "${project}_cvat_logs":/home/django/logs \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  bash -c "cd /home/django/logs && tar -xvf /backup/cvat_logs.tar.gz --strip 3"

# 4.5 ClickHouse events database
docker run --rm --name temp_restore_clickhouse \
  -v "${project}_cvat_events_db":/var/lib/clickhouse \
  -v "$BACKUP_DIR":/backup \
  ubuntu \
  bash -c "cd /var/lib/clickhouse && tar -xvf /backup/cvat_events_db.tar.gz --strip 3"

# 5. Start the Docker containers
echo "Starting Docker containers..."
docker compose up -d

# 6. Print total elapsed time
echo " "
echo "Total execution time: $SECONDS seconds"
echo "Done!"
