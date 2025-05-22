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

# 4. Backup databases and data
echo "Backup databases and data from Docker containers..."
docker run --rm --name temp_backup_db -v "$project"_cvat_db:/var/lib/postgresql/data -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_db.tar.gz /var/lib/postgresql/data
docker run --rm --name temp_backup_cvat -v "$project"_cvat_server:/home/django/data -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_data.tar.gz /home/django/data
docker run --rm --name temp_backup_clickhouse -v "$project"_cvat_events_db:/var/lib/clickhouse/ -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_events_db.tar.gz /var/lib/clickhouse

# 5. Start the Docker containers
echo "Starting Docker containers..."
docker compose up -d

# 6. Print total elapsed time
echo " "
echo "Total execution time: $SECONDS seconds"
echo "Done!"