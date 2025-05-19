#!/usr/bin/env bash
set -euo pipefail

# 0. Start timer
SECONDS=0

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
docker run --rm --name temp_backup_db --volumes-from cvat_db -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_db.tar.gz /var/lib/postgresql/data
docker run --rm --name temp_backup_cvat --volumes-from cvat_server -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_data.tar.gz /home/django/data
docker run --rm --name temp_backup_keys --volumes-from cvat_keys -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_keys.tar.gz /home/django/keys
docker run --rm --name temp_backup_logs --volumes-from cvat_logs -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_logs.tar.gz /home/django/logs
docker run --rm --name temp_backup_clickhouse --volumes-from cvat_clickhouse -v "$BACKUP_DIR":/backup ubuntu tar -czvf /backup/cvat_events_db.tar.gz /var/lib/clickhouse

# 5. Start the Docker containers
echo "Starting Docker containers..."
docker compose up -d

# 6. Print total elapsed time
echo " "
echo "Total execution time: $SECONDS seconds"
echo "Done!"