#!/usr/bin/env bash
set -euo pipefail

# 0. Start timer
SECONDS=0

# 1. Ensure backup folder is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <path_to_backup_folder>" >&2
  echo "Standard folder for Satellogic WS03: /Nas/cvat_data_storage"
  exit 1
fi

BACKUP_DIR="$1"

# 2. Validate that it exists and is a directory
if [ ! -d "$BACKUP_DIR" ]; then
  echo "Error: Backup folder not found or not a directory: $BACKUP_DIR" >&2
  echo "Standard folder for Satellogic WS03: /Nas/cvat_data_storage"
  exit 1
fi

# 3. Stop the Docker containers
echo "Stopping Docker containers..."
docker compose stop

# 4. Restore databases and data
echo "Restoring databases and data from Docker containers..."
docker run --rm --name temp_backup_db --volumes-from cvat_db -v $BACKUP_DIR:/backup ubuntu bash -c "cd /var/lib/postgresql/data && tar -xvf /backup/cvat_db.tar.gz --strip 4"
docker run --rm --name temp_backup_cvat --volumes-from cvat_server -v $BACKUP_DIR:/backup ubuntu bash -c "cd /home/django/data && tar -xvf /backup/cvat_data.tar.gz --strip 3"
docker run --rm --name temp_backup_clickhouse --volumes-from cvat_clickhouse -v $BACKUP_DIR:/backup ubuntu bash -c "cd /var/lib/clickhouse && tar -xvf /backup/cvat_events_db.tar.gz --strip 3"

# 5. Start the Docker containers
echo "Starting Docker containers..."
docker compose up -d

# 6. Print total elapsed time
echo " "
echo "Total execution time: $SECONDS seconds"
echo "Done!"