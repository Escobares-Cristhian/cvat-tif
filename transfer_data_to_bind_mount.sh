#!/usr/bin/env bash
# transfer_data_to_bind_mount.sh

# 0. Declare your volume suffixes in a list
volumes=( db data keys logs clickhouse events_db)

# 1. Ensure backup folder is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <path_to_backup_folder>" >&2
  echo "Standard folder for Satellogic WS03: /Nas/cvat_data_storage/manual_backup"
  exit 1
fi

main_dir="$1"

# 2. Validate that it exists and is a directory
if [ ! -d "$main_dir" ]; then
  echo "Error: Backup folder not found or not a directory: $main_dir" >&2
  echo "Standard folder for Satellogic WS03: /Nas/cvat_data_storage/manual_backup"
  exit 1
fi

# If exist, ddelate all content. And create the subfolders.
# If not existing, create the folder.
if [ -d "${main_dir}" ]; then
  echo "Deleting all content in ${main_dir}..."
  rm -rf "${main_dir:?}"/*
  echo "Creating subfolders..."
  for vol in "${volumes[@]}"; do
    mkdir -p "${main_dir}/cvat_${vol}"
  done

else
  echo "Creating directory ${main_dir}..."
  mkdir -p "${main_dir}"
fi


# 3. Figure out the Compose project name.
#    - If you set COMPOSE_PROJECT_NAME in an .env, that’ll be used.
#    - Otherwise docker-compose will default to the cwd basename.
project="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}"

# 4. Iterate over each and copy into your NAS path
for vol in "${volumes[@]}"; do
  src="${project}_cvat_${vol}"               # e.g. "cvat-tif_cvat_db"
  dst="${events_db}/cvat_${vol}"
  echo "Copying $src → $dst …"

  docker run --rm \
    -v "${src}":/from \
    -v "${dst}":/to \
    ubuntu:24.04 \
    bash -c "cp -r /from/. /to/"
done

# # Postgres official image runs as UID 999
# sudo chown -R 999:999 ${events_db}/cvat_db

# # ClickHouse official image runs as UID 101
# sudo chown -R 101:101 ${events_db}/cvat_events_db