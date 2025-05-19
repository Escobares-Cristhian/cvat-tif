#!/usr/bin/env bash
# transfer_data_to_bind_mount.sh

# 0. Select folder of storage
# main_dir="/Nas/cvat_data_storage/current_storage"
main_dir="/share/cescobares/cvat_data_storage/current_storage"

# If not existing, create the folder
if [ ! -d "${main_dir}" ]; then
  mkdir -p "${main_dir}"
fi

# 1. Figure out the Compose project name.
#    - If you set COMPOSE_PROJECT_NAME in an .env, that’ll be used.
#    - Otherwise docker-compose will default to the cwd basename.
project="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}"

# 2. Declare your volume suffixes in a list
volumes=( db data keys logs clickhouse events_db)

# 3. Iterate over each and copy into your NAS path
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

# Postgres official image runs as UID 999
sudo chown -R 999:999 ${events_db}/cvat_db

# ClickHouse official image runs as UID 101
sudo chown -R 101:101 ${events_db}/cvat_events_db