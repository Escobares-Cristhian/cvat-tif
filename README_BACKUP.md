# README

These two scripts let you create and restore manual backups of a Satellogic CVAT deployment. They only work together if both the source (make_backup) and target (restore_backup) CVAT instances are the **same version**.  You can upgrade the target CVAT to a newer version before restoring by following the official upgrade instructions.


## Prerequisites

- Docker & docker-compose installed and configured
- Identical CVAT versions on backup and restore environments
- Writable backup directory (e.g. `/Nas/cvat_data_storage/manual_backup`)

---


## make_backup.sh

Creates compressed archives of:

1. **PostgreSQL database** (`cvat_db.tar.gz`)
2. **CVAT data** (`cvat_data.tar.gz`)
3. **ClickHouse events** (`cvat_events_db.tar.gz`)


**Usage**
```bash
./make_backup.sh <backup_folder_path>
```

## restore_backup.sh

Restores the three compressed archives into a CVAT deployment. Ensure the target CVAT instance is at the same version before running.

**Usage**
```bash
./restore_backup.sh <backup_folder_path>
