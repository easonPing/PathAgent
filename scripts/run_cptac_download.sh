#!/usr/bin/env bash
set -u

project_root=/sfs/weka/scratch/efk4ps/PathAgent
ruby_root=/apps/software/standard/core/ruby/3.4.3
gem_root="$project_root/.tools/ruby-gems"
driver_log="$project_root/data/raw/cptac/download_lists/download_driver.log"

cd "$project_root" || exit 1
mkdir -p "$project_root/data/raw/cptac/download_lists"
export GEM_HOME="$gem_root"
export GEM_PATH="$gem_root"
export PATH="$gem_root/bin:$ruby_root/bin:$PATH"
export LD_LIBRARY_PATH="$ruby_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec >>"$driver_log" 2>&1
echo "[$(date --iso-8601=seconds)] Starting/resuming SlideBench-VQA-CPTAC WSI download"
python scripts/download_cptac_wsi.py all --workers 4
status=$?
echo "[$(date --iso-8601=seconds)] Downloader exit status: $status"
exit "$status"
