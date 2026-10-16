#!/usr/bin/env bash
# Runs ON the GPU box. Works the SEASON QUEUE (what the laptop's cv-season-w0..w2 work) with N
# parallel workers, detached. ONLY while the laptop's season workers are STOPPED: the shards are
# computed per machine, so both running at once can take the same game. Bring results home with
# pull_box.py --apply BEFORE restarting the laptop workers.
#
#   bash scripts/gpu_box/run_season.sh 4
# Progress: tail ~/season_w*.log ; per game: output/miss_season/logs/<g>.log
set -e
N=${1:-4}
cd ~/command-cv
# threads from the cgroup QUOTA (64 visible cores, 16 usable), split across workers
Q=$(awk '{ if ($1=="max") print 16; else print int($1/$2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 16)
T=$(( Q / N )); [ "$T" -ge 1 ] || T=1
for K in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T nohup bash -c \
    "date +START\ %F\ %T; ~/cv-venv/bin/python -u scripts/miss_season.py --workers $N --worker $K --setup-first --trials-per-park 3 --idle-exit-min 600 --min-free-gb 6; date +END\ %F\ %T" \
    > ~/season_w$K.log 2>&1 < /dev/null &
done
echo "started $N season workers, $T threads each (quota $Q cores)"
