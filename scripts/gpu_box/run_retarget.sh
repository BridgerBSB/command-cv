#!/usr/bin/env bash
# Runs ON the GPU box. Starts retarget_season.py shards FIRST..LAST of TOTAL, detached (nohup), so
# the laptop can run the other shards at the same time without ever touching the same game
# (a game belongs to shard game_pk % TOTAL). Each worker holds at most 2 games of clips on disk
# and stops under 20 GB free (retarget_season.py). Logs: ~/retarget_w<K>.log
#
#   bash scripts/gpu_box/run_retarget.sh 6 1 5       # box = shards 1-5 of 6, laptop = shard 0
set -e
TOTAL=${1:?total}; FIRST=${2:?first}; LAST=${3:?last}
cd ~/command-cv
N=$((LAST - FIRST + 1))
Q=$(awk '{ if ($1=="max") print 16; else print int($1/$2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 16)
T=$(( Q / N )); [ "$T" -ge 1 ] || T=1
for K in $(seq "$FIRST" "$LAST"); do
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T nohup bash -c \
    "date +START\ %F\ %T; ~/cv-venv/bin/python -u scripts/retarget_season.py --workers $TOTAL --worker $K; date +END\ %F\ %T" \
    > ~/retarget_w$K.log 2>&1 < /dev/null &
done
echo "started shards $FIRST-$LAST of $TOTAL, $T threads each (quota $Q cores)"
