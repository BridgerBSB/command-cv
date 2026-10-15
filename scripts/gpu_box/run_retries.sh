#!/usr/bin/env bash
# Runs ON the GPU box. Reruns a list of FAILED season games with N parallel workers, detached.
# The laptop never reruns a FAIL on its own (done_games counts FAIL as done), so a box working the
# failures and the laptop working new games can never take the same game.
#
#   bash scripts/gpu_box/run_retries.sh 822501,822514,... 4
# Progress: tail ~/retries_w*.log ; per game: output/miss_season/logs/<g>.log
set -e
GAMES=${1:?comma-separated gamePks}; N=${2:-4}
cd ~/command-cv
# threads from the cgroup QUOTA (64 visible cores, 16 usable), split across workers
Q=$(awk '{ if ($1=="max") print 16; else print int($1/$2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 16)
T=$(( Q / N )); [ "$T" -ge 1 ] || T=1
for K in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T nohup bash -c \
    "date +START\ %F\ %T; ~/cv-venv/bin/python -u scripts/miss_season.py --games $GAMES --retry-games $GAMES --workers $N --worker $K --min-free-gb 6; date +END\ %F\ %T" \
    > ~/retries_w$K.log 2>&1 < /dev/null &
done
echo "started $N workers, $T threads each (quota $Q cores) on $(echo $GAMES | tr ',' '\n' | wc -l) games"
