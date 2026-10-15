#!/usr/bin/env bash
# Runs on the LAPTOP (Git Bash). Puts the code + everything gitignored that a season game needs on
# the GPU box: the repo (as a git bundle - no GitHub key needed on the box), the ball weights, the
# OpenCommand glove model, every cvpull CSV, the season list, and the WORKING-TREE ledger + registry
# (the running workers change them between commits).  ~220 MB, ~15 s on home Wi-Fi.
# NOT over a phone hotspot (Zac's data plan) and NOT from stadium Wi-Fi (MLB-Press / Daikin Park
# WiFi block the box outright).
#
#   bash scripts/gpu_box/push_box.sh 69.162.73.23 30674
set -e
HOST=${1:?host}; PORT=${2:?port}
KEY="$HOME/.ssh/hostinger_vps"
CV="$(cd "$(dirname "$0")/../.." && pwd)"
SSH="ssh -4 -p $PORT -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new ubuntu@$HOST"
cd "$CV"
TMP="$(mktemp -d)"
git bundle create "$TMP/command-cv.bundle" main >/dev/null 2>&1
scp -4 -q -P "$PORT" -i "$KEY" "$TMP/command-cv.bundle" "ubuntu@$HOST:~/"
$SSH 'set -e; if [ -d ~/command-cv/.git ]; then git -C ~/command-cv fetch -q ~/command-cv.bundle main:refs/remotes/bundle/main && git -C ~/command-cv checkout -q -B main bundle/main; else git clone -q ~/command-cv.bundle ~/command-cv && git -C ~/command-cv checkout -q -B main origin/main; fi; git -C ~/command-cv log --oneline -1'
tar -cf - data/yolo_ball_runs/milb4_v4/weights/best.pt data/yolo_ball_runs/milb8r_proj/weights/best.pt \
    data/yolo_ball_runs/milb9s_segra/weights/best.pt models/od/YOLO/glove_tracking/model_weights/glove_tracking.pt \
    output/season_pull_2026.csv output/cvpull docs/miss_season_ledger.csv docs/miss_park_registry.csv \
  | $SSH 'tar -xf - -C ~/command-cv && cd ~/command-cv && echo "on box: $(ls data/yolo_ball_runs/*/weights/best.pt | wc -l) weights, $(ls output/cvpull | wc -l) pulls, ledger $(($(wc -l < docs/miss_season_ledger.csv)-1)) rows"'
rm -rf "$TMP"
