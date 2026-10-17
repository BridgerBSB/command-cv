#!/usr/bin/env bash
# Runs on the LAPTOP (Git Bash). Everything the GPU box needs to run retarget_season.py: the code,
# weights, cvpull CSVs and ledger (push_box.sh), then each PASS game's camera/ball inputs (~181 MB
# for 300 games, 2026-10-17) and the done-markers (command_miss_<g>_bo_hold.csv) of games already
# switched here, so the box never redoes one.
#
#   bash scripts/gpu_box/push_retarget.sh 69.162.73.23 30674
set -e
HOST=${1:?host}; PORT=${2:?port}
KEY="$HOME/.ssh/hostinger_vps"
CV="$(cd "$(dirname "$0")/../.." && pwd)"
SSH="ssh -4 -p $PORT -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new ubuntu@$HOST"
bash "$CV/scripts/gpu_box/push_box.sh" "$HOST" "$PORT"
cd "$CV"
LIST="$(mktemp)"
"$HOME/.venvs/cv-rocm/Scripts/python.exe" - "$LIST" <<'PY'
import os, sys
import pandas as pd
led = pd.read_csv("docs/miss_season_ledger.csv", dtype=str)
names = ["detect_tracks_%d_v4.csv", "stab2_H_%dc.csv", "park_pose_%d_ballonly.npz", "detect_tracks_%d_v4stab2c.csv",
         "holdout_%d.csv", "command_miss_%d_bo.csv", "pitch_context_%d.csv", "command_miss_%d_bo_hold.csv"]
files = [("output/" + n % g) for g in sorted({int(x) for x in led.loc[led.verdict == "PASS", "game_pk"]})
         for n in names if os.path.exists("output/" + n % g)]
open(sys.argv[1], "w").write("\n".join(files) + "\n")
print("%d per-game files" % len(files))
PY
tar -cf - -T "$LIST" | $SSH 'tar -xf - -C ~/command-cv && cd ~/command-cv && echo "on box: $(ls output/stab2_H_*c.csv | wc -l) games with a camera, $(ls output/command_miss_*_bo_hold.csv 2>/dev/null | wc -l) already switched"'
rm -f "$LIST"
