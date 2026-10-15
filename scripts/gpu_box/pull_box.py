"""Bring GPU-box results home: the box's NEW ledger rows + each of those games' step outputs.

Runs on the LAPTOP. For every game the box finished since the last push (a ledger row the laptop
does not have), it moves the laptop's own outputs of that game aside (miss_season.retry_aside -
numbered, never deleted), copies the box's outputs in, then appends the box's row to the laptop
ledger under the same lock the running workers use. The board counts a game when ANY of its rows
is PASS, so a rerun that passes joins the board on the next rebuild; one that fails again changes
nothing on it.

    python scripts/gpu_box/pull_box.py 69.162.73.23 30674            # dry run: what would come home
    python scripts/gpu_box/pull_box.py 69.162.73.23 30674 --apply
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import miss_season as MS  # noqa: E402
from miss_new_park import STEP_OUTPUTS  # noqa: E402

KEY = os.path.join(os.path.expanduser("~"), ".ssh", "hostinger_vps")


def ssh(host, port, cmd, binary=False):
    p = subprocess.run(["ssh", "-4", "-p", str(port), "-i", KEY, "-o", "BatchMode=yes",
                        "ubuntu@%s" % host, cmd], capture_output=True, text=not binary)
    if p.returncode:
        sys.exit("ssh failed (%d): %s" % (p.returncode, (p.stderr or b"")[-300:]))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    assert os.path.isdir(os.path.join(ROOT, "output")), ROOT

    box = pd.read_csv(io.StringIO(ssh(a.host, a.port, "cat ~/command-cv/docs/miss_season_ledger.csv")),
                      dtype={"eye": str, "note": str})
    home = MS.read_ledger()
    seen = set(zip(home.game_pk.astype(int), home.finished_ct.astype(str)))
    new = box[[(int(g), str(t)) not in seen for g, t in zip(box.game_pk, box.finished_ct)]]
    print("box ledger %d rows, laptop %d -> %d new row(s) from the box" % (len(box), len(home), len(new)))
    if new.empty:
        return 0
    print(new[["game_pk", "venue", "verdict", "bok_gap_in", "window_used", "run_min"]].to_string(index=False))
    if not a.apply:
        print("\n(dry run - add --apply to bring them home)")
        return 0

    for r in new.itertuples(index=False):
        g = int(r.game_pk)
        names = [n.format(g=g) for n in STEP_OUTPUTS] + [
            "miss_scope_%d.txt" % g, "miss_season_marker_%d.txt" % g, "miss_window_%d.txt" % g,
            "pitch_context_%d.csv" % g]
        files = ["output/" + n for n in names] + ["output/miss_season/logs/%d.log" % g]
        listing = ssh(a.host, a.port, "cd ~/command-cv && ls -d %s 2>/dev/null; true" % " ".join(files))
        have = [f for f in listing.split() if f]
        MS.retry_aside(g)                                   # the laptop's FAIL outputs -> _aside_<g>_retryN
        raw = ssh(a.host, a.port, "cd ~/command-cv && tar -cf - %s" % " ".join(have), binary=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as t:
            for m in t.getmembers():
                assert m.isfile() and m.name.startswith("output/") and ".." not in m.name, m.name
                if m.name.endswith("/%d.log" % g):          # keep the laptop's own log; box log beside it
                    m.name = m.name[:-4] + ".gpu.log"
                t.extract(m, ROOT)
        MS.append_ledger({k: ("" if pd.isna(getattr(r, k)) else getattr(r, k))   # blank, as run_game writes
                          for k in MS.LEDGER_COLS if hasattr(r, k)})
        print("  %d  %s  %.2f in  <- %d files" % (g, r.verdict, r.bok_gap_in if r.bok_gap_in == r.bok_gap_in else float("nan"), len(have)))
    print("\ndone. The board picks up new PASSes on its next rebuild (worker 0, every 10 passes) or "
          "`python scripts/miss_season.py --board`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
