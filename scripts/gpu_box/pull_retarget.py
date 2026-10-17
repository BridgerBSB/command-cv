"""Bring the GPU box's retarget results home: every game the box SWITCHED (has
command_miss_<g>_bo_hold.csv there) and the laptop has not. Never overwrites a game the laptop
already switched. Copies the switched miss file, the kept old one, the new set-ups/targets, and
appends the box's per-game log rows to output/retarget_season_box.csv.

    python scripts/gpu_box/pull_retarget.py 69.162.73.23 30674            # dry run
    python scripts/gpu_box/pull_retarget.py 69.162.73.23 30674 --apply
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
OUT = os.path.join(ROOT, "output")
KEY = os.path.join(os.path.expanduser("~"), ".ssh", "hostinger_vps")
FILES = ["command_miss_%d_bo.csv", "command_miss_%d_bo_hold.csv", "command_miss_%d_bo_oc.csv",
         "setups_%d_oc.csv", "targets_%d_stab2c_oc.csv"]


def ssh(host, port, cmd, binary=False):
    p = subprocess.run(["ssh", "-4", "-p", str(port), "-i", KEY, "-o", "BatchMode=yes", "ubuntu@%s" % host, cmd],
                       capture_output=True, text=not binary)
    if p.returncode:
        sys.exit("ssh failed (%d): %s" % (p.returncode, (p.stderr or b"")[-300:]))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    box = {int(s.split("_")[2]) for s in ssh(a.host, a.port, "cd ~/command-cv/output && ls command_miss_*_bo_hold.csv 2>/dev/null || true").split()}
    here = {g for g in box if os.path.exists(os.path.join(OUT, "command_miss_%d_bo_hold.csv" % g))}
    new = sorted(box - here)
    print("box switched %d game(s); %d already switched here; %d to bring home" % (len(box), len(here), len(new)))
    logs = ssh(a.host, a.port, "cd ~/command-cv/output && cat retarget_season_w*.csv 2>/dev/null || true")
    if not a.apply or not new:
        print("(dry run)" if not a.apply else "nothing to do")
        return
    names = " ".join(f % g for g in new for f in FILES)
    blob = ssh(a.host, a.port, "cd ~/command-cv/output && tar -cf - %s 2>/dev/null || true" % names, binary=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        members = t.getmembers()
        assert all(os.path.basename(m.name) == m.name for m in members), "unexpected path in the box tar"
        t.extractall(OUT, members=members)
    got = {g for g in new if os.path.exists(os.path.join(OUT, "command_miss_%d_bo_hold.csv" % g))}
    assert got == set(new), "missing after copy: %s" % sorted(set(new) - got)
    rows = [l for l in logs.splitlines() if l and not l.startswith("game_pk")]
    keep = [l for l in rows if l.split(",")[0].isdigit() and int(l.split(",")[0]) in set(new)]
    p = os.path.join(OUT, "retarget_season_box.csv")
    hdr = next((l for l in logs.splitlines() if l.startswith("game_pk")), None)
    need_hdr = bool(hdr) and (not os.path.exists(p) or os.path.getsize(p) == 0)
    with open(p, "a", encoding="utf-8") as f:
        if need_hdr:
            f.write(hdr + "\n")
        f.write("\n".join(keep) + ("\n" if keep else ""))
    print("brought home %d game(s), %d log row(s) -> %s" % (len(got), len(keep), p))


if __name__ == "__main__":
    main()
