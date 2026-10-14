"""One screen: is the season run alive, how far along, and what needs a person.

Built 2026-10-13 so a check-in costs a few lines, not a session of log reading.
The only hand step the season run has is the eye check of a setup park's PASSING trial
(--eye-ok + --ready); this lists exactly those, with the render to look at.

    python scripts/season_status.py
"""
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "docs", "miss_season_ledger.csv")
REGISTRY = os.path.join(ROOT, "docs", "miss_park_registry.csv")
CARDS = os.path.join(ROOT, "output", "season_cards", "index.csv")


def workers():
    """Live miss_season.py launchers, by worker id, from the process table."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine "
          "-match 'miss_season' -and $_.CommandLine -match 'venvs' } | ForEach-Object { $_.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception as e:                           # noqa: BLE001
        return "unknown (%s)" % e
    ids = sorted(l.split("--worker ")[1].split()[0] for l in out.splitlines() if "--worker " in l)
    return ", ".join("w" + i for i in ids) or "NONE RUNNING"


def last_game(w):
    p = os.path.join(ROOT, "output", "detached", "cv-season-w%d.log" % w)
    if not os.path.exists(p):
        return "-"
    lines = [l for l in open(p, encoding="utf-8", errors="replace") if l.startswith("== ")]
    return lines[-1][3:60].strip() if lines else "-"


def main():
    led = pd.read_csv(LEDGER, dtype={"eye": str})
    last = led.drop_duplicates("game_pk", keep="last")
    reg = pd.read_csv(REGISTRY)
    print("workers alive: %s" % workers())
    for w in range(3):
        print("  w%d on %s" % (w, last_game(w)))
    print("games graded %d: %d PASS, %d FAIL; last finished %s"
          % (len(last), (last.verdict == "PASS").sum(), (last.verdict != "PASS").sum(), led.finished_ct.iloc[-1]))
    print("parks: %s" % ", ".join("%s %d" % kv for kv in reg.status.value_counts().items()))
    if os.path.exists(CARDS):
        ix = pd.read_csv(CARDS)
        print("cards: %d of %d in-org pitchers, %d of %d pitches measured (%.0f%%)"
              % ((ix.card.fillna("") != "").sum(), len(ix), ix.measured.sum(), ix.pitches.sum(),
                 100 * ix.measured.sum() / max(ix.pitches.sum(), 1)))
    du = shutil.disk_usage(ROOT)
    print("disk free %.1f GB" % (du.free / 2 ** 30))
    setup = reg[reg.status == "setup"].venue_id
    todo = last[(last.role == "setup") & (last.verdict == "PASS") & (last.eye.fillna("") == "")
                & last.venue_id.isin(setup)]
    if len(todo):
        print("NEEDS AN EYE CHECK (then --eye-ok <game> and --ready <venue_id>):")
        for r in todo.itertuples(index=False):
            print("  %d %s (venue %d) held-out %.2f in -> output/park_pose_%d_ref.png"
                  % (r.game_pk, r.venue, r.venue_id, r.bok_gap_in, r.game_pk))
    else:
        print("nothing waiting on an eye check")


if __name__ == "__main__":
    sys.exit(main())
