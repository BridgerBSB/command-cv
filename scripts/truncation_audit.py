"""Which banked games had their ball flights CUT OFF by the scan window.

WHY THIS EXISTS. The scan window is `--lo/--hi` on rolling_pull and
detect_ball_tracks. When `--hi` lands before a park's plate crossing, chains
stop at the ceiling instead of at the plate, and the loss is silent: the clips
still produce labels, the labels are still on the ball, there is just less of
each flight and some clips yield nothing at all. On 2026-09-19 Greensboro's
plate turned out to be at frame 277 against a window ending at 250 (3 of 33
clips), and Constellation went from 31% to 47% yield off that one change. Both
had been diagnosed as gate failures before anyone measured the window.

THE INSTRUMENT. For every clip, take the LAST frame of its chain. A flight
that ends because the ball reached the plate stops wherever the plate is. A
flight that ends because it ran out of window stops exactly AT the ceiling. So
the fraction of clips whose chain ends at the ceiling separates the two, and
the p90 of the ending frame says how much headroom is left.

  healthy     p90 end sits well below the ceiling, <3% of clips at it
              (AdventHealth 821925: ceiling 221, p90 217, 2%)
  truncating  p90 end IS the ceiling, double-digit % piled on it
              (Vystar 816447: ceiling 300, p50 260, p90 300, 18%)

WHAT IT DOES NOT TELL YOU. A chain ending at the ceiling is not proof of a real
flight -- a junk track can also run to the ceiling. This ranks games for
re-running, it does not certify any single clip. And it cannot see a park whose
window is so short that no chain forms at all; that park shows up as a low clip
count, not as truncation. Read the clip count beside the percentage.

  python scripts/truncation_audit.py
  python scripts/truncation_audit.py --min-pct 8     # only what is worth re-running
"""
import argparse
import collections
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def venue_lookup(cvpull_dir):
    """game_pk -> venue, from the banked pull CSVs. Missing is fine, printed as '?'."""
    out = {}
    for path in glob.glob(os.path.join(cvpull_dir, "*.csv")):
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("game_pk") and row.get("venue"):
                        out[row["game_pk"]] = row["venue"]
                    break
        except OSError:
            continue
    return out


def audit_one(path):
    """-> (n_clips, ceiling, n_at_ceiling, p50_end, p90_end) or None if empty."""
    last = collections.defaultdict(int)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frame = int(row["frame"])
            if frame > last[row["stem"]]:
                last[row["stem"]] = frame
    if not last:
        return None
    ends = sorted(last.values())
    ceiling = ends[-1]
    # within 1 of the ceiling, not equal to it: a chain can lose its final frame
    # to the gate and still have been truncated.
    at = sum(1 for v in ends if v >= ceiling - 1)
    return len(ends), ceiling, at, ends[len(ends) // 2], ends[int(len(ends) * 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--cvpull-dir", default=os.path.join(ROOT, "output", "cvpull"))
    ap.add_argument("--min-pct", type=float, default=0.0,
                    help="only show games with at least this %% of chains on the ceiling")
    ap.add_argument("--warn-pct", type=float, default=8.0,
                    help="at or above this, the game is called TRUNCATING")
    # A percentage over 7 clips is one clip. Without this, a game where a single
    # junk chain ran to the ceiling reads 14%% and outranks a 184-clip game that
    # is genuinely losing a fifth of its flights.
    ap.add_argument("--min-clips", type=int, default=30,
                    help="below this, report the number but never call it TRUNCATING")
    a = ap.parse_args()

    venue = venue_lookup(a.cvpull_dir)
    files = sorted(glob.glob(os.path.join(a.labels_dir, "det_labels_*.csv")))
    # The denominator, printed whether or not anything is found -- a scan that
    # looked at nothing and a scan that found nothing are otherwise identical.
    print(f"scanned {len(files)} det_labels_*.csv in {a.labels_dir}")
    if not files:
        print("NOTHING TO AUDIT -- wrong directory, or no game has been labelled yet.")
        return

    rows = []
    for path in files:
        game = os.path.basename(path)[len("det_labels_"):-len(".csv")]
        res = audit_one(path)
        if res is None:
            continue
        n, ceiling, at, p50, p90 = res
        rows.append((100.0 * at / n, game, venue.get(game.split("_")[0], "?"),
                     n, ceiling, at, p50, p90))

    shown = [r for r in rows if r[0] >= a.min_pct]
    print(f"{len(rows)} games carry labels; {len(shown)} at or above --min-pct {a.min_pct}\n")
    print(f"{'game':>10}  {'venue':<24} {'clips':>5} {'ceil':>5} {'@ceil':>10} "
          f"{'p50end':>6} {'p90end':>6}  verdict")
    for pct, game, ven, n, ceiling, at, p50, p90 in sorted(shown, reverse=True):
        if n < a.min_clips:
            verdict = f"too few clips (n={n}) -- no verdict"
        elif pct >= a.warn_pct:
            verdict = "TRUNCATING -- re-run wider"
        else:
            verdict = "ok"
        print(f"{game:>10}  {ven[:24]:<24} {n:>5} {ceiling:>5} {at:>4} {pct:>4.0f}% "
              f"{p50:>6} {p90:>6}  {verdict}")

    bad = [r for r in shown if r[0] >= a.warn_pct and r[3] >= a.min_clips]
    if bad:
        print(f"\n{len(bad)} game(s) still truncating. Delete their det_labels_*.csv "
              f"first or rolling_pull skips them, then re-run at a --hi well past the\n"
              f"current ceiling and re-audit: the percentage should fall to low single "
              f"digits, which is what an untruncated park looks like.")
    else:
        print("\nNo game is piling chains on its ceiling.")


if __name__ == "__main__":
    main()
