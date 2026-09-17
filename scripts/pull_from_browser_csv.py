"""Download a game's clips from a CSV produced by the browser-side authed pull.

WHY THE BROWSER DOES THE AUTHED HALF. The analytics endpoint is Okta-gated and its bearer
expires in 1 hour, so every previous pull needed a fresh token pasted in by hand. Instead,
Chrome holds the signed-in session and runs the authed fetches itself, writing one CSV of
play_id + trajectory + clip URL to the Downloads folder. Nothing in this process ever
handles the credential -- it only reads the DATA that came back. The session also outlives
the 1 h bearer.

THE CLIP URLS ARE NOT OKTA-GATED (docs/2026-07-03-statsapi-video-auth.md): sporty-clips.mlb.com
is behind Cloudflare, which only rejects a non-browser User-Agent. So the mp4s download
fine from Python with `_cdn_headers()`, and that is what this script does.

PROVENANCE CHECK RUNS FIRST AND IS NOT OPTIONAL. The 9-param polynomial must reproduce the
payload's own reported plate_x / plate_z when evaluated at zone_time. On the first pull
(816698, 212 pitches) it matched to 0.000000. If that check fails, the columns have been
mis-mapped and nothing downstream is trustworthy -- this is the join-and-count discipline
from .claude/rules/index-and-time-origins.md applied to our own extraction.

DISK. Clips run 5-6.5 MB, a full game is ~1.0-1.9 GB, and this box has single-digit GB free.
--max-clips bounds a pull; check free space before starting and prune finished games
(the mp4s are disposable once frames + labels are extracted; the CSV is the durable part).

    python scripts/pull_from_browser_csv.py --csv ~/Downloads/cvpull_816698.csv --name aaa_816698
"""
import argparse
import csv as _csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ingest.video import download  # noqa: E402

TRAJ = ["traj_x0", "traj_x1", "traj_x2", "traj_y0", "traj_y1", "traj_y2",
        "traj_z0", "traj_z1", "traj_z2", "t_start", "t_end", "zone_time",
        "rel_x", "rel_y", "rel_z"]
OUT_COLS = (["stem", "game_pk", "date", "venue", "play_id", "pitch_type", "hand",
             "plate_x", "plate_z", "sz_top", "sz_bot", "call", "angle",
             "clip_downloaded", "cf_url"] + TRAJ)


MISMAP_MEDIAN_FT = 0.05      # 0.6 in -- an order below our label precision, far above noise
MISMAP_MAX_FT = 0.5          # 6 in -- a real mis-map is off by FEET


def check_provenance(rows):
    """The polynomial evaluated at zone_time must reproduce the payload's own plate_x/z.

    WHAT THIS IS FOR: catching a COLUMN MIS-MAP, where traj_z1 has landed in traj_y1 or the
    axes are transposed. That failure is gross -- the crossing would be off by FEET.

    THE THRESHOLD WAS ORIGINALLY 1e-3 ft AND THAT WAS WRONG, corrected with evidence rather
    than to make a run pass. Across six pulled games the disagreement is BINARY PER GAME:

      Isotopes / Durham / Frawley      0 rows off, max 0.000000 ft
      Riders / Hops / L.P. Frans       ~every row off, median 0.0036-0.0046, max 0.036 ft

    Same parser, same columns, different games. A mis-map on our side would be identical
    across all six and would be measured in feet; a per-GAME pattern at 0.004 ft (0.05 in,
    about a fourteenth of our ~2.5 px label precision) is MLB's own data -- those games'
    reported `location` evidently comes from a slightly different computation than the
    polynomial. So the check now separates the failure it exists to catch from numerical
    disagreement, and PRINTS the magnitude either way so a real drift cannot hide inside a
    pass. See .claude/rules/grep-guard-criteria.md on adjusting a criterion honestly.
    """
    ex, ez = [], []
    for r in rows:
        try:
            zt = float(r["zone_time"])
            px = float(r["traj_x0"]) + float(r["traj_x1"]) * zt + float(r["traj_x2"]) * zt ** 2
            pz = float(r["traj_z0"]) + float(r["traj_z1"]) * zt + float(r["traj_z2"]) * zt ** 2
            ex.append(abs(px - float(r["plate_x"])))
            ez.append(abs(pz - float(r["plate_z"])))
        except (TypeError, ValueError, KeyError):
            continue
    n = len(ex)
    assert n, "no row had a usable trajectory + plate crossing -- columns are mis-mapped"
    import statistics
    med = max(statistics.median(ex), statistics.median(ez))
    mx = max(max(ex), max(ez))
    assert med < MISMAP_MEDIAN_FT and mx < MISMAP_MAX_FT, (
        f"PROVENANCE FAILED: polynomial at zone_time misses the reported plate crossing by "
        f"median {med:.4f} ft, max {mx:.4f} ft over {n} rows. That is mis-map territory, not "
        f"numerical noise. Do not use this pull.")
    return n, med, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True, help="clip dir name, e.g. aaa_816698")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    src = os.path.expanduser(a.csv)
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    n, med, mx = check_provenance(rows)
    print(f"provenance OK on {n} rows: polynomial vs reported plate crossing "
          f"median {med:.6f} ft, max {mx:.6f} ft "
          f"({'exact' if mx == 0 else 'within MLB data noise'})")

    rows = [r for r in rows if r.get("cf_url")]
    if a.max_clips:
        rows = rows[:a.max_clips]
    out_dir = os.path.join(ROOT, "data", "clips", a.name)
    os.makedirs(out_dir, exist_ok=True)

    free_gb = shutil.disk_usage(ROOT).free / 2 ** 30
    est_gb = len(rows) * 5.8 / 1024
    print(f"{len(rows)} clips to fetch, ~{est_gb:.2f} GB estimated; {free_gb:.1f} GB free")
    assert free_gb - est_gb > 1.5, (
        f"REFUSING: that would leave under 1.5 GB free. Prune a finished game's mp4s first "
        f"(the CSV is the durable artifact, the clips are not).")

    for i, r in enumerate(rows):
        r["stem"] = f"{a.name}_{i:04d}"

    def get(r):
        p = os.path.join(out_dir, r["stem"] + ".mp4")
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            return r["stem"], True, "cached"
        try:
            download(r["cf_url"], p)
            return r["stem"], True, ""
        except Exception as e:                       # noqa: BLE001
            return r["stem"], False, type(e).__name__

    ok = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(get, r): r for r in rows}
        for k, fu in enumerate(as_completed(futs), 1):
            stem, good, note = fu.result()
            futs[fu]["clip_downloaded"] = int(good)
            ok += good
            if k % 25 == 0 or not good:
                print(f"  [{k}/{len(rows)}] {stem} {'ok' if good else 'FAIL ' + note}",
                      flush=True)

    man = os.path.join(out_dir, f"{a.name}_manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    used = sum(os.path.getsize(os.path.join(out_dir, p)) for p in os.listdir(out_dir)
               if p.endswith(".mp4")) / 2 ** 30
    print(f"\n{ok}/{len(rows)} clips downloaded, {used:.2f} GB on disk")
    print(man)
    print(f"disk now {shutil.disk_usage(ROOT).free / 2**30:.1f} GB free")


if __name__ == "__main__":
    main()
