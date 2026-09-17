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


def check_provenance(rows):
    """The polynomial evaluated at zone_time must reproduce the payload's own plate_x/z."""
    worst_x = worst_z = 0.0
    n = 0
    for r in rows:
        try:
            zt = float(r["zone_time"])
            px = float(r["traj_x0"]) + float(r["traj_x1"]) * zt + float(r["traj_x2"]) * zt ** 2
            pz = float(r["traj_z0"]) + float(r["traj_z1"]) * zt + float(r["traj_z2"]) * zt ** 2
            worst_x = max(worst_x, abs(px - float(r["plate_x"])))
            worst_z = max(worst_z, abs(pz - float(r["plate_z"])))
            n += 1
        except (TypeError, ValueError, KeyError):
            continue
    assert n, "no row had a usable trajectory + plate crossing -- columns are mis-mapped"
    assert worst_x < 1e-3 and worst_z < 1e-3, (
        f"PROVENANCE FAILED: polynomial at zone_time misses the reported plate crossing by "
        f"{worst_x:.4f} ft in x, {worst_z:.4f} ft in z over {n} rows. The columns do not mean "
        f"what they are labelled. Do not use this pull.")
    return n, worst_x, worst_z


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
    n, wx, wz = check_provenance(rows)
    print(f"provenance OK: polynomial reproduces plate_x/plate_z on {n} rows "
          f"(worst {wx:.6f} ft x, {wz:.6f} ft z)")

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
