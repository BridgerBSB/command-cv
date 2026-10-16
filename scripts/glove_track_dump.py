"""Save the RAW glove track of every target-review clip, once, so set-up rules can be tested
with no GPU and no re-download.

Why: the set-up rule in src/auto_glove.py picks one hold out of the glove's path. Zac graded
180 of its picks (output/target_review/key.csv + Downloads/target_review_results*.json) and
a quarter of the charted ones came from the WRONG MOMENT. Testing a replacement rule needs
the whole path each pick was chosen from, and setups_<g>.csv keeps only the pick.

Reads every frame of each clip (no early stop), same model, same sampling (step 3) and same
sequential decode as the season run, so the old rule re-derived from this track must land on
key.csv's pick. scripts/setup_rule_test.py checks that first.

Writes output/target_review/glove_tracks/<rid>.csv (frame,x,y,conf) + <rid>.json (n_frames,
stem, game). Resumable: an item already written is skipped.

    python scripts/glove_track_dump.py [--only 26,45] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.auto_glove import _glove_track  # noqa: E402
from src.ingest.video import download  # noqa: E402  (browser UA + MLB Referer; a bare GET is 403)

KEY = os.path.join(ROOT, "output", "target_review", "key.csv")
OUT = os.path.join(ROOT, "output", "target_review", "glove_tracks")
GLOVE = os.path.join(ROOT, "models", "od", "YOLO", "glove_tracking", "model_weights", "glove_tracking.pt")


def clip_url(game: int, stem: str) -> str | None:
    man = os.path.join(ROOT, "data", "clips", "pull_%d" % game, "pull_%d_manifest.csv" % game)
    if not os.path.exists(man):
        return None
    m = pd.read_csv(man, usecols=["stem", "cf_url"])
    hit = m.loc[m.stem == stem, "cf_url"]
    return str(hit.iloc[0]) if len(hit) and str(hit.iloc[0]).startswith("http") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated rids")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    assert os.path.exists(KEY), KEY
    assert os.path.exists(GLOVE), GLOVE
    os.makedirs(OUT, exist_ok=True)
    key = pd.read_csv(KEY)
    if a.only:
        key = key[key.rid.isin([int(r) for r in a.only.split(",")])]
    if a.limit:
        key = key.head(a.limit)

    from ultralytics import YOLO
    model = YOLO(GLOVE)
    done = failed = 0
    t_all = time.time()
    for _, r in key.iterrows():
        rid, game, stem = int(r.rid), int(r.game_pk), str(r.stem)
        csv_p, meta_p = os.path.join(OUT, "%d.csv" % rid), os.path.join(OUT, "%d.json" % rid)
        if os.path.exists(meta_p):
            done += 1
            continue
        url = clip_url(game, stem)
        if url is None:
            print("rid %d  %s: no clip URL in the manifest" % (rid, stem), flush=True)
            failed += 1
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        t0 = time.time()
        try:
            for attempt in range(4):              # the CDN throttles bursts (target_review.py)
                try:
                    download(url, tmp.name)
                    break
                except Exception:                 # noqa: BLE001
                    if attempt == 3:
                        raise
                    time.sleep((3, 10, 30)[attempt])
            track, n = _glove_track(tmp.name, model)
        except Exception as e:                    # noqa: BLE001
            print("rid %d  %s: FAILED %s" % (rid, stem, e), flush=True)
            failed += 1
            continue
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        pd.DataFrame(track, columns=["frame", "x", "y", "conf"]).to_csv(csv_p, index=False,
                                                                        float_format="%.6g")
        json.dump(dict(rid=rid, game=game, stem=stem, n_frames=int(n), n_track=len(track)),
                  open(meta_p, "w"))
        done += 1
        print("rid %3d  %s  %d glove frames of %d  %.1f s   [%d done, %d failed]"
              % (rid, stem, len(track), n, time.time() - t0, done, failed), flush=True)
    print("FINISHED: %d of %d written, %d failed, %.1f min" % (done, len(key), failed,
                                                               (time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
