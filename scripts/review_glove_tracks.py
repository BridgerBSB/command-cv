"""Save the every-frame glove track of each GRADED clip-review pitch, so set-up rules can be
re-scored against Zac's 2026-10-16 grades offline, without another sheet.

target_review_clips.py drew the track into the video and kept only the two picks. This re-runs
the same detector pass (same clip, same end frame, same model and settings) and writes
output/target_review_clips/tracks/<rid>.csv (frame, x, y, conf). It ASSERTS both picks come
back exactly as graded (docs/target_review/clips_key_2026-10-16.csv), so a saved track is the
track the grades were made on.

    python scripts/review_glove_tracks.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.auto_glove import _glove_track, _select_setup_anchored, anchored_setup_point  # noqa: E402
import target_review as r1  # noqa: E402
import target_review_r2 as r2  # noqa: E402
import target_review_clips as rc  # noqa: E402

KEY = os.path.join(ROOT, "docs", "target_review", "clips_key_2026-10-16.csv")
TRK = os.path.join(rc.OUT, "tracks")


def main():
    os.makedirs(TRK, exist_ok=True)
    k = pd.read_csv(KEY)
    pool, _ = r1.candidates()
    url = pool.set_index(["game_pk", "stem"]).cf_url
    k["cf_url"] = [url.loc[(g, s)] for g, s in zip(k.game_pk, k.stem)]
    todo = k[[not os.path.exists(os.path.join(TRK, "%d.csv" % r)) for r in k.rid]].to_dict("records")
    print("%d graded pitches, %d tracks to build" % (len(k), len(todo)), flush=True)
    from ultralytics import YOLO
    model = YOLO(r2.GLOVE)
    ppi = {g: rc.px_per_inch(g) for g in k.game_pk.unique()}
    t0, bad = time.time(), []
    with ThreadPoolExecutor(3) as ex:
        for i, (it, clip) in enumerate(zip(todo, ex.map(lambda r: r2.fetch(r["cf_url"]), todo)), 1):
            if clip is None:
                bad.append((it["rid"], "download failed"))
                continue
            try:
                end = int(it["anchor"]) + int(round(rc.POST_S * it["fps"]))
                track1, _ = _glove_track(clip, model, step=1, stop=lambda tr, fi: fi >= end)
            finally:
                os.remove(clip)
            reason, run = _select_setup_anchored([p for p in track1 if p[0] % 3 == 0], int(it["anchor"]),
                                                 rc.RADIUS, rc.MAX_GAP, rc.MIN_LEN)
            ox, oy, of = anchored_setup_point(run)
            assert np.hypot(ox - it["ours_x"], oy - it["ours_y"]) <= 0.5 and of == it["ours_frame"], it["rid"]
            oc = rc.oc_pick(track1, int(it["anchor"]), it["fps"], ppi[it["game_pk"]])
            if pd.isna(it["oc_x"]):
                assert oc is None, it["rid"]                  # graded as 'no OpenCommand target'
            else:
                assert oc and np.hypot(oc[0] - it["oc_x"], oc[1] - it["oc_y"]) <= 0.5 \
                    and oc[2] == it["oc_frame"], it["rid"]
            pd.DataFrame(track1, columns=["frame", "x", "y", "conf"]).to_csv(
                os.path.join(TRK, "%d.csv" % it["rid"]), index=False)
            print("  %d / %d  rid %d  %d detections  %.0f s" % (i, len(todo), it["rid"], len(track1),
                                                             time.time() - t0), flush=True)
    print("done: %d tracks, failures %s, %.1f min" % (len(os.listdir(TRK)), bad, (time.time() - t0) / 60))
    json.dump({"failures": bad}, open(os.path.join(TRK, "_run.json"), "w"))


if __name__ == "__main__":
    main()
