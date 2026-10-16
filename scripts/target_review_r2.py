"""Round-2 blind target review: grade the RELEASE-ANCHORED set-up rule.

Round 1 (scripts/target_review.py, Zac 2026-10-15) graded the longest-hold rule: 30 of 117
charted targets wrong, 27 of them from the wrong moment. scripts/setup_rule_test.py then
checked a replacement against those answers on saved glove tracks. Passing that test proves
the new rule MOVES the wrong ones and LEAVES the good ones; it cannot say the new spots are
right, and the rule was designed while looking at round-1 items. This sheet answers both:

  r1_wrong_time  round-1 wrong-moment items the new rule re-picked (moved > MOVE_PX)
  r1_ok_moved    round-1 OK items the new rule moved (a regression check; ideally none)
  fresh          pitches round 1 never sampled, from the same PASS-game pool. Their
                 wrong-rate is the honest rate for the new rule.

Shuffled together, stratum only in key.csv (blind). Each clip is downloaded once, the glove is
read up to the ball's first frame (src.auto_glove detect_setup_catch, anchor_frame), and the
pick is rendered at its own frame with the same crop and ring as round 1. The fresh sample is
FROZEN on first run (pick.csv), for the reason target_review.py gives.

    python scripts/target_review_r2.py --n-fresh 80
    -> output/target_review_r2/index.html (+ a copy on Zac's Desktop)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.auto_glove import detect_setup_catch  # noqa: E402
from src.ingest.video import download  # noqa: E402
import target_review as r1  # noqa: E402  (candidates(), PAGE, crop size: one copy of each)

OUT = os.path.join(ROOT, "output", "target_review_r2")
DESK = r"C:\Users\zbrid\OneDrive\Desktop\command-cv-target-review-r2"
RULE_TEST = os.path.join(ROOT, "output", "target_review", "rule_test.csv")
GLOVE = os.path.join(ROOT, "models", "od", "YOLO", "glove_tracking", "model_weights", "glove_tracking.pt")
KEEP_PX, MOVE_PX = 5.0, 10.0          # setup_rule_test.py's marks
# The setup_rule_test.py variant whose settings equal src.auto_glove's defaults. The live pick
# below uses those defaults; the per-item assert proves the two agree.
RULE = "primary"


def anchors(game):
    p = os.path.join(ROOT, "output", "detect_tracks_%d_v4.csv" % game)
    if not os.path.exists(p):
        return {}, None
    s = pd.read_csv(p, usecols=["stem", "frame"]).groupby("stem").frame.min()
    return s.to_dict(), float(s.median())


def fetch(url):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    for attempt in range(4):                          # the CDN throttles bursts
        try:
            download(url, tmp.name)
            return tmp.name
        except Exception:                             # noqa: BLE001
            if attempt == 3:
                os.remove(tmp.name)
                return None
            time.sleep((3, 10, 30)[attempt])


def crop_ring(clip, frame_idx, x, y):
    cap = cv2.VideoCapture(clip)
    fi, frame = -1, None
    while fi < frame_idx:                             # sequential, never seeked
        ok, fr = cap.read()
        if not ok:
            break
        fi, frame = fi + 1, fr
    cap.release()
    if frame is None or fi != frame_idx:
        return None, None, None
    h, w = frame.shape[:2]
    x0 = int(min(max(x - r1.CROP_W / 2, 0), max(w - r1.CROP_W, 0)))
    y0 = int(min(max(y - r1.CROP_H / 2, 0), max(h - r1.CROP_H, 0)))
    crop = frame[y0:y0 + r1.CROP_H, x0:x0 + r1.CROP_W].copy()
    c = (int(round(x - x0)), int(round(y - y0)))
    cv2.circle(crop, c, 20, (31, 110, 235), 2, cv2.LINE_AA)       # Astros orange, BGR, as round 1
    cv2.circle(crop, c, 2, (31, 110, 235), -1, cv2.LINE_AA)
    return crop, x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fresh", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    assert os.path.exists(RULE_TEST), "run scripts/setup_rule_test.py first"
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    pool, n_games = r1.candidates()
    print("pool: %d PASS games, %d targets" % (n_games, len(pool)))

    rt = pd.read_csv(RULE_TEST)
    px = rt[RULE + "_px"]
    old = pd.concat([rt[(rt.verdict == "wrong_time") & (px > MOVE_PX)].assign(stratum="r1_wrong_time"),
                     rt[(rt.verdict == "ok") & (px > KEEP_PX)].assign(stratum="r1_ok_moved")])
    print("round-1 items re-graded: %s" % old.stratum.value_counts().to_dict())

    frozen = os.path.join(OUT, "pick.csv")
    if os.path.exists(frozen):
        pick = pd.read_csv(frozen)
        print("frozen sample reused: %d items" % len(pick))
    else:
        seen = set(pd.read_csv(os.path.join(ROOT, "output", "target_review", "pick.csv")).stem)
        fresh = pool[~pool.stem.isin(seen)].sample(a.n_fresh, random_state=a.seed)
        pick = pd.concat([old[["game", "stem", "stratum", "rid"]].rename(columns={"game": "game_pk", "rid": "r1_rid"}),
                          fresh[["game_pk", "stem"]].assign(stratum="fresh")], ignore_index=True)
        pick = pick.sample(frac=1, random_state=a.seed).reset_index(drop=True)
        pick["rid"] = range(1, len(pick) + 1)
        pick.to_csv(frozen, index=False)
    pick = pick.merge(pool[["game_pk", "stem", "cf_url", "venue", "date", "pitch_type"]],
                      on=["game_pk", "stem"], how="left")
    assert pick.cf_url.notna().all(), "a picked pitch left the pool"

    from ultralytics import YOLO
    model = YOLO(GLOVE)
    anc = {g: anchors(g) for g in pick.game_pk.unique()}
    want = {(r.r1_rid): r for r in rt.itertuples()} if "r1_rid" in pick else {}
    items, skipped, keyrows = [], {}, []
    recs = pick.to_dict("records")
    with ThreadPoolExecutor(3) as ex:                 # downloads run ahead; the GPU stays serial
        for it, clip in zip(recs, ex.map(lambda r: fetch(r["cf_url"]), recs)):
            meta = os.path.join(OUT, "img", "%d.json" % it["rid"])
            if os.path.exists(meta):
                items.append(json.load(open(meta, encoding="utf-8")))
                if clip:
                    os.remove(clip)
                continue
            if clip is None:
                skipped["download failed"] = skipped.get("download failed", 0) + 1
                continue
            try:
                own, med = anc[it["game_pk"]]
                anchor = own.get(it["stem"], med)
                if anchor is None:
                    skipped["no ball tracks for the game"] = skipped.get("no ball tracks for the game", 0) + 1
                    continue
                r = detect_setup_catch(clip, model, setup_only=True, anchor_frame=int(anchor))
                if not r.get("ok"):
                    skipped["new rule: " + r["reason"].split(" ")[0]] = skipped.get(
                        "new rule: " + r["reason"].split(" ")[0], 0) + 1
                    continue
                x, y, f = r["setup"]
                if not pd.isna(it.get("r1_rid", np.nan)):          # same pick the saved-track test made?
                    t = want[int(it["r1_rid"])]
                    d = float(np.hypot(x - getattr(t, RULE + "_x"), y - getattr(t, RULE + "_y")))
                    assert d <= 1.0, "rid %d: live pick %.1f px from the test's" % (it["rid"], d)
                crop, x0, y0 = crop_ring(clip, int(f), x, y)
                if crop is None:
                    skipped["frame not decoded"] = skipped.get("frame not decoded", 0) + 1
                    continue
                name = "%d.jpg" % it["rid"]
                cv2.imwrite(os.path.join(OUT, "img", name), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                out = dict(rid=int(it["rid"]), img="img/" + name, x0=x0, y0=y0, w=crop.shape[1],
                           h=crop.shape[0], game=int(it["game_pk"]), stem=it["stem"], frame=int(f),
                           tx=round(x, 2), ty=round(y, 2),
                           label="%s %s, %s" % (it["venue"], it["date"], it["pitch_type"]))
                json.dump(out, open(meta, "w", encoding="utf-8"))
                items.append(out)
            finally:
                if clip and os.path.exists(clip):
                    os.remove(clip)
            print("  %d / %d rendered" % (len(items), len(recs)), flush=True)
    print("rendered %d of %d; skipped %s" % (len(items), len(recs), skipped))
    done = {i["rid"]: i for i in items}
    k = pick[pick.rid.isin(done)][["rid", "stratum", "game_pk", "stem"]].copy()
    k["frame"] = [done[r]["frame"] for r in k.rid]
    k["x"] = [done[r]["tx"] for r in k.rid]
    k["y"] = [done[r]["ty"] for r in k.rid]
    k.to_csv(os.path.join(OUT, "key.csv"), index=False)

    page = r1.PAGE
    for old_s, new_s in (('"target_review_v1"', '"target_review_r2"'),
                         ('"target_review_results.json"', '"target_review_r2_results.json"'),
                         ("<title>Target Review</title>", "<title>Target Review 2</title>")):
        assert old_s in page, old_s                   # a replace that matches nothing is a silent no-op
        page = page.replace(old_s, new_s)
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(
        page.replace("__ITEMS__", json.dumps(sorted(items, key=lambda i: i["rid"]))))
    print("-> %s" % os.path.join(OUT, "index.html"))
    if os.path.isdir(os.path.dirname(DESK)):
        shutil.copytree(OUT, DESK, ignore=shutil.ignore_patterns("key.csv", "pick.csv"),
                        dirs_exist_ok=True)           # blind: no key on the Desktop
        print("-> %s" % os.path.join(DESK, "index.html"))


if __name__ == "__main__":
    main()
