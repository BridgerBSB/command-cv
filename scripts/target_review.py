"""Blind review sheet for the catcher's SET-UP TARGET - the one input no tracking system can check.

Zac, 2026-10-15: "I can annotate and tell you the errors later and we can train." The per-pitch
B_minus_A_in error covers the camera + ball; nothing covers the glove target except the
plausibility screen. This samples set-up targets from PASS games, renders each at its own set-up
frame with the target ringed, and writes a one-page review: click OK if the ring is on the
catcher's glove, click the real glove if it is not, or mark "can't tell".

TWO STRATA, SHUFFLED TOGETHER SO THE REVIEW IS BLIND:
  plausible   - targets that reach the board. Their wrong-rate IS the board's target error rate.
  implausible - targets the screen threw out. Tells us what the screen is catching.
The stratum of each item lives in key.csv, NOT in the page.

Video is re-fetched from the pull manifests' public sporty-clips URLs (no token), one clip at a
time, and deleted after its frame is read. Frames are read SEQUENTIALLY from 0, never seeked -
the project's frame-index convention (index-and-time-origins.md #2), the one setups_<g>.csv uses.

    python scripts/target_review.py --n-plaus 120 --n-implaus 60
    -> output/target_review/index.html (+ a copy on Zac's Desktop)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import sys

import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ingest.video import download  # noqa: E402  (browser UA + MLB Referer; a bare GET is 403)
OUT = os.path.join(ROOT, "output")
LEDGER = os.path.join(ROOT, "docs", "miss_season_ledger.csv")
DESK = r"C:\Users\zbrid\OneDrive\Desktop\command-cv-target-review"
CROP_W, CROP_H = 640, 400


def candidates():
    led = pd.read_csv(LEDGER, dtype={"eye": str, "note": str})
    games = sorted(set(led.loc[led.verdict == "PASS", "game_pk"].astype(int)))
    rows = []
    for g in games:
        mp = os.path.join(OUT, "command_miss_%d_bo.csv" % g)
        sp = os.path.join(OUT, "setups_%d.csv" % g)
        man = os.path.join(ROOT, "data", "clips", "pull_%d" % g, "pull_%d_manifest.csv" % g)
        if not (os.path.exists(mp) and os.path.exists(sp) and os.path.exists(man)):
            continue
        m = pd.read_csv(mp)[["stem", "plaus", "pitch_type"]]
        s = pd.read_csv(sp)
        u = pd.read_csv(man, usecols=["stem", "cf_url", "venue", "date"])
        d = m.merge(s, on="stem").merge(u, on="stem")
        d["game_pk"] = g
        rows.append(d)
    c = pd.concat(rows, ignore_index=True)
    return c[c.cf_url.astype(str).str.startswith("http")], len(games)


FAILS = {}


def render(item, img_dir):
    """Download one clip, read sequentially to the set-up frame, crop around the target, ring it.
    Resumable: an item rendered by an earlier run is reused from its img/<rid>.json."""
    meta = os.path.join(img_dir, "%s.json" % item["rid"])
    if os.path.exists(meta):
        return json.load(open(meta, encoding="utf-8"))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        for attempt in range(4):                      # the CDN throttles bursts (2026-10-15: 141/180 failed)
            try:
                download(item["cf_url"], tmp.name)
                break
            except Exception:                         # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep((3, 10, 30)[attempt])
        cap = cv2.VideoCapture(tmp.name)
        want, fi, frame = int(item["frame"]), -1, None
        while fi < want:
            ok, fr = cap.read()
            if not ok:
                break
            fi += 1
            frame = fr
        cap.release()
        if frame is None or fi != want:
            return None
        h, w = frame.shape[:2]
        x, y = float(item["x"]), float(item["y"])
        x0 = int(min(max(x - CROP_W / 2, 0), max(w - CROP_W, 0)))
        y0 = int(min(max(y - CROP_H / 2, 0), max(h - CROP_H, 0)))
        crop = frame[y0:y0 + CROP_H, x0:x0 + CROP_W].copy()
        cx, cy = int(round(x - x0)), int(round(y - y0))
        cv2.circle(crop, (cx, cy), 20, (31, 110, 235), 2, cv2.LINE_AA)      # Astros orange, BGR
        cv2.circle(crop, (cx, cy), 2, (31, 110, 235), -1, cv2.LINE_AA)
        name = "%s.jpg" % item["rid"]
        cv2.imwrite(os.path.join(img_dir, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out = dict(rid=item["rid"], img="img/" + name, x0=x0, y0=y0, w=crop.shape[1], h=crop.shape[0],
                    game=int(item["game_pk"]), stem=item["stem"], frame=want,
                    tx=round(x, 2), ty=round(y, 2), label="%s %s, %s" % (item["venue"], item["date"],
                                                                          item["pitch_type"]))
        json.dump(out, open(meta, "w", encoding="utf-8"))
        return out
    except Exception as e:                                             # noqa: BLE001
        k = type(e).__name__ + ": " + str(e)[:60]
        FAILS[k] = FAILS.get(k, 0) + 1
        return None
    finally:
        os.remove(tmp.name)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Target Review</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#14213d;--mut:#5b6474;--navy:#002D62;--orange:#EB6E1F;--ok:#1b7f3b;--bad:#b3261e}
body{margin:0;font:14px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:var(--navy);color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:2}
header b{color:var(--orange)} button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid #ccd;background:#fff;cursor:pointer}
main{padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr))}
.card{background:var(--card);border-radius:10px;box-shadow:0 1px 3px #0002;padding:10px}
.card.ok{outline:3px solid var(--ok)}.card.wrong{outline:3px solid var(--bad)}.card.unclear{outline:3px solid #999}
.imgw{position:relative;cursor:crosshair}.imgw img{width:100%;display:block;border-radius:6px}
.fix{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border:3px solid var(--bad);border-radius:50%;pointer-events:none}
.row{display:flex;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap}.lab{color:var(--mut);font-size:12px;flex:1}
</style></head><body>
<header><span>Target review: is the <b>orange ring</b> on the catcher's glove where he SET UP?</span>
<span id="cnt"></span><button id="dl">Download results</button>
<span style="font-size:12px;opacity:.8">Right = <b>OK</b> &middot; wrong = click the real glove &middot; saves as you go</span></header>
<main id="m"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "target_review_v1";
let R = {}; try { R = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} paint(); }
function paint(){
  const n = Object.keys(R).length; document.getElementById("cnt").textContent = n + " of " + ITEMS.length + " reviewed";
  for (const it of ITEMS){ const c = document.getElementById("c" + it.rid), r = R[it.rid];
    c.className = "card" + (r ? " " + r.verdict : "");
    const f = c.querySelector(".fix"); if (r && r.verdict === "wrong"){ f.style.display = "block";
      f.style.left = (100 * (r.fx - it.x0) / it.w) + "%"; f.style.top = (100 * (r.fy - it.y0) / it.h) + "%"; } else f.style.display = "none"; }
}
const m = document.getElementById("m");
for (const it of ITEMS){
  const c = document.createElement("div"); c.id = "c" + it.rid; c.className = "card";
  c.innerHTML = '<div class="imgw"><img loading="lazy" src="' + it.img + '"><div class="fix" style="display:none"></div></div>' +
    '<div class="row"><span class="lab">#' + it.rid + ' &middot; ' + it.label + '</span>' +
    '<button data-v="ok">OK</button><button data-v="unclear">Can\\'t tell</button><button data-v="clear">Undo</button></div>';
  c.querySelector(".imgw").addEventListener("click", e => { const b = e.currentTarget.getBoundingClientRect();
    R[it.rid] = {verdict: "wrong", fx: +(it.x0 + (e.clientX - b.left) / b.width * it.w).toFixed(1),
                 fy: +(it.y0 + (e.clientY - b.top) / b.height * it.h).toFixed(1)}; save(); });
  c.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    if (b.dataset.v === "clear") delete R[it.rid]; else R[it.rid] = {verdict: b.dataset.v}; save(); }));
  m.appendChild(c);
}
document.getElementById("dl").addEventListener("click", () => {
  const out = ITEMS.map(it => Object.assign({rid: it.rid, game: it.game, stem: it.stem, frame: it.frame, tx: it.tx, ty: it.ty}, R[it.rid] || {verdict: ""}));
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "target_review_results.json"; a.click(); });
paint();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-plaus", type=int, default=120)
    ap.add_argument("--n-implaus", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(OUT, "target_review"))
    a = ap.parse_args()
    c, n_games = candidates()
    plaus = c[c.plaus.astype(bool)]
    impl = c[~c.plaus.astype(bool)]
    print("pool: %d PASS games, %d targets (%d plausible / %d implausible)"
          % (n_games, len(c), len(plaus), len(impl)))
    pick = pd.concat([plaus.sample(min(a.n_plaus, len(plaus)), random_state=a.seed).assign(stratum="plausible"),
                      impl.sample(min(a.n_implaus, len(impl)), random_state=a.seed).assign(stratum="implausible")])
    pick = pick.sample(frac=1, random_state=a.seed).reset_index(drop=True)       # blind: shuffle strata
    pick["rid"] = range(1, len(pick) + 1)
    img_dir = os.path.join(a.out, "img")
    os.makedirs(img_dir, exist_ok=True)
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda r: render(r, img_dir), pick.to_dict("records")))
    items = [r for r in res if r]
    print("rendered %d of %d (%d clip(s) failed to fetch or decode)" % (len(items), len(pick), len(pick) - len(items)))
    for k, v in sorted(FAILS.items(), key=lambda kv: -kv[1]):
        print("   %3d x %s" % (v, k))
    pick[pick.rid.isin([i["rid"] for i in items])][
        ["rid", "stratum", "game_pk", "stem", "frame", "x", "y", "plaus", "pitch_type", "venue", "date"]
    ].to_csv(os.path.join(a.out, "key.csv"), index=False)
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(PAGE.replace("__ITEMS__", json.dumps(items)))
    print("-> %s" % os.path.join(a.out, "index.html"))
    if os.path.isdir(os.path.dirname(DESK)):
        # copy OVER the top: OneDrive holds the folder open, so deleting it first fails (WinError 5)
        shutil.copytree(a.out, DESK, ignore=shutil.ignore_patterns("key.csv"), dirs_exist_ok=True)  # blind
        print("-> %s" % os.path.join(DESK, "index.html"))


if __name__ == "__main__":
    main()
