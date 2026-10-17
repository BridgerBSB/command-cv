"""Round-2 CLIP sheet: the same 134 pitches as target_review_r2.py, as playable clips.

WHY (Zac, 2026-10-16, partway through round 2): a still frame cannot show the glove settle.
Round 2 pictured every pick at the hold's LAST frame (round 1's middle frame had read as too
early, rid 52), and the last frame of a hold is usually mid-delivery. 36 of his first 48
grades came back Wrong moment, and at least some of those are a verdict on the PICTURE, not
on the spot - in his #16 the glove is still where the catcher set it while the pitcher
throws. A clip from before the hold to the ball shows when the glove settled, and the grade
becomes the only thing that matters for a miss distance: is the ring on the target SPOT.

EACH ITEM
  * The clip from before the set-up hold to just after the ball is first seen, with the
    glove detection drawn on EVERY frame as a thin white circle. That is how OpenCommand's
    README gif shows it (artifacts/rogers_sinker.gif: "thin white circle: catcher glove
    detection"); they draw no settle marker, and their rule below does not look for one.
  * Two rings, A and B, which rule gets which letter is random per item (key.csv only), so
    the comparison is blind:
      ours        release-anchored set-up (src/auto_glove._select_setup_anchored): the
                  median spot over the longest hold still going within 60 frames before
                  the ball, started >= 30 before it. Same pick as round 2, asserted to 1 px.
      opencommand v1.1's naive target, tomdoyo/open-command src/target_inference.py
                  select_target (read 2026-10-16, commit d96ed0a): the HIGHEST glove in
                  [release - 2.0 s, release - 0.3 s], discounted 15 in per second before
                  the window's last detection, so a late target beats an earlier, higher
                  one. Recomputed here on OUR glove detections.
  * A timeline: a tick per ring at the frame its rule read, and the ball's first frame.
  * Pause and click the glove to mark the true spot (x, y and the frame) when neither ring
    is right. Marks are scored offline, so a later rule can be checked without a new sheet.

PORTING OPENCOMMAND'S RULE (now src.auto_glove.select_setup_opencommand) - their step-2 teleport
cleaning comes first (clean_teleports;
without it "highest glove" picks the pitcher's glove), then three differences, all stated:
  * release: we do not have their release_s. The ball's first frame in flight (the
    detect_tracks chain start, the same anchor our rule uses) stands in for it, which is a
    few frames after release, so their window lands that much later here.
  * inches: their height and discount are world inches. Converted with the game's
    px-per-inch slope of target_py on target_z_in in command_miss_<g>_bo.csv (median
    residual 0.0-0.3 px on the first 12 games checked, slopes 3.3-5.8 px/in).
  * every frame: they detect every frame and so does this pass. Our rule reads every 3rd
    (auto_glove._glove_track step 3); fi % 3 == 0 of the same sequential pass IS that track.
  v1.2's per-pitcher model (glove dependence + offset, empirical Bayes) is downstream of the
  per-clip pick and needs a pitcher's season, so it is not something a clip can show.

    python scripts/target_review_clips.py               (needs target_review_r2.py run first)
    python scripts/target_review_clips.py --limit 3     (a timing / look test)
    -> output/target_review_clips/index.html (+ a copy on Zac's Desktop)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.auto_glove import (_glove_track, _select_setup_anchored, anchored_setup_point,  # noqa: E402
                            select_setup_opencommand, OC_WINDOW_S)
import target_review as r1  # noqa: E402  (candidates(): the pool with each pitch's clip URL)
import target_review_r2 as r2  # noqa: E402  (anchors(), fetch(), GLOVE: one copy of each)

R2 = os.path.join(ROOT, "output", "target_review_r2")
OUT = os.path.join(ROOT, "output", "target_review_clips")
DESK = r"C:\Users\Owner\OneDrive\Desktop\command-cv-target-review-clips"
# imageio-ffmpeg's bundled binary. Installed in the system Python, not in cv-rocm; this laptop
# only (the work laptop blocks unapproved executables - CLAUDE.md IT constraints).
FFMPEG = r"C:\Users\Owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"

OUT_W, OUT_H = 640, 400         # round 1/2's crop size, so a ring sits at the same scale
MARGIN = 110                    # px kept around both rings when they are far apart
SAME_PX = 10.0                  # rings closer than this are drawn as one (A=B); half a ring radius
# auto_glove.detect_setup_catch's defaults, which round 2's live pick used
RADIUS, MAX_GAP, MIN_LEN = 22.0, 8, 4
PRE_S, POST_S, MAX_S = 0.4, 0.3, 6.0   # clip shown: from PRE_S before the earliest pick to POST_S past the ball


def px_per_inch(game):
    """|d target_py / d target_z_in| over the game's pitches (see module docstring)."""
    c = pd.read_csv(os.path.join(ROOT, "output", "command_miss_%d_bo.csv" % game)).dropna(
        subset=["target_py", "target_z_in"])
    if len(c) < 20:
        return None
    return abs(float(np.polyfit(c.target_z_in, c.target_py, 1)[0]))


def oc_pick(track1, anchor, fps, ppi):
    """OpenCommand's pick (src.auto_glove.select_setup_opencommand, the one copy), or None."""
    reason, setup = select_setup_opencommand(track1, anchor, fps, ppi)
    return None if reason else setup


def crop_box(points, w, h):
    """(x0, y0, cw, ch): at least OUT_W x OUT_H, OUT_W:OUT_H aspect, holding every point."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    cw = max(OUT_W, max(xs) - min(xs) + 2 * MARGIN)
    ch = max(OUT_H, max(ys) - min(ys) + 2 * MARGIN)
    cw, ch = max(cw, ch * OUT_W / OUT_H), max(ch, cw * OUT_H / OUT_W)
    cw, ch = min(cw, w), min(ch, h)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    x0 = int(min(max(cx - cw / 2, 0), w - cw))
    y0 = int(min(max(cy - ch / 2, 0), h - ch))
    return x0, y0, int(cw), int(ch)


def write_clip(clip, path, f0, f1, box, track1, fps):
    """Frames f0..f1 (sequential decode from 0, never seeked), cropped, glove circle drawn."""
    x0, y0, cw, ch = box
    s = OUT_W / cw
    det = {p[0]: p for p in track1}
    ff = subprocess.Popen([FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", "%dx%d" % (OUT_W, OUT_H), "-r", "%.3f" % fps, "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "6",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                          stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(clip)
    fi, n = -1, 0
    while fi < f1:
        if fi + 1 < f0:                                # before the window: decode, do not keep
            if not cap.grab():
                break
            fi += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        im = cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        if fi in det:
            c = (int(round((det[fi][1] - x0) * s)), int(round((det[fi][2] - y0) * s)))
            cv2.circle(im, c, 15, (255, 255, 255), 1, cv2.LINE_AA)
        ff.stdin.write(im.tobytes())
        n += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    return n


def build(it, clip, model, anc, ppi, rng_flip):
    own, med = anc
    anchor = int(own.get(it["stem"], med))
    cap = cv2.VideoCapture(clip)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    end = anchor + int(round(POST_S * fps))
    t0 = time.time()
    track1, _ = _glove_track(clip, model, step=1, stop=lambda tr, fi: fi >= end)
    t_det = time.time() - t0
    track3 = [p for p in track1 if p[0] % 3 == 0]                  # == _glove_track(step=3)
    reason, run = _select_setup_anchored(track3, anchor, RADIUS, MAX_GAP, MIN_LEN)
    assert reason is None, "rid %d: round 2 rendered this pick, now: %s" % (it["rid"], reason)
    ox, oy, of = anchored_setup_point(run)
    d = float(np.hypot(ox - it["tx"], oy - it["ty"]))
    assert d <= 1.0 and of == it["frame"], "rid %d: pick %.1f px / f%d vs round 2's f%d" % (
        it["rid"], d, of, it["frame"])
    oc = oc_pick(track1, anchor, fps, ppi)
    pts = [(ox, oy)] + ([oc[:2]] if oc else [])
    box = crop_box(pts, w, h)
    f0 = max(0, int(min(run[0][0], oc[2] if oc else anchor, anchor - OC_WINDOW_S * fps) - PRE_S * fps))
    f0 = max(f0, end - int(MAX_S * fps))
    vid = "vid/%d.mp4" % it["rid"]
    n = write_clip(clip, os.path.join(OUT, vid), f0, end, box, track1, fps)
    x0, y0, cw, ch = box
    pct = lambda x, y: (round(100 * (x - x0) / cw, 3), round(100 * (y - y0) / ch, 3))  # noqa: E731
    rules = {"ours": dict(x=ox, y=oy, frame=of, hold=[run[0][0], run[-1][0]])}
    if oc:
        rules["opencommand"] = dict(x=oc[0], y=oc[1], frame=oc[2])
    order = ["ours", "opencommand"] if not rng_flip else ["opencommand", "ours"]
    order = [r for r in order if r in rules]
    same = bool(oc is not None and float(np.hypot(ox - oc[0], oy - oc[1])) < SAME_PX)
    rings = []
    for k, rule in zip("AB", order):
        px, py = pct(rules[rule]["x"], rules[rule]["y"])
        rings.append(dict(k=k, px=px, py=py, frame=rules[rule]["frame"]))
    if same:
        rings = [dict(rings[0], k="A=B")]
    page = dict(rid=int(it["rid"]), vid=vid, label=it["label"], fps=round(fps, 3), f0=f0, n=n,
                anchor=anchor, x0=x0, y0=y0, cw=cw, ch=ch, rings=rings)
    key = dict(rid=int(it["rid"]), game_pk=int(it["game"]), stem=it["stem"], A=order[0],
               B=order[1] if len(order) > 1 else "", same=same, anchor=anchor, fps=fps,
               ppi=ppi, ours_x=ox, ours_y=oy, ours_frame=of, hold_start=run[0][0],
               hold_end=run[-1][0], oc_x=oc[0] if oc else np.nan, oc_y=oc[1] if oc else np.nan,
               oc_frame=oc[2] if oc else np.nan, f0=f0, n_frames=n, det_s=round(t_det, 1))
    return page, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N items only (timing / look test)")
    ap.add_argument("--seed", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "vid"), exist_ok=True)
    pick = pd.read_csv(os.path.join(R2, "pick.csv"))[["rid", "stratum"]]
    items = [json.load(open(os.path.join(R2, "img", "%d.json" % r), encoding="utf-8"))
             for r in pd.read_csv(os.path.join(R2, "key.csv")).rid]       # the 134 round 2 rendered
    pool, _ = r1.candidates()
    url = pool.set_index(["game_pk", "stem"]).cf_url
    for it in items:
        it["cf_url"] = url.loc[(it["game"], it["stem"])]
    if a.limit:
        items = items[:a.limit]
    flip = dict(zip([i["rid"] for i in items],
                    np.random.default_rng(a.seed).random(len(items)) < 0.5))
    from ultralytics import YOLO
    model = YOLO(r2.GLOVE)
    games = sorted({i["game"] for i in items})
    anc = {g: r2.anchors(g) for g in games}
    ppi = {g: px_per_inch(g) for g in games}
    pages, keys, skipped = [], [], {}
    t_start = time.time()
    with ThreadPoolExecutor(3) as ex:                  # downloads run ahead; the GPU stays serial
        for it, clip in zip(items, ex.map(lambda r: r2.fetch(r["cf_url"]), items)):
            meta = os.path.join(OUT, "vid", "%d.json" % it["rid"])
            if os.path.exists(meta):
                m = json.load(open(meta, encoding="utf-8"))
                pages.append(m["page"])
                keys.append(m["key"])
                if clip:
                    os.remove(clip)
                continue
            if clip is None:
                skipped["download failed"] = skipped.get("download failed", 0) + 1
                continue
            try:
                page, key = build(it, clip, model, anc[it["game"]], ppi[it["game"]], flip[it["rid"]])
            finally:
                os.remove(clip)
            json.dump({"page": page, "key": key}, open(meta, "w", encoding="utf-8"), default=float)
            pages.append(page)
            keys.append(key)
            print("  %d / %d  rid %d  %d frames, detect %.1f s, %.0f s elapsed"
                  % (len(pages), len(items), it["rid"], key["n_frames"], key["det_s"],
                     time.time() - t_start), flush=True)
    print("built %d of %d; skipped %s; %.1f min" % (len(pages), len(items), skipped,
                                                    (time.time() - t_start) / 60))
    k = pd.DataFrame(keys).merge(pick, on="rid", how="left")
    k.to_csv(os.path.join(OUT, "key.csv"), index=False)
    print("one ring (A=B): %d   no OpenCommand target: %d   two rings: %d"
          % (k.same.sum(), (k.B == "").sum(), ((~k.same) & (k.B != "")).sum()))
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(
        PAGE.replace("__ITEMS__", json.dumps(sorted(pages, key=lambda p: p["rid"]))))
    print("-> %s" % os.path.join(OUT, "index.html"))
    if not a.limit and os.path.isdir(os.path.dirname(DESK)):
        shutil.copytree(OUT, DESK, ignore=shutil.ignore_patterns("key.csv", "*.json"),
                        dirs_exist_ok=True)            # blind: no key, no per-item rule names
        print("-> %s" % os.path.join(DESK, "index.html"))


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Target Clips</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#14213d;--mut:#5b6474;--navy:#002D62;--orange:#EB6E1F;--cyan:#22b8cf;--ok:#1b7f3b;--bad:#b3261e}
body{margin:0;font:14px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:var(--navy);color:#fff;padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header b.a{color:var(--orange)} header b.b{color:var(--cyan)}
button{font:inherit;padding:5px 10px;border-radius:6px;border:1px solid #ccd;background:#fff;cursor:pointer}
button.on{background:var(--navy);color:#fff;border-color:var(--navy)}
main{padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr))}
.card{background:var(--card);border-radius:10px;box-shadow:0 1px 3px #0002;padding:10px}
.card.v-a,.card.v-b,.card.v-both{outline:3px solid var(--ok)}.card.v-neither{outline:3px solid var(--bad)}.card.v-unclear{outline:3px solid #999}
.vw{position:relative;cursor:crosshair;aspect-ratio:640/400;background:#000;border-radius:6px;overflow:hidden}
.vw video{width:100%;height:100%;display:block}
.ring{position:absolute;width:6.25%;aspect-ratio:1;transform:translate(-50%,-50%);border:3px solid var(--orange);border-radius:50%;pointer-events:none}
.ring.b{border-color:var(--cyan)} .ring span{position:absolute;left:105%;top:-14px;font:700 13px system-ui;color:var(--orange);text-shadow:0 0 3px #000}
.ring.b span{color:var(--cyan)} .hide .ring{display:none}
.mark{position:absolute;width:14px;height:14px;transform:translate(-50%,-50%);border:3px solid var(--bad);background:#fff8;border-radius:50%;pointer-events:none;display:none}
canvas.tl{width:100%;height:30px;display:block;margin-top:6px;cursor:pointer}
.row{display:flex;gap:6px;margin-top:6px;align-items:center;flex-wrap:wrap}.lab{color:var(--mut);font-size:12px;flex:1;min-width:160px}
.t{font:12px ui-monospace,monospace;color:var(--mut)}
</style></head><body>
<header><span>Which ring sits where the catcher <b>gave the target</b>: <b class="a">A</b> or <b class="b">B</b>?</span>
<span id="cnt"></span><button id="dl">Download results</button>
<span style="font-size:12px;opacity:.85;flex-basis:100%">Thin white circle = the glove the detector saw that frame. Timeline ticks = the frame each ring was read at; red = ball first seen.
Neither right = pause on the moment the target is set and <b>click the glove</b> (that marks the true spot). Saves as you go.</span></header>
<main id="m"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "target_review_clips_v1";
let R = {}; try { R = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(R)); } catch (e) {} paint(); }
const COL = {"A": "#EB6E1F", "A=B": "#EB6E1F", "B": "#22b8cf"};
function frameOf(it, v){ return it.f0 + Math.min(it.n - 1, Math.max(0, Math.floor(v.currentTime * it.fps + 1e-3))); }
function seek(it, v, f){ v.pause(); v.currentTime = (Math.min(it.n - 1, Math.max(0, f - it.f0)) + 0.5) / it.fps; }
function drawTL(it, c, v){
  const cv = c.querySelector("canvas"), W = cv.clientWidth, H = 30, dpr = window.devicePixelRatio || 1;
  if (cv.width !== W * dpr){ cv.width = W * dpr; cv.height = H * dpr; }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const X = f => 4 + (W - 8) * (f - it.f0) / Math.max(1, it.n - 1);
  g.fillStyle = "#e3e7ee"; g.fillRect(4, 12, W - 8, 6);
  g.fillStyle = "#b3261e"; g.fillRect(X(it.anchor) - 1, 4, 3, 22);
  g.font = "10px system-ui"; g.fillText("ball", X(it.anchor) + 4, 10);
  for (const r of it.rings){ g.fillStyle = COL[r.k]; g.fillRect(X(r.frame) - 2, 6, 4, 18); }
  const m = (R[it.rid] || {}).mark; if (m){ g.strokeStyle = "#b3261e"; g.lineWidth = 2; g.strokeRect(X(m.frame) - 3, 8, 6, 14); }
  if (v){ g.fillStyle = "#14213d"; g.fillRect(X(frameOf(it, v)), 2, 1.5, 26); }
}
function paint(){
  let n = 0;
  for (const it of ITEMS){ const c = document.getElementById("c" + it.rid), r = R[it.rid];
    if (r && r.verdict) n++;
    c.className = "card" + (r && r.verdict ? " v-" + r.verdict : "");
    c.querySelectorAll("button[data-v]").forEach(b => b.classList.toggle("on", !!r && r.verdict === b.dataset.v));
    const mk = c.querySelector(".mark");
    if (r && r.mark){ mk.style.display = "block"; mk.style.left = (100 * (r.mark.x - it.x0) / it.cw) + "%"; mk.style.top = (100 * (r.mark.y - it.y0) / it.ch) + "%"; }
    else mk.style.display = "none";
    drawTL(it, c, c.querySelector("video")); }
  document.getElementById("cnt").textContent = n + " of " + ITEMS.length + " graded";
}
const m = document.getElementById("m");
const io = new IntersectionObserver(es => { for (const e of es){ const v = e.target;
  if (e.isIntersecting){ if (!v.src) v.src = v.dataset.src; v.playbackRate = +(v.dataset.rate || 0.5); v.play().catch(() => {}); } else v.pause(); } }, {threshold: 0.4});
for (const it of ITEMS){
  const c = document.createElement("div"); c.id = "c" + it.rid; c.className = "card";
  const two = it.rings.length === 2;
  const rings = it.rings.map(r => '<div class="ring' + (r.k === "B" ? " b" : "") + '" style="left:' + r.px + '%;top:' + r.py + '%"><span>' + r.k + '</span></div>').join("");
  const verdicts = two ? [["a", "A right"], ["b", "B right"], ["both", "Both right"], ["neither", "Neither"], ["unclear", "Can\\'t tell"]]
                       : [["both", "Right"], ["neither", "Wrong"], ["unclear", "Can\\'t tell"]];
  c.innerHTML = '<div class="vw"><video muted loop playsinline preload="none" data-src="' + it.vid + '"></video>' + rings + '<div class="mark"></div></div>' +
    '<canvas class="tl"></canvas>' +
    '<div class="row"><button data-c="play">Pause</button><button data-c="back">&#9664; frame</button><button data-c="fwd">frame &#9654;</button>' +
    '<button data-c="rate">0.5x</button><button data-c="rings">Hide rings</button><span class="t"></span></div>' +
    '<div class="row"><span class="lab">#' + it.rid + ' &middot; ' + it.label + '</span>' +
    verdicts.map(([v, t]) => '<button data-v="' + v + '">' + t + '</button>').join("") + '<button data-v="clear">Undo</button></div>';
  m.appendChild(c);
  const v = c.querySelector("video"), t = c.querySelector(".t"), pb = c.querySelector('[data-c="play"]');
  io.observe(v);
  const tick = () => { t.textContent = "frame " + frameOf(it, v) + "  (" + ((frameOf(it, v) - it.anchor) / it.fps).toFixed(2) + " s to ball)"; drawTL(it, c, v); };
  v.addEventListener("timeupdate", tick); v.addEventListener("seeked", tick);
  v.addEventListener("play", () => { pb.textContent = "Pause"; (function loop(){ if (!v.paused){ tick(); requestAnimationFrame(loop); } })(); });
  v.addEventListener("pause", () => { pb.textContent = "Play"; tick(); });
  c.querySelector('[data-c="play"]').onclick = () => v.paused ? v.play() : v.pause();
  c.querySelector('[data-c="back"]').onclick = () => seek(it, v, frameOf(it, v) - 1);
  c.querySelector('[data-c="fwd"]').onclick = () => seek(it, v, frameOf(it, v) + 1);
  c.querySelector('[data-c="rate"]').onclick = e => { const r = {"0.5": 0.25, "0.25": 1, "1": 0.5}[String(v.dataset.rate || 0.5)];
    v.dataset.rate = r; v.playbackRate = r; e.target.textContent = r + "x"; };
  c.querySelector('[data-c="rings"]').onclick = e => { const h = c.querySelector(".vw").classList.toggle("hide"); e.target.textContent = h ? "Show rings" : "Hide rings"; };
  c.querySelector("canvas").addEventListener("click", e => { const b = e.currentTarget.getBoundingClientRect();
    seek(it, v, Math.round(it.f0 + (e.clientX - b.left - 4) / (b.width - 8) * (it.n - 1))); });
  c.querySelector(".vw").addEventListener("click", e => { const b = e.currentTarget.getBoundingClientRect();
    v.pause(); const r = R[it.rid] || {};
    r.mark = {x: +(it.x0 + (e.clientX - b.left) / b.width * it.cw).toFixed(1), y: +(it.y0 + (e.clientY - b.top) / b.height * it.ch).toFixed(1), frame: frameOf(it, v)};
    if (!r.verdict) r.verdict = "neither"; R[it.rid] = r; save(); });
  c.querySelectorAll("button[data-v]").forEach(b => b.addEventListener("click", () => {
    if (b.dataset.v === "clear") delete R[it.rid]; else R[it.rid] = Object.assign(R[it.rid] || {}, {verdict: b.dataset.v}); save(); }));
}
window.addEventListener("resize", paint);
document.getElementById("dl").addEventListener("click", () => {
  const out = ITEMS.map(it => Object.assign({rid: it.rid, rings: it.rings.map(r => r.k).join(",")}, R[it.rid] || {verdict: ""}));
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "target_review_clips_results.json"; a.click(); });
paint();
</script></body></html>"""


if __name__ == "__main__":
    main()
