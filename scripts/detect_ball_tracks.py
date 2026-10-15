"""Build ball tracks from the DETECTOR instead of from frame differencing.

WHAT THIS REPLACES. Every track in this project so far came from absdiff + top-hat
(onboard_park -> seeds) or from projecting a pose (harvest_park_labels -> snap). Both are
cadence-sensitive: a feed carrying 30 fps in a 60 fps container has an empty diff on every
other frame, which is why L.P. Frans yielded 13 of 50 clips and Hops 1 of 40. A detector runs
PER FRAME and has no cadence assumption at all, so this path sidesteps the problem rather
than fixing three copies of it.

It also needs NO POSE. Seeds needed a pose to be validated; harvest needed a pose to know
where to look. This needs neither, which inverts the bootstrap: detector -> tracks -> pose,
instead of pose -> tracks.

TOP-K, NOT TOP-1, AND THAT IS NOT OPTIONAL. A stationary high-confidence distractor takes
top-1 on most frames at some parks -- a crowd object held top-1 on 35 of 56 flight frames at
Asheville in September, and top-1-only reporting made that read as blindness. So candidates
are taken K deep and the FLIGHT is chosen by physics: consecutive-ish frames, a step inside a
plausible px/frame band, roughly monotonic direction, and a preference for ACCELERATING
motion because a pitch speeds up in pixels as it approaches while a pan or a return throw
does not.

NO LABELS ARE INVENTED. A clip that yields no plausible chain is reported and skipped, never
back-filled. The per-clip chain length and step profile are printed so a park that is
"working" for the wrong reason is visible.

    python scripts/detect_ball_tracks.py --dir data/clips/aaa_815419 --game 815419 \
        --weights data/yolo_ball_runs/milb4park/weights/best.pt

SPEED. Inference runs ~0.4 s/frame at imgsz 1280, so the 100-frame default window costs about
40 s a clip and a 600-clip park is hours. Two levers, and only one is safe by default:
  --batch      already on; per-call overhead dominated before it.
  --lo / --hi  the flight occupies about 45 of those 100 frames. Plate frames measured from
               the harvest scans: Isotopes 213-220, Frawley 213-223, Asheville 214-224, and a
               flight is ~26 frames, so f180-230 covers all of them with margin.
THE DEFAULT STAYS WIDE ON PURPOSE. Those parks agree because they share a broadcast cut
convention; Hops and Riders have never produced a plate frame, so a narrow default would make
an unseen park yield nothing and read as 'the detector fails there' -- exactly the false
blocker pattern in .claude/rules/probe-capability-before-accepting-a-blocker.md. Narrow it per
park, from the plate-frame distribution this script prints at the end of every run.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

STEP_LO, STEP_HI = 2.0, 30.0      # px per frame, wide enough for both camera archetypes
MIN_RUN = 6
MIN_WINDOW_CLIPS = 8      # pass 2 needs a real distribution of plate frames, not 3 points
WINDOW_MAX_TOL = 18.0     # a flight is ~26 frames; a wider window cannot exclude the delivery
MAX_GAP = 3               # frames a chain may skip without a candidate
# --bridge N (0 = off, the default, byte-identical to before): a chain may also skip up to N
# frames, but ONLY to a candidate where constant velocity from its last two points puts the
# ball - within BRIDGE_R0 + BRIDGE_R1 px per skipped frame, at 0.6-1.6x its last step speed.
# SEGRA 827289 (2026-10-13): 60 of 62 chains that stopped before the plate stopped on the gap
# rule - the ball undetected for a median 6 frames mid-flight, then detected again IN LINE
# (direction cos ~1.0, same step) and at rank 1. A wider plain gap would let a chain jump to
# anything moving; the prediction gate only lets it continue the flight it is on.
BRIDGE = 0
BRIDGE_R0, BRIDGE_R1 = 4.0, 1.5


def best_chain(dets, step_lo=STEP_LO, step_hi=STEP_HI):
    """Longest physically plausible chain through per-frame candidates.

    dets: {frame: [(x, y, conf), ...]}. Greedy from each possible start, keeping the longest;
    the search is tiny (a few hundred candidates per clip) so there is no reason to be clever.
    """
    ch = all_chains(dets, step_lo, step_hi)
    return max(ch, key=len) if ch else []


def all_chains(dets, step_lo=STEP_LO, step_hi=STEP_HI):
    """Every greedy chain, one per start candidate, in start order.

    best_chain keeps only the longest, and on 2026-09-24 that was the failure: at Isotopes
    the shortstop stands in the foreground and the chain through his white cleats is LONGER
    than the pitch, so v2 followed his shoes on 53 of 156 clips (plate misses of 1-2 ft) while
    the pitch chain existed in the same candidate set. Keeping every chain lets a later step
    choose the one that ends where this park's pitches end. Ties in length resolve to the
    earliest start, exactly as best_chain always did (strict `>` below).
    """
    frames = sorted(dets)
    out = []
    for si, f0 in enumerate(frames):
        for c0 in dets[f0]:
            chain = [(f0, c0[0], c0[1], c0[2])]
            for f1 in frames[si + 1:]:
                gap = f1 - chain[-1][0]
                if gap > MAX_GAP and not (BRIDGE and len(chain) >= 2 and gap <= BRIDGE):
                    break
                prev = chain[-1]
                cands = []
                for x, y, cf in dets[f1]:
                    dt = f1 - prev[0]
                    st = np.hypot(x - prev[1], y - prev[2]) / dt
                    if not (step_lo <= st <= step_hi):
                        continue
                    if dt > MAX_GAP:         # a bridge: only where the flight it is on predicts
                        p0 = chain[-2]
                        v = ((prev[1] - p0[1]) / (prev[0] - p0[0]), (prev[2] - p0[2]) / (prev[0] - p0[0]))
                        sp = np.hypot(*v)
                        if np.hypot(x - prev[1] - v[0] * dt, y - prev[2] - v[1] * dt) > BRIDGE_R0 + BRIDGE_R1 * dt                                 or not (0.6 * sp <= st <= 1.6 * sp):
                            continue
                    if len(chain) >= 2:      # keep going the same way
                        vx, vy = chain[-1][1] - chain[-2][1], chain[-1][2] - chain[-2][2]
                        nx, ny = x - prev[1], y - prev[2]
                        n1 = np.hypot(vx, vy) * np.hypot(nx, ny)
                        if n1 > 0 and (vx * nx + vy * ny) / n1 < 0.80:
                            continue
                    cands.append((st, x, y, cf))
                if not cands:
                    continue
                st, x, y, cf = max(cands, key=lambda c: c[3])   # strongest that fits
                chain.append((f1, x, y, cf))
            out.append(chain)
    return out


def _flush(model, buf, dets, a):
    """Run one batched predict over buffered (frame_index, image) pairs."""
    res = model.predict([b[1] for b in buf], conf=a.conf, imgsz=a.imgsz,
                        verbose=False, device=a.device)
    for (fi, _), r in zip(buf, res):
        if r.boxes is None or not len(r.boxes):
            continue
        xy = r.boxes.xywh.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-cf)[:a.topk]
        dets[fi] = [(float(xy[k, 0]), float(xy[k, 1]), float(cf[k])) for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--lo", type=int, default=150)
    # 300, not 250. Measured 2026-09-19: Greensboro's plate frame is f277 (median,
    # spanning 230-337) and Constellation piled 25% of its chains onto the old f250
    # ceiling. Both parks were previously diagnosed as gate failures; both were the
    # window. The extra 50 frames cost ~50% more inference per clip, which is the right
    # trade against silently losing two thirds of a park. Narrow it per park AFTER the
    # run prints that park's own chain-end distribution.
    ap.add_argument("--hi", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16,
                    help="frames per predict call; 16 fits a 4 GB card at 1280")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0",
                    help="'auto' = '0' when torch sees a GPU (CUDA, or ROCm on the AMD laptop), "
                         "else 'cpu'. The Hostinger VPS has no GPU (2026-10-14)")
    ap.add_argument("--two-pass", action="store_true",
                    help="admit SHORT chains that end inside the consensus window pass 1 "
                         "defined. For tight-camera parks where the ball is separable for "
                         "only a handful of frames.")
    ap.add_argument("--min-run-2", type=int, default=4,
                    help="minimum chain length in pass 2 (pass 1 uses %d)" % MIN_RUN)
    ap.add_argument("--tag", default="")
    ap.add_argument("--from-dets", default="",
                    help="replay chain selection from a saved dets_<game><tag>.csv: no video, "
                         "no GPU, seconds instead of ~27 min for a 327-clip park")
    ap.add_argument("--window-pick", action="store_true",
                    help="for a clip whose longest chain ends outside the consensus window, "
                         "take its longest chain that ends INSIDE it instead of dropping it")
    ap.add_argument("--plate-spot", action="store_true",
                    help="pick, per clip, the longest chain that ends WHERE this park's "
                         "pitches end (spatial consensus from pass 1), not the longest overall")
    ap.add_argument("--stems-file", default="",
                    help="CSV with a `stem` column: run on those clips only (e.g. a game's "
                         "holdout_<game>.csv, so a model trained on the other half is graded "
                         "without touching its own training clips). The pass-1 consensus window "
                         "is then fitted on that subset, so compare two models only when both "
                         "ran with the SAME file")
    ap.add_argument("--bridge", type=int, default=0,
                    help="let a chain skip up to N frames to a candidate its own constant-velocity "
                         "flight predicts (0 = off). Per park, from docs/miss_park_registry.csv")
    a = ap.parse_args()
    if a.device == "auto":
        import torch
        a.device = "0" if torch.cuda.is_available() else "cpu"
        print("device: %s" % a.device, flush=True)
    global BRIDGE
    BRIDGE = a.bridge
    if BRIDGE:
        assert BRIDGE > MAX_GAP, "--bridge %d is not longer than the plain gap rule (%d)" % (BRIDGE, MAX_GAP)
        print("bridge: chains may skip up to %d frames along their own flight" % BRIDGE, flush=True)

    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    man = pd.read_csv(os.path.join(d, os.path.basename(d) + "_manifest.csv"))
    # TWO MANIFEST SHAPES ON DISK, and the older one has no `stem` column at all. The MLB
    # pulls (mlb_823850/824018/824259/824341) predate that convention and name their files
    # <play_id>_<ANGLE>.mp4; the MiLB browser pulls carry an explicit `stem`. Deriving the
    # older shape here rather than rewriting four manifests keeps the raw pulls untouched.
    # Without this the MLB dirs died with `'DataFrame' object has no attribute 'stem'` --
    # loud, at least, because detect_all_parks reports a per-park return code.
    import fit_park_pose as _fpp
    man = man.assign(_stem=_fpp.manifest_stems(man))

    # --game FILTERS the manifest, it does not just label the output.
    #
    # Two clip dirs on disk are PITCHER dirs, not park dirs: ogando_2026 holds 149 clips from
    # 13 games and langford_2026 holds 30 from 10, across SEVEN different A+ ballparks
    # between them. Stamping --game on every row was silently wrong there -- most rows got a
    # game_pk they did not come from -- and, worse, the consensus flight window was being
    # fitted across cameras that share no cut convention, so those parks were unusable and
    # nobody could see why.
    #
    # Filtering makes each game inside a shared directory its own park, which is how four
    # ballparks we ALREADY have on disk become reachable: Greensboro 37 clips, Rome 24,
    # Winston-Salem 10, Greenville 7. Greensboro and Rome are the behind-the-pitcher
    # archetype that differencing could never track and the detector cracked at Hops today.
    #
    # No-op on a single-game dir, and on a manifest with no game_pk column at all.
    if "game_pk" in man.columns:
        mine = man[man.game_pk.astype(str) == str(a.game)]
        if len(mine) != len(man):
            print("manifest holds %d games; --game %d selects %d of %d rows"
                  % (man.game_pk.nunique(), a.game, len(mine), len(man)), flush=True)
        assert len(mine), (
            "game %d is not in this manifest. It holds: %s. Stamping the label on another "
            "game's clips would put a wrong game_pk on every row."
            % (a.game, sorted(man.game_pk.astype(str).unique())))
        man = mine

    if a.stems_file:
        keep = set(pd.read_csv(a.stems_file).stem.astype(str))
        sub = man[man["_stem"].astype(str).isin(keep)]
        print("--stems-file %s keeps %d of %d clips" % (os.path.basename(a.stems_file),
                                                       len(sub), len(man)), flush=True)
        assert len(sub), "--stems-file matched none of this game's clips"
        man = sub

    cand = list(man["_stem"])
    stems = [s for s in cand if os.path.exists(os.path.join(d, str(s) + ".mp4"))]
    assert stems, (
        "no manifest row matched a file on disk in %s. %d rows, first candidate %r -- the "
        "naming convention is wrong, not the directory." % (d, len(cand), cand[:1]))
    if a.limit:
        stems = stems[:a.limit]

    # RAW DETECTIONS ARE SAVED, because chain selection is where the bugs have been and
    # inference is where the time goes. Every change to the chain logic used to cost a full
    # re-run of the network (27 min at Isotopes); with the detections on disk it costs seconds,
    # and the old and new selection can be compared on IDENTICAL candidates.
    dets_path = os.path.join(ROOT, "output", f"dets_{a.game}{a.tag}.csv")
    all_dets = {}
    if a.from_dets:
        # round_trip: pandas' default fast float parser can land 1 ULP off, which is enough
        # to make a replay differ byte-wise from the live run it is meant to reproduce.
        dd = pd.read_csv(a.from_dets, float_precision="round_trip")
        for (stem, f), g in dd.groupby(["stem", "frame"], sort=True):
            all_dets.setdefault(stem, {})[int(f)] = [
                (float(r.x), float(r.y), float(r.conf)) for r in g.itertuples()]
        for s in stems:                  # clips with no detection at all still count
            all_dets.setdefault(s, {})
        print(f"replaying {len(all_dets)} clips from {a.from_dets}", flush=True)
    else:
        from ultralytics import YOLO
        model = YOLO(a.weights)
        for j, stem in enumerate(stems, 1):
            cap = cv2.VideoCapture(os.path.join(d, stem + ".mp4"))
            dets, i, buf = {}, 0, []
            while True:
                ok, fr = cap.read()
                if not ok or i > a.hi:
                    break
                if i >= a.lo:
                    buf.append((i, fr))
                    # BATCHED, because one predict() per frame is dominated by per-call
                    # overhead rather than by the network: ~100 frames a clip x 600 clips is
                    # 60,000 calls. Frames are flushed in chunks instead of all held, since a
                    # full window of 1280x720x3 is ~276 MB and this box runs near its memory
                    # watchdog.
                    if len(buf) >= a.batch:
                        _flush(model, buf, dets, a)
                        buf = []
                i += 1
            cap.release()
            if buf:
                _flush(model, buf, dets, a)
                buf = []
            all_dets[stem] = dets
            if j % 20 == 0:
                print(f"  [{j}/{len(stems)}] inferred", flush=True)
        pd.DataFrame([dict(stem=s, frame=f, x=x, y=y, conf=c)
                      for s, dts in all_dets.items() for f, cs in dts.items()
                      for x, y, c in cs]).to_csv(dets_path, index=False, float_format="%.17g")
        # %.17g, not the default: pandas writes 16 significant digits, which drops the last
        # digit of a float64 (309.76849365234375 -> 309.7684936523437). Harmless in pixels
        # (~1e-13) but it made a --from-dets replay differ BYTE-wise from the live run, and
        # a replay that cannot be byte-identical cannot prove a refactor neutral.
        print(f"raw detections -> {dets_path}", flush=True)

    rows, kept, lens = [], 0, []
    short = {}          # stem -> chain, for clips that fail MIN_RUN but are not empty
    for stem in stems:
        dets = all_dets.get(stem, {})
        chain = best_chain(dets) if len(dets) >= MIN_RUN else []
        if len(chain) < MIN_RUN:
            # HOLD the detections instead of discarding them, so a second pass can revisit
            # this clip without paying for inference again. A clip that fails MIN_RUN is
            # exactly the case --two-pass exists for, and re-running the network over a
            # whole park to try a looser gate costs 10-20 minutes per attempt.
            if a.two_pass and len(chain) >= a.min_run_2:
                short[stem] = chain
            continue
        kept += 1
        lens.append(len(chain))
        for f, x, y, cf in chain:
            rows.append(dict(stem=stem, game_pk=a.game, frame=int(f), x=x, y=y, conf=cf))

    assert rows, "no clip produced a plausible flight -- check --lo/--hi and the weights"
    out = pd.DataFrame(rows)

    # CONSENSUS FLIGHT WINDOW, ported from onboard_park for the same reason it exists there.
    # A park's clips share a broadcast cut convention, so real plate crossings cluster hard --
    # measured 213-220 at Isotopes, 213-223 at Frawley, 214-224 at Asheville. A chain whose
    # END sits far outside that cluster is not a pitch.
    #
    # THIS IS NOT A TUNING KNOB. Rendering Riders showed its chains split cleanly in two: the
    # ones ending near f213 are unmistakable baseballs, and the ones ending at f157-168 are
    # dark murky blobs. The detector's own confidence does NOT separate them (junk ran
    # 0.23-0.58, real ran 0.22-0.74) -- only WHEN the chain ends does. Derived from this
    # park's own distribution, never carried between parks.
    ends = out.groupby("stem").frame.max()

    # IS THE SCAN WINDOW CUTTING FLIGHTS OFF? A chain that ends at the last frame we looked
    # at has not necessarily finished -- it may simply be where we stopped looking. That
    # failure is completely silent: the clip still yields a (short) chain, the short chain
    # then fails MIN_RUN, and the park reports a low yield that reads as "hard park".
    #
    # Measured 2026-09-19. Greensboro's plate is at f277 median, spanning 230-337, while the
    # default window stops at 250 -- so it scored 3/33 and I diagnosed it as a CHAINING
    # failure. Re-run at f200-350 it gives 10/33 and every render is a clean baseball at the
    # plate. Constellation then showed 25% of chains ending EXACTLY at f250, which is the
    # same defect and also inflated its chain-end IQR to 49.5 against 5-12 elsewhere.
    #
    # Both parks were previously written up as gate problems. They were window problems.
    at_ceiling = int((ends >= a.hi - 4).sum())
    if at_ceiling and at_ceiling / len(ends) > 0.05:
        print("")
        print("*** WINDOW TRUNCATION: %d of %d chains (%.0f%%) end within 4 frames of --hi "
              "%d." % (at_ceiling, len(ends), 100 * at_ceiling / len(ends), a.hi))
        print("    Those are flights CUT OFF, not flights that finished, and the clips that")
        print("    lost more frames failed MIN_RUN and are not in this count at all.")
        print("    Re-run with --hi %d (and --lo %d) before trusting this park's yield."
              % (int(a.hi + 90), int(max(0, a.lo + 40))))

    centre = float(ends.median())
    spread = float((ends - centre).abs().median()) or 1.0
    tol = max(12.0, 4.0 * spread)
    keep = set(ends[(ends - centre).abs() <= tol].index)
    dropped = sorted(set(ends.index) - keep)
    if dropped:
        print("")
        print(f"consensus flight window: chains must end within +-{tol:.0f} frames "
              f"of f{centre:.0f}")
        print("  dropped: " + ", ".join(f"{s}@f{ends[s]:.0f}" for s in dropped[:12])
              + (f" (+{len(dropped) - 12} more)" if len(dropped) > 12 else ""))
        out = out[out.stem.isin(keep)]
        # WINDOW PICK (2026-09-28, opt-in). A dropped clip lost only its LONGEST chain; the
        # pitch chain is often still in the same candidate set, just shorter than the return
        # throw that follows it (Asheville 822515: 8 of 60 dropped, longest chains ending
        # f256-300, while e.g. ogando_si_9's pitch runs f193-217 at conf ~0.8). Re-choose the
        # longest chain that ENDS inside the consensus window. Plate spot cannot do this at a
        # side-view park -- it refuses there on spatial spread.
        if a.window_pick:
            added = []
            for s in dropped:
                ok = [c for c in all_chains(all_dets.get(s, {}))
                      if len(c) >= MIN_RUN and abs(c[-1][0] - centre) <= tol]
                if ok:
                    c = max(ok, key=len)
                    added.append(s)
                    out = pd.concat([out, pd.DataFrame(
                        [dict(stem=s, game_pk=a.game, frame=int(f), x=x, y=y, conf=cf,
                              window_pick=1) for f, x, y, cf in c])], ignore_index=True)
            if "window_pick" in out:
                out["window_pick"] = out["window_pick"].fillna(0).astype(int)
            print(f"  window pick: {len(added)}/{len(dropped)} recovered: {', '.join(added)}")
        kept = out.stem.nunique()
        lens = list(out.groupby("stem").size())

    # PLATE SPOT: WHERE a chain ends, not only WHEN. Added 2026-09-24.
    #
    # At Isotopes the shortstop stands in the foreground of the broadcast shot. His white
    # cleats move for as long as a pitch does and end at the same TIME, so the consensus
    # window above cannot see them -- and their chain is often the LONGER one, so best_chain
    # took it. Graded against the tracking system, v2 missed the plate by 1-2 FEET on 53 of
    # 156 clips this way while the pitch chain sat in the same candidate set. Real pitches at
    # one park end in one small patch of the frame (the catcher's glove); the cleats end a few
    # hundred pixels away at the bottom of the shot.
    #
    # The spot is derived from THIS park's pass-1 chains, like the time window, so nothing is
    # carried between parks. It needs a majority of pass-1 chains to be pitches -- at a park
    # where most chains are junk (Route 66, 2026-09-24) the median lands on junk, which is why
    # a new park is still rendered and admitted by eye before its crops train anything.
    #
    # REFUSES rather than admitting on a spot that cannot discriminate: fewer than
    # MIN_WINDOW_CLIPS pass-1 clips, or a tolerance wider than HALF the park's median flight
    # length in pixels -- a spot that wide contains most of the flight, so it cannot separate
    # a pitch from anything else moving through the same half of the frame.
    spot = None
    if a.plate_spot:
        srt = out.sort_values("frame")
        last = srt.groupby("stem").tail(1).set_index("stem")
        first = srt.groupby("stem").head(1).set_index("stem")
        cx, cy = float(last.x.median()), float(last.y.median())
        spread_px = float(np.hypot(last.x - cx, last.y - cy).median()) or 1.0
        tol_px = max(12.0, 4.0 * spread_px)
        flight_px = float(np.median(np.hypot(last.x - first.x.reindex(last.index),
                                             last.y - first.y.reindex(last.index))))
        n1 = len(last)
        print("")
        if n1 < MIN_WINDOW_CLIPS or tol_px > flight_px / 2:
            print(f"PLATE SPOT REFUSED: ({cx:.0f},{cy:.0f}) +-{tol_px:.0f}px from {n1} pass-1 "
                  f"clip(s); needs >={MIN_WINDOW_CLIPS} clips and a tolerance <= half the "
                  f"median flight ({flight_px:.0f}px). Keeping the longest chain per clip.")
        else:
            spot = (cx, cy, tol_px)
            before = {s: tuple(g.frame) for s, g in out.groupby("stem")}
            rows, n_same, n_switch, n_add = [], 0, 0, 0
            for stem in stems:
                dets = all_dets.get(stem, {})
                if len(dets) < MIN_RUN:
                    continue
                ok = [c for c in all_chains(dets) if len(c) >= MIN_RUN
                      and abs(c[-1][0] - centre) <= tol
                      and np.hypot(c[-1][1] - cx, c[-1][2] - cy) <= tol_px]
                if not ok:
                    continue
                ch = max(ok, key=len)
                if stem not in before:
                    n_add += 1
                elif tuple(f for f, *_ in ch) == before[stem]:
                    n_same += 1
                else:
                    n_switch += 1
                for f, x, y, cf in ch:
                    rows.append(dict(stem=stem, game_pk=a.game, frame=int(f), x=x, y=y,
                                     conf=cf))
            out = pd.DataFrame(rows)
            keep = set(out.stem)
            n_drop = len(set(before) - keep)
            kept = out.stem.nunique()
            lens = list(out.groupby("stem").size())
            print(f"PLATE SPOT ({cx:.0f},{cy:.0f}) +-{tol_px:.0f}px, flight {flight_px:.0f}px, "
                  f"from {n1} clips: {n_same} unchanged, {n_switch} SWITCHED to a chain ending "
                  f"at the spot, {n_add} added, {n_drop} dropped (no chain ends there)")

    # SECOND PASS: short chains, admitted ONLY if they land in the window the LONG chains
    # defined. Four parks now show the same shape -- Greensboro 3/33, Constellation 4/16,
    # Hops 29/50, L.P. Frans 15/50 -- and in every one the rendered tracks are unmistakable
    # baseballs while the failures have detections in 59-100 of 101 frames at confidence up
    # to 0.80. The detector is not blind at those parks; the chain is short because the
    # camera is tight and low, so the ball is only separable for a handful of frames.
    #
    # WHY THIS IS NOT JUST LOWERING MIN_RUN. A global MIN_RUN of 4 would admit short junk
    # everywhere, including the five parks that are already fine. Here the extra evidence is
    # the park's OWN consensus window, which was derived from chains that passed the strict
    # gate -- so a 4-frame run ending where this park's real pitches end is a different claim
    # from a 4-frame run ending anywhere. The window must come from pass 1; deriving it from
    # the relaxed set would let the short chains vote on their own admission criterion, which
    # is the dead-test shape in .claude/rules/tautological-display.md.
    #
    # REPORTED SEPARATELY, ALWAYS. Pass-2 clips are tagged in the output so their yield, and
    # any damage they do, can be attributed rather than blended into the headline number.
    if a.two_pass and short:
        # THE WINDOW HAS TO BE A WINDOW. Measured at Greensboro: pass 1 kept only 3 clips,
        # whose ends (219, 243, 244) gave a spread so wide that tol came out at +-50 frames
        # on a 100-frame scan. That admits nearly anything, and what it admitted was the ball
        # AT RELEASE -- four chains ending f186-189, about 55 frames and 0.9 s before the
        # plate. Every one is a real baseball; none is a flight.
        #
        # AND WHY GREENSBORO ONLY HAD 3 PASS-1 CLIPS IN THE FIRST PLACE, established later
        # the same day: its plate frame is f277 and the window stopped at f250, so most of
        # its flights were truncated before they finished. The thin window was a SYMPTOM of
        # the scan being wrong, not a property of the park. With f200-350 pass 1 keeps 10.
        # So these two conditions are still right -- a window built on 3 points cannot
        # exclude the delivery -- but they were protecting against damage whose root cause
        # was upstream. Fix the window first; the refusal is the backstop, not the diagnosis.
        #
        # Two conditions, both physical rather than tuned:
        #   n >= MIN_WINDOW_CLIPS  -- three samples is not a distribution, and a median plus
        #     MAD computed from three points cannot describe where this park's pitches land.
        #   tol <= WINDOW_MAX_TOL  -- a pitch flight is ~26 frames, so a tolerance wider than
        #     half of one lets a "flight" be entirely disjoint from the real flight. A window
        #     that cannot exclude the delivery is not evidence.
        # Failing either, pass 2 REFUSES rather than admitting on a gate that cannot fail
        # (.claude/rules/grep-guard-criteria.md).
        n_pass1 = int(out.stem.nunique())
        if n_pass1 < MIN_WINDOW_CLIPS or tol > WINDOW_MAX_TOL:
            print("")
            print("PASS 2 REFUSED: window is f%.0f +-%.0f from %d pass-1 clip(s). Needs >=%d "
                  "clips and a tolerance <=%d" % (centre, tol, n_pass1, MIN_WINDOW_CLIPS,
                                                  WINDOW_MAX_TOL))
            print("  A tolerance wider than half a flight (~26 frames) cannot tell a pitch "
                  "from the delivery,\n  and at Greensboro that exact failure admitted four "
                  "tracks of the ball still in the hand.")
            short = {}
        added, rejected = [], 0
        for stem, chain in short.items():
            if stem in keep:          # the plate-spot pass already found it a full chain
                continue
            end = chain[-1][0]
            if abs(end - centre) > tol:
                rejected += 1
                continue
            if spot and np.hypot(chain[-1][1] - spot[0], chain[-1][2] - spot[1]) > spot[2]:
                rejected += 1
                continue
            added.append(stem)
            for f, x, y, cf in chain:
                rows.append(dict(stem=stem, game_pk=a.game, frame=int(f), x=x, y=y,
                                 conf=cf, pass2=1))
        if added:
            out = pd.DataFrame(rows)
            out["pass2"] = out.get("pass2", 0)
            out["pass2"] = out["pass2"].fillna(0).astype(int)
            out = out[out.stem.isin(keep | set(added))]
            kept = out.stem.nunique()
            lens = list(out.groupby("stem").size())
        print("")
        print(f"PASS 2 (min run {a.min_run_2}, must end within +-{tol:.0f} of f{centre:.0f}): "
              f"{len(added)} clip(s) added, {rejected} rejected on the window, "
              f"{len(short) - len(added) - rejected} had no chain at all")
        print("  These are SHORTER and therefore weaker. Render them before the dataset "
              "build -- they are tagged pass2=1 in the csv.")
    p = os.path.join(ROOT, "output", f"detect_tracks_{a.game}{a.tag}.csv")
    out.to_csv(p, index=False)
    print(f"\ngame {a.game}: {kept}/{len(stems)} clips with a flight, {len(out)} points")
    print(f"  chain length: median {int(np.median(lens))}, "
          f"range {min(lens)}-{max(lens)}   mean conf {out.conf.mean():.2f}")
    ends = out.groupby("stem").frame.max()
    print(f"  chain ends (proxy for the plate crossing): p05 {ends.quantile(.05):.0f}  "
          f"median {ends.median():.0f}  p95 {ends.quantile(.95):.0f}   -> "
          f"--lo {int(ends.quantile(.05)) - 32} --hi {int(ends.quantile(.95)) + 6} "
          f"would have sufficed")
    print(f"{p}\n  These are DETECTIONS, not yet verified. Render before using them.")


if __name__ == "__main__":
    main()
    # Same guard as train_ball_detector.py: the ROCm build can spin at interpreter teardown,
    # which would hang any orchestrator waiting on this process (v6_run.py).
    if "torch" in sys.modules and getattr(sys.modules["torch"].version, "hip", None):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
