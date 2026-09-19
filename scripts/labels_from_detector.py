"""Turn detector tracks into training labels WITHOUT letting the pose veto them.

THE MEASUREMENT THAT MADE THIS NECESSARY (2026-09-18, Isotopes AAA 815419).
Running the documented next step -- `audit_seed_tracks --seeds detect_tracks_815419.csv
--write-labels` -- kept 131 clips and FLAGGED 81. Rendering the sheet settled it in one
look: the six WORST-reprojecting tracks, the ones the filter threw away, each hold an
unmistakable baseball in every zoom tile, with a clean green arc across the full frame. The
yellow projected path diverges from them, which is the pose being wrong, not the track.

The script's own printed numbers said so before the render did: worst kept 17.69 px, best
flagged 18.96 px -- a 1.07x gap. Durham's known-junk batting-helmet track sat at 6x. A cut
whose boundary is 1.07x is not separating two populations, it is slicing one.

WHY THE FILTER WAS RIGHT AND IS NOW WRONG. Reprojection RMS compares an OBSERVED track to
a 7-parameter model of the camera. It was the correct junk filter when tracks came from
frame differencing, because differencing produced real junk -- helmets, umpires, a track on
the outfield wall -- and the pose was the only thing that could tell a pitch from a person.
Detector tracks arrive already filtered, twice: every candidate is a ball to a network that
reads 2.15 px on a held-out park, and every chain has to be consecutive-ish, inside a
plausible px/frame band, roughly monotonic, accelerating, and end inside the park's
consensus flight window. What reprojection measures on THAT input is almost entirely the
POSE's error. Isotopes' pose is railed (Cy 600 ft, f 18983 px, f/Cy 31.6 against OC-MLB's
40.8 and Durham's recovered 39.4), so it fails good tracks in bulk.

SO THE POSE IS KEPT FOR EXACTLY ONE JOB: the ball DIAMETER, which a label needs and a
2-D track cannot supply. And that use is independently checked -- `measured_diameter`
reads the ball's actual width off the pixels, and the ratio against the pose-derived value
lands 0.94-1.02 on five parks while the two known-bad poses sit at 1.37 and 1.78, with
nothing in between. Isotopes reads 0.95. A pose can be useless for miss distance and still
correct for boxes, because only f/depth enters a diameter -- Frawley is exactly that case.

REPORTED, NEVER ENFORCED. The reprojection distribution is still printed, because it is the
best cheap read on pose quality we have. It just does not delete anything any more. No
sample floor, no auto-reject; the n is printed and the call is the reader's.

THE VERIFICATION IS A RANDOM SAMPLE, NOT THE EXTREMES. Rendering the worst and the best
answers "does the sort order work". Rendering a random draw answers "is what I am shipping
a baseball", which is the question. Seeded so the sheet is reproducible.

    python scripts/labels_from_detector.py --game 815419 --dir data/clips/aaa_815419
    python scripts/labels_from_detector.py --game 817407 --dir data/clips/aa_817407 --diameter-px 7.9
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import pinhole as ph  # noqa: E402
import fit_park_pose as fpp  # noqa: E402
from audit_seed_tracks import fit_t0, measured_diameter, panel  # noqa: E402

FPS = 59.69


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tracks", default=None, help="default output/detect_tracks_<game>.csv")
    ap.add_argument("--pose", default=None, help="default output/park_pose_<game>.npz")
    ap.add_argument("--diameter-px", type=float, default=0.0,
                    help="constant fallback when no pose exists; REPORTED in the output")
    ap.add_argument("--measure-diameter", action="store_true",
                    help="no pose: measure the ball width off the pixels and use its median "
                         "as a constant. For a park that has never been solved.")
    ap.add_argument("--rescale-to-measured", action="store_true",
                    help="scale the pose-derived diameters so their median matches the "
                         "pixel measurement; use when the size ratio is outside 0.94-1.02")
    ap.add_argument("--report-clip-size", action="store_true", default=True,
                    help="print each clip's measured ball width and flag outliers")
    ap.add_argument("--no-report-clip-size", dest="report_clip_size",
                    action="store_false")
    ap.add_argument("--sample", type=int, default=10, help="random clips to render")
    ap.add_argument("--check-size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    tracks = a.tracks or os.path.join(ROOT, "output", "detect_tracks_%d.csv" % a.game)
    assert os.path.exists(tracks), "no tracks at %s -- run detect_ball_tracks.py first" % tracks
    pose = a.pose or os.path.join(ROOT, "output", "park_pose_%d.npz" % a.game)
    have_pose = os.path.exists(pose)
    assert have_pose or a.diameter_px > 0 or a.measure_diameter, (
        "no pose at %s, no --diameter-px and no --measure-diameter. A label needs a box "
        "width and this script will not invent one." % pose)

    manifest = os.path.join(d, os.path.basename(d) + "_manifest.csv")
    clips = fpp.build(a.game, tracks, manifest, fps_default=FPS)
    assert clips, "no clips built -- check the tracks csv and the manifest"

    C = ang = f = None
    if have_pose:
        z = np.load(pose)
        C, ang, f = z["C"], z["ang"], float(z["f"])
        print("pose C=(%.0f,%.0f,%.0f) f=%.0f  f/Cy=%.1f   (OC-MLB reads 40.8)"
              % (C[0], C[1], C[2], f, f / max(1e-6, C[1])))

    rows, reproj = [], []
    for c in clips:
        if have_pose:
            pan, tilt, roll = ang
            t0, rms, e = fit_t0(c, C, pan, tilt, roll, f)
            t = t0 + c["fr"] / c["fps"]
            dia = fpp.ball_diameter_px(C, pan, tilt, roll, f, c["traj"](t))
            reproj.append(rms)
            c["_proj"] = ph.project(C, pan, tilt, roll, f, c["traj"](t))
            c["_e"] = e
            c["_rms"] = rms
        else:
            dia = np.full(len(c["fr"]), a.diameter_px, float)  # filled below if measuring
        for k, fr_i in enumerate(c["fr"]):
            rows.append(dict(stem=c["stem"], game_pk=a.game, frame=int(fr_i),
                             x=float(c["obs"][k, 0]), y=float(c["obs"][k, 1]),
                             diameter_px=float(dia[k])))

    lab = pd.DataFrame(rows)

    # NO POSE AT ALL: measure the ball off the pixels and use the median as a constant.
    #
    # This is what a park that has never been solved gets, and it is worth being explicit
    # about what it costs. The pose version gives a per-frame diameter that GROWS as the ball
    # approaches -- measured spans are 7.4-8.5 px at Isotopes and 11.1-13.4 at Frawley, so a
    # constant is wrong by roughly +-10% at the two ends of a flight. For a YOLO box scaled
    # 1.8x that is inside the slop, and it is the difference between a park contributing and
    # a park sitting out. The n and the spread are printed so the cost is visible, and the
    # park should be re-labelled from a pose once it has one.
    if a.measure_diameter and not have_pose:
        s = lab.sample(min(200, len(lab)), random_state=a.seed)
        ms = []
        for _i, r in s.iterrows():
            m = measured_diameter(os.path.join(d, r["stem"] + ".mp4"),
                                  int(r["frame"]), r["x"], r["y"])
            if m:
                ms.append(m)
        assert ms, ("could not measure a single ball width in %d samples -- not inventing a "
                    "diameter on no evidence" % len(s))
        ms = np.array(ms, float)
        med = float(np.median(ms))
        lab["diameter_px"] = med
        print("MEASURED diameter %.2f px (n=%d, p25 %.2f p75 %.2f) used as a CONSTANT -- no "
              "pose for this park,\n  so the per-frame growth through the flight is not "
              "modelled. Re-label from a pose when one exists."
              % (med, len(ms), float(np.percentile(ms, 25)), float(np.percentile(ms, 75))))

    # SHAPE FROM THE POSE, SCALE FROM THE PIXELS.
    # diameter = f * D_ball / depth, so a wrong f (or a railed depth) multiplies the WHOLE
    # flight profile by one constant -- it does not distort the way the ball grows as it
    # approaches, because that comes from the trajectory. Riders AA is the case that forced
    # this: its pose reads f/Cy 21.5 against OC-MLB's 40.8 and hands out boxes 43% too small
    # (measured 7.69 px vs pose-derived 5.40), while its POSITIONS are fine. Rescaling keeps
    # the profile the pose gets right and takes the one number it gets wrong from an
    # independent measurement of the pixels.
    #
    # Deliberately opt-in. Applying it everywhere would quietly paper over a pose that is
    # wrong for a reason worth finding, and the ratio is printed either way.
    if a.rescale_to_measured and have_pose:
        s = lab.sample(min(max(a.check_size, 60), len(lab)), random_state=a.seed)
        meas, posed = [], []
        for _i, r in s.iterrows():
            m = measured_diameter(os.path.join(d, r["stem"] + ".mp4"),
                                  int(r["frame"]), r["x"], r["y"])
            if m:
                meas.append(m)
                posed.append(r["diameter_px"])
        assert meas, "could not measure a single ball width -- not rescaling on no evidence"
        k = float(np.median(meas)) / float(np.median(posed))
        lab["diameter_px"] = lab["diameter_px"] * k
        print("\nRESCALED diameters by %.3f (measured %.2f px / pose-derived %.2f px, n=%d). "
              "Profile\n  from the pose, scale from the pixels." %
              (k, float(np.median(meas)), float(np.median(posed)), len(meas)))

    out = os.path.join(ROOT, "output", "det_labels_%d%s.csv" % (a.game, a.tag))
    lab.to_csv(out, index=False)
    print("\n%s\n  %d clips, %d labelled points, ball %.1f-%.1f px  (NOTHING excluded)"
          % (out, lab.stem.nunique(), len(lab),
             lab.diameter_px.min(), lab.diameter_px.max()))

    if reproj:
        q = np.percentile(reproj, [5, 25, 50, 75, 95])
        print("\nreprojection vs the pose, REPORTED NOT ENFORCED: p05 %.1f  p25 %.1f  "
              "med %.1f  p75 %.1f  p95 %.1f px" % tuple(q))
        print("  This is a read on the POSE, not on the tracks. A detector chain that is a "
              "clean\n  baseball can reproject badly against a railed pose -- 81 of 212 "
              "Isotopes tracks did,\n  and every rendered one held a ball. High numbers "
              "here say refit the camera.")

    if a.check_size and have_pose:
        # A DIFFERENT DRAW FROM THE ONE THAT SET THE SCALE. With --rescale-to-measured the
        # same sample would return 1.00 by construction -- a number that cannot come out any
        # other way, which is the dead test in .claude/rules/tautological-display.md. Offset
        # the seed so this stays an actual read on the rest of the labels.
        s = lab.sample(min(a.check_size, len(lab)),
                       random_state=a.seed + (1000 if a.rescale_to_measured else 0))
        meas, posed = [], []
        for _i, r in s.iterrows():
            m = measured_diameter(os.path.join(d, r["stem"] + ".mp4"),
                                  int(r["frame"]), r["x"], r["y"])
            if m:
                meas.append(m)
                posed.append(r["diameter_px"])
        if meas:
            ratio = float(np.median(meas)) / float(np.median(posed))
            print("\nsize check (n=%d): measured-from-pixels %.2f px vs pose-derived %.2f px"
                  ", ratio %.2f" % (len(meas), float(np.median(meas)),
                                    float(np.median(posed)), ratio))
            print("  0.94-1.02 is what a correct pose reads (5 parks); the two known-bad "
                  "ones sit at\n  1.37 and 1.78, nothing in between. This is the ONLY thing "
                  "the pose is trusted for here.")

    # PER-CLIP BALL WIDTH, REPORTED. A broadcast that cuts to a full-screen graphic with a
    # small video INSET is still video, the detector still finds the ball in it, and the
    # resulting track is real -- at a completely different scale. Two of ten randomly drawn
    # Riders clips were exactly that (defensive-alignment card, pitch playing in a corner
    # inset), and they read 4.4 px and 5.9 px against a park median of 7.8 with their tracks
    # sitting in a region of the frame no other clip touches. Undetected, they teach the
    # network a ball half the right size.
    #
    # FLAGGED, NOT DROPPED, and deliberately so: a cut on ball width is a threshold nobody
    # has agreed to, and a clip can be small for honest reasons (a wide park, a high camera).
    # The n and the list are printed; the call is the reader's.
    if a.report_clip_size:
        per = []
        for c in clips:
            k = max(1, len(c["fr"]) // 5)
            ms = []
            for j in range(0, len(c["fr"]), k):
                m = measured_diameter(os.path.join(d, c["stem"] + ".mp4"),
                                      int(c["fr"][j]), c["obs"][j, 0], c["obs"][j, 1])
                if m:
                    ms.append(m)
            if ms:
                per.append((c["stem"], len(c["fr"]), float(np.median(ms)),
                            float(np.median(c["obs"][:, 1]))))
        if per:
            med = float(np.median([p[2] for p in per]))
            ys = np.array([p[3] for p in per], float)
            ymed = float(np.median(ys))
            ymad = float(np.median(np.abs(ys - ymed))) or 1.0
            # TWO SIGNALS, BECAUSE ONE OF THEM MISSES. On Riders the width test caught the
            # obvious inset (0.56x) and missed the second one at 0.75x, which sat with it in
            # a region of the frame no other clip touches. Position is the stronger signal
            # here and it costs nothing: a robust z on the track's median y (MAD-based, so
            # one outlier cannot widen its own gate) separates both cleanly.
            odd = [p for p in per
                   if p[2] < 0.65 * med or p[2] > 1.55 * med
                   or abs(p[3] - ymed) > 5.0 * ymad]
            print("\nper-clip geometry: ball width median %.2f px, track median y %.0f "
                  "(MAD %.0f), %d clips" % (med, ymed, ymad, len(per)))
            if odd:
                print("  %d clip(s) off that -- check for a picture-in-picture cut or a "
                      "second camera. FLAGGED, NOT DROPPED:" % len(odd))
                for stem, n, dia, yy in sorted(odd, key=lambda p: p[2]):
                    print("    %-22s n=%-3d width %5.2f px (%.2fx)  y %4.0f (%+.1f MAD)"
                          % (stem, n, dia, dia / med, yy, (yy - ymed) / ymad))
            else:
                print("  nothing off it.")

    # RANDOM sample, not the extremes -- the question is whether what ships is a baseball.
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(clips), size=min(a.sample, len(clips)), replace=False)
    sheets = []
    for i in sorted(idx):
        c = clips[i]
        proj = c.get("_proj", c["obs"])
        e = c.get("_e", np.zeros(len(c["fr"])))
        tag = "random %d/%d" % (list(sorted(idx)).index(i) + 1, len(idx))
        if "_rms" in c:
            tag += "  reproj %.1f px" % c["_rms"]
        p = panel(os.path.join(d, c["stem"] + ".mp4"), c, proj, e, tag)
        if p is not None:
            sheets.append(p)
    if sheets:
        w = max(s.shape[1] for s in sheets)
        sheets = [cv2.copyMakeBorder(s, 2, 2, 0, w - s.shape[1], cv2.BORDER_CONSTANT,
                                     value=(40, 40, 40)) for s in sheets]
        sp = os.path.join(ROOT, "output", "det_labels_%d%s.png" % (a.game, a.tag))
        cv2.imwrite(sp, np.vstack(sheets))
        print("\n%s\n  A RANDOM draw, seed %d. Green = the shipped label. Every tile should "
              "hold a\n  baseball; if one does not, that is the yield number, not a rounding "
              "error." % (sp, a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
