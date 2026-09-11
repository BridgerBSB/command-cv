"""Render the step 1 audit pack: every image a reviewer needs to check the two
claims (no drawn zone box on the A+ broadcast; the camera holds still during the
flight), plus a SOURCES.txt that says where each input came from.

Uses ONLY labels that pass probe_aplus_camera.label_on_api_clip (12 of 55 were
drawn on a different video). All frames are decoded in order, never sought.

    python scripts/audit_step1.py [--out C:\\Users\\Owner\\Desktop\\command-cv-audit]

Outputs
  audit1a_zonebox_around_pitch.png  one verified clip per ballpark, catch-30 .. catch+120
  audit1b_zonebox_whole_clip.png    same clips, one frame per second, whole clip
  audit1c_plate_area_fullres.png    native-resolution crop around the labeled catch glove
  audit2_camera_motion.png          3 parks: tracked corners (kept/rejected), frame diff,
                                    per-frame shift: feature tracking vs phase correlation
  venues.json, SOURCES.txt
"""
import argparse
import csv
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_aplus_camera import CLIPS, HERE, flight_motion, read_range, verified_labels  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MANIFEST = os.path.join(CLIPS, "ogando_2026_manifest.csv")
NAVY_BGR, TW, TH = (98, 45, 0), 384, 216


def venues_for(stems):
    """Ballpark per game from the PUBLIC game feed (no auth): gameData.venue."""
    man = {r["stem"]: r for r in csv.DictReader(open(MANIFEST))}
    out = {}
    for pk in sorted({man[s]["game_pk"] for s in stems}):
        url = (f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
               "?fields=gameData,venue,name,teams,away,home,datetime,officialDate")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        d = json.load(urllib.request.urlopen(req, timeout=30))["gameData"]
        out[pk] = dict(venue=d["venue"]["name"], date=d["datetime"]["officialDate"],
                       away=d["teams"]["away"]["name"], home=d["teams"]["home"]["name"])
    return {s: dict(game_pk=man[s]["game_pk"], play_id=man[s]["play_id"], **out[man[s]["game_pk"]])
            for s in stems}


def _label(img, text, w):
    cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), 0, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def _header(width, text):
    h = np.full((34, width, 3), NAVY_BGR, np.uint8)
    cv2.putText(h, text, (8, 23), 0, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return h


def zonebox_sheets(picks, anns, ven, out):
    offs = [-30, -10, 0, 10, 20, 30, 45, 60, 90, 120]
    rows_a, rows_b, rows_c = [], [], []
    for s in picks:
        cf = anns[s]["catch"]["frame"]
        clip = os.path.join(CLIPS, s + ".mp4")
        cap = cv2.VideoCapture(clip)
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        allf = read_range(clip, 0, n)
        v = ven[s]
        lab = f"{v['venue']} | {v['date']} | {s} | catch f{cf}"
        blank = np.zeros((720, 1280, 3), np.uint8)
        rows_a += [_header(TW * len(offs), lab),
                   np.hstack([_label(cv2.resize(allf.get(cf + o, blank), (TW, TH)), f"catch{o:+d}", TW)
                              for o in offs])]
        idx = list(range(0, len(allf), int(round(fps))))[:10]
        idx += [idx[-1]] * (10 - len(idx))
        rows_b += [_header(TW * 10, lab + f" | {len(allf)} frames, 1 per second"),
                   np.hstack([_label(cv2.resize(allf[i], (TW, TH)),
                                     f"t={i / fps:.1f}s f{i}" + (" CATCH" if abs(i - cf) < fps / 2 else ""), TW)
                              for i in idx])]
        cw, ch = 520, 300
        cx, cy = anns[s]["catch"]["cx"], anns[s]["catch"]["cy"]
        x0, y0 = int(np.clip(cx - cw // 2, 0, 1280 - cw)), int(np.clip(cy - ch // 2, 0, 720 - ch))
        rows_c += [_header(cw * 5, lab + f" | native-res crop around labeled catch glove ({cx},{cy})"),
                   np.hstack([_label(allf.get(cf + o, blank)[y0:y0 + ch, x0:x0 + cw].copy(), f"catch{o:+d}", cw)
                              for o in (-6, 0, 15, 30, 60)])]
    for name, rows in (("audit1a_zonebox_around_pitch.png", rows_a),
                       ("audit1b_zonebox_whole_clip.png", rows_b),
                       ("audit1c_plate_area_fullres.png", rows_c)):
        cv2.imwrite(os.path.join(out, name), np.vstack(rows))
        print(os.path.join(out, name))


def motion_figure(clips, anns, ven, out):
    win = cv2.createHanningWindow((640, 360), cv2.CV_32F)
    fig, axs = plt.subplots(len(clips), 3, figsize=(18, 4.6 * len(clips)),
                            gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for r, s in enumerate(clips):
        m = flight_motion(os.path.join(CLIPS, s + ".mp4"), anns[s]["catch"]["frame"], per_frame=True)
        fr = m["frames"]
        x0, y0, x1, y1 = m["band"]
        ax = axs[r, 0]
        ax.imshow(cv2.cvtColor(fr[0], cv2.COLOR_BGR2RGB))
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#EB6E1F", lw=2))
        a0s, a1s = m["p0"][m["keep"]].reshape(-1, 2), m["p1"][m["keep"]].reshape(-1, 2)
        for a0, a1, ok in zip(a0s, a1s, m["inliers"]):
            col = "#39FF14" if ok else "red"
            ax.annotate("", xy=a0 + 40 * (a1 - a0), xytext=a0,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
        ax.set_title(f"{s} | {ven[s]['venue']} | {ven[s]['date']}\n"
                     f"{int(m['inliers'].sum())} corners KEPT by the fit (green), "
                     f"{int((~m['inliers']).sum())} REJECTED (red: fans, batter)\n"
                     "arrows = movement from catch-30 to the catch, drawn x40", fontsize=10)
        ax.axis("off")
        ax = axs[r, 1]
        ax.imshow(cv2.absdiff(fr[-1], fr[0]).max(axis=2), cmap="inferno", vmin=0, vmax=80)
        ax.axis("off")
        ax.set_title("|catch frame - catch-30 frame|\nstill camera = dark background; only people who moved light up",
                     fontsize=10)
        g = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (640, 360)).astype(np.float32) for f in fr]
        pc, resp = [0.0], [1.0]
        for i in range(1, len(g)):
            (dx, dy), rr = cv2.phaseCorrelate(g[0], g[i], win)
            pc.append(2 * float(np.hypot(dx, dy)))
            resp.append(float(rr))
        pc, resp = np.array(pc), np.array(resp)
        t = np.arange(len(g)) / 60 * 1000
        ax = axs[r, 2]
        ax.plot(t, m["t_series"], c="#002D62", lw=2.2, label=f"feature tracking (used): max {m['t_px']:.2f} px")
        ax.plot(t, pc, c="#EB6E1F", lw=1.2, ls="--", label=f"phase correlation (rejected): max {pc.max():.1f} px")
        low = resp < 0.15
        ax.scatter(t[low], pc[low], s=22, facecolors="none", edgecolors="red",
                   label="phase corr. confidence < 0.15 (unreliable)")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, 1000)
        ax.set_xlabel("ms into window (catch at 500)")
        ax.set_ylabel("camera shift, px (log above 1)")
        ax.axhline(1, c="grey", lw=0.6, ls=":")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8.5, loc="upper left")
        big = pc > 3
        note = (f"phase corr. readings > 3 px: {int(big.sum())}, max confidence among them "
                f"{resp[big].max():.2f}" if big.any() else "phase corr. never exceeds 3 px here")
        ax.set_title("camera shift vs the window's first frame\n" + note, fontsize=10)
    fig.suptitle("Audit 2 - A+ broadcast camera during ball flight "
                 "(verified hand-labeled catch frames, 60 fps, decoded in order)",
                 fontsize=14, color="#002D62", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97])
    p = os.path.join(out, "audit2_camera_motion.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(p)


SOURCES = """STEP 1 AUDIT PACK - sources and method
==========================================

Regenerate: python command-cv/scripts/audit_step1.py --out <dir>
Numbers:    python command-cv/scripts/probe_aplus_camera.py

INPUTS
1. Video clips: data/clips/ogando_2026/<stem>.mp4 (gitignored). Pulled Jul 3-8 2026 by
   scripts/pull_pitcher_clips.py from the AUTHED MLB Stats API analytics endpoint
   game/{{game_pk}}/{{play_id}}/analytics?hydrate=analytics(video), angle BROADCAST
   (the only angle offered at A+). 1280x720, ~59.7 fps.
2. Setup / catch frames and glove boxes: data/annotations/<stem>.json, drawn by Zac in
   tools/annotate.html (files saved Jul 2 and Jul 5 2026). Ingest
   (scripts/ingest_annotation.py) saved the frame it read as
   data/yolo/images/<split>/<stem>_<tag>_f<frame>.png.
3. Ballpark and date: PUBLIC MLB game feed, api/v1.1/game/{{game_pk}}/feed/live,
   gameData.venue.name and datetime.officialDate (venues.json here).
4. OpenCommand claims being tested: github.com/tomdoyo/open-command at v1.2.0, commit
   984d26e1, summarized in command-cv/docs/2026-09-01-opencommand-v12-and-cross-camera-accuracy.md.
   Their per-clip camera refit uses the drawn strike-zone box, and excludes the ball
   because "camera often moves mid-ball flight".

LABEL PROVENANCE (why some labels are not used)
{provenance}

CLAIM 1 - no drawn strike-zone box
  Looked at by eye: audit1a (catch-30 .. catch+120), audit1b (whole clip, 1 fps),
  audit1c (native-res crop around the labeled glove), one verified clip per ballpark.
  This is a visual check, not an automated detector.

CLAIM 2 - camera holds still during the flight
  Window: catch-30 .. catch (0.5 s; the pitch flight is ~0.42 s).
  Method: track ~300 corners in the crowd/wall band (orange box, scorebugs excluded),
  RANSAC similarity fit each frame vs the first -> translation px + zoom %.
  Rejected method: whole-frame phase correlation. Its big readings come only with
  near-zero confidence and do not reproduce run to run on identical frames.
  Approx inches = px x 12 / labeled glove width in px (glove ~12 in), glove depth only.

WHAT THIS DOES NOT SHOW
  - camera movement BETWEEN pitches (pan/zoom per pitch)
  - whether the ball can be detected on ~16 frames per flight (the solve needs that)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\Owner\Desktop\command-cv-audit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    good, bad = verified_labels()
    anns = dict(good)
    ven = venues_for(list(anns))
    json.dump(ven, open(os.path.join(a.out, "venues.json"), "w"), indent=1)
    picks = {}
    for s in sorted(anns):
        picks.setdefault(ven[s]["venue"], s)
    print("one verified clip per ballpark:", picks)
    zonebox_sheets(list(picks.values()), anns, ven, a.out)
    by_venue = sorted(picks.items())
    motion_clips = ["ogando_si_17"] + [s for v, s in by_venue if s != "ogando_si_17"
                                       and ven[s]["venue"] != ven["ogando_si_17"]["venue"]][:2]
    motion_figure(motion_clips, anns, ven, a.out)
    prov = f"  {len(good)} of {len(good) + len(bad)} labels verified on the API clip.\n" + "\n".join(
        f"  REJECTED {s}: {why}" for s, why in bad)
    open(os.path.join(a.out, "SOURCES.txt"), "w", encoding="utf-8").write(SOURCES.format(provenance=prov))
    print(os.path.join(a.out, "SOURCES.txt"))


if __name__ == "__main__":
    main()
