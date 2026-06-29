"""Method B on a real clip: setup vs catch mitt displacement -> command miss.

Proposes the mitt with the color+motion detector; override with explicit pixel
coords for a trustworthy number (until the YOLO detector is trained). Renders an
overlay on the catch frame (orange = setup target, navy = where it ended up) +
the miss readout, and appends to the command profile.

Examples:
  python scripts/methodb_pitch.py --clip ogando_ff_1.mp4 --setup-frame 90 --catch-frame 113 --hand R
  python scripts/methodb_pitch.py --clip ogando_ff_1.mp4 --setup-frame 90 --catch-frame 113 \
      --setup-xy 862 478 --catch-xy 861 470 --diam 54 --hand R --pitcher Ogando
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.glove_track import detect_mitt, quiet_reference
from src.miss_methodb import GlovePx, compute_miss_b


def _load_frames(clip):
    import cv2
    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit(f"could not read frames from {clip}")
    return frames


def _resolve(frames, frame_idx, xy, diam, ref):
    if xy is not None:
        d = diam or (detect_mitt(frames[frame_idx], ref_frame=ref).diameter_px
                     if detect_mitt(frames[frame_idx], ref_frame=ref) else 50.0)
        return GlovePx(cx=xy[0], cy=xy[1], diameter_px=d), True
    g = detect_mitt(frames[frame_idx], ref_frame=ref)
    if g is None:
        raise SystemExit(f"detector found no mitt at frame {frame_idx}; "
                         f"pass --setup-xy/--catch-xy to set it manually")
    if diam:
        g = GlovePx(g.cx, g.cy, diam)
    return g, False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--setup-frame", type=int, required=True)
    p.add_argument("--catch-frame", type=int, required=True)
    p.add_argument("--setup-xy", type=float, nargs=2)
    p.add_argument("--catch-xy", type=float, nargs=2)
    p.add_argument("--diam", type=float, help="override mitt diameter px (the ruler)")
    p.add_argument("--glove-in", type=float, default=12.0)
    p.add_argument("--hand", default="R")
    p.add_argument("--pitcher", default="")
    p.add_argument("--pitch-id", default="")
    p.add_argument("--outdir", default=None)
    a = p.parse_args()

    frames = _load_frames(a.clip)
    ref = quiet_reference(frames)
    setup, s_manual = _resolve(frames, a.setup_frame, a.setup_xy, a.diam, ref)
    catch, c_manual = _resolve(frames, a.catch_frame, a.catch_xy, a.diam, ref)

    miss = compute_miss_b(setup, catch, pitcher_throws=a.hand,
                          glove_diameter_in=a.glove_in)
    src = ("manual" if s_manual else "auto") + "/" + ("manual" if c_manual else "auto")
    print(f"{a.pitcher or os.path.basename(a.clip)}: {miss.direction}  "
          f"[setup f{a.setup_frame} -> catch f{a.catch_frame}, {src}, "
          f"scale={miss.scale_in_per_px:.3f} in/px]")

    outdir = a.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.clip))[0]
    out = os.path.join(outdir, f"methodb_{base}.png")
    _render_overlay(frames[a.catch_frame], setup, catch, miss, out,
                    label=f"{a.pitcher or base}: {miss.miss_in:.1f}in "
                          f"{miss.horiz_label}/{miss.vert_label}")
    print("wrote", out)


def _render_overlay(frame_bgr, setup, catch, miss, out_path, label=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = frame_bgr[:, :, ::-1]
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / 110, h / 110), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(rgb); ax.axis("off")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    NAVY, ORANGE = "#002D62", "#EB6E1F"
    ax.plot([setup.cx, catch.cx], [setup.cy, catch.cy], color="#C9A23A",
            lw=2.4, ls=(0, (6, 5)))
    # setup target (orange reticle)
    r = setup.diameter_px / 2
    ax.add_artist(plt.Circle((setup.cx, setup.cy), r, fill=False, ec=ORANGE, lw=2.6))
    ax.plot([setup.cx - r, setup.cx + r], [setup.cy, setup.cy], color=ORANGE, lw=1.8)
    ax.plot([setup.cx, setup.cx], [setup.cy - r, setup.cy + r], color=ORANGE, lw=1.8)
    # catch (navy dot)
    ax.scatter([catch.cx], [catch.cy], s=160, c=NAVY, edgecolors="white", lw=1.6)
    if label:
        ax.text(12, 26, label, color="white", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=NAVY, ec="none", alpha=0.85))
    ax.text(12, h - 16, "orange = setup target   navy = catch (actual)",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5))
    fig.savefig(out_path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
