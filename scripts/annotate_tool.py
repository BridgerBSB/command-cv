"""Interactive mitt annotator — tag the settled glove (setup) and the catch.

Per clip: scrub to the frame where the glove has SETTLED, click the mitt center
then click its edge (sets the box), press 's' to lock it as SETUP; scrub to the
CATCH frame, click + 'c'. Press 'w' to write the pitch annotation (JSON) plus
YOLO training samples for both frames. This is the label set that trains the
mitt detector.

Controls (focus the image window):
  a / d        prev / next frame            A / D      jump back / fwd 10
  left-click   1st = mitt center, 2nd = mitt edge (radius -> square box)
  s            lock current box+frame as SETUP target
  c            lock current box+frame as CATCH
  t            type Trackman pX pZ szTop szBot in the console (optional, Method A)
  w            write annotation + YOLO samples (needs setup + catch)
  q            quit

Run: python scripts/annotate_tool.py --clip "path/ogando_ff_1.mp4" --hand R
Filename "<pitcher>_<type>_<n>.mp4" sets pitcher + pitch_type automatically.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Box, save_sample, write_data_yaml


def _meta_from_name(clip: str):
    stem = os.path.splitext(os.path.basename(clip))[0]
    parts = stem.split("_")
    pitcher = parts[0] if parts else stem
    ptype = parts[1] if len(parts) > 1 else ""
    return stem, pitcher, ptype


class Annotator:
    def __init__(self, clip, hand):
        import cv2
        self.clip = clip
        self.hand = hand
        self.stem, self.pitcher, self.ptype = _meta_from_name(clip)
        cap = cv2.VideoCapture(clip)
        self.frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            self.frames.append(fr)
        cap.release()
        if not self.frames:
            raise SystemExit(f"no frames in {clip}")
        self.i = len(self.frames) // 2
        self.click_center = None
        self.cur_box = None
        self.setup = None   # (frame_idx, Box)
        self.catch = None
        self.trackman = {}

    def run(self):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.draw()
        print(f"annotating {self.stem}  pitcher={self.pitcher} type={self.ptype} "
              f"hand={self.hand}  ({len(self.frames)} frames)")
        plt.show()

    def draw(self):
        self.ax.clear()
        self.ax.imshow(self.frames[self.i][:, :, ::-1])
        self.ax.axis("off")
        s = f"setup={'f%d' % self.setup[0] if self.setup else '-'}  " \
            f"catch={'f%d' % self.catch[0] if self.catch else '-'}"
        self.ax.set_title(f"frame {self.i}/{len(self.frames)-1}   {s}   "
                          f"[a/d step, s=setup c=catch w=write q=quit]")
        if self.cur_box:
            b = self.cur_box
            self.ax.add_patch(self.plt.Rectangle(
                (b.cx - b.w / 2, b.cy - b.h / 2), b.w, b.h,
                fill=False, ec="#EB6E1F", lw=2))
        self.fig.canvas.draw_idle()

    def on_click(self, e):
        if e.xdata is None:
            return
        if self.click_center is None:
            self.click_center = (e.xdata, e.ydata)
            print(f"  center ({e.xdata:.0f},{e.ydata:.0f}) — now click the mitt edge")
        else:
            cx, cy = self.click_center
            r = ((e.xdata - cx) ** 2 + (e.ydata - cy) ** 2) ** 0.5
            self.cur_box = Box.from_center_radius(cx, cy, max(r, 6))
            self.click_center = None
            print(f"  box r={r:.0f}px — press 's' (setup) or 'c' (catch)")
            self.draw()

    def on_key(self, e):
        k = e.key
        if k in ("d", "right"):
            self.i = min(self.i + 1, len(self.frames) - 1); self.cur_box = None; self.draw()
        elif k in ("a", "left"):
            self.i = max(self.i - 1, 0); self.cur_box = None; self.draw()
        elif k == "D":
            self.i = min(self.i + 10, len(self.frames) - 1); self.cur_box = None; self.draw()
        elif k == "A":
            self.i = max(self.i - 10, 0); self.cur_box = None; self.draw()
        elif k == "s" and self.cur_box:
            self.setup = (self.i, self.cur_box); print(f"  SETUP locked @ f{self.i}"); self.draw()
        elif k == "c" and self.cur_box:
            self.catch = (self.i, self.cur_box); print(f"  CATCH locked @ f{self.i}"); self.draw()
        elif k == "t":
            self._read_trackman()
        elif k == "w":
            self.write()
        elif k == "q":
            self.plt.close(self.fig)

    def _read_trackman(self):
        try:
            raw = input("  Trackman pX pZ szTop szBot (blank=skip): ").split()
            if len(raw) >= 2:
                self.trackman = {"plate_x": float(raw[0]), "plate_z": float(raw[1])}
                if len(raw) >= 4:
                    self.trackman["sz_top"] = float(raw[2])
                    self.trackman["sz_bot"] = float(raw[3])
                print("  trackman set:", self.trackman)
        except Exception as ex:
            print("  skipped:", ex)

    def write(self):
        if not (self.setup and self.catch):
            print("  need BOTH setup and catch before writing"); return
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data")
        ann_dir = os.path.join(root, "annotations")
        os.makedirs(ann_dir, exist_ok=True)

        def box_dict(fb):
            i, b = fb
            return {"frame": i, "cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h}

        ann = {"clip": os.path.basename(self.clip), "stem": self.stem,
               "pitcher": self.pitcher, "pitch_type": self.ptype, "hand": self.hand,
               "setup": box_dict(self.setup), "catch": box_dict(self.catch),
               "trackman": self.trackman}
        with open(os.path.join(ann_dir, f"{self.stem}.json"), "w") as f:
            json.dump(ann, f, indent=2)

        # YOLO training samples for both frames
        write_data_yaml()
        for tag, (fi, b) in (("setup", self.setup), ("catch", self.catch)):
            save_sample(self.frames[fi], b, f"{self.stem}_{tag}_f{fi}")
        print(f"  WROTE annotation + 2 YOLO samples for {self.stem}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--hand", default="R")
    a = p.parse_args()
    Annotator(a.clip, a.hand).run()


if __name__ == "__main__":
    main()
