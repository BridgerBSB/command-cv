"""Does OpenCommand's glove_tracking.pt (class 0) put the SET-UP target where our validated
Asheville targets are? Gate for swapping it in as --mitt-weights at new parks (the in-house mitt
model lands on the umpire's mask off Asheville). Usage: python scripts/check_glove_model_setup.py 640
"""
import sys, os
sys.path.insert(0, r"C:\Users\Owner\bsb-resources\command-cv")
import numpy as np, pandas as pd
from ultralytics import YOLO
from src.auto_glove import detect_setup_catch
oc = YOLO(r"C:\Users\Owner\bsb-resources\command-cv\models\od\YOLO\glove_tracking\model_weights\glove_tracking.pt")
class GloveOnly:
    def __init__(self, imgsz): self.imgsz = imgsz
    def predict(self, fr, imgsz=None, **k):
        k.pop("device", None)
        return oc.predict(fr, imgsz=self.imgsz, classes=[0], device=0, **k)
tg = pd.read_csv(r"C:\Users\Owner\bsb-resources\command-cv\output\targets_822515_stab2c.csv").set_index("stem")
imgsz = int(sys.argv[1])
d = []
for s in tg.index:
    r = detect_setup_catch(rf"C:\Users\Owner\bsb-resources\command-cv\data\clips\ogando_2026\{s}.mp4", GloveOnly(imgsz))
    if not r["ok"]:
        d.append(np.nan); continue
    x, y, _ = r["setup"]
    d.append(np.hypot(x - tg.loc[s, "raw_px"], y - tg.loc[s, "raw_py"]))
d = pd.Series(d, index=tg.index)
print(f"imgsz {imgsz}: {d.notna().sum()}/{len(d)} set-ups found; px from our target: median {d.median():.1f}, p90 {d.quantile(.9):.1f}, >10px {int((d>10).sum())}")
d.to_csv(rf"C:\Users\Owner\bsb-resources\command-cv\output\oc_setup_check_{imgsz}.csv")
