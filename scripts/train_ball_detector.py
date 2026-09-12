"""Train the ball detector on the auto-generated labels.

The labels come from geometry, not from anyone clicking: `ball_clock_align.py` projects the
known trajectory through the solved camera, so every flight frame carries a ball position and
size. `build_ball_dataset.py` turns those into crops. This trains on them.

WHAT THIS SCRIPT'S NUMBER IS WORTH. mAP on these crops is a weak gate and should not be quoted
as the result. Two reasons:
  1. with one game, train and val share a camera and a stadium, so val is near-duplicate data
  2. the labels are geometry, so the detector is being graded against the same projection that
     produced them, not against an independent truth
The REAL gate is downstream: feed this detector's ball positions into the joint camera solve
(scripts/ablate_no_box.py) and see whether the camera still lands near the box-assisted answer
and whether the held-out box corners still come back inside an inch. Until that runs, this is
plumbing with a loss curve.

CPU reality on this laptop: torch is CPU-only and single-threaded, ~0.25 s per image per epoch
at 320 px. 5,600 images x 60 epochs is about a day, so the defaults here are deliberately small
and --epochs is the dial. Run it overnight or cut --fraction.

    python scripts/train_ball_detector.py --data data/yolo_ball/data.yaml --epochs 30
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "yolo_ball", "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt", help="n is the only sane choice on CPU")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=320, help="crops are 320 px at native resolution")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="<1 to trial on part of the set")
    ap.add_argument("--name", default="ball")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-amp", action="store_true",
                    help="skip the Automatic Mixed Precision check. That check allocates a CUDA "
                         "workspace and runs an inference probe at STARTUP, and on this box it is "
                         "where the 4th watchdog kill landed - before a single epoch ran. Trading "
                         "some speed for a process that survives startup is worth it here.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pt of --name. This box's memory watchdog kills "
                         "long runs (it took one at epoch 3/30), and ultralytics writes last.pt "
                         "every epoch, so a kill should cost ONE epoch and not the run. Wrap "
                         "this in a restart loop and it grinds forward on its own.")
    a = ap.parse_args()
    assert os.path.exists(a.data), f"no dataset: {a.data} (run build_ball_dataset.py first)"
    from ultralytics import YOLO

    out = os.path.join(ROOT, "data", "yolo_ball_runs")
    last = os.path.join(out, a.name, "weights", "last.pt")
    amp = not a.no_amp
    if a.resume and os.path.exists(last):
        # NOTE: resume restores the ORIGINAL run's args (batch, epochs, imgsz) from the
        # CHECKPOINT by design - amp=True included. So amp must be passed EXPLICITLY here or
        # --no-amp is silently ignored on exactly the path that needs it: the 4th watchdog
        # kill landed during the AMP check on a resume, before a single epoch ran.
        print(f"resuming from {last}  (amp={amp})", flush=True)
        r = YOLO(last).train(resume=True, amp=amp)
        print(f"\nweights: {os.path.join(str(r.save_dir), 'weights', 'best.pt')}")
        return
    if a.resume:
        print(f"--resume given but no {last}; starting fresh", flush=True)
    m = YOLO(a.model)
    r = m.train(data=os.path.abspath(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                fraction=a.fraction, device=a.device, workers=0, project=out, name=a.name,
                exist_ok=True, plots=True, val=True, amp=amp,
                # the ball is a small, roughly round, colour-stable object: geometric warping
                # helps, colour jitter and mosaic mostly do not
                mosaic=0.0, mixup=0.0, erasing=0.0, hsv_s=0.3, hsv_v=0.3,
                degrees=0.0, translate=0.15, scale=0.3, fliplr=0.5)
    w = os.path.join(str(r.save_dir), "weights", "best.pt")
    print(f"\nweights: {w}")
    print("NEXT, and this is the actual test: swap these detections into the joint camera solve "
          "and compare the camera position and the held-out box corners against the run that "
          "used OpenCommand's detections. mAP above is not the result.")


if __name__ == "__main__":
    main()
