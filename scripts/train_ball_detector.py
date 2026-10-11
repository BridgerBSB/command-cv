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

TRAIN ON THE GPU: pass --device 0. The default is still cpu, and that default is the trap that
had this project declared "cannot train here" for six days (probe-capability-before-accepting-
a-blocker.md). Measured on data/yolo_milb (2,345 imgs, yolo11n, 320 px, batch 16):
    Surface, RTX 3050 Ti, CUDA              ~54 s/epoch   (data/yolo_ball_runs/milb4park)
    ASUS PX13, Radeon 8060S, ROCm 7.2.1      ~20 s/epoch   (amd_bench_1010_bn, 2026-10-10)
On the AMD laptop run it from the ROCm venv: ~\\.venvs\\cv-rocm\\Scripts\\python.exe

    python scripts/train_ball_detector.py --data data/yolo_ball/data.yaml --epochs 30 --device 0
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _native_batchnorm_on_rocm(device) -> bool:
    """Route batch-norm to PyTorch's own kernel on AMD ROCm. No-op on CUDA and CPU.

    MIOpen compiles its batch-norm TRAINING kernel at runtime with hipRTC, and the ROCm
    Windows wheel (torch 2.9.1+rocm7.2.1, gfx1151) cannot find <type_traits>:
    HIPRTC_ERROR_COMPILATION -> miopenStatusUnknownError on the first training step
    (2026-10-10). Inference never hits it, so the AMP check passes and the crash comes
    after. Convolutions keep MIOpen; only batch-norm goes native, and its backward follows
    the forward's kernel. Measured the same learning curve as the CUDA baseline.
    """
    import torch
    if str(device).lower() == "cpu" or not getattr(torch.version, "hip", None):
        return False
    from torch.nn.modules.batchnorm import _BatchNorm
    if getattr(_BatchNorm.forward, "_native_bn", False):      # blocking rule #21
        return True
    orig = _BatchNorm.forward

    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            return orig(self, x)
    forward._native_bn = True
    _BatchNorm.forward = forward
    return True


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
    if _native_batchnorm_on_rocm(a.device):
        print("ROCm: batch-norm on PyTorch's native kernel (MIOpen's runtime compile is broken "
              "on Windows)", flush=True)

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
    # The ROCm build can spin at interpreter teardown (2026-10-06), which leaves a 1 GB
    # process holding the GPU after the run has finished. Skip the teardown there.
    import torch
    if getattr(torch.version, "hip", None):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
