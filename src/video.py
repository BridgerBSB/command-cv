"""Video frame extraction for the provided-clip path.

Decode a clip (the user-supplied behind-plate video) and pull the single
glove-flash frame by index or timestamp. Returns RGB ndarrays so matplotlib and
the report overlay consume them directly. cv2 is imported lazily so the rest of
the engine never depends on it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClipInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str) -> ClipInfo:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open clip: {path}")
    info = ClipInfo(
        path=path,
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def frame_at_index(path: str, index: int):
    """Return the RGB ndarray at frame `index` (0-based)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open clip: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"could not read frame {index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_at_time(path: str, seconds: float):
    """Return the RGB ndarray nearest timestamp `seconds`."""
    info = probe(path)
    return frame_at_index(path, int(round(seconds * info.fps)))


def save_frame(frame, out_path: str) -> str:
    import os
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return out_path
