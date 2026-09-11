"""The clip-resolution guard and frame accuracy in scripts/ingest_annotation.py.

Both tests are written from the real defect: 12 of 55 labels were ingested against a
same-named video from a different pull, and frame seeks land early. Each test fails on
the pre-fix behaviour (picking the first candidate; cap.set seeking).
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from ingest_annotation import _read_frame, resolve_clip  # noqa: E402


def _write_clip(path, n=40, tag=0):
    """n frames whose grey level encodes the frame index (+tag), so a decoded frame
    identifies itself."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for i in range(n):
        vw.write(np.full((64, 64, 3), (i * 3 + tag) % 256, np.uint8))
    vw.release()
    return path


def test_same_name_in_two_folders_is_refused(tmp_path, monkeypatch):
    import ingest_annotation as mod
    monkeypatch.setattr(mod, "ROOT", str(tmp_path))
    _write_clip(str(tmp_path / "data" / "clips" / "ogando_ff_1.mp4"), tag=0)
    _write_clip(str(tmp_path / "data" / "clips" / "ogando_2026" / "ogando_ff_1.mp4"), tag=90)
    js = tmp_path / "ann.json"
    js.write_text(json.dumps({"stem": "ogando_ff_1", "clip": "ogando_ff_1.mp4"}))
    path, why = resolve_clip({"stem": "ogando_ff_1", "clip": "ogando_ff_1.mp4"}, str(js))
    assert path is None and "AMBIGUOUS" in why and "ogando_2026" in why


def test_clips_dir_disambiguates(tmp_path, monkeypatch):
    import ingest_annotation as mod
    monkeypatch.setattr(mod, "ROOT", str(tmp_path))
    _write_clip(str(tmp_path / "data" / "clips" / "ogando_ff_1.mp4"), tag=0)
    api = _write_clip(str(tmp_path / "data" / "clips" / "ogando_2026" / "ogando_ff_1.mp4"), tag=90)
    js = tmp_path / "ann.json"
    js.write_text("{}")
    path, why = resolve_clip({"stem": "ogando_ff_1", "clip": "ogando_ff_1.mp4"}, str(js),
                             clips_dir=str(tmp_path / "data" / "clips" / "ogando_2026"))
    assert why is None and os.path.realpath(path) == os.path.realpath(api)


def test_unique_name_still_resolves(tmp_path, monkeypatch):
    import ingest_annotation as mod
    monkeypatch.setattr(mod, "ROOT", str(tmp_path))
    only = _write_clip(str(tmp_path / "data" / "clips" / "ogando_2026" / "ogando_si_17.mp4"))
    js = tmp_path / "ann.json"
    js.write_text("{}")
    path, why = resolve_clip({"stem": "ogando_si_17", "clip": "ogando_si_17.mp4"}, str(js))
    assert why is None and os.path.realpath(path) == os.path.realpath(only)


def _decode_all(clip):
    cap = cv2.VideoCapture(clip)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


@pytest.mark.parametrize("idx", [0, 7, 23, 39])
def test_read_frame_returns_that_exact_frame(tmp_path, idx):
    """Ground truth is the clip decoded in order: frame idx is the (idx+1)-th frame out.
    Asserting a grey VALUE instead would test the codec, not the reader (mp4v is lossy)."""
    clip = _write_clip(str(tmp_path / "c.mp4"), n=40)
    assert np.array_equal(_read_frame(clip, idx), _decode_all(clip)[idx])
