"""Per-pitcher command profile: append each measured miss + session rollups.

CSV-backed so it survives across sessions and is trivially readable by an Arm
Farm tab later. Full float precision is carried through; rounding happens only
at display (BLOCKING rule: never round until display).
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass

from .geometry import MissResult

FIELDS = [
    "pitch_id", "pitcher_id", "pitcher_name", "pitcher_throws",
    "session_date", "pitch_type",
    "intended_x", "intended_z", "actual_x", "actual_z",
    "miss_in", "horiz_in", "vert_in", "horiz_label", "vert_label",
]


@dataclass
class CommandRow:
    pitch_id: str
    pitcher_id: str
    pitcher_name: str
    pitcher_throws: str
    session_date: str
    pitch_type: str
    intended_x: float
    intended_z: float
    actual_x: float
    actual_z: float
    miss_in: float
    horiz_in: float
    vert_in: float
    horiz_label: str
    vert_label: str

    @classmethod
    def from_miss(cls, miss: MissResult, *, pitch_id: str, pitcher_id: str,
                  pitcher_name: str, pitcher_throws: str, session_date: str,
                  pitch_type: str = "") -> "CommandRow":
        return cls(
            pitch_id=pitch_id, pitcher_id=pitcher_id, pitcher_name=pitcher_name,
            pitcher_throws=pitcher_throws, session_date=session_date,
            pitch_type=pitch_type,
            intended_x=miss.intended_x, intended_z=miss.intended_z,
            actual_x=miss.actual_x, actual_z=miss.actual_z,
            miss_in=miss.miss_in, horiz_in=miss.horiz_in, vert_in=miss.vert_in,
            horiz_label=miss.horiz_label, vert_label=miss.vert_label,
        )


def append_row(csv_path: str, row: CommandRow) -> None:
    """Append one command row, writing a header if the file is new."""
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(asdict(row))


def session_summary(rows: list[CommandRow]) -> dict:
    """Avg miss + directional bias for a set of rows (full precision)."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    avg_miss = sum(r.miss_in for r in rows) / n
    avg_horiz = sum(r.horiz_in for r in rows) / n   # signed: bias
    avg_vert = sum(r.vert_in for r in rows) / n
    return {
        "n": n,
        "avg_miss_in": avg_miss,
        "horiz_bias_in": avg_horiz,
        "vert_bias_in": avg_vert,
    }
