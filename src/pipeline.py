"""Orchestrator: a glove-flash frame's detections + a Trackman row -> miss + report.

Stages [3]-[6] of the pipeline (calibrate, compute, report, profile). Detection
(stage 2) is supplied via a Detector; ingest (stage 0/1) supplies the Trackman
row + frame upstream.
"""
from __future__ import annotations

from dataclasses import dataclass

from .calibration import home_plate_world_points, solve_homography, strikezone_world_points
from .detect import FrameDetections
from .geometry import MissResult, StrikeZone, compute_miss
from .profile import CommandRow, append_row
from .report import render_strike_zone_view


@dataclass
class TrackmanPitch:
    """The fields we need from a Trackman/Statcast pitch row."""
    pitch_id: str
    plate_x: float            # ft, catcher view
    plate_z: float            # ft
    pitcher_throws: str = "R"
    sz_top: float | None = None
    sz_bot: float | None = None
    pitcher_id: str = ""
    pitcher_name: str = ""
    pitch_type: str = ""
    session_date: str = ""


@dataclass
class PitchResult:
    miss: MissResult
    zone: StrikeZone
    report_path: str | None


def process_pitch(
    det: FrameDetections,
    tm: TrackmanPitch,
    report_path: str | None = None,
    profile_csv: str | None = None,
) -> PitchResult:
    """Run calibrate -> compute -> (report) -> (profile) for one pitch."""
    zone = StrikeZone.from_trackman(tm.sz_top, tm.sz_bot)

    # Build pixel<->world correspondences from whatever detections we have.
    img_pts = list(det.plate_pts)
    world_pts = list(home_plate_world_points())
    if det.zone_pts:
        img_pts += list(det.zone_pts)
        world_pts += strikezone_world_points(zone.z_bot, zone.z_top)
    if len(img_pts) < 4:
        raise ValueError(
            "calibration needs >=4 reference points; supply strike-zone corners "
            "in addition to plate corners (2 plate pts alone is insufficient)."
        )

    homography = solve_homography(img_pts, world_pts)
    intended_x, intended_z = homography.pixel_to_world(*det.glove_px)

    miss = compute_miss(intended_x, intended_z, tm.plate_x, tm.plate_z,
                        pitcher_throws=tm.pitcher_throws)

    out = None
    if report_path:
        subtitle = " · ".join(p for p in [tm.pitcher_name, tm.pitch_type] if p)
        out = render_strike_zone_view(
            miss, zone, report_path,
            pitch_label=f"Pitch #{tm.pitch_id}" if tm.pitch_id else "",
            subtitle=subtitle,
        )

    if profile_csv:
        append_row(profile_csv, CommandRow.from_miss(
            miss, pitch_id=tm.pitch_id, pitcher_id=tm.pitcher_id,
            pitcher_name=tm.pitcher_name, pitcher_throws=tm.pitcher_throws,
            session_date=tm.session_date, pitch_type=tm.pitch_type))

    return PitchResult(miss=miss, zone=zone, report_path=out)
