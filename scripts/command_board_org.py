"""Per-pitcher command board for OUR pitchers, pooled across every game that has a miss.

Same per-pitch numbers as command_board_park.py (it imports board() and summarise(), so the
arm-side sign, the plausibility screen and the two roots have ONE definition):
  miss  = set-up glove (target, from video) to where the ball crossed, inches
          video = the detector's ball at the plate (only where the track reaches it)
          data  = the tracking system's plate_x / plate_z (every pitch)
          Both are reported, never blended (Zac, 09-27); one PNG per root.
  arm   = signed miss toward the pitcher's arm side; up = above the target.
  n     = pitches with a plausible target. No sample floor: read small n as small n.

OURS = pitch_team_org == 117 (Houston) from pitch_context.py, i.e. the pitching team's parent
org in the game feed, so road games count as well as home games. A game whose context file
predates that column is refused rather than guessed.

Each game is listed with the camera it was measured through, so a board never mixes in a
park whose camera has not been graded. --games takes game:tag pairs.

    python scripts/command_board_org.py --name pilot --games 814874:_bo 815419:_s2c
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from command_board_park import NAVY, PT_COLOR, PT_ORDER, board, summarise  # noqa: E402

HOU = 117
# Panel titles carry the LEVEL: club names ran off the right edge on a 5-wide row, and a pitcher
# who pitched for two affiliates ("Corpus/Sugar Land") still did after shortening them.
SHORT = {"Houston Astros": "MLB", "Sugar Land Space Cowboys": "AAA", "Corpus Christi Hooks": "AA",
         "Asheville Tourists": "A+", "Fayetteville Woodpeckers": "A"}
LEVEL_ORDER = ["MLB", "AAA", "AA", "A+", "A"]


def load(games):
    parts, log = [], []
    for g, tag in games:
        ctx = pd.read_csv(os.path.join(ROOT, "output", "pitch_context_%d.csv" % g))
        assert "pitch_team_org" in ctx, "pitch_context_%d.csv predates pitch_team_org -- rerun pitch_context.py" % g
        d, n_all = board(g, tag, extra_ctx=("pitcher_id", "pitch_team", "pitch_team_org"))
        ours = d[d.pitch_team_org == HOU].assign(game=g, tag=tag)
        log.append((g, tag, n_all, len(d), len(ours), ", ".join(sorted(ours.pitch_team.dropna().unique()))))
        parts.append(ours)
    return pd.concat(parts, ignore_index=True), log


def draw(d, pit, M, title, png):
    names = list(pit.index.get_level_values(0))
    ncol = min(5, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    # cells near square: the panels are equal-aspect, so a taller cell is only white space
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(3.1 * ncol, 8.0), 3.0 * nrow + 2.15), squeeze=False)
    lim = 30
    for ax, name in zip(axes.ravel(), names):
        s = d[d.pitcher_name == name]
        ax.add_patch(plt.Circle((0, 0), 6, fill=False, ls=":", lw=0.8, color="#888"))
        ax.add_patch(plt.Circle((0, 0), 12, fill=False, ls=":", lw=0.8, color="#bbb"))
        ax.axhline(0, color="#ddd", lw=0.6, zorder=0)
        ax.axvline(0, color="#ddd", lw=0.6, zorder=0)
        for pt, q in s.groupby("pitch_type"):
            q = q[q[M + "_miss"].notna()]
            ax.scatter(q[M + "_dx"].clip(-lim, lim), q[M + "_dz"].clip(-lim, lim), s=14,
                       color=PT_COLOR.get(pt, "#777"), edgecolor="k", lw=0.3, label=pt, zorder=3)
        ax.plot(0, 0, marker="+", color=NAVY, ms=10, mew=2, zorder=4)
        r = pit.loc[name].iloc[0]
        hand = pit.loc[name].index[0]
        ax.set_title("%s (%sHP), %s\nn %d   median miss %.1f in"
                     % (name, hand, r["team"], r["n_" + M], r[M + "_miss_med"]), fontsize=8.5, color=NAVY)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5)
        armx = -lim * 0.95 if hand == "R" else lim * 0.95
        ax.text(armx, -lim * 0.92, "arm side", fontsize=6.5, color="#666",
                ha="left" if hand == "R" else "right")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labs = [p for p in PT_ORDER if p in handles] + sorted(set(handles) - set(PT_ORDER))
    fig.legend([handles[l] for l in labs], labs, loc="lower center", ncol=min(len(labs), 6),
               fontsize=7.5, frameon=False)
    # pinned in INCHES from the top: at 5 rows the default y=0.98 put its second line on top of
    # the first row of panel titles
    H = fig.get_figheight()
    fig.suptitle(title, fontsize=10, color=NAVY, y=1 - 0.12 / H, va="top")
    fig.subplots_adjust(left=0.04, right=0.99, top=1 - 1.25 / H,
                        bottom=0.9 / fig.get_figheight(), hspace=0.38, wspace=0.28)
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output name: output/command_board_org_<name>_*")
    ap.add_argument("--games", nargs="+", required=True, help="game:tag pairs, e.g. 814874:_bo")
    a = ap.parse_args()
    games = [(int(x.split(":")[0]), x.split(":")[1]) for x in a.games]
    d, log = load(games)
    print("%-8s %-7s %8s %8s %6s  %s" % ("game", "tag", "targets", "plaus", "ours", "our team"))
    for g, t, n_all, n_pl, n_ours, team in log:
        print("%-8d %-7s %8d %8d %6d  %s" % (g, t, n_all, n_pl, n_ours, team))
    print("our pitches with a plausible target: %d, pitchers %d" % (len(d), d.pitcher_id.nunique()))

    pit = summarise(d, ["pitcher_name", "p_throws"])
    meta = d.groupby("pitcher_name").agg(team=("pitch_team", lambda s: "/".join(sorted({SHORT.get(v, v) for v in s.unique()},
                                                    key=lambda x: LEVEL_ORDER.index(x) if x in LEVEL_ORDER else 9))),
                                         games=("game", "nunique"))
    pit = pit.join(meta, on="pitcher_name").sort_values("n_data", ascending=False)
    typ = summarise(d, ["pitcher_name", "pitch_type"])
    pd.set_option("display.width", 200)
    print(pit.round(1).to_string())
    base = os.path.join(ROOT, "output", "command_board_org_%s" % a.name)
    pit.round(3).to_csv(base + "_pitchers.csv")
    typ.round(3).to_csv(base + "_types.csv")
    d.to_csv(base + "_pitches.csv", index=False)
    for M in ("video", "data"):
        # three short lines: one long line ran off both edges of the 8 in single-pitcher figure
        title = ("Command miss vs the catcher's set-up glove - Astros pitchers, %d games\nball location from %s\n"
                 "catcher's view, inches; dotted rings 6 and 12 in; points past 30 in drawn at the edge"
                 % (len(games), "VIDEO" if M == "video" else "plate_x / plate_z DATA"))
        draw(d, pit, M, title, base + "_" + M + ".png")
        print("->", base + "_" + M + ".png")


if __name__ == "__main__":
    main()
