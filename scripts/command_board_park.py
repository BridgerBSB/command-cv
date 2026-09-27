"""Per-pitcher command board for one game, from command_miss.py output.

WHAT EACH NUMBER IS, so nobody has to guess:
  miss    = distance from the catcher's SET-UP glove (the target) to where the ball crossed,
            inches. The crossing is the video ball where its track reaches the plate, else
            the tracking system's plate_x / plate_z (command_miss.py FALLBACK RULE).
  arm     = signed miss toward the pitcher's ARM side, inches (negative = glove side).
            plate_x is catcher's view, + = catcher's right, so a RHP's arm side is -x.
  up      = signed miss above the target, inches (negative = below).
  n       = pitches with a plausible target. Printed on every row; no sample floor is
            applied -- read small n as small n.
Only pitches whose target passes OpenCommand's plausibility screen are used (the same
screen command_miss prints); the count dropped by it is reported.

WHAT IT IS NOT. Where the catcher sets up is not always where the pitcher was trying to
throw (a pitchout, a set-up the pitcher shakes off, a "just miss" intent). This is miss vs
the GLOVE, which is the standard OpenCommand-style definition, not miss vs intent.

    python scripts/command_board_park.py --game 821819 --tag _final
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVY, ORANGE = "#002D62", "#EB6E1F"
PT_ORDER = ["Four-seam FB", "Sinker", "Cutter", "Slider", "Sweeper", "Curveball",
            "Knuckle Curve", "Changeup", "Splitter"]
PT_COLOR = {"Four-seam FB": "#D22D49", "Sinker": "#FE9D00", "Cutter": "#933F2C",
            "Slider": "#EEE716", "Sweeper": "#DDB33A", "Curveball": "#00D1ED",
            "Knuckle Curve": "#6236CD", "Changeup": "#1DBE3A", "Splitter": "#3BACAC"}


def board(game, tag, src="auto"):
    d = pd.read_csv(os.path.join(ROOT, "output", "command_miss_%d%s.csv" % (game, tag)))
    ctx = pd.read_csv(os.path.join(ROOT, "output", "pitch_context_%d.csv" % game))
    d = d[d.target_src == src].merge(ctx[["stem", "pitcher_name", "p_throws", "bat_side"]],
                                     on="stem", how="left")
    assert d.pitcher_name.notna().all(), "pitches with no pitcher -- rerun pitch_context.py"
    n_all = len(d)
    d = d[d.plaus].copy()
    # final crossing minus target, per the fallback rule
    d["dx"] = np.where(d.final_src == "B", d.B_dx, d.A_dx)
    d["dz"] = np.where(d.final_src == "B", d.B_dz, d.A_dz)
    d["arm"] = np.where(d.p_throws == "R", -d.dx, d.dx)
    d["up"] = d.dz
    return d, n_all


def summarise(d, keys):
    g = d.groupby(keys)
    return pd.DataFrame(dict(n=g.size(), miss_med=g.final_miss.median(),
                             arm_mean=g.arm.mean(), up_mean=g.up.mean(),
                             video_share=g.final_src.apply(lambda s: (s == "B").mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--tag", default="_final")
    ap.add_argument("--park", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d, n_all = board(a.game, a.tag)
    print("game %d%s: %d pitches with a target, %d pass the plausibility screen"
          % (a.game, a.tag, n_all, len(d)))
    pit = summarise(d, ["pitcher_name", "p_throws"]).sort_values("n", ascending=False)
    typ = summarise(d, ["pitcher_name", "pitch_type"])
    pd.set_option("display.width", 160)
    print(pit.round(1).to_string())
    print(typ.round(1).to_string())
    base = a.out or os.path.join(ROOT, "output", "command_board_%d" % a.game)
    pit.round(3).to_csv(base + "_pitchers.csv")
    typ.round(3).to_csv(base + "_types.csv")

    # one panel per pitcher: every pitch's miss vector from the target (catcher's view, so
    # the picture matches the broadcast zone), coloured by pitch type
    names = list(pit.index.get_level_values(0))
    ncol = min(5, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(3.1 * ncol, 8.0), 3.6 * nrow + 1.9), squeeze=False)
    lim = 30
    for ax, name in zip(axes.ravel(), names):
        s = d[d.pitcher_name == name]
        ax.add_patch(plt.Circle((0, 0), 6, fill=False, ls=":", lw=0.8, color="#888"))
        ax.add_patch(plt.Circle((0, 0), 12, fill=False, ls=":", lw=0.8, color="#bbb"))
        ax.axhline(0, color="#ddd", lw=0.6, zorder=0)
        ax.axvline(0, color="#ddd", lw=0.6, zorder=0)
        for pt, q in s.groupby("pitch_type"):
            ax.scatter(q.dx.clip(-lim, lim), q.dz.clip(-lim, lim), s=14,
                       color=PT_COLOR.get(pt, "#777"), edgecolor="k", lw=0.3, label=pt, zorder=3)
        ax.plot(0, 0, marker="+", color=NAVY, ms=10, mew=2, zorder=4)
        r = pit.loc[name].iloc[0]
        hand = pit.loc[name].index[0]
        ax.set_title("%s (%sHP)\nn %d   median miss %.1f in" % (name, hand, r.n, r.miss_med),
                     fontsize=8.5, color=NAVY)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5)
        # arm side label: RHP arm side is catcher's LEFT (-x)
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
    fig.suptitle("Command miss vs the catcher's set-up glove - %s%s\n"
                 "catcher's view, inches; dotted rings 6 and 12 in; points past 30 in drawn at the edge"
                 % (a.park + " " if a.park else "", "game %d" % a.game), fontsize=10, color=NAVY)
    fig.subplots_adjust(left=0.04, right=0.99, top=1 - 0.95 / fig.get_figheight(),
                        bottom=0.9 / fig.get_figheight(), hspace=0.32, wspace=0.28)
    png = base + ".png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("->", png)


if __name__ == "__main__":
    main()
