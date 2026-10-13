"""One season command card per Houston pitcher still in the org, plus an index of ALL of them.

Zac, 2026-10-13: the goal is the miss distances and the chart for the WHOLE MiLB season for each
Houston pitcher still on eBIS. The org board (command_board_org.py) draws every pitcher on one
tall image; this is the per-pitcher page, built from the board's own per-pitch rows, so the miss
definitions, the arm-side sign and the plausibility screen are the board's (one definition).

  pitches   output/command_board_org_<board>_pitches.csv (written by miss_season.py --board)
  who       output/season_pitchers_2026.csv (season_pitchers.py): in_org, pitches thrown
  coverage  every game he pitched (box scores) with where it stands in the season run:
            measured / failed the camera gate / queued at a ready park / park in setup /
            park blocked / park needs boxes / not pulled

A pitcher with nothing measured yet gets NO page - and a row in the index saying where his
pitches are waiting, because an absent page reads as a broken run (tautological-display.md 4).

    python scripts/season_cards.py                    # -> output/season_cards/
    python scripts/season_cards.py --pitchers 694297  # one card
"""
import argparse
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from command_board_org import LIM, draw_panel  # noqa: E402
from command_board_park import NAVY, ORANGE, PT_COLOR, PT_ORDER, summarise  # noqa: E402

OUT = os.path.join(ROOT, "output")
LEVEL_ORDER = ["MLB", "AAA", "AA", "A+", "A"]
SHORT = {"aaa": "AAA", "aa": "AA", "aplus": "A+", "a": "A"}
# where a game he pitched stands; the order is the legend order on the strip
BUCKETS = [("measured", NAVY), ("camera failed", "#B23A3A"), ("queued", "#9AA5B1"),
           ("park in setup", "#C9CED6"), ("park blocked", ORANGE), ("not run yet", "#E4E7EB")]


def game_buckets(season):
    """game_pk -> bucket for every season game, from the season run's own plan + ledger."""
    import argparse as _ap
    import miss_season as S
    a = _ap.Namespace(games="", venue_ids="", retry_failed=False, no_setup=False,
                      trials_per_park=2, retry_games="")
    led = S.read_ledger()
    verdict = dict(zip(led.game_pk.astype(int), led.verdict))      # last row wins
    out = {}
    for j in S.plan(season, S.read_registry(), led, a):
        st, g = j["status"], j["game_pk"]
        if st == "done":
            out[g] = "camera failed" if verdict.get(g) in ("FAIL", "ERROR") else "measured"
        elif st == "run":
            out[g] = "queued"
        elif st in ("run setup", "park in setup, awaiting review", "park in setup (--no-setup)"):
            out[g] = "park in setup"
        elif st == "park blocked":
            out[g] = "park blocked"
        else:
            out[g] = "not run yet"                 # awaiting pull / needs scorebug boxes
    return out


def appearances(season):
    from season_pitchers import houston_pitchers
    rows = [r for g in season.itertuples(index=False) for r in houston_pitchers(int(g.game_pk), g.level)]
    app = pd.DataFrame(rows).merge(season[["game_pk", "date", "venue"]], on="game_pk")
    app["date"] = pd.to_datetime(app.date)
    return app


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def type_table(ax, d):
    t = summarise(d, ["pitch_type"])
    order = [p for p in PT_ORDER if p in t.index] + sorted(set(t.index) - set(PT_ORDER))
    t = t.loc[order]
    ax.axis("off")
    cols = ["Pitch", "n", "Miss", "Arm", "Up", "Video (n)"]

    def f(v, fmt):
        return "-" if pd.isna(v) else fmt % v

    def video(r):
        return "-" if not r.n_video else "%.1f (%d)" % (r.video_miss_med, r.n_video)
    cells = []
    for pt, r in t.iterrows():
        cells.append([pt, "%d" % r.n_data, f(r.data_miss_med, "%.1f"), f(r.data_arm_mean, "%+.1f"),
                      f(r.data_up_mean, "%+.1f"), video(r)])
    allr = summarise(d.assign(_all=1), ["_all"]).iloc[0]
    cells.append(["All", "%d" % allr.n_data, f(allr.data_miss_med, "%.1f"), f(allr.data_arm_mean, "%+.1f"),
                  f(allr.data_up_mean, "%+.1f"), video(allr)])
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[0.31, 0.09, 0.115, 0.115, 0.115, 0.235])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.25)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#D5D9DF")
        if i == 0:
            c.set_facecolor(NAVY)
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        elif i == len(cells):
            c.set_facecolor("#EEF1F5")
            c.get_text().set_fontweight("bold")
        if j == 0 and 0 < i < len(cells):
            c.get_text().set_color("black")
            c.set_facecolor("white")
            c.get_text().set_text("  " + c.get_text().get_text())
            c.set_text_props(ha="left")
            # a small swatch of the pitch colour, the same colour as the scatter
            c.set_facecolor(matplotlib.colors.to_rgba(PT_COLOR.get(cells[i - 1][0], "#777777"), 0.2))
    ax.set_title("By pitch type, inches\nData: Miss = median, Arm / Up = mean signed miss\n"
                 "Video = median miss (n)",
                 fontsize=8, color=NAVY, loc="left")


def season_strip(ax, d, games):
    """Every game he pitched, left to right by date: measured games show their median miss; the
    rest are ticks coloured by where the game stands in the season run."""
    per = d.groupby("game").agg(med=("data_miss", "median"), n=("data_miss", "count"))
    g = games.copy()
    g["med"] = g.game_pk.map(per.med)
    g["n"] = g.game_pk.map(per.n).fillna(0)
    m = g[g.bucket == "measured"]
    ax.scatter(m.date, m.med, s=12 + 2.2 * m.n, color=NAVY, alpha=0.85, zorder=3)
    top = max(20.0, float(np.nanmax(m.med)) + 2) if len(m) and m.med.notna().any() else 20.0
    ax.set_ylim(-2.6, top)
    for b, col in BUCKETS:
        s = g[g.bucket == b]
        ax.scatter(s.date, np.full(len(s), -1.5), marker="|", s=110, lw=2.2, color=col,
                   label="%s (%d)" % (b, len(s)), zorder=2)
    ax.axhline(0, color="#ddd", lw=0.6)
    ax.set_ylabel("median miss, in", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_xlim(pd.Timestamp("2026-03-25"), pd.Timestamp("2026-09-25"))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=6.5, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    ax.set_title("Season: each game he pitched (dot = median miss that game, size = pitches; "
                 "ticks = where the game stands)", fontsize=8, color=NAVY, loc="left")


def card(d, row, games, png, pdf):
    hand = d.p_throws.dropna().iloc[0] if d.p_throws.notna().any() else row.get("throws", "R")
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    meas_games = games[games.bucket == "measured"]
    fig.text(0.05, 0.965, "%s  (%sHP)" % (row.pitcher_name, hand), fontsize=17, fontweight="bold",
             color=NAVY, va="top")
    fig.text(0.05, 0.922, "2026 command: miss from the catcher's set-up glove   |   %s   |   "
             "measured %d of %d pitches thrown (%.0f%%) in %d of %d games"
             % (row.levels, len(d), row.pitches, 100 * len(d) / max(row.pitches, 1),
                len(meas_games), len(games)), fontsize=9.5, color="#333", va="top")
    labs = [p for p in PT_ORDER if p in set(d.pitch_type)] + sorted(set(d.pitch_type.dropna()) - set(PT_ORDER))
    h = [plt.Line2D([], [], marker="o", ls="", color=PT_COLOR.get(p, "#777"), mec="k", mew=0.3, ms=6) for p in labs]
    fig.legend(h, labs, loc="center left", bbox_to_anchor=(0.045, 0.877), ncol=min(len(labs), 9),
               fontsize=7.5, frameon=False, handletextpad=0.2, columnspacing=0.9)
    # fixed axes, not a gridspec: the miss panels are equal-aspect, so a gridspec cell taller than
    # it is wide only adds white space between them and the season strip
    for k, (M, lab) in enumerate((("data", "Ball location: tracking data (every pitch)"),
                                   ("video", "Ball location: video (track reaches the plate)"))):
        ax = fig.add_axes([0.05 + 0.315 * k, 0.405, 0.28, 0.40])
        n = int(d[M + "_miss"].notna().sum())
        med = d[M + "_miss"].median()
        draw_panel(ax, d, M, hand, "%s\nn %d   median miss %s" % (lab, n, "-" if pd.isna(med) else "%.1f in" % med),
                   ms=16, fs=(8.5, 7))
    type_table(fig.add_axes([0.685, 0.405, 0.295, 0.40]), d)
    season_strip(fig.add_axes([0.07, 0.125, 0.91, 0.19]), d, games)
    fig.text(0.05, 0.030, "Catcher's view, inches. Miss = set-up glove (video) to where the ball crossed the plate. "
             "Dotted rings 6 and 12 in; points past %d in drawn at the edge." % LIM, fontsize=7, color="#555")
    fig.text(0.05, 0.013, "Houston-affiliate games only. Each game is measured through its own camera, "
             "admitted on a held-out check.", fontsize=7, color="#555")
    fig.text(0.98, 0.013, "built %s" % datetime.now().strftime("%Y-%m-%d"), fontsize=7, color="#888", ha="right")
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)


def coverage_page(ix, png, pdf):
    """Every in-org pitcher, one bar: his pitches thrown, split by where each game stands. The
    page that says who has no card yet and why."""
    ix = ix.copy()
    ix["lvl"] = ix.levels.str.split("/").str[0].map({v: k for k, v in enumerate(LEVEL_ORDER)})
    ix = ix.sort_values(["lvl", "pitches"], ascending=[True, False]).reset_index(drop=True)
    n = len(ix)
    fig = plt.figure(figsize=(11, 0.165 * n + 1.9))
    H = fig.get_figheight()
    ax = fig.add_axes([0.30, 0.75 / H, 0.50, 1 - 1.55 / H])
    left = np.zeros(n)
    for b, col in BUCKETS:
        v = ix["pitches_" + b.replace(" ", "_")].to_numpy(float)
        ax.barh(np.arange(n), v, left=left, color=col, height=0.78,
                label="%s (%d)" % (b, int(v.sum())), edgecolor="none")
        left += v
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(["%s  %s" % (r.pitcher, r.levels) for r in ix.itertuples()], fontsize=6.6)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("pitches thrown in Houston-affiliate games", fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    xmax = left.max()
    for k, r in ix.iterrows():
        txt = ("%d of %d measured   median miss %.1f in" % (r.measured, r.pitches, r.miss_med_data)
               if r.measured else "no card yet")
        ax.text(xmax * 1.02, k, txt, va="center", fontsize=6.4, color=NAVY if r.measured else "#888",
                clip_on=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, fontsize=6.8, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    fig.text(0.03, 1 - 0.18 / H, "2026 command miss: season coverage, Houston pitchers still in the org",
             fontsize=12, fontweight="bold", color=NAVY, va="top")
    fig.text(0.03, 1 - 0.45 / H, "%d pitchers, %d of %d pitches measured (%.0f%%), %d cards. Bars split each "
             "pitcher's pitches by where the game stands in the season run."
             % (n, ix.measured.sum(), ix.pitches.sum(), 100 * ix.measured.sum() / max(ix.pitches.sum(), 1),
                (ix.card != "").sum()), fontsize=8, color="#333", va="top")
    fig.text(0.97, 0.25 / H, "built %s" % datetime.now().strftime("%Y-%m-%d %H:%M"), fontsize=6.5,
             color="#888", ha="right")
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="season2026")
    ap.add_argument("--pitchers", default="", help="comma-separated pitcher ids (default: all in org)")
    ap.add_argument("--out", default=os.path.join(OUT, "season_cards"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(os.path.join(OUT, "command_board_org_%s_pitches.csv" % a.board))
    who = pd.read_csv(os.path.join(OUT, "season_pitchers_2026.csv"))
    season = pd.read_csv(os.path.join(OUT, "season_pull_2026.csv"))
    app = appearances(season)
    app["bucket"] = app.game_pk.map(game_buckets(season))
    assert app.bucket.notna().all(), "a season game with no plan status"
    io = who[who.in_org]
    if a.pitchers:
        io = io[io.pitcher_id.isin({int(x) for x in a.pitchers.split(",")})]
    idx = []
    for r in io.itertuples(index=False):
        mine = d[d.pitcher_id == r.pitcher_id]
        games = app[app.pitcher_id == r.pitcher_id].sort_values("date")
        bp = games.groupby("bucket").pitches.sum()
        rec = dict(pitcher_id=r.pitcher_id, pitcher=r.pitcher_name, levels=r.levels, games=len(games),
                   pitches=r.pitches, measured=len(mine), games_measured=int((games.bucket == "measured").sum()),
                   miss_med_data=mine.data_miss.median() if len(mine) else np.nan,
                   miss_med_video=mine.video_miss.median() if len(mine) else np.nan, card="")
        for b, _ in BUCKETS:
            rec["pitches_" + b.replace(" ", "_")] = int(bp.get(b, 0))
        if len(mine):
            base = os.path.join(a.out, "%d_%s" % (r.pitcher_id, slug(r.pitcher_name)))
            card(mine, pd.Series(r._asdict()), games, base + ".png", base + ".pdf")
            rec["card"] = os.path.relpath(base + ".pdf", ROOT)
        idx.append(rec)
    ix = pd.DataFrame(idx)
    assert len(ix), "no in-org pitcher matched --pitchers"
    ix["share"] = ix.measured / ix.pitches.where(ix.pitches > 0)
    ix = ix.sort_values(["levels", "pitches"], ascending=[True, False])
    ix.to_csv(os.path.join(a.out, "index.csv"), index=False)
    if not a.pitchers:
        coverage_page(ix, os.path.join(a.out, "_coverage.png"), os.path.join(a.out, "_coverage.pdf"))
    print("in-org pitchers %d: %d cards, %d with nothing measured yet; %d of %d pitches measured (%.0f%%)"
          % (len(ix), (ix.card != "").sum(), (ix.card == "").sum(), ix.measured.sum(), ix.pitches.sum(),
             100 * ix.measured.sum() / max(ix.pitches.sum(), 1)))
    print("pitches thrown by where the game stands:")
    for b, _ in BUCKETS:
        print("  %-15s %6d" % (b, ix["pitches_" + b.replace(" ", "_")].sum()))
    print(a.out)


if __name__ == "__main__":
    main()
