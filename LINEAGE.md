# LINEAGE - command-cv

> READ FIRST. Chronological record: what this project is now, what superseded
> what, and what's deprecated (STILL-WIRED vs REMOVED). Newest on top.
> Day-to-day results live in `docs/PROGRESS-LOG.md`; this file is for structural changes.
> Entries before 2026-10-14 live in `bsb-resources/LINEAGE.md` (Command CV entries there).

---

## 2026-10-15 - A quarter of CHARTED set-up targets come from the wrong moment; the 0.47 gate is a coin flip at the margin

**What changed:** three findings that change what is true about the season numbers, plus compute.
**Idea shift:**
  - WAS: a target that passes the plausibility screen is the catcher's set-up.
  - NOW: Zac's blind review (scripts/target_review.py, 172/180 answered): plausible targets are wrong
    26% of the time (30/117, CI 19-34%) - 27 WRONG MOMENT (catcher not set yet / another pitch), 3 wrong
    spot (~15 px). The screen itself is right (0/47 rejected targets were OK). Wrong-moment targets sit
    low (22 vs 29 in) and their frames scatter (IQR 45-162 vs 84-120).
  - WAS: 0.47 in held-out gate separates good cameras from bad.
  - NOW: near-miss FAILs pass 13-49% of resamples (session 4c bootstrap) - the gate is a coin flip
    between 0.47 and ~0.75. Charts keep 0.47; the gate gets tested inside the miss-vs-BB% model
    (strict / 0.75 / per-pitch-error weighted). Do NOT rerun near misses to "rescue" them - a rerun is a reroll.
  - Asheville (2807) failures were the DETECT WINDOW (opened after the ball, f177-186; the ladder moved it
    later), not the camera fit start. Window 175-260 (`b1557ce`); 822469 passed 0.36 at it.
**Deprecated / still wired:**
  - STILL-WIRED  src/auto_glove.py:32 SETUP_CUTOFF_FRAC + "longest low-motion run" set-up rule (the
    wrong-moment source). Proposed fix, not built: the hold that ends just before the ball's release.
  - STILL-WIRED  scripts/miss_season.py:79 GATE_BOK_IN = 0.47 (kept on purpose until the BB% test).
**Compute:** CPU VPS is ~23x slower on detection and hours/game on the yolo11x glove model - not a
season-run home. Hostinger RTX 4090 box: same answers as the laptop (median miss 12.71 both; per-pitch
0.06 in), detect 1.7 min. `scripts/gpu_box/` rebuilds it in one command; it bills while it EXISTS.
Fixes found on the way: `--device auto` (073f745), glove FP16 only on GPU (a157965).
**Active truth now:** charts = 0.47 gate + plausibility screen, with ~1 in 4 targets from the wrong moment.
**Next:** build + test the release-anchored set-up (must move the 27 wrong-moment items, leave the 87 OK);
pull the 25 Asheville GPU reruns home (`gpu_box/pull_box.py --apply`), then destroy the GPU box.

## 2026-10-14 - Command CV is its own repo: BridgerBSB/command-cv (personal GitHub)

**What changed:** Command CV moved out of `bsb-resources` (work GitHub `zbridger_astros`, branch
`feature/pd-goals`, folder `command-cv/`) into its own repo `github.com/BridgerBSB/command-cv`,
branch `main`. Full history kept: 254 commits via `git subtree split --prefix=command-cv`.

**Idea shift:**
  - WAS: one folder inside the work analytics monorepo, committed and pushed with the four apps.
  - NOW: a standalone repo on the personal GitHub. It uses only the MLB Stats API (no
    GroundControl2 database), so nothing ties it to the work repo, and it can run on the
    personal laptop and the Hostinger VPS without the work repo.
  - WHY: Zac's call - easier to run and share from personal infrastructure, then hand to
    work once proven. Do NOT re-add `command-cv/` to `bsb-resources` tracking; that makes two
    copies that drift.

**How the move was done (no files moved):** the folder stays at
`C:\Users\Owner\bsb-resources\command-cv`. It was converted IN PLACE: `git init`, remote
`git@github-personal:BridgerBSB/command-cv.git`, fetch, `git reset origin/main` (index only, working
tree untouched). `bsb-resources` stopped tracking it (`git rm -r --cached command-cv`) and ignores
`command-cv/`. The season workers kept running through the switch; no script calls git or
resolves paths from a repo root.

**Commit IDs changed.** Every SHA cited before 2026-10-14 (memory, PROGRESS-LOG, docs, the
`bsb-resources` LINEAGE) is a `bsb-resources` SHA. `docs/sha-map-from-bsb-resources.csv` maps all
254 to their new IDs (0 unmatched). Example: `e6708ede7` (bsb-resources) = `0e318ab58` (here).

**Deprecated:**
  - STILL-WIRED  `bsb-resources` branches other than `feature/pd-goals` (main, topic branches)
    still contain `command-cv/` history. Frozen; expected.
  - STILL-WIRED  `bsb-resources/.claude/rules/` holds the 4 rules that govern this code
    (index-and-time-origins, long-jobs-on-this-laptop, metric-must-separate-failure-modes,
    probe-capability-before-accepting-a-blocker). They auto-load for sessions started in
    `bsb-resources` because the path is unchanged. Not copied here, to avoid two copies.
  - REMOVED      `bsb-resources` tracking of `command-cv/` (feature/pd-goals).

**Active truth now:** commit CV work from inside this folder (`git -C .../command-cv`), push to
`origin` = BridgerBSB/command-cv. A `git add command-cv/...` from `bsb-resources` now fails as an
ignored path - that error means "wrong repo", not a broken file.

**Next:** move the season run to the Hostinger VPS (time one game there first; plan in
`bsb-resources/docs/2026-10-14-compute-storage-backup-plan.md`).
