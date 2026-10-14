# LINEAGE - command-cv

> READ FIRST. Chronological record: what this project is now, what superseded
> what, and what's deprecated (STILL-WIRED vs REMOVED). Newest on top.
> Day-to-day results live in `docs/PROGRESS-LOG.md`; this file is for structural changes.
> Entries before 2026-10-14 live in `bsb-resources/LINEAGE.md` (Command CV entries there).

---

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
