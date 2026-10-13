"""miss_season.py decides which games run, which resume and which wait. DB-, GPU- and
network-free: every check builds its own season / registry / ledger in a temp folder.

Each check has been failed on purpose against the defect it exists for (grep-guard-criteria.md
section 2); the injections are listed in INJECTIONS at the bottom and `--inject` replays them.

    python tests/test_miss_season.py            # plain script, no pytest needed
    python tests/test_miss_season.py --inject   # every injected defect must turn a check red
"""
import argparse
import math
import os
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import miss_new_park  # noqa: E402
import miss_season as S  # noqa: E402


class Box:
    """Points miss_season / miss_new_park at a temp output folder for one check."""

    def __enter__(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "output")
        self.cvpull = os.path.join(self.out, "cvpull")
        os.makedirs(self.cvpull)
        self.saved = (S.OUT, S.CVPULL, miss_new_park.OUT)
        S.OUT, S.CVPULL, miss_new_park.OUT = self.out, self.cvpull, self.out
        return self

    def __exit__(self, *exc):
        S.OUT, S.CVPULL, miss_new_park.OUT = self.saved

    def touch(self, name, text="x"):
        p = os.path.join(self.out, name)
        open(p, "w").write(text)
        return p

    def pull_csv(self, g):
        open(os.path.join(self.cvpull, "cvpull_%d.csv" % g), "w").write("play_id\n")


def season(rows):
    return pd.DataFrame([dict(level="aa", team="T", game_pk=g, date=d, venue_id=v, venue="V%d" % v,
                              side="home", opponent="O", day_night="night", banked=0)
                         for g, d, v in rows])


REG = {1: dict(status="ready", lo=165, hi=235), 2: dict(status="setup", lo=pd.NA, hi=pd.NA),
       3: dict(status="blocked", lo=pd.NA, hi=pd.NA)}
ARGS = argparse.Namespace(games="", venue_ids="", retry_failed=False, no_setup=False)


def statuses(rows, ledger=None):
    led = ledger if ledger is not None else pd.DataFrame(columns=S.LEDGER_COLS)
    return {j["game_pk"]: j["status"] for j in S.plan(season(rows), REG, led, ARGS)}


def check_plan_classifies_every_case():
    with Box() as b:
        for g in (101, 102, 201, 202, 301, 401):
            b.pull_csv(g)
        st = statuses([(101, "2026-04-01", 1), (102, "2026-04-02", 1), (103, "2026-04-03", 1),
                       (201, "2026-04-01", 2), (202, "2026-04-02", 2), (301, "2026-04-01", 3),
                       (401, "2026-04-01", 4)])
    assert st[101] == "run" and st[102] == "run", st
    assert st[103] == "awaiting pull", st
    assert st[201] == "run setup", st                 # first game at a setup park is its trial
    assert st[202] == "park in setup, awaiting review", st
    assert st[301] == "park blocked", st
    assert st[401] == "park needs scorebug boxes", st


def check_setup_park_runs_once_across_runs():
    """A park whose trial already ran (ledger role=setup) must not run a SECOND trial."""
    with Box() as b:
        b.pull_csv(202)
        led = pd.DataFrame([dict(game_pk=201, venue_id=2, role="setup", verdict="PASS")])
        st = statuses([(201, "2026-04-01", 2), (202, "2026-04-02", 2)], led)
    assert st[201] == "done" and st[202] == "park in setup, awaiting review", st


def check_failed_trial_reruns_only_under_retry():
    """A FAILED trial leaves its park waiting; --retry-failed reruns that same game."""
    led = pd.DataFrame([dict(game_pk=201, venue_id=2, role="setup", verdict="FAIL")])
    with Box() as b:
        b.pull_csv(201)
        b.pull_csv(202)
        plain = statuses([(201, "2026-04-01", 2), (202, "2026-04-02", 2)], led)
        retry = {j["game_pk"]: j["status"] for j in S.plan(
            season([(201, "2026-04-01", 2), (202, "2026-04-02", 2)]), REG, led,
            argparse.Namespace(games="", venue_ids="", retry_failed=True, no_setup=False))}
    assert plain[201] == "done" and plain[202] == "park in setup, awaiting review", plain
    assert retry[201] == "run setup" and retry[202] == "park in setup, awaiting review", retry


def check_killed_game_resumes_but_pilot_game_does_not():
    """miss_new_park writes its scope marker at step 0. A SEASON game killed after that has a
    season marker too and must resume; a PILOT game (scope marker only) is done."""
    with Box() as b:
        for g in (501, 502):
            b.pull_csv(g)
            b.touch("miss_scope_%d.txt" % g, "ours-only")
        b.touch("miss_season_marker_502.txt", "season")
        st = statuses([(501, "2026-04-01", 1), (502, "2026-04-02", 1)])
    assert st[501] == "done", st
    assert st[502] == "run", st


def check_stale_outputs_only_without_scope_marker():
    with Box() as b:
        b.touch("holdout_601.csv")
        b.touch("command_miss_601_bo.csv")
        stale = miss_new_park.stale_outputs(601)
        b.touch("miss_scope_601.txt", "ours-only")
        after = miss_new_park.stale_outputs(601)
    assert set(stale) == {"holdout_601.csv", "command_miss_601_bo.csv"}, stale
    assert after == [], after


def check_gate_fails_on_no_evidence():
    assert S.gate(0.26) == "PASS" and S.gate(S.GATE_BOK_IN) == "PASS"
    assert S.gate(0.48) == "FAIL" and S.gate(1.48) == "FAIL"
    assert S.gate(float("nan")) == "FAIL"


def check_window_measured_reads_last_line():
    with Box() as b:
        p = b.touch("g.log",
                    "  chain ends: p05 174  median 189  p95 195   -> --lo 142 --hi 201 would have sufficed\n"
                    "resumed\n"
                    "  chain ends: p05 170  median 190  p95 199   -> --lo 140 --hi 205 would have sufficed\n")
        got = S.window_measured(p)
    assert got == "--lo 140 --hi 205", got


def check_board_season_pass_supersedes_pilot():
    led = pd.DataFrame([dict(game_pk=822515, verdict="PASS"), dict(game_pk=822512, verdict="FAIL")])
    games = S.board_games(led)
    assert games[822515] == "_bo", games            # full-game season run replaces the Ogando pull
    assert games[822512] == "_s2c", games           # a FAIL keeps the pilot entry
    assert games[814874] == "_bo", games            # admitted pilot cameras stay on


def _ladder_run(true_end):
    """Drive miss_new_park.detect_with_ladder with a fake detector whose every chain ends at
    min(true_end, --hi). Returns (final window file, the windows it detected at)."""
    calls = []
    with Box() as b:
        def fake_sh(args):
            lo, hi = miss_new_park._flag(args, "--lo"), miss_new_park._flag(args, "--hi")
            calls.append((lo, hi))
            end = min(true_end, hi)
            pd.DataFrame([dict(stem="s%02d" % k, game_pk=9, frame=f, x=0, y=0, conf=1, window_pick=0)
                          for k in range(20) for f in range(end - 12, end + 1)]).to_csv(
                os.path.join(b.out, "detect_tracks_9_v4.csv"), index=False)
        saved = miss_new_park.sh
        miss_new_park.sh = fake_sh
        try:
            a = argparse.Namespace(detect_flags="--window-pick --lo 146 --hi 225", dir="x", weights="w")
            miss_new_park.detect_with_ladder(9, a)
            final = open(os.path.join(b.out, "miss_window_9.txt")).read()
        finally:
            miss_new_park.sh = saved
    return final, calls


def check_ladder_shifts_a_cut_off_window():
    """Flights ending at f235 under --hi 225 are cut off: shift once to 186/265 and stop there.
    Flights ending at f200 are not: one detection, window unchanged."""
    final, calls = _ladder_run(235)
    assert calls == [(146, 225), (186, 265)], calls
    assert final == "--window-pick --lo 186 --hi 265", final
    final, calls = _ladder_run(200)
    assert calls == [(146, 225)] and final == "--window-pick --lo 146 --hi 225", (calls, final)


def check_order_ready_parks_before_trials():
    """A ready-park game (straight onto the board) outranks an EARLIER setup trial."""
    jobs = [dict(game_pk=1, date="2026-04-01", status="run setup"),
            dict(game_pk=2, date="2026-06-01", status="run"),
            dict(game_pk=3, date="2026-05-01", status="run"),
            dict(game_pk=4, date="2026-04-01", status="awaiting pull")]
    got = [j["game_pk"] for j in S.order(jobs)]
    assert got == [3, 2, 1], got


def check_delete_counts_hard_links_once():
    """hou_<g> is hard links into pull_<g>: 2 files of 1000 bytes linked twice free 2000."""
    tmp = tempfile.mkdtemp()
    saved = S.ROOT
    S.ROOT = tmp
    try:
        a = os.path.join(tmp, "data", "clips", "pull_9")
        b = os.path.join(tmp, "data", "clips", "hou_9")
        os.makedirs(a)
        os.makedirs(b)
        for k in range(2):
            open(os.path.join(a, "c%d.mp4" % k), "wb").write(b"x" * 1000)
            os.link(os.path.join(a, "c%d.mp4" % k), os.path.join(b, "c%d.mp4" % k))
        freed = S.delete_clips(9) * 2 ** 30
        left = os.listdir(a) + os.listdir(b)
    finally:
        S.ROOT = saved
    assert left == [], left
    assert math.isclose(freed, 2000), freed


def check_shards_partition_the_season():
    """Every runnable game lands in exactly ONE worker's shard, and a setup park's trial is
    chosen over the WHOLE season (one trial per park), not once per shard."""
    rows = [(1000 + k, "2026-04-%02d" % (k + 1), 1) for k in range(7)] +            [(2001, "2026-04-01", 2), (2002, "2026-04-02", 2)]
    with Box() as b:
        for g, _, _ in rows:
            b.pull_csv(g)
        jobs = S.plan(season(rows), REG, pd.DataFrame(columns=S.LEDGER_COLS), ARGS)
    picks = []
    for k in range(3):
        a = argparse.Namespace(workers=3, worker=k)
        picks.append({j["game_pk"] for j in S.order([j for j in jobs if S.in_shard(j["game_pk"], a)])})
    allrun = {j["game_pk"] for j in S.order(jobs)}
    assert set().union(*picks) == allrun and sum(map(len, picks)) == len(allrun), (picks, allrun)
    assert sum(j["status"] == "run setup" for j in jobs) == 1, jobs


def check_second_trial_runs_at_measured_window():
    """--trials-per-park 2: a park whose first trial FAILED gets one more trial, on its NEXT
    game, carrying the window the failed trial measured. One is the default."""
    led = pd.DataFrame([dict(game_pk=201, venue_id=2, role="setup", verdict="FAIL",
                             window_measured="--lo 158 --hi 237")])
    rows = [(201, "2026-04-01", 2), (202, "2026-04-02", 2), (203, "2026-04-03", 2)]
    with Box() as b:
        for g in (201, 202, 203):
            b.pull_csv(g)
        two = {j["game_pk"]: j for j in S.plan(season(rows), REG, led, argparse.Namespace(
            games="", venue_ids="", retry_failed=False, no_setup=False, trials_per_park=2))}
        one = {j["game_pk"]: j["status"] for j in S.plan(season(rows), REG, led, ARGS)}
        led2 = pd.concat([led, pd.DataFrame([dict(game_pk=202, venue_id=2, role="setup",
                                                  verdict="FAIL", window_measured="")])])
        after = {j["game_pk"]: j["status"] for j in S.plan(season(rows), REG, led2, argparse.Namespace(
            games="", venue_ids="", retry_failed=False, no_setup=False, trials_per_park=2))}
    assert two[202]["status"] == "run setup", two
    assert two[202]["trial_window"] == "--lo 158 --hi 237", two[202]
    assert two[203]["status"] == "park in setup, awaiting review", two
    assert one[202] == "park in setup, awaiting review", one
    assert after[203] == "park in setup, awaiting review", after      # two used, none passed


def check_setup_first_order():
    jobs = [dict(game_pk=1, date="2026-05-01", status="run setup"),
            dict(game_pk=2, date="2026-04-01", status="run")]
    assert [j["game_pk"] for j in S.order(jobs, setup_first=True)] == [1, 2]
    assert [j["game_pk"] for j in S.order(jobs)] == [2, 1]


def check_run_lock_shards():
    """Worker locks coexist; a whole (unsharded) run refuses while any worker holds its byte, and
    a second copy of one worker refuses. Each case runs in a child process (a lock is per
    process on Windows, and sys.exit is the refusal)."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import miss_season as S; S.OUT = %r; "
            "f = S.acquire_run_lock(%d, %d); print('GOT'); sys.stdout.flush(); "
            "import time; time.sleep(%d)")
    with Box() as b:
        scripts = os.path.join(ROOT, "scripts")
        hold = subprocess.Popen([sys.executable, "-c", code % (scripts, b.out, 1, 2, 20)],
                                stdout=subprocess.PIPE, text=True)
        assert hold.stdout.readline().strip() == "GOT"
        try:
            got = {}
            for name, (w, n) in {"w0": (0, 2), "w1": (1, 2), "whole": (0, 1)}.items():
                r = subprocess.run([sys.executable, "-c", code % (scripts, b.out, w, n, 0)],
                                   capture_output=True, text=True)
                got[name] = "GOT" in r.stdout
        finally:
            hold.kill()
    assert got == {"w0": True, "w1": False, "whole": False}, got


CHECKS = [check_plan_classifies_every_case, check_setup_park_runs_once_across_runs,
          check_failed_trial_reruns_only_under_retry,
          check_killed_game_resumes_but_pilot_game_does_not,
          check_stale_outputs_only_without_scope_marker, check_gate_fails_on_no_evidence,
          check_window_measured_reads_last_line, check_board_season_pass_supersedes_pilot,
          check_order_ready_parks_before_trials, check_delete_counts_hard_links_once,
          check_ladder_shifts_a_cut_off_window, check_shards_partition_the_season,
          check_second_trial_runs_at_measured_window, check_setup_first_order,
          check_run_lock_shards]


# pytest discovery, same functions
def test_all():
    for c in CHECKS:
        c()


def run():
    bad = 0
    for c in CHECKS:
        try:
            c()
            print("  ok    " + c.__name__)
        except Exception as e:                           # noqa: BLE001  (attribute, never crash)
            bad += 1
            print("  FAIL  %s: %s: %s" % (c.__name__, type(e).__name__, e))
    return bad


# Each entry re-introduces one defect and names the check that must go red.
INJECTIONS = {
    "season marker written AFTER the run (killed game marked done)":
        ("pilot_games", lambda: set(int(os.path.basename(p)[11:-4]) for p in
                                    __import__("glob").glob(os.path.join(S.OUT, "miss_scope_*.txt"))),
         "check_killed_game_resumes_but_pilot_game_does_not"),
    "gate passes NaN":
        ("gate", lambda x: "FAIL" if x > S.GATE_BOK_IN else "PASS",
         "check_gate_fails_on_no_evidence"),
    "every game at a setup park runs":
        ("plan", None, "check_plan_classifies_every_case"),
    "order by date only (trials ahead of ready games)":
        ("order", lambda jobs: sorted([j for j in jobs if j["status"] in ("run", "run setup")],
                                      key=lambda j: (j["date"], j["game_pk"])),
         "check_order_ready_parks_before_trials"),
    "every unlink counted as freed space":
        ("delete_clips", lambda g: sum(
            (os.path.getsize(p), os.remove(p))[0] for d in ("pull_%d" % g, "hou_%d" % g)
            for p in __import__("glob").glob(os.path.join(S.ROOT, "data", "clips", d, "*.mp4")))
            / 2 ** 30,
         "check_delete_counts_hard_links_once"),
    "ladder never shifts (the pre-fix behaviour)":
        ("_ladder_noop", None, "check_ladder_shifts_a_cut_off_window"),
    "shard by game_pk % workers ignored (every worker takes every game)":
        ("in_shard", lambda g, a: True, "check_shards_partition_the_season"),
    "a whole run locks only byte 0 (the pre-worker lock)":
        ("_lock_byte0", None, "check_run_lock_shards"),
    "window parser keeps the first line":
        ("window_measured", lambda p: next((l.split("->")[-1].replace("would have sufficed", "").strip()
                                            for l in open(p) if "would have sufficed" in l), ""),
         "check_window_measured_reads_last_line"),
}


def inject():
    bad = 0
    for name, (attr, fake, must_fail) in INJECTIONS.items():
        if attr == "_ladder_noop":                  # lives in miss_new_park, not miss_season
            import miss_new_park as M
            orig_t = M.truncation
            M.truncation = lambda g, hi: (0.0, 0)
            try:
                check_ladder_shifts_a_cut_off_window()
                red = False
            except Exception:                        # noqa: BLE001
                red = True
            finally:
                M.truncation = orig_t
            bad += not red
            print("  %s  %s -> %s" % ("red " if red else "GREEN(!)", name, must_fail))
            continue
        if attr == "_lock_byte0":                   # the child processes import miss_season fresh,
            src = os.path.join(ROOT, "scripts", "miss_season.py")     # so inject on disk
            text = open(src, encoding="utf-8").read()
            assert "MAX_WORKERS = 16" in text
            open(src, "w", encoding="utf-8").write(text.replace("MAX_WORKERS = 16", "MAX_WORKERS = 1", 1))
            try:
                check_run_lock_shards()
                red = False
            except Exception:                        # noqa: BLE001
                red = True
            finally:
                open(src, "w", encoding="utf-8").write(text)
            bad += not red
            print("  %s  %s -> %s" % ("red " if red else "GREEN(!)", name, must_fail))
            continue
        orig = getattr(S, attr)
        if attr == "plan":
            def fake(season_, reg, ledger, a, _orig=orig):
                rows = _orig(season_, reg, ledger.iloc[0:0], a)
                for r in rows:
                    if r["status"] == "park in setup, awaiting review":
                        r["status"] = "run setup"
                return rows
        setattr(S, attr, fake)
        try:
            fn = next(c for c in CHECKS if c.__name__ == must_fail)
            fn()
            red = False
        except Exception:                                # noqa: BLE001
            red = True
        finally:
            setattr(S, attr, orig)
        bad += not red
        print("  %s  %s -> %s" % ("red " if red else "GREEN(!)", name, must_fail))
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", action="store_true")
    a = ap.parse_args()
    n = run()
    if a.inject:
        print("injected defects (each must be red):")
        n += inject()
    print("ALL PASS" if not n else "%d PROBLEM(S)" % n)
    sys.exit(1 if n else 0)
