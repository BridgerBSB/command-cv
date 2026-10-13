/* command-cv browser-side pull -- paste into the Chrome console on statsapi.mlb.com.
 *
 * WHY THIS FILE EXISTS AT ALL, AND WHY IT IS COMMITTED.
 * The analytics endpoint that carries a pitch's clip URL is Okta-gated and its bearer dies
 * in 3600 s, so a Python pull needs a token pasted by hand every hour. The browser already
 * holds the signed-in session, so it can run the authed half itself and write out a CSV of
 * play_id + trajectory + clip URL that Python then consumes with no credential at all
 * (scripts/pull_from_browser_csv.py). A version of this existed on 2026-09-17, lived only
 * in a scratchpad, and was destroyed by a PC restart -- taking the whole MiLB pull path
 * with it. It is committed now so that cannot happen twice.
 *
 * NOTHING HERE HANDLES A CREDENTIAL ON DISK. The token, if one is used, stays in
 * sessionStorage in the tab and dies with it. Only the returned DATA is written out.
 *
 * WHAT IS PUBLIC AND WHAT IS NOT (measured 2026-09-18, three games, three levels):
 *   feed/live playId        PUBLIC at every level -- 276/276 A+, 298/298 AA, 327/327 AAA
 *   feed/live pitchData     AAA and MLB only -- 0/276 at A+, 0/298 at AA
 *   analytics(video)        OKTA at every level; this is the only thing that needs auth
 *   sporty-clips .mp4       Cloudflare UA check only, NO bearer -- Python downloads these
 * So the browser's whole job is the third row. The playIds come from the public feed and
 * the mp4s are fetched later by Python.
 *
 * THE SESSION COOKIE ALONE IS NOT ENOUGH -- MEASURED 2026-09-18, DO NOT ASSUME OTHERWISE.
 * A fetch to the analytics endpoint from a statsapi.mlb.com page with credentials:"include"
 * and 66 cookies on the origin returns **401**. So this file does NOT remove the bearer from
 * the loop; what it removes is having to move data between the browser and Python by hand.
 * The token still has to reach it, by one of:
 *   - land on the Okta redirect (statsapi.mlb.com/#access_token=...) with this file loaded,
 *     which caches it in sessionStorage; or
 *   - __CV_BEARER = "<access_token>" in the console.
 * One token covers a lot of ground -- the authed half is JSON only and a whole game is a
 * couple of minutes, so an hour is many games. The mp4s are fetched later by Python and need
 * no token at all.
 * STILL OPEN: whether the research tool can silently re-issue a token (an implicit
 * prompt=none refresh) while its IdP session is alive. That needs the authorize URL, which
 * is not recorded anywhere in this repo. Ask before guessing at an auth endpoint.
 *
 * USAGE
 *   1. Sign in to the statsapi research tool in this tab.
 *   2. Paste this file into the console.
 *   3. __cvAuth()                 -- reports which auth path works; do this FIRST.
 *   4. __cvGame(815419)           -- one game  -> Downloads/cvpull_815419.csv
 *      __cvGames([815419, 821819])-- many, sequential, one CSV each.
 *   5. Hand the CSVs to scripts/pull_from_browser_csv.py.
 *
 * The gamePks come from scripts/affiliate_schedule.py, which orders them so an interrupted
 * run still spans every level and every ballpark rather than finishing one park.
 */

(function () {
  "use strict";

  var FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live";
  var ANALYTICS = "https://statsapi.mlb.com/api/v1/game/{pk}/{guid}/analytics?hydrate=analytics(video)";
  var CONCURRENCY = 6;          // polite; the endpoint is not ours to hammer
  var COLS = ["play_id", "game_pk", "date", "venue", "pitch_type", "hand", "plate_x",
              "plate_z", "sz_top", "sz_bot", "call", "angle", "cf_url",
              "traj_x0", "traj_x1", "traj_x2", "traj_y0", "traj_y1", "traj_y2",
              "traj_z0", "traj_z1", "traj_z2", "t_start", "t_end", "zone_time",
              "rel_x", "rel_y", "rel_z"];

  function tokenFromHash() {
    var m = /access_token=([^&]+)/.exec(location.hash || "");
    return m ? decodeURIComponent(m[1]) : null;
  }

  function bearer() {
    if (window.__CV_BEARER) return window.__CV_BEARER;
    var h = tokenFromHash();
    if (h) { sessionStorage.setItem("cvBearer", h); return h; }
    return sessionStorage.getItem("cvBearer");
  }

  /* Auth is PROBED, never assumed. Two paths can work -- the session cookie alone, or an
   * explicit bearer -- and which one is live decides whether the hourly refresh matters at
   * all. Reporting "signed in" without a request that actually returns a clip URL is the
   * false-capability pattern in .claude/rules/probe-capability-before-accepting-a-blocker.md. */
  window.__cvAuth = async function (pk) {
    pk = pk || 815419;
    var feed = await fetch(FEED.replace("{pk}", pk)).then(function (r) { return r.json(); });
    var guid = null;
    feed.liveData.plays.allPlays.some(function (p) {
      return (p.playEvents || []).some(function (e) {
        if (e.isPitch && e.playId) { guid = e.playId; return true; }
        return false;
      });
    });
    if (!guid) { console.log("no playId in game " + pk + " -- pick another game"); return; }
    var url = ANALYTICS.replace("{pk}", pk).replace("{guid}", guid);
    var out = {};
    try {
      var rc = await fetch(url, { credentials: "include" });
      out.cookie = rc.status;
    } catch (e) { out.cookie = "ERR " + e; }
    var b = bearer();
    if (b) {
      try {
        var rb = await fetch(url, { headers: { Authorization: "Bearer " + b } });
        out.bearer = rb.status;
      } catch (e2) { out.bearer = "ERR " + e2; }
    } else {
      out.bearer = "no token: sign in, then re-paste, or __CV_BEARER = '<access_token>'";
    }
    console.log("AUTH PROBE on " + pk + " / " + guid, out);
    console.log("200 on either line means pulls will work. 401/403 on both means the tab " +
                "is not signed in to the research tool.");
    return out;
  };

  async function analytics(pk, guid) {
    var url = ANALYTICS.replace("{pk}", pk).replace("{guid}", guid);
    var b = bearer();
    var init = b ? { headers: { Authorization: "Bearer " + b } } : { credentials: "include" };
    var r = await fetch(url, init);
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  /* Angle selection mirrors src/ingest/video.py: prefer the named angle, fall back to the
   * lowest rank actually served. MiLB serves BROADCAST only, so the preference is a no-op
   * there -- recorded per row rather than assumed, because an angle change would silently
   * alter every camera we solve for. */
  function pickVideo(payload, want) {
    var groups = (((payload || {}).video || {}).playbackGroups) || [];
    var best = null;
    groups.forEach(function (g) {
      (g.playbackRenditions || []).forEach(function (rd) {
        if (!rd.playbackUrl || rd.playbackUrl.indexOf(".mp4") < 0) return;
        var exact = want && g.mediaSourceType === want ? 0 : 1;
        var rank = typeof rd.rank === "number" ? rd.rank : 999;
        // Explicit two-key compare. `[a,b] < [c,d]` in JS compares the STRINGS "a,b" and
        // "c,d", so rank 10 sorts before rank 9 -- a silent mis-pick, not an error.
        var better = !best || exact < best.exact || (exact === best.exact && rank < best.rank);
        if (better) {
          best = { exact: exact, rank: rank, url: rd.playbackUrl, angle: g.mediaSourceType };
        }
      });
    });
    return best;
  }

  /* SCHEMA IS PINNED, NOT GUESSED. Ported field for field from
   * scripts/pull_pitcher_clips.py::pitch_truth + trajectory_truth, which was verified on
   * Ogando 821922 (2026-09-10): the polynomial evaluated at zoneTime reproduces the payload's
   * own `location` to 1e-13. The public feed's x0/vX0/aX are a DIFFERENT parameterisation
   * (origin y=50 ft, not release) and are stripped below AAA anyway, so they are not used
   * here even when present -- mixing the two is the shared-name/different-origin defect in
   * .claude/rules/index-and-time-origins.md.
   *
   *   metaData.stat.play.pitchData              -> strike zone, type, hand
   *   ballSegments.pitchSegment.trajectoryData  -> trajectoryPolynomial{X,Y,Z}[3],
   *                                                validTimeInterval[2], zoneTime, location
   *   ballSegments.pitchSegment.releaseData     -> releasePosition{x,y,z}
   *
   * A missing piece comes back EMPTY, never a fake 0 -- Python's provenance assert would
   * happily "pass" on a row of zeros if we invented them. */
  function traj(payload) {
    var p = payload || {};
    var play = ((((p.metaData || {}).stat || {}).play) || {});
    var pd = play.pitchData || {};
    var seg = ((p.ballSegments || {}).pitchSegment) || {};
    var td = seg.trajectoryData || {};
    var loc = td.location || {};
    var rp = (seg.releaseData || {}).releasePosition || {};
    var tv = td.validTimeInterval || [];
    function n(v) { return (v === 0 || v) ? v : ""; }
    var out = {
      plate_x: n(loc.x), plate_z: n(loc.z),
      sz_top: n(pd.strikeZoneTop), sz_bot: n(pd.strikeZoneBottom),
      t_start: n(tv[0]), t_end: n(tv[1]), zone_time: n(td.zoneTime),
      rel_x: n(rp.x), rel_y: n(rp.y), rel_z: n(rp.z),
      pitch_type: (((play.details || {}).type) || {}).description || "",
      hand: (((play.details || {}).pitchHand) || {}).code || ""
    };
    ["x", "y", "z"].forEach(function (ax) {
      var c = td["trajectoryPolynomial" + ax.toUpperCase()] || [];
      for (var i = 0; i < 3; i++) out["traj_" + ax + i] = n(c[i]);
    });
    return out;
  }

  /* Dump one raw payload. The 09-17 loss means nobody here has SEEN this JSON in this
   * session; if a field ever moves, this is how it gets re-pinned from evidence instead of
   * from memory. */
  window.__cvProbe = async function (pk, guid) {
    if (!guid) {
      var feed = await fetch(FEED.replace("{pk}", pk)).then(function (r) { return r.json(); });
      feed.liveData.plays.allPlays.some(function (p) {
        return (p.playEvents || []).some(function (e) {
          if (e.isPitch && e.playId) { guid = e.playId; return true; }
          return false;
        });
      });
    }
    var pay = await analytics(pk, guid);
    console.log("top-level keys:", Object.keys(pay));
    console.log("trajectoryData:", ((pay.ballSegments || {}).pitchSegment || {}).trajectoryData);
    console.log("parsed row:", traj(pay));
    console.log("video:", pickVideo(pay, "CENTERFIELD"));
    window.__cvLastPayload = pay;
    return pay;
  };

  function csv(rows) {
    var head = COLS.join(",");
    var body = rows.map(function (r) {
      return COLS.map(function (c) {
        var v = r[c];
        if (v === undefined || v === null) v = "";
        v = String(v);
        return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
      }).join(",");
    });
    return head + "\n" + body.join("\n") + "\n";
  }

  function save(name, text) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  window.__cvGame = async function (pk, want) {
    want = want === undefined ? "CENTERFIELD" : want;
    var feed = await fetch(FEED.replace("{pk}", pk)).then(function (r) { return r.json(); });
    var venue = feed.gameData.venue.name;
    var date = feed.gameData.datetime.officialDate;
    var pitches = [];
    feed.liveData.plays.allPlays.forEach(function (p) {
      (p.playEvents || []).forEach(function (e) {
        if (!e.isPitch || !e.playId) return;
        pitches.push({
          play_id: e.playId,
          pitch_type: (e.details && e.details.type && e.details.type.description) || "",
          call: (e.details && e.details.description) || "",
          hand: (p.matchup && p.matchup.pitchHand && p.matchup.pitchHand.code) || ""
        });
      });
    });
    console.log(pk + " " + venue + " " + date + ": " + pitches.length + " pitches");

    var rows = [], done = 0, failed = 0, i = 0;
    async function worker() {
      while (i < pitches.length) {
        var me = pitches[i++];
        try {
          var pay = await analytics(pk, me.play_id);
          var vid = pickVideo(pay, want);
          var t = traj(pay);
          var row = { play_id: me.play_id, game_pk: pk, date: date, venue: venue,
                      pitch_type: me.pitch_type, hand: me.hand, call: me.call,
                      angle: vid ? vid.angle : "", cf_url: vid ? vid.url : "" };
          // The analytics payload fills in what the feed omits below AAA (type, hand), but
          // must not blank a field the feed already answered -- so non-empty wins.
          for (var k in t) { if (t[k] !== "" || !row[k]) row[k] = t[k]; }
          rows.push(row);
        } catch (e) {
          failed++;
          if (failed <= 3) console.warn("  " + me.play_id + " " + e.message);
        }
        if (++done % 50 === 0) console.log("  " + done + "/" + pitches.length +
                                           " (" + rows.length + " ok, " + failed + " failed)");
      }
    }
    var pool = [];
    for (var w = 0; w < CONCURRENCY; w++) pool.push(worker());
    await Promise.all(pool);

    var withUrl = rows.filter(function (r) { return r.cf_url; }).length;
    console.log(pk + " DONE: " + rows.length + " rows, " + withUrl + " with a clip URL, " +
                failed + " failed");
    if (!withUrl) {
      console.warn("NO CLIP URLS. Run __cvAuth() -- this is almost always an expired session, " +
                   "and writing the CSV anyway would hand Python a file that looks fine and " +
                   "downloads nothing.");
      return { rows: rows, withUrl: 0 };
    }
    save("cvpull_" + pk + ".csv", csv(rows));
    return { rows: rows, withUrl: withUrl, venue: venue, date: date };
  };

  window.__cvGames = async function (pks, want) {
    var log = [];
    for (var j = 0; j < pks.length; j++) {
      console.log("=== [" + (j + 1) + "/" + pks.length + "] game " + pks[j] + " ===");
      try {
        var r = await window.__cvGame(pks[j], want);
        log.push([pks[j], r.venue || "", r.withUrl]);
        if (!r.withUrl) {
          // A game with NO VIDEO looks exactly like a dead session (zero clip URLs).
          // Re-probe on a known game: stop only if the session is really gone.
          // 2026-10-12: 817475 (Final, 312 pitches, no video) stopped a 105-game batch.
          var a = await window.__cvAuth();
          if (!a || (a.cookie !== 200 && a.bearer !== 200)) {
            console.warn("stopping: session looks dead");
            break;
          }
          console.warn("game " + pks[j] + ": no clip URLs but the session is live -- no video, continuing");
        }
      } catch (e) {
        log.push([pks[j], "ERROR", String(e)]);
      }
    }
    console.table(log);
    return log;
  };

  console.log("command-cv browser pull loaded. __cvAuth() first, then __cvGame(pk) / __cvGames([pk,...]).");
})();
