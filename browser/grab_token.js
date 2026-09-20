/* command-cv -- grab the Okta bearer out of the statsapi tab and save it as a file.
 *
 * WHY THIS IS ONE LINE AND WHY IT CONTAINS NO SPACES.
 * Zac's copy/paste path into the Chrome console STRIPS SPACES. So a snippet is only
 * usable here if it still parses with every space removed -- which rules out `var x`,
 * `let x`, `function f`, `new Blob`, and every other keyword-then-space construct.
 * The line below has ZERO spaces in it, so stripping them is a no-op. Verified by
 * parsing both the original and a space-stripped copy, and by driving both against a
 * fake DOM in four cases (token in the URL hash / token in sessionStorage / hash wins
 * over sessionStorage / no token at all -> downloads nothing and says so).
 *
 * It avoids `new Blob` by encoding the token into a data: URI on an <a download>.
 * Nothing is written to disk by the page except the token file the browser downloads.
 *
 * USE
 *   1. Sign in to the statsapi research tool in a Chrome tab (Okta).
 *   2. Open the console on that tab and paste THE ONE LINE below.
 *   3. It logs TOKEN_OK_len=<n> and downloads Downloads/mlb_token.txt.
 *      NO_TOKEN_sign_in_first means it found nothing -- you are not signed in, or you
 *      are on the wrong tab, or the redirect hash has already been cleared.
 *   4. Tell Claude; it moves the file to command-cv/data/.mlb_token (gitignored).
 *
 * The token dies in 3600 s. The mp4 URLs it leads to need no auth and do not expire,
 * so one token is worth many games -- the authed half is JSON only, a couple of minutes
 * per game. See docs/2026-07-03-statsapi-video-auth.md for the two-step auth model and
 * browser/cv_pull.js for the fuller in-browser pull.
 */

((t,a)=>{t?(a.href='data:text/plain;charset=utf-8,'+encodeURIComponent(t),a.download='mlb_token.txt',a.click(),console.log('TOKEN_OK_len='+t.length)):console.log('NO_TOKEN_sign_in_first')})((/access_token=([^&]+)/.exec(location.hash)||[])[1]||sessionStorage.getItem('cvBearer')||'',document.createElement('a'))
