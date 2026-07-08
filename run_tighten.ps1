# Command CV — one-command TIGHTEN: pull a full game + calibrate, hands-off.
#
# Runs on any Windows box (personal OR work laptop) — CPU only, no GPU needed.
# The Okta Bearer token goes in command-cv\data\.mlb_token (gitignored) first:
#   grab it from the statsapi browser-login redirect (#access_token=...), paste
#   it into that file, then run this.
#
# Usage (from the command-cv folder):
#   .\run_tighten.ps1 -GamePk 822515
#   .\run_tighten.ps1 -GamePk 822515 -Pitcher ogando -Mlbam 808010
#
param(
    [Parameter(Mandatory = $true)][string]$GamePk,
    [string]$Pitcher = "ogando",
    [string]$Mlbam = "808010"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$tokenFile = Join-Path $PSScriptRoot "data\.mlb_token"
if (-not (Test-Path $tokenFile)) {
    Write-Host "MISSING token: paste the Okta Bearer access_token into data\.mlb_token first." -ForegroundColor Red
    Write-Host "  (browser login to statsapi -> copy #access_token=... from the redirect URL)"
    exit 1
}
$bearer = (Get-Content $tokenFile -Raw).Trim()

Write-Host "== 1/2  Pulling ALL pitches from game $GamePk ($Pitcher) ==" -ForegroundColor Cyan
python scripts\pull_pitcher_clips.py --name $Pitcher --mlbam $Mlbam --games $GamePk --bearer $bearer
if ($LASTEXITCODE -ne 0) { Write-Host "pull failed (token expired? re-grab + re-run)" -ForegroundColor Red; exit 1 }

Write-Host "`n== 2/2  Auto-detect gloves + RANSAC calibrate (this game) ==" -ForegroundColor Cyan
python scripts\tighten.py --pitcher $Pitcher --game $GamePk

Write-Host "`nDone. Compare the TIGHTENED LOO median to the 4.2in baseline above." -ForegroundColor Green
