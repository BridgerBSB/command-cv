# Overnight watchdog for the v4 run. Runs detached (Task Scheduler), so a Claude Code
# memory reap cannot touch it.
#
# Every -IntervalMin minutes until -StopAtCT (US Central, whatever zone the laptop is in):
#   1. keeps the machine from IDLE-sleeping (SetThreadExecutionState; per-process, no
#      settings changed, released when this script exits). Closing the lid still sleeps.
#   2. writes a heartbeat line to output/detached/watchdog.log: epoch, which v4 step is
#      alive, free disk, free RAM
#   3. if NO v4 python process is alive and the run is not finished, relaunches
#      scripts/v4_finish.py, which resumes from wherever it stopped (at most -MaxRestarts)
#
#   .\scripts\overnight_watchdog.ps1                       # defaults: every 10 min until 9:00 CT
param([int]$IntervalMin = 10, [string]$StopAtCT = "09:00", [int]$MaxRestarts = 4, [int]$StaleMin = 20)

$Root = Split-Path -Parent $PSScriptRoot
$Log  = Join-Path $Root "output\detached\watchdog.log"
$Done = Join-Path $Root "output\v4_grades.txt"
$Res  = Join-Path $Root "data\yolo_ball_runs\milb4_v4\results.csv"

Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][W.P]::SetThreadExecutionState([uint32]"0x80000001")    # CONTINUOUS | SYSTEM_REQUIRED

$ct = [TimeZoneInfo]::FindSystemTimeZoneById("Central Standard Time")
$nowCT = [TimeZoneInfo]::ConvertTime([DateTime]::Now, $ct)
$stopCT = [DateTime]::ParseExact($nowCT.ToString("yyyy-MM-dd") + " " + $StopAtCT, "yyyy-MM-dd HH:mm", $null)
if ($stopCT -le $nowCT) { $stopCT = $stopCT.AddDays(1) }

function Say($m) { $t = [TimeZoneInfo]::ConvertTime([DateTime]::Now, $ct).ToString("MM-dd HH:mm") + " CT"; Add-Content -Path $Log -Value "[$t] $m"; Write-Host "[$t] $m" }
# ALIVE = a v4 log was written in the last $StaleMin minutes. NOT a process lookup: run
# from its own scheduled task, Win32_Process returned no command lines and the first version
# of this watchdog decided a healthy training run was dead and launched a second GPU job on
# top of it (2026-09-26 00:56 CT, caught and stopped within a minute). Training rewrites its
# progress bar every second and detection prints every ~20 clips, so a quiet log is a real
# signal. Slowest normal gap measured: the 53k-image dataset scan, ~9.5 min.
function V4Alive {
  $cut = (Get-Date).AddMinutes(-$StaleMin)
  @(Get-ChildItem (Join-Path $Root "output\detached") -Filter "cv-v4*.log" | Where-Object { $_.LastWriteTime -gt $cut })
}

Say "watchdog start: every $IntervalMin min until $($stopCT.ToString('MM-dd HH:mm')) CT, max $MaxRestarts restarts"
$restarts = 0
while ($true) {
  $nowCT = [TimeZoneInfo]::ConvertTime([DateTime]::Now, $ct)
  $ep = if (Test-Path $Res) { (Get-Content $Res | Measure-Object -Line).Lines - 1 } else { 0 }
  $procs = V4Alive
  $step = if ($procs.Count) { ($procs | ForEach-Object { $_.BaseName + ' (' + [int]((Get-Date) - $_.LastWriteTime).TotalMinutes + ' min ago)' }) -join ',' } else { 'none - no v4 log written in ' + $StaleMin + ' min' }
  $disk = '{0:N1}' -f ((Get-PSDrive C).Free / 1GB)
  $ram  = '{0:N1}' -f ((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)
  $finished = Test-Path $Done
  Say "epoch $ep/15 | alive: $step | disk $disk GB | RAM free $ram GB | done: $finished"

  if ($finished -and -not $procs.Count) { Say "v4 COMPLETE - results in output\v4_grades.txt. Watchdog exiting."; break }
  if (-not $procs.Count -and -not $finished) {
    if ($restarts -ge $MaxRestarts) { Say "NOT RELAUNCHING: $restarts restarts used. Needs a human look." }
    else {
      $restarts++
      Say "nothing alive and not finished -> relaunch #$restarts via v4_finish.py (resumes from last checkpoint)"
      & (Join-Path $PSScriptRoot "run_detached.ps1") -Command "python scripts\v4_finish.py" -Name "cv-v4-resume$restarts" 3>$null | Out-Null
      Start-Sleep ($StaleMin * 60)     # let the relaunch start writing before judging again
    }
  }
  if ($nowCT -ge $stopCT) { Say "reached $StopAtCT CT - watchdog exiting (the job itself keeps running if still going)."; break }
  Start-Sleep ($IntervalMin * 60)
}
[void][W.P]::SetThreadExecutionState([uint32]"0x80000000")
