<#
  Run a long command-cv job OUTSIDE Claude Code's process tree.

  WHY THIS EXISTS.
  When free RAM gets critical, Claude Code kills its own background shells to
  recover it, and the job it kills is usually the only long-lived thing it is
  allowed to touch. That happened eight times on 2026-09-19 to the same crop
  recovery. The documented fixes are to set
  CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1 at launch (needs a restart) or
  to close applications (not always possible -- on 09-19 VS Code could not be
  closed, and ten idle Claude sessions were living inside it).

  This is the third option and it needs neither. The Task Scheduler starts the
  process, so its parent is the scheduler service and NOT claude.exe. A reaper
  that walks Claude Code's own process tree never sees it. The job keeps
  running across a /clear, across a session restart, and across a reap.

  WHAT IT DOES NOT DO. It does not create memory. If Windows itself runs out,
  Windows will page or kill something, and this job is no longer protected by
  anything. The reaper exists for a reason; this only takes THIS job out of its
  reach. Check the footprint it reports and keep an eye on free RAM.

  MEASURE THE FOOTPRINT LATE, NOT AT STARTUP. The rolling pull read 77 MB
  thirty seconds in, which was quoted as evidence the job was never what filled
  the machine. Eight minutes later it had spawned a detector child at 1,271 MB.
  The parent is a thin orchestrator and the real cost arrives with inference,
  so an early reading understates it by more than an order of magnitude. That
  is why -Status walks the whole process TREE and prints every worker rather
  than just the one the scheduler started.

  USAGE
    .\scripts\run_detached.ps1 -Command "python scripts\rolling_pull.py ..." -Name cv-rolling
    .\scripts\run_detached.ps1 -Status  -Name cv-rolling     # alive? log tail?
    .\scripts\run_detached.ps1 -Stop    -Name cv-rolling

  The log goes to output\detached\<name>.log and is the only place the job's
  output lands -- a detached process has no console to print to. Tail it with
    Get-Content output\detached\<name>.log -Tail 30
#>
param(
  [string]$Command = "",
  [string]$Name    = "cv-detached",
  [switch]$Status,
  [switch]$Stop,
  # The Task Scheduler stops the task at this limit. 12 h was the old fixed value, and a
  # season run measured in days hit it without saying so (2026-10-13).
  [int]$Hours = 12
)

$ErrorActionPreference = "Stop"
$Root    = Split-Path -Parent $PSScriptRoot
$LogDir  = Join-Path $Root "output\detached"
$Log     = Join-Path $LogDir "$Name.log"
$Wrapper = Join-Path $LogDir "$Name.cmd"
$Task    = "cv-detached-$Name"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# FINDING THE JOB AGAIN AFTER DELIBERATELY ORPHANING IT.
# The first version of this looked for a marker ENV VAR in the process command
# line. An env var is not on the command line, so -Status reported NOT RUNNING
# while the job was visibly writing to its own log -- a check that could only
# ever say one thing. Anchor on the WRAPPER PATH instead: it IS on the shim
# cmd.exe's command line, it is unique per job name, and the real worker is a
# descendant of that cmd.
function Get-JobTree {
  param([string]$WrapperPath)
  $shims = @(Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
               Where-Object { $_.CommandLine -and $_.CommandLine -like "*$WrapperPath*" })
  if (-not $shims) { return @() }
  $all  = Get-CimInstance Win32_Process
  $seen = [System.Collections.ArrayList]::new()
  $queue = [System.Collections.Queue]::new()
  foreach ($s in $shims) { [void]$queue.Enqueue($s) ; [void]$seen.Add($s) }
  while ($queue.Count) {
    $cur = $queue.Dequeue()
    foreach ($kid in ($all | Where-Object { $_.ParentProcessId -eq $cur.ProcessId })) {
      if ($seen.ProcessId -notcontains $kid.ProcessId) { [void]$seen.Add($kid); [void]$queue.Enqueue($kid) }
    }
  }
  return $seen
}

if ($Stop) {
  # FIND THE TREE BEFORE STOPPING THE TASK. Stop-ScheduledTask ends the cmd shim -- the
  # only process Get-JobTree can anchor on -- and leaves the python worker orphaned, so
  # the old order found nothing, killed nothing and still printed "task removed".
  # 2026-10-12: a season job kept running 5 h after such a -Stop and collided with its
  # replacement on the same games' downloads.
  $tree = @(Get-JobTree -WrapperPath $Wrapper)
  # Children first, so a shell cannot respawn a worker we already killed.
  foreach ($p in ($tree | Sort-Object CreationDate -Descending)) {
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    "stopped $($p.Name) pid=$($p.ProcessId)"
  }
  try { Stop-ScheduledTask       -TaskName $Task -ErrorAction Stop } catch {}
  try { Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction Stop } catch {}
  if (-not $tree.Count) {
    "WARNING: no process tree found for '$Name'. If it was running, its worker may be orphaned:"
    "  Get-CimInstance Win32_Process -Filter ""Name='python.exe'"" | Select ProcessId, CommandLine"
  }
  "task '$Task' removed"
  return
}

if ($Status) {
  $tree = @(Get-JobTree -WrapperPath $Wrapper)
  $work = @($tree | Where-Object { $_.Name -ne 'cmd.exe' })
  if ($work) {
    foreach ($p in $work) {
      $o   = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
      $par = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ParentProcessId)" -ErrorAction SilentlyContinue).Name
      "ALIVE  {0} pid={1}  {2} MB  parent={3}  started {4}" -f `
        $p.Name, $p.ProcessId, [math]::Round($o.WorkingSet64/1MB), $par, $p.CreationDate
    }
  } elseif ($tree) {
    "SHIM ALIVE but no worker yet (starting, or the command exited)"
  } else {
    "NOT RUNNING"
  }
  if (Test-Path $Log) { ""; "--- last 20 log lines ---"; Get-Content $Log -Tail 20 }
  return
}

if (-not $Command) { throw "give -Command, or -Status / -Stop" }

# PYTHON BLOCK-BUFFERS WHEN ITS STDOUT IS A FILE.
# Interactively you get line buffering and see progress as it happens. Redirected
# to a log you get 8 KB blocks, so a job that is working perfectly writes nothing
# for tens of minutes -- which looks exactly like a hang, and the only way to tell
# the difference is to go read CPU time and directory sizes. That happened on the
# first real run of this script. `-u` costs nothing here; the job is I/O bound.
if ($Command -match '(^|\s)python(\.exe)?\s' -and $Command -notmatch '(^|\s)python(\.exe)?\s+-u(\s|$)') {
  Write-Warning ("no -u: python will block-buffer into the log, so progress will not " +
                 "appear and a working job will look hung. Inserting -u.")
  $Command = $Command -replace '(^|\s)python(\.exe)?\s', '$1python -u '
  "  command is now: $Command"
}

# THE REDIRECT GOES FIRST ON THE ECHO LINES, AND IT MUST.
# `echo ... EXIT %ERRORLEVEL%>> "$Log"` expands to `echo ... EXIT 0>> "$Log"`,
# and cmd reads `0>>` as an append-redirect of file descriptor 0 (stdin), not
# as the digit zero followed by a redirect. The text then goes to a console no
# one is watching and the log simply has no EXIT line -- which is
# indistinguishable from the job having been killed. The START line has the
# same trap whenever the command happens to end in a digit, and the real one
# ends in `--min-free-gb 6`. Leading `>>"file" echo ...` cannot be misparsed.
# THE START LINE ECHOES THE COMMAND AS TEXT, SO ITS METACHARACTERS MUST BE ESCAPED.
# Unescaped, `echo START a & b` is not a log line: cmd runs `echo START a` and then
# EXECUTES `b`. On 2026-09-24 a six-step `&` chain ran twice this way -- once inside
# the echo, once for real -- with the first pass's output going to no log at all,
# which looked like a step silently failing. `^` escapes & | < > ^ for cmd.
$EchoText = $Command -replace '([&|<>^])', '^$1'
$cmd = @"
@echo off
cd /d "$Root"
>> "$Log" echo [%DATE% %TIME%] START $EchoText
($Command) >> "$Log" 2>&1
>> "$Log" echo [%DATE% %TIME%] EXIT %ERRORLEVEL%
"@
Set-Content -Path $Wrapper -Value $cmd -Encoding ascii

try { Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction Stop } catch {}

$action   = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$Wrapper`"" -WorkingDirectory $Root
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -ExecutionTimeLimit ([TimeSpan]::FromHours($Hours)) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName $Task -Action $action -Settings $settings -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $Task

"registered and started task '$Task'"
"  log : $Log"
"  stop: .\scripts\run_detached.ps1 -Stop -Name $Name"
