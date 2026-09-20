<#
  Wait for the crop recovery to finish, then run the whole v2 decision chain
  unattended: baseline -> train -> compare.

  WHY IT WAITS RATHER THAN JUST RUNNING. Two inference processes at once throw
  `SystemError: copyMakeBorder returned a result with an exception set`, which
  names neither memory nor the GPU and reads like a corrupt install. So this
  polls until the rolling pull's worker is gone before touching the GPU.

  WHY IT RE-MEASURES v1 INSTEAD OF QUOTING 2.15. The 2.15 px figure comes from
  a checkpoint written days ago, by an earlier version of the eval, against a
  val split built at the time. Re-scoring v1's own weights against the split
  that is on disk RIGHT NOW costs about two minutes and removes every one of
  those differences, so the only thing separating the two rows in the final
  table is the model. If the re-measured baseline does NOT come out near 2.15,
  that is itself the finding and the comparison should stop until it is
  explained -- do not average them and move on.

  WHAT IT DOES NOT DECIDE. It prints the table; it does not conclude. The
  median sits under the labels' own ~2.8 px noise floor, so a flat median is
  expected and is not evidence either way. The tail (p90), the within-8px rate,
  the median rank and the no-detection count are where 13 parks can show up.

  LAUNCH IT WITH AN ABSOLUTE POWERSHELL PATH. Neither `cmd` nor `powershell`
  resolves on PATH in every shell this project is driven from, and the failure
  is a CommandNotFoundException that reads like the script is missing rather
  than the interpreter.

    .\scripts\run_detached.ps1 -Name chain -Command `
      "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\after_recovery_train.ps1"
#>
param(
  [string]$Data      = "data/yolo_milb3/data.yaml",
  [string]$Dataset   = "data/yolo_milb3",
  [string]$V1Weights = "data/yolo_ball_runs/milb4park/weights/best.pt",
  [string]$RunName   = "milb13park",
  [int]$Epochs       = 30,
  [int]$MaxWaitMin   = 360
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py  = "python"
$out = Join-Path $Root "output"

function Rolling-Alive {
  $wrapper = Join-Path $Root "output\detached\rolling.cmd"
  $shims = @(Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
               Where-Object { $_.CommandLine -and $_.CommandLine -like "*$wrapper*" })
  if (-not $shims) { return $false }
  $all = Get-CimInstance Win32_Process
  foreach ($s in $shims) {
    if ($all | Where-Object { $_.ParentProcessId -eq $s.ProcessId -and $_.Name -eq 'python.exe' }) { return $true }
  }
  return $false
}

Write-Host "[chain] waiting for the crop recovery to release the GPU (max $MaxWaitMin min)"
$waited = 0
while ((Rolling-Alive) -and $waited -lt ($MaxWaitMin * 2)) {
  Start-Sleep -Seconds 30; $waited++
  if ($waited % 10 -eq 0) {
    $n = (Get-ChildItem (Join-Path $Root "$Dataset\images\train") -File).Count
    Write-Host ("[chain] still waiting, {0} min elapsed, train crops {1}" -f ($waited/2), $n)
  }
}
if (Rolling-Alive) { Write-Host "[chain] TIMED OUT waiting. Not starting the GPU work."; exit 2 }
Write-Host "[chain] recovery finished."

$nTrain = (Get-ChildItem (Join-Path $Root "$Dataset\images\train") -File).Count
$nVal   = (Get-ChildItem (Join-Path $Root "$Dataset\images\val")   -File).Count
Write-Host "[chain] dataset: train $nTrain / val $nVal"
# A val split that vanished or grew is a different exam. Stop rather than report.
if ($nVal -ne 139) { Write-Host "[chain] ABORT: val is $nVal images, expected 139 (v1's Asheville split)."; exit 3 }

Write-Host "`n[chain] === STEP 1/3: re-measure v1 on TODAY's split ==="
& $py scripts/eval_ball_detector.py --data $Dataset --weights $V1Weights --device 0 `
      --json-out "$out\eval_v1_on_current_val.json"
if ($LASTEXITCODE -ne 0) { Write-Host "[chain] baseline eval FAILED ($LASTEXITCODE). Stopping."; exit 4 }

Write-Host "`n[chain] === STEP 2/3: train v2 on $nTrain crops ==="
& $py scripts/train_ball_detector.py --data $Data --epochs $Epochs --batch 16 --imgsz 320 `
      --device 0 --name $RunName
if ($LASTEXITCODE -ne 0) { Write-Host "[chain] training FAILED ($LASTEXITCODE). Stopping."; exit 5 }

$v2 = Join-Path $Root "data\yolo_ball_runs\$RunName\weights\best.pt"
if (-not (Test-Path $v2)) { Write-Host "[chain] ABORT: no weights at $v2"; exit 6 }

Write-Host "`n[chain] === STEP 3/3: score v2 on the SAME split, against the baseline ==="
& $py scripts/eval_ball_detector.py --data $Dataset --weights $v2 --device 0 `
      --json-out "$out\eval_v2_on_current_val.json" --vs "$out\eval_v1_on_current_val.json"

Write-Host "`n[chain] DONE. Read the better? column, not the median alone."
Write-Host "[chain] Do NOT read mAP -- it has misled this project three times."
