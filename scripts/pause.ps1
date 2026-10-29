# Met PumpML en pause : arret propre de tous les processus (collecte, IA, dashboard).
# La pause survit a un redemarrage du PC : rien ne repart tant que tu n'as pas lance resume.ps1.
$root = Split-Path -Parent $PSScriptRoot
$flag = Join-Path $root "data\PAUSE"
New-Item -ItemType File -Force $flag | Out-Null
Write-Host "Pause demandee, arret propre en cours..."

$deadline = (Get-Date).AddSeconds(40)
do {
    Start-Sleep -Seconds 1
    $procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match '-m pumpml' }
} while ($procs -and (Get-Date) -lt $deadline)

Stop-ScheduledTask -TaskName "PumpML" -ErrorAction SilentlyContinue
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host "PumpML est en pause. CPU et GPU liberes. Pour reprendre : PumpML-Reprendre.bat"
