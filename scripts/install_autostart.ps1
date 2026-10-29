# Lance PumpML automatiquement a l'ouverture de session Windows, et le relance s'il s'arrete.
# Usage (PowerShell, dans le dossier du projet) :  .\scripts\install_autostart.ps1
# Desinstaller :                                     Unregister-ScheduledTask -TaskName PumpML -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $py)) { throw "venv introuvable : $py" }

$action = New-ScheduledTaskAction -Execute $py -Argument "-m pumpml supervise" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "PumpML" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Collecte pump.fun, paper trading et entrainement quotidien" -Force | Out-Null
Write-Host "Tache 'PumpML' installee. Demarrage maintenant..."
Start-ScheduledTask -TaskName "PumpML"
Write-Host "Dashboard : http://127.0.0.1:8787"
Write-Host "Pense a desactiver la mise en veille du PC (Parametres > Alimentation) pour tourner 24h/24."
