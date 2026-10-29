# Reprend PumpML apres une pause.
$root = Split-Path -Parent $PSScriptRoot
Remove-Item -Force (Join-Path $root "data\PAUSE") -ErrorAction SilentlyContinue
Start-ScheduledTask -TaskName "PumpML"
Write-Host "PumpML redemarre. Dashboard : http://127.0.0.1:8787"
