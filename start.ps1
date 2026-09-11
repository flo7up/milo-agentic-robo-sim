param([int]$Port = 8000, [switch]$SkipBuild, [switch]$BackendOnly)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path .runtime/env/python.exe)) {
    throw 'Run ./scripts/setup-windows.ps1 first.'
}
if (-not $SkipBuild -and -not $BackendOnly) {
    if (-not (Test-Path frontend/node_modules)) {
        npm --prefix frontend ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed' }
    }
    npm --prefix frontend run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
}
Write-Host "Embodied Robot Lab: http://127.0.0.1:$Port"
if ($BackendOnly) { Write-Host "Backend API: http://127.0.0.1:$Port/docs" }
& ./.runtime/env/python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port $Port