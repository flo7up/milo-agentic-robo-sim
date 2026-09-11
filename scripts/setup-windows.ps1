$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force .runtime | Out-Null
if (-not (Test-Path .runtime/micromamba.exe)) {
    Invoke-WebRequest 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile .runtime/micromamba.tar.bz2
    tar -xjf .runtime/micromamba.tar.bz2 -C .runtime Library/bin/micromamba.exe
    Move-Item .runtime/Library/bin/micromamba.exe .runtime/micromamba.exe
}
$env:MAMBA_ROOT_PREFIX = "$HOME/.robosim-mamba"
& ./.runtime/micromamba.exe create -y -p .runtime/env -f environment.yml
if ($LASTEXITCODE -ne 0) { throw 'Binary physics environment installation failed' }
uv pip install --python .runtime/env/python.exe -r pyproject.toml --group dev
if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed' }
& ./.runtime/env/python.exe -m pytest tests/test_robot_asset.py -q
if ($LASTEXITCODE -ne 0) { throw 'Asset verification failed' }