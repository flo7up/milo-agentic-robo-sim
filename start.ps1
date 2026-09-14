[CmdletBinding(DefaultParameterSetName = 'Lab', SupportsShouldProcess = $true, ConfirmImpact = 'Low')]
param(
    [Parameter(ParameterSetName = 'Lab')]
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [Parameter(ParameterSetName = 'Lab')]
    [switch]$SkipBuild,
    [Parameter(ParameterSetName = 'Lab')]
    [switch]$BackendOnly,
    [Parameter(ParameterSetName = 'Lab')]
    [switch]$Nav2,
    [Parameter(Mandatory = $true, ParameterSetName = 'Navigation')]
    [switch]$NavigationTest,
    [Parameter(ParameterSetName = 'Navigation')]
    [ValidateNotNullOrEmpty()]
    [string]$Checkpoint = '.runtime/navigation-stop-balanced-1500/checkpoint',
    [Parameter(ParameterSetName = 'Navigation')]
    [ValidateRange(0, 19)]
    [int]$Case = 15,
    [Parameter(ParameterSetName = 'Navigation')]
    [int]$Seed = 716,
    [Parameter(ParameterSetName = 'Navigation')]
    [ValidateRange(1, 40)]
    [int]$Requests = 30,
    [Parameter(ParameterSetName = 'Navigation')]
    [ValidateNotNullOrEmpty()]
    [string]$Output,
    [Parameter(ParameterSetName = 'Navigation')]
    [ValidateNotNullOrEmpty()]
    [string]$ModelPython = '.runtime/smolvla-env/Scripts/python.exe'
)

$ErrorActionPreference = 'Stop'
$nav2Container = $null
$previousRosEnabled = [Environment]::GetEnvironmentVariable('MILO_ROS_ENABLED', 'Process')
Push-Location $PSScriptRoot
try {
    $physicsPython = './.runtime/env/python.exe'
    if (-not (Test-Path -LiteralPath $physicsPython -PathType Leaf)) {
        throw 'Physics Python is missing. Run ./scripts/setup-windows.ps1 first.'
    }
    if ($PSCmdlet.ParameterSetName -eq 'Navigation') {
        if (-not $NavigationTest) { throw 'Specify -NavigationTest to run the local model evaluation.' }
        if (-not (Test-Path -LiteralPath $ModelPython -PathType Leaf)) {
            throw "Model Python is missing: $ModelPython. See docs/SMOLVLA.md for the separate Windows CUDA environment."
        }
        if (-not (Test-Path -LiteralPath $Checkpoint -PathType Container)) {
            throw "Navigation checkpoint directory is missing: $Checkpoint"
        }
        $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
        foreach ($filename in @('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $checkpointPath $filename) -PathType Leaf)) {
                throw "Navigation checkpoint is missing $filename. Select a completed navigation fine-tuning checkpoint."
            }
        }
        $trainingReport = Join-Path (Split-Path $checkpointPath -Parent) 'report.json'
        if (-not (Test-Path -LiteralPath $trainingReport -PathType Leaf)) {
            throw 'Navigation checkpoint has no training report. Select the checkpoint subfolder of a completed training run.'
        }
        $training = Get-Content -LiteralPath $trainingReport -Raw | ConvertFrom-Json
        if ($training.status -ne 'trained_and_reloaded' -or $training.embodiment -ne 'milo-navigation-v1') {
            throw 'This mode requires a completed milo-navigation-v1 checkpoint, not an arm or base-model checkpoint.'
        }
        if (-not $Output) {
            $runName = 'navigation-{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
            $Output = Join-Path '.runtime/navigation-tests' $runName
        }
        $outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
        if (Test-Path -LiteralPath $outputPath) {
            throw "Output already exists: $outputPath. Choose a new -Output directory; test results are never overwritten."
        }
        $modelPythonPath = (Resolve-Path -LiteralPath $ModelPython).Path
        Write-Host 'Local navigation test: native Windows CUDA, separate physics scene.'
        Write-Host 'This does not start the browser arm server or move the robot in an existing browser session.'
        Write-Host "Checkpoint: $checkpointPath"
        Write-Host "Case: $Case | Seed: $Seed | Request limit: $Requests"
        Write-Host "Results: $outputPath"
        if (-not $PSCmdlet.ShouldProcess($outputPath, 'Run fine-tuned navigation model in an isolated simulation')) { return }
        $arguments = @('-m', 'scripts.navigation_policy', '--stage', 'evaluate', '--checkpoint', $checkpointPath,
            '--python', $modelPythonPath, '--case', "$Case", '--seed', "$Seed", '--requests', "$Requests", '--output', $outputPath)
        & $physicsPython @arguments
        if ($LASTEXITCODE -ne 0) { throw "Navigation test process failed (exit $LASTEXITCODE). Review results in $outputPath." }
        $resultPath = Join-Path $outputPath 'report.json'
        if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) { throw 'Navigation test exited without producing a report.' }
        $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
        Write-Host "Outcome: $($result.status) | Task success: $($result.success)"
        Write-Host "Report: $resultPath"
        Write-Host "Camera timelapse (not real-time): $(Join-Path $outputPath 'motion.gif')"
        return
    }

    if ($SkipBuild -and -not $BackendOnly -and -not (Test-Path -LiteralPath 'frontend/dist/index.html' -PathType Leaf)) {
        throw 'Built frontend is missing. Run ./start.ps1 without -SkipBuild, or use -BackendOnly for the API.'
    }
    $url = "http://127.0.0.1:$Port"
    if (-not $PSCmdlet.ShouldProcess($url, 'Build as needed and start the browser UI/API; no automatic model inference')) { return }
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if ($listeners | Where-Object { $_.Port -eq $Port }) {
        throw "Port $Port is already in use. Use the existing app if it is Milo, or choose another -Port. Port 8001 is reserved for automated tests."
    }
    if (-not $SkipBuild -and -not $BackendOnly) {
        if (-not (Test-Path -LiteralPath 'frontend/node_modules' -PathType Container)) {
            npm --prefix frontend ci --registry=https://packagefeedproxy.microsoft.io/npm/
            if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed' }
        }
        npm --prefix frontend run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
    }
    if ($Nav2) {
        if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker Desktop is required for -Nav2.' }
        docker version --format '{{.Server.Version}}'
        if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with its Linux engine, then retry -Nav2.' }
        docker image inspect milo-nav2:jazzy --format '{{.Id}}' 2>$null
        if ($LASTEXITCODE -ne 0) {
            docker build --tag milo-nav2:jazzy ros
            if ($LASTEXITCODE -ne 0) { throw 'Nav2 image build failed; normal startup still supports Built-in backup.' }
        }
        $nav2Container = "milo-nav2-$Port-$([guid]::NewGuid().ToString('N').Substring(0,6))"
        $env:MILO_ROS_ENABLED = '1'
        docker run --detach --name $nav2Container --restart unless-stopped --env "ROS_DOMAIN_ID=$($Port % 200)" --mount "type=bind,source=$PSScriptRoot/ros,target=/opt/milo,readonly" milo-nav2:jazzy bash -lc "source /opt/ros/jazzy/setup.bash && exec ros2 launch /opt/milo/milo.launch.py backend:=http://host.docker.internal:$Port"
        if ($LASTEXITCODE -ne 0) { throw 'Could not start the Nav2 container.' }
        Write-Host "Nav2 primary: $nav2Container. Run settings retains Built-in backup."
    }
    Write-Host "Embodied Robot Lab: $url"
    if ($BackendOnly) { Write-Host "Backend API: $url/docs" }
    Write-Host 'Browser driving: configure Luna, load a challenge, then click Start LLM control.'
    Write-Host 'Luna selects observed destinations; -Nav2 enables Nav2 primary, with Built-in backup in Run settings.'
    Write-Host 'Select SmolVLA primitives for learned local velocities; that model loads on its first Start.'
    Write-Host 'SmolVLA stays loaded between browser runs. Use Unload local model to release GPU memory.'
    Write-Host 'Optional isolated evaluation: ./start.ps1 -NavigationTest.'
    Write-Host 'No arm-policy server is required. Luna deployment access is required for supervision.'
    & $physicsPython -m uvicorn backend.app:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Robot Lab server exited with code $LASTEXITCODE." }
}
finally {
    if ($nav2Container) {
        docker rm --force $nav2Container | Out-Null
        [Environment]::SetEnvironmentVariable('MILO_ROS_ENABLED', $previousRosEnabled, 'Process')
    }
    Pop-Location
}