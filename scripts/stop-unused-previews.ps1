[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateRange(1, 65535)]
    [int]$KeepPort = 8000,
    [int[]]$Ports = @(8019, 8020, 8021, 8022, 8023, 8024, 8025, 8026)
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root '.runtime/env/python.exe'
$output = Join-Path $root ('.runtime/unused-previews-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$results = @()
foreach ($port in $Ports) {
    if ($port -eq $KeepPort) { continue }
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if (-not $listener -or @($listener).Count -ne 1) { continue }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
    if ($process.ExecutablePath -ne $python) { continue }
    $url = "http://127.0.0.1:$port"
    try {
        $response = Invoke-WebRequest "$url/api/state" -TimeoutSec 5
        $state = $response.Content | ConvertFrom-Json
        if (-not $state.run_id -or $state.agent.active -or $state.busy -or -not $state.stopped -or
                $state.agent.message -ne 'Operator disconnected') { continue }
        $identity = "$port / process $($process.ProcessId) / run $($state.run_id)"
        if (-not $PSCmdlet.ShouldProcess($identity, 'Preserve unused simulator state and stop its process tree')) { continue }
        $directory = Join-Path $output "$port"
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $directory 'state.json'), $response.Content)
        Invoke-WebRequest "$url/api/home" -TimeoutSec 10 -OutFile (Join-Path $directory 'home.json')
        if ($state.agent.session_id) {
            Invoke-WebRequest "$url/api/agent/trace?session_id=$($state.agent.session_id)" -TimeoutSec 10 -OutFile (Join-Path $directory 'trace.json')
        }
        if ($state.camera.url) {
            Invoke-WebRequest ($url + $state.camera.url) -TimeoutSec 10 -OutFile (Join-Path $directory 'camera.png')
        }
        $current = Invoke-RestMethod "$url/api/state" -TimeoutSec 5
        $owner = Get-NetTCPConnection -State Listen -LocalPort $port
        $sameProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ProcessId)"
        if ($owner.OwningProcess -ne $process.ProcessId -or $sameProcess.CreationDate -ne $process.CreationDate -or
                $current.run_id -ne $state.run_id -or $current.agent.session_id -ne $state.agent.session_id -or
                $current.snapshot.simulated_time_s -ne $state.snapshot.simulated_time_s -or
                $current.agent.active -or $current.busy -or -not $current.stopped -or
                $current.agent.message -ne 'Operator disconnected') {
            throw 'Simulator changed during preservation; not stopping it.'
        }
        $descendants = @($process.ProcessId)
        $all = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, CreationDate)
        do {
            $children = @($all | Where-Object { $_.ParentProcessId -in $descendants -and $_.ProcessId -notin $descendants })
            $descendants += @($children.ProcessId)
        } while ($children.Count)
        taskkill /PID $process.ProcessId /T /F
        if ($LASTEXITCODE -ne 0) { throw 'Process-tree shutdown failed.' }
        if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
            throw 'Port is still listening after shutdown.'
        }
        $results += [pscustomobject]@{ port = $port; process = $process.ProcessId; run_id = $state.run_id;
            preserved = $directory; stopped_processes = $descendants; status = 'stopped' }
    }
    catch {
        $results += [pscustomobject]@{ port = $port; status = 'not_stopped'; error_type = $_.Exception.GetType().Name }
        Write-Warning "Port $port was not stopped: $($_.Exception.GetType().Name)"
    }
}
if ($results.Count) {
    $results | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $output 'cleanup.json')
}
$results | Select-Object port, process, run_id, status, preserved | Format-Table -AutoSize