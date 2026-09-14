param(
    [int]$KeepPort = 8022,
    [Parameter(Mandatory = $true)][string]$Output,
    [switch]$CloseInactive
)

$ErrorActionPreference = 'Stop'
if (Test-Path $Output) { throw 'Use a new output directory to preserve earlier snapshots.' }
$directory = New-Item -ItemType Directory -Path $Output
$records = @()
$listeners = Get-NetTCPConnection -State Listen | Where-Object { $_.LocalAddress -eq '127.0.0.1' -and $_.LocalPort -ge 8000 -and $_.LocalPort -le 8023 } | Sort-Object LocalPort
foreach ($listener in $listeners) {
    $port = $listener.LocalPort
    $processId = $listener.OwningProcess
    $entry = [ordered]@{ port = $port; process_id = $processId; status = 'preserved' }
    try {
        $state = Invoke-RestMethod "http://127.0.0.1:$port/api/state" -TimeoutSec 5
        $entry.run_id = $state.run_id
        $entry.simulated_s = $state.snapshot.simulated_time_s
        $entry.active = $state.agent.active
        $entry.busy = $state.busy
        if ($port -eq $KeepPort -or $state.agent.active -or $state.busy) {
            $entry.reason = 'Current app or active session'
            $records += [pscustomobject]$entry
            continue
        }
        if (-not $state.run_id -or $null -eq $state.observation -or $null -eq $state.agent.active) { throw 'Unrecognized application state' }
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$processId"
        if ($process.Name -notmatch '^python(w)?\.exe$' -or $process.CommandLine -notmatch '(backend\.app|uvicorn|tests\.test_agent)') { throw 'Listener is not a recognized simulator process' }
        $saved = New-Item -ItemType Directory -Path (Join-Path $directory.FullName "port-$port")
        $state | ConvertTo-Json -Depth 60 | Set-Content -LiteralPath (Join-Path $saved.FullName 'state.json') -Encoding utf8
        $trace = Invoke-RestMethod "http://127.0.0.1:$port/api/agent/trace" -TimeoutSec 10
        $trace | ConvertTo-Json -Depth 60 | Set-Content -LiteralPath (Join-Path $saved.FullName 'trace.json') -Encoding utf8
        $urls = @($state.camera.url)
        foreach ($event in $trace.events) {
            $urls += @($event.image_url)
            $urls += @($event.image_urls)
        }
        $imageIndex = 0
        foreach ($url in ($urls | Where-Object { $_ } | Select-Object -Unique)) {
            if ($url -notmatch '^/api/(camera|agent/trace|frames)/' -or $url -match '[\r\n]') { throw 'Unexpected camera reference' }
            Invoke-WebRequest "http://127.0.0.1:$port$url" -TimeoutSec 10 -OutFile (Join-Path $saved.FullName "image-$imageIndex.png")
            $imageIndex++
        }
        $entry.images_saved = $imageIndex
        $entry.status = 'snapshotted'
        if ($CloseInactive) {
            $current = Invoke-RestMethod "http://127.0.0.1:$port/api/state" -TimeoutSec 5
            $owner = Get-NetTCPConnection -State Listen -LocalAddress '127.0.0.1' -LocalPort $port
            if ($current.agent.active -or $current.busy -or $current.run_id -ne $state.run_id -or $owner.OwningProcess -ne $processId) { throw 'Session changed during snapshot; preserving it' }
            $stopped = Invoke-RestMethod "http://127.0.0.1:$port/api/stop" -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 15
            if (-not $stopped.stopped) { throw 'Stop was not confirmed; preserving process' }
            & taskkill.exe /PID $processId /T /F
            if ($LASTEXITCODE -ne 0) { throw 'Process-tree shutdown failed' }
            $entry.status = 'closed_after_snapshot'
        }
    } catch {
        $entry.status = 'preserved_error'
        $entry.reason = 'Snapshot or safety checks failed; process left untouched'
    }
    $records += [pscustomobject]$entry
    $records | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $directory.FullName 'cleanup.json') -Encoding utf8
}
$records | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $directory.FullName 'cleanup.json') -Encoding utf8
$records | Format-Table port, process_id, status, images_saved, reason