param(
    [string]$Python = "D:\Miniconda3\envs\aml-evidence\python.exe",
    [int]$Port = 8011,
    [int]$Requests = 20,
    [int]$Concurrency = 5,
    [int]$DuplicateReviews = 8,
    [string]$OutputRoot = "artifacts/multiworker_shared_v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$absoluteOutput = Join-Path $repoRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $absoluteOutput | Out-Null

$env:PYTHONPATH = "src"
$env:AML_EVIDENCE_STORE_PATH = Join-Path $OutputRoot "evidence.sqlite"
$env:AML_AGENT_CHECKPOINT_PATH = Join-Path $OutputRoot "checkpoint.sqlite"
$env:AML_AGENT_AUDIT_PATH = Join-Path $OutputRoot "audit.sqlite"
$env:AML_AGENT_COORDINATION_PATH = Join-Path $OutputRoot "coordination.sqlite"
$baseUrl = "http://127.0.0.1:$Port"
$stdoutPath = Join-Path $absoluteOutput "uvicorn.stdout.log"
$stderrPath = Join-Path $absoluteOutput "uvicorn.stderr.log"

$server = Start-Process -FilePath $Python `
    -ArgumentList @(
        "-m", "uvicorn", "aml_evidence_graph.api.app:create_default_app",
        "--factory", "--host", "127.0.0.1", "--port", "$Port", "--workers", "2"
    ) `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri "$baseUrl/healthz" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ready) {
        throw "Two-worker API did not become ready; inspect $stderrPath"
    }

    & $Python scripts/probe_multiworker_evidence.py `
        --base-url $baseUrl `
        --partition-ref mock-partition `
        --attempts 40 `
        --output (Join-Path $OutputRoot "evidence_probe.json")
    if ($LASTEXITCODE -ne 0) { throw "Evidence visibility probe failed." }

    & $Python scripts/benchmark_controlled_agent.py `
        --base-url $baseUrl `
        --requests $Requests `
        --concurrency $Concurrency `
        --duplicate-reviews $DuplicateReviews `
        --timeout 60 `
        --output (Join-Path $OutputRoot "agent_benchmark")
    if ($LASTEXITCODE -ne 0) { throw "Controlled-Agent benchmark failed." }
}
finally {
    $childIds = @(
        Get-CimInstance Win32_Process |
            Where-Object { $_.ParentProcessId -eq $server.Id } |
            Select-Object -ExpandProperty ProcessId
    )
    foreach ($childId in $childIds) {
        Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}
