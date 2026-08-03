param(
    [string]$PythonExecutable = "D:\Miniconda3\envs\aml-evidence\python.exe",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [ValidateRange(1, 16)]
    [int]$MaxGpus = 1,
    [switch]$ForceRetrain,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$features = Join-Path $repoRoot "artifacts\pit_features_fe_v2"
$tableBase = Join-Path $repoRoot "artifacts\table_baseline_fe_v2"
$gatOutput = Join-Path $repoRoot "artifacts\gat_fe_v2"
$tableOofOutput = Join-Path $repoRoot "artifacts\table_oof_fe_v2"
$graphOofOutput = Join-Path $repoRoot "artifacts\graph_oof_fe_v2"
$fusionOutput = Join-Path $repoRoot "artifacts\fusion_fe_v2"
$fusionTestOutput = Join-Path $repoRoot "artifacts\fusion_test_fe_v2"
$fusionBootstrapOutput = Join-Path $repoRoot "artifacts\fusion_test_fe_v2_bootstrap"
$investigationOutput = Join-Path $repoRoot "artifacts\test_investigation_views_fe_v2"
$modelConfig = Join-Path $repoRoot "configs\models.gat.yaml"
$tableConfig = Join-Path $repoRoot "configs\models.yaml"
$logRoot = Join-Path $repoRoot "artifacts\logs"
$statusPath = Join-Path $logRoot "fe_v2_gat_oof_fusion_status.json"
$chainLog = Join-Path $logRoot "fe_v2_gat_oof_fusion_windows.log"

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Set-Location -LiteralPath $repoRoot
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$script:pipelineState = [ordered]@{
    pipeline = "fe_v2_gat_oof_fusion"
    platform = "windows"
    features = "artifacts/pit_features_fe_v2"
    model_config_gat = "configs/models.gat.yaml"
    max_gpus = $MaxGpus
    current_step = "preflight"
    current_state = "starting"
    steps = [ordered]@{}
    outputs = [ordered]@{
        gat = "artifacts/gat_fe_v2"
        table_oof = "artifacts/table_oof_fe_v2"
        graph_oof = "artifacts/graph_oof_fe_v2"
        fusion = "artifacts/fusion_fe_v2"
        fusion_test = "artifacts/fusion_test_fe_v2"
        fusion_bootstrap = "artifacts/fusion_test_fe_v2_bootstrap"
        investigation_views = "artifacts/test_investigation_views_fe_v2"
    }
}

function Get-UtcTimestamp {
    return [DateTimeOffset]::UtcNow.ToString("o")
}

function Write-ChainLog {
    param([string]$Message)
    $line = "[$(Get-UtcTimestamp)] $Message"
    Write-Host $line
    Add-Content -LiteralPath $chainLog -Value $line -Encoding utf8
}

function Save-PipelineState {
    $script:pipelineState.updated_at = Get-UtcTimestamp
    $json = $script:pipelineState | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText(
        $statusPath,
        $json,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Set-StepState {
    param(
        [string]$Name,
        [string]$State,
        [string]$Detail = "",
        [string]$StartedAt = "",
        [string]$FinishedAt = ""
    )
    $entry = [ordered]@{ state = $State; updated_at = Get-UtcTimestamp }
    if ($Detail) { $entry.detail = $Detail }
    if ($StartedAt) { $entry.started_at = $StartedAt }
    if ($FinishedAt) { $entry.finished_at = $FinishedAt }
    $script:pipelineState.steps[$Name] = $entry
    $script:pipelineState.current_step = $Name
    $script:pipelineState.current_state = $State
    Save-PipelineState
}

function Assert-FileExists {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Description`: $Path"
    }
}

function Invoke-ExperimentStep {
    param(
        [string]$Name,
        [scriptblock]$CompletionCheck,
        [string[]]$Arguments
    )
    if (-not $ForceRetrain -and (& $CompletionCheck)) {
        Set-StepState -Name $Name -State "skipped" -Detail "complete artifact present"
        Write-ChainLog "SKIP $Name (complete artifact present)"
        return
    }

    $startedAt = Get-UtcTimestamp
    Set-StepState -Name $Name -State "running" -Detail ($Arguments -join " ") -StartedAt $startedAt
    Write-ChainLog "START $Name"
    # Windows PowerShell 5.1 wraps native stderr records as non-terminating
    # NativeCommandError objects. Structured Python logging uses stderr even for INFO,
    # so keep streaming both channels without letting those records abort model saving.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExecutable @Arguments 2>&1 | Tee-Object -FilePath $chainLog -Append
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $finishedAt = Get-UtcTimestamp
    if ($exitCode -ne 0) {
        Set-StepState -Name $Name -State "failed" -Detail "exit=$exitCode" -StartedAt $startedAt -FinishedAt $finishedAt
        throw "$Name failed with exit code $exitCode"
    }
    if (-not (& $CompletionCheck)) {
        Set-StepState -Name $Name -State "failed" -Detail "expected artifact missing" -StartedAt $startedAt -FinishedAt $finishedAt
        throw "$Name exited successfully but its expected artifact is missing."
    }
    Set-StepState -Name $Name -State "complete" -StartedAt $startedAt -FinishedAt $finishedAt
    Write-ChainLog "COMPLETE $Name"
}

Assert-FileExists -Path $PythonExecutable -Description "Python executable"
Assert-FileExists -Path (Join-Path $features "_run_manifest.json") -Description "FE v2 PIT manifest"
Assert-FileExists -Path (Join-Path $tableBase "metrics.json") -Description "FE v2 table metrics"
Assert-FileExists -Path (Join-Path $tableBase "scores\table_validation_scores.parquet") -Description "FE v2 table validation scores"
Assert-FileExists -Path (Join-Path $tableBase "scores\table_test_scores.parquet") -Description "FE v2 table test scores"
Assert-FileExists -Path $modelConfig -Description "GAT model config"
Assert-FileExists -Path $tableConfig -Description "table model config"

$steps = @(
    [ordered]@{
        name = "01_gat"
        complete = { Test-Path -LiteralPath (Join-Path $gatOutput "metrics.json") -PathType Leaf }
        arguments = @(
            "-u", "-m", "aml_evidence_graph.training.run_graphsage",
            "--features", $features,
            "--output", $gatOutput,
            "--model-config", $modelConfig,
            "--device", $Device,
            "--max-gpus", "$MaxGpus",
            "--overwrite"
        )
    },
    [ordered]@{
        name = "02_table_oof"
        complete = { Test-Path -LiteralPath (Join-Path $tableOofOutput "table_oof_scores.parquet") -PathType Leaf }
        arguments = @(
            "-u", "-m", "aml_evidence_graph.training.oof",
            "--features", $features,
            "--output", $tableOofOutput,
            "--model", "table",
            "--model-config", $tableConfig,
            "--splits", "3",
            "--minimum-training-months", "2",
            "--overwrite"
        )
    },
    [ordered]@{
        name = "03_graph_oof"
        complete = { Test-Path -LiteralPath (Join-Path $graphOofOutput "graphsage_oof_scores.parquet") -PathType Leaf }
        arguments = @(
            "-u", "-m", "aml_evidence_graph.training.oof",
            "--features", $features,
            "--output", $graphOofOutput,
            "--model", "graphsage",
            "--model-config", $modelConfig,
            "--splits", "3",
            "--minimum-training-months", "2",
            "--device", $Device,
            "--max-gpus", "$MaxGpus",
            "--overwrite"
        )
    },
    [ordered]@{
        name = "04_fusion"
        complete = { Test-Path -LiteralPath (Join-Path $fusionOutput "run_manifest.json") -PathType Leaf }
        arguments = @(
            "-u", "-m", "aml_evidence_graph.training.fusion",
            "--oof", (Join-Path $tableOofOutput "table_oof_scores.parquet"),
            "--oof", (Join-Path $graphOofOutput "graphsage_oof_scores.parquet"),
            "--validation", (Join-Path $tableBase "scores\table_validation_scores.parquet"),
            "--validation", (Join-Path $gatOutput "scores\graphsage_validation_scores.parquet"),
            "--components", "catboost,graphsage",
            "--output", $fusionOutput,
            "--model-config", $tableConfig,
            "--overwrite"
        )
    },
    [ordered]@{
        name = "05_fusion_test"
        complete = { Test-Path -LiteralPath (Join-Path $fusionTestOutput "metrics.json") -PathType Leaf }
        arguments = @(
            "-u", "-m", "aml_evidence_graph.training.evaluate_fusion",
            "--fusion-dir", $fusionOutput,
            "--test", (Join-Path $tableBase "scores\table_test_scores.parquet"),
            "--test", (Join-Path $gatOutput "scores\graphsage_test_scores.parquet"),
            "--features", $features,
            "--output", $fusionTestOutput,
            "--overwrite"
        )
    },
    [ordered]@{
        name = "06_fusion_bootstrap"
        complete = { Test-Path -LiteralPath (Join-Path $fusionBootstrapOutput "metrics.json") -PathType Leaf }
        arguments = @(
            "-u", "-m", "aml_evidence_graph.training.evaluate_fusion",
            "--fusion-dir", $fusionOutput,
            "--test", (Join-Path $tableBase "scores\table_test_scores.parquet"),
            "--test", (Join-Path $gatOutput "scores\graphsage_test_scores.parquet"),
            "--features", $features,
            "--output", $fusionBootstrapOutput,
            "--bootstrap-iterations", "200",
            "--overwrite"
        )
    },
    [ordered]@{
        name = "07_investigation_views"
        complete = { Test-Path -LiteralPath (Join-Path $investigationOutput "summary.json") -PathType Leaf }
        arguments = @(
            "-u", "-m", "aml_evidence_graph.aggregation.views",
            "--transactions", $features,
            "--scores", (Join-Path $fusionTestOutput "test_fusion_scores.parquet"),
            "--score-column", "fusion_calibrated_probability",
            "--as-of-ts", "2023-08-23T23:59:59Z",
            "--output", $investigationOutput
        )
    }
)

if ($PlanOnly) {
    Write-Host "FE v2 experiment plan ($Device, max-gpus=$MaxGpus):"
    foreach ($step in $steps) {
        $state = if (& $step.complete) { "complete" } else { "pending" }
        Write-Host "  $($step.name): $state"
    }
    exit 0
}

try {
    Write-ChainLog "===== FE v2 GAT -> OOF -> fusion START ====="
    foreach ($step in $steps) {
        Invoke-ExperimentStep -Name $step.name -CompletionCheck $step.complete -Arguments $step.arguments
    }

    $tableMetrics = Get-Content (Join-Path $tableBase "metrics.json") -Raw -Encoding utf8 | ConvertFrom-Json
    $gatMetrics = Get-Content (Join-Path $gatOutput "metrics.json") -Raw -Encoding utf8 | ConvertFrom-Json
    $fusionMetrics = Get-Content (Join-Path $fusionTestOutput "metrics.json") -Raw -Encoding utf8 | ConvertFrom-Json
    $bootstrapMetrics = Get-Content (Join-Path $fusionBootstrapOutput "metrics.json") -Raw -Encoding utf8 | ConvertFrom-Json
    $investigationSummary = Get-Content (Join-Path $investigationOutput "summary.json") -Raw -Encoding utf8 | ConvertFrom-Json
    $catboostPrAuc = [double]$tableMetrics.test_metrics.catboost.pr_auc
    $gatPrAuc = [double]$gatMetrics.test_metrics.pr_auc
    $fusionPrAuc = [double]$fusionMetrics.test_metrics.pr_auc
    $script:pipelineState.summary = [ordered]@{
        catboost_fe_v2_test_pr_auc = $catboostPrAuc
        gat_fe_v2_test_pr_auc = $gatPrAuc
        fusion_fe_v2_test_pr_auc = $fusionPrAuc
        gat_delta_vs_v1 = $gatPrAuc - 0.9483
        fusion_delta_vs_v1 = $fusionPrAuc - 0.9175
        fusion_exceeds_gat = $fusionPrAuc -gt $gatPrAuc
        fusion_bootstrap = $bootstrapMetrics.test_bootstrap_intervals
        investigation_account_count = $investigationSummary.account_count
        investigation_funds_path_count = $investigationSummary.funds_path_count
        investigation_case_count = $investigationSummary.investigation_case_count
    }
    $script:pipelineState.current_step = "pipeline"
    $script:pipelineState.current_state = "complete"
    Save-PipelineState
    Write-ChainLog "SUMMARY CatBoost=$catboostPrAuc GAT=$gatPrAuc fusion=$fusionPrAuc"
    Write-ChainLog "===== FE v2 GAT -> OOF -> fusion COMPLETE ====="
} catch {
    $script:pipelineState.current_state = "failed"
    $script:pipelineState.error = $_.Exception.Message
    Save-PipelineState
    Write-ChainLog "FAILED: $($_.Exception.Message)"
    throw
}
