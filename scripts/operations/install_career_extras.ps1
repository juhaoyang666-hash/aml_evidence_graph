param(
    [ValidateSet("agent", "retrieval", "mlops", "spark", "all")]
    [string]$Group = "agent",
    [string]$Python = "D:\Miniconda3\envs\aml-evidence\python.exe"
)

$ErrorActionPreference = "Stop"
$IndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$Groups = if ($Group -eq "all") {
    @("agent-persistence", "retrieval", "mlops", "spark")
} else {
    @{ agent = "agent-persistence"; retrieval = "retrieval"; mlops = "mlops"; spark = "spark" }[$Group]
}

foreach ($Extra in $Groups) {
    & $Python -m pip install --index-url $IndexUrl -e ".[${Extra}]"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install optional dependency group: $Extra"
    }
}

& $Python scripts/operations/check_career_environment.py
exit $LASTEXITCODE
