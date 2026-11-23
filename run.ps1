param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root;$env:PYTHONPATH" } else { $Root }
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    & $VenvPy -m nlu_authorship @ArgsList
} else {
    python -m nlu_authorship @ArgsList
}
exit $LASTEXITCODE
