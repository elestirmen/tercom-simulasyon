[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $SimulationArgs
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$entryPoint = Join-Path $projectRoot 'simulasyon_yonlendirme_uclu_dashboard.py'

if (-not (Test-Path -LiteralPath $entryPoint)) {
    throw "Giriş dosyası bulunamadı: $entryPoint"
}

Push-Location $projectRoot
try {
    & python $entryPoint @SimulationArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
