[CmdletBinding()]
param(
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$entryPoint = Join-Path $projectRoot 'simulasyon_yonlendirme_uclu_dashboard.py'

$deployCommand = Get-Command pyside6-deploy -ErrorAction SilentlyContinue
if ($null -eq $deployCommand) {
    throw 'pyside6-deploy bulunamadı. PySide6 kurulu bir sanal ortamda bu komutu yeniden çalıştırın.'
}

$arguments = @(
    $entryPoint,
    '--name',
    'GPS-Denied-Mission-Control'
)
if ($DryRun) {
    $arguments += '--dry-run'
}

Push-Location $projectRoot
try {
    & $deployCommand.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "pyside6-deploy hata kodu döndürdü: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
