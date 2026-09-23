[CmdletBinding()]
param(
    [switch]$NoUi,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$projectRoot = $PSScriptRoot
$coreSource = Join-Path $projectRoot "cpp_core"
$buildDir = Join-Path $coreSource "build"
$coreExe = Join-Path $buildDir "bin\graph_core.exe"
$pipeline = Join-Path $projectRoot "repos\starter\starter.py"

if (-not (Test-Path -LiteralPath $coreExe -PathType Leaf)) {
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        throw "graph_core.exe отсутствует, а CMake не найден в PATH. Установите CMake или запустите скрипт из Developer PowerShell for Visual Studio."
    }
    Write-Host "C++ core отсутствует: выполняется сборка Release..."
    & cmake -S $coreSource -B $buildDir -A x64
    if ($LASTEXITCODE -ne 0) { throw "Конфигурация CMake завершилась с кодом $LASTEXITCODE." }
    & cmake --build $buildDir --config Release
    if ($LASTEXITCODE -ne 0) { throw "Сборка CMake завершилась с кодом $LASTEXITCODE." }
}

if (-not (Test-Path -LiteralPath $coreExe -PathType Leaf)) {
    throw "После сборки не найден C++ core: $coreExe"
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python не найден в PATH."
}

$pipelineArgs = @($pipeline, "--core", $coreExe, "--host", $HostAddress, "--port", $Port)
if ($NoUi) { $pipelineArgs += "--no-ui" }
& python @pipelineArgs
if ($LASTEXITCODE -ne 0) { throw "Python pipeline завершился с кодом $LASTEXITCODE." }
