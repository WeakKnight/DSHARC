param([string]$Dxc = 'dxc')

$ErrorActionPreference = 'Stop'
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Run from an x64 Visual Studio developer PowerShell with the Windows SDK installed.'
}
$testDirectory = Join-Path ([IO.Path]::GetTempPath()) ('dsharc-runtime-' + [Guid]::NewGuid().ToString('N'))
& (Join-Path $PSScriptRoot 'Compile.ps1') -Dxc $Dxc -OutputDirectory $testDirectory
$executable = Join-Path $testDirectory 'Runtime.exe'
$object = Join-Path $testDirectory 'Runtime.obj'
& cl.exe /nologo /std:c++17 /EHsc /W4 /WX /O2 (Join-Path $PSScriptRoot 'Runtime.cpp') "/Fe:$executable" "/Fo:$object" /link d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'Runtime test harness compilation failed' }
& $executable $testDirectory
if ($LASTEXITCODE -ne 0) { throw 'DSHARC runtime tests failed' }
Write-Output "Runtime test artifacts: $testDirectory"
