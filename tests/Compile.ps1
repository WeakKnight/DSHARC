param(
    [string]$Dxc = 'dxc',
    [switch]$Spirv,
    [string]$OutputDirectory = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrEmpty($OutputDirectory)) {
    $OutputDirectory = Join-Path ([IO.Path]::GetTempPath()) ('dsharc-compile-' + [Guid]::NewGuid().ToString('N'))
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$entries = @('ClearCS', 'ResetCountCS', 'BeginCS', 'RequestCS', 'CompactCS', 'ArgumentsCS',
    'MaterialCS', 'LightingCS', 'ResolveCS', 'GatherCS', 'LookupCS', 'SamplingCS', 'KeyCS')
$targets = @('DXIL')
if ($Spirv) { $targets += 'SPIRV' }

foreach ($target in $targets) {
    foreach ($entry in $entries) {
        $compilerArguments = @('-T', 'cs_6_6', '-E', $entry, '-HV', '2021', '-Ges', '-WX',
            '-I', (Join-Path $repoRoot 'include'), (Join-Path $PSScriptRoot 'Compile.hlsl'),
            '-Fo', (Join-Path $OutputDirectory ($entry + '.' + $target.ToLowerInvariant())))
        if ($target -eq 'SPIRV') {
            $compilerArguments += @('-spirv', '-fspv-target-env=vulkan1.2',
                '-fvk-u-shift', '0', '0', '-fvk-t-shift', '16', '0', '-fvk-b-shift', '32', '0')
        }
        & $Dxc @compilerArguments
        if ($LASTEXITCODE -ne 0) { throw "$target compilation failed: $entry" }
        Write-Output "$target PASS $entry"
    }
}
Write-Output "Compiled $($entries.Count * $targets.Count) entry points. Artifacts: $OutputDirectory"
