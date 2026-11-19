# Fetches NVIDIA's DLSS SDK files the engine's DLSS upscaler needs (UPSCALING_PLAN.md stage 4) into this folder: the
# redistributable DLSS model (win-x64\nvngx_dlss.dll), the static NGX library and headers the shim is built against
# (sdk\) and the licence. They stay out of git; run this once after a fresh clone, then build.cmd (the shim,
# win-x64\freefall_dlss.dll), then build the engine. Without them the engine runs and DLSS reads unavailable.
#
# Source: github.com/NVIDIA/DLSS, tag v310.9.16 (2026-10-16), NVIDIA RTX SDK licence (sdk\LICENSE.txt: not open
# source; it allows shipping nvngx_dlss.dll with an application). The hashes below are of the files first fetched on
# 2026-11-19; nvngx_dlss.dll's Authenticode signature read valid for NVIDIA Corporation.
#
#   powershell -ExecutionPolicy Bypass -File ThirdParty\DLSS\fetch.ps1

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$base = 'https://raw.githubusercontent.com/NVIDIA/DLSS/v310.9.16'

$files = @(
    @{ From = 'lib/Windows_x86_64/rel/nvngx_dlss.dll';   To = 'win-x64/nvngx_dlss.dll';   Sha256 = 'd00b5b0ab29c3ca101b70c3a5ecb3a01b7ef33766a3383a28a18623083e6123e' },
    @{ From = 'lib/Windows_x86_64/x64/nvsdk_ngx_s.lib';  To = 'sdk/lib/nvsdk_ngx_s.lib';  Sha256 = '7b9a5239c1145af5f645ab472aa066f9ea79db36ac819562312aae69d65f3765' },
    @{ From = 'include/nvsdk_ngx.h';                     To = 'sdk/include/nvsdk_ngx.h' },
    @{ From = 'include/nvsdk_ngx_defs.h';                To = 'sdk/include/nvsdk_ngx_defs.h' },
    @{ From = 'include/nvsdk_ngx_params.h';              To = 'sdk/include/nvsdk_ngx_params.h' },
    @{ From = 'include/nvsdk_ngx_helpers.h';             To = 'sdk/include/nvsdk_ngx_helpers.h' },
    @{ From = 'include/nvsdk_ngx_helpers_d3d.h';         To = 'sdk/include/nvsdk_ngx_helpers_d3d.h' },
    @{ From = 'LICENSE.txt';                             To = 'sdk/LICENSE.txt' }
)

foreach ($f in $files) {
    $target = Join-Path $here $f.To
    New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
    Write-Host "Fetching $($f.From)"
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$($f.From)" -OutFile $target
    if ($f.Sha256) {
        $hash = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLowerInvariant()
        if ($hash -ne $f.Sha256) {
            Remove-Item $target
            throw "$($f.To): SHA-256 $hash is not the expected $($f.Sha256); removed"
        }
        if ($target.EndsWith('.dll')) {
            $signature = Get-AuthenticodeSignature $target
            if ($signature.Status -ne 'Valid') { Write-Warning "$($f.To): signature $($signature.Status)" }
        }
    }
}
Write-Host "Done: $here. Now run build.cmd for the shim."
