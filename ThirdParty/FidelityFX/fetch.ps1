# Fetches AMD's FidelityFX SDK files the engine's FSR upscaler needs (UPSCALING_PLAN.md stage 3) into this folder:
# the signed DX12 DLLs (win-x64), the FFX API headers the C# bindings follow (include) and the licence. They stay out of
# git (57.7 MB); run this once after a fresh clone, then build. Without them the engine runs and FSR reads unavailable.
#
# Source: github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag v2.4.0 (2026-10-28), MIT licence. AMD requires its
# signed DLLs: they are not built from source. The hashes below are of the files first fetched on 2026-11-19, whose
# Authenticode signatures read valid for Advanced Micro Devices.
#
#   powershell -ExecutionPolicy Bypass -File ThirdParty\FidelityFX\fetch.ps1

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$base = 'https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX'

$files = @(
    @{ From = 'signedbin/amd_fidelityfx_upscaler_dx12.dll'; To = 'win-x64/amd_fidelityfx_upscaler_dx12.dll'; Sha256 = 'aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace' },
    @{ From = 'signedbin/amd_fidelityfx_loader_dx12.dll';   To = 'win-x64/amd_fidelityfx_loader_dx12.dll';   Sha256 = 'e2191d3c0828be45eb8affd4e3ec0378f2293adc345b772a7170edcf3b021ecb' },
    @{ From = 'api/include/ffx_api.h';                      To = 'include/ffx_api.h' },
    @{ From = 'api/include/ffx_api_types.h';                To = 'include/ffx_api_types.h' },
    @{ From = 'api/include/dx12/ffx_api_dx12.h';            To = 'include/dx12/ffx_api_dx12.h' },
    @{ From = 'upscalers/include/ffx_upscale.h';            To = 'include/ffx_upscale.h' },
    @{ From = 'docs/license.md';                            To = 'license.md' }
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
        $signature = Get-AuthenticodeSignature $target
        if ($signature.Status -ne 'Valid') { Write-Warning "$($f.To): signature $($signature.Status)" }
    }
}
Write-Host "Done: $here"
