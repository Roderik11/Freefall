using System;
using System.Numerics;
using Vortice.Direct3D12;

namespace Freefall.Graphics.Upscaling
{
    /// <summary>
    /// What the renderer asks of a temporal upscaler once it is made (FSR: <see cref="Fsr"/>, DLSS: <see cref="Dlss"/>;
    /// UPSCALING_PLAN.md): a frame through it, its jitter count, what it is and what it holds. Making and letting go
    /// differ by SDK and happen at the renderer's resize point (DeferredRenderer.CreateDisplayTextures).
    /// </summary>
    internal interface ITemporalUpscaler : IDisposable
    {
        /// <summary>Whether it runs: its context is made (FSR), or NGX is up for it and its feature can be made (DLSS).</summary>
        bool Created { get; }

        /// <summary>Which upscaler and version for the log and the stats ("FSR 3.1.6", "DLSS 310.9.16").</summary>
        string Version { get; }

        /// <summary>The video memory its own resources take (outside the GpuMemory ledger), 0 if it cannot say yet.</summary>
        ulong GpuBytes { get; }

        /// <summary>How many jitter offsets go round for this render and display width.</summary>
        int JitterPhaseCount(int renderWidth, int displayWidth);

        /// <summary>
        /// Upscale this frame into <paramref name="output"/> (display size, R16G16B16A16_Float, in UnorderedAccess). The
        /// inputs are the render size and readable by compute: the pre-exposed HDR colour, the reverse-Z depth (R32
        /// typeless), the motion vectors (R16G16_Float, last frame's UV minus this frame's, no jitter), the exposure (1x1
        /// R32_Float holding 1), and the reactive mask (R8_UNorm, how far to trust this frame over the history; null for
        /// none). The SDK may set its own descriptor heaps and root signature: the caller sets the engine's again after
        /// it. False if it failed.
        /// </summary>
        bool Dispatch(ID3D12GraphicsCommandList list, ID3D12Resource color, ID3D12Resource depth, ID3D12Resource motion,
            ID3D12Resource exposure, ID3D12Resource? reactive, ID3D12Resource output, int renderWidth, int renderHeight, int displayWidth, int displayHeight,
            Vector2 jitterPixels, bool reset, float near, float far, float fovYRadians, float frameMs, float sharpness);
    }
}
