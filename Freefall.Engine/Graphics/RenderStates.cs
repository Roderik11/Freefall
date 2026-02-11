namespace Freefall.Graphics
{
    /// <summary>
    /// Blend state presets for DX12 pipelines.
    /// These are used when creating Pipeline State Objects.
    /// </summary>
    public enum BlendState
    {
        /// <summary>No blending, writes directly.</summary>
        Opaque,
        /// <summary>Alpha blending: src * srcAlpha + dst * (1 - srcAlpha)</summary>
        AlphaBlend,
        /// <summary>Additive blending: src + dst</summary>
        Additive
    }

    /// <summary>
    /// Depth stencil state presets for DX12 pipelines.
    /// </summary>
    public enum DepthStencilState
    {
        /// <summary>Read and write depth, comparison Less.</summary>
        Default,
        /// <summary>Read depth only, no write.</summary>
        ZReadNoWrite,
        /// <summary>Depth disabled.</summary>
        None
    }

    /// <summary>
    /// Rasterizer state presets for DX12 pipelines.
    /// </summary>
    public enum RasterizerState
    {
        /// <summary>Cull back faces (default).</summary>
        BackCull,
        /// <summary>Cull front faces (for inside-out geometry).</summary>
        FrontCull,
        /// <summary>No culling.</summary>
        NoCull,
        /// <summary>Wireframe mode.</summary>
        Wireframe
    }
}
