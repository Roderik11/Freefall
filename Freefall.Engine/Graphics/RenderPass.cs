namespace Freefall.Graphics
{
    public enum RenderPass
    {
        Shadow,
        ShadowMap, // Alias?
        Opaque,
        Sky,
        Transparent,
        Light,
        Forward,
        PostProcess,
        Overlay,
        Debug,
        LocalShadow, // point light cube faces into the local shadow atlas (LocalShadows), drawn like Shadow
        OITMoments,  // order-independent transparency: the absorbance's moments (OrderIndependentTransparency, oit.hlsli)
        OIT,         // order-independent transparency: the transparents' light, weighted, after the Forward pass's water
    }
}
