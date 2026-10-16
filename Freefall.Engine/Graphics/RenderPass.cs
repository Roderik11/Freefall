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
    }
}
