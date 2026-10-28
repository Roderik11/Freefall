// The transparents drawn without order (oit.hlsli, Graphics/OrderIndependentTransparency.cs) laid over the HDR
// composite after the forward pass's water:
//     composite × T + (1 − T) × Σ(w·C) / Σ(w·alpha) + additive light,   T = exp(−Σ absorbance)
// Only the columns [ResolveFrom, ResolveTo): in a split compare the other side shows the sorted draws.

#pragma kernel CSResolve

cbuffer PushConstants : register(b3)
{
    uint CompositeUAVIdx;       // RWTexture2D<float4>: the HDR composite, pre-exposed
    uint AccumIdx;              // Texture2D<float4>: rgb Σ w·C, a Σ w·alpha
    uint ExtraIdx;              // Texture2D<float4>: rgb Σ additive light, a Σ absorbance
    uint ResolveFrom;           // first pixel column the transparents are resolved into
    uint ResolveTo;             // one past the last
    uint Height;
};

[numthreads(8, 8, 1)]
void CSResolve(uint3 id : SV_DispatchThreadID)
{
    if (id.x < ResolveFrom || id.x >= ResolveTo || id.y >= Height) return;

    Texture2D<float4> accumTex = ResourceDescriptorHeap[AccumIdx];
    Texture2D<float4> extraTex = ResourceDescriptorHeap[ExtraIdx];
    float4 accum = accumTex[id.xy];
    float4 extra = extraTex[id.xy];
    if (extra.a <= 0.0 && all(extra.rgb <= 0.0)) return;      // nothing transparent here

    RWTexture2D<float4> composite = ResourceDescriptorHeap[CompositeUAVIdx];
    float4 behind = composite[id.xy];
    float t = exp(-extra.a);
    float3 over = accum.a > 1e-7 ? accum.rgb * ((1.0 - t) / accum.a) : 0.0;
    composite[id.xy] = float4(behind.rgb * t + over + extra.rgb, behind.a);
}
