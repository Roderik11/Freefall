// terrain_height_bake.hlsl — Composites terrain HeightLayers and StampGroups
// into a final R32_Float heightmap.
//
// HeightLayers: dispatched per-layer (CS_ImportLayer)
// StampGroups: dispatched per-group with a StructuredBuffer of instances (CS_StampGroup)
//
// Uses push constants (b3) matching engine convention.

#pragma kernel CS_ImportLayer
#pragma kernel CS_StampGroup
#pragma kernel CS_Clear

// Push constants (root parameter 0, register b3) — bindless indices + params
cbuffer PushConstants : register(b3)
{
    uint SourceIdx;       // slot 0 — SRV: source heightmap texture (import) / brush texture (stamp)
    uint OutputIdx;       // slot 1 — UAV: output RWTexture2D<float>
    uint StampBufIdx;     // slot 2 — SRV: StructuredBuffer<StampData> (stamp group)
    uint BlendMode;       // slot 3 — 0=Set, 1=Add, 2=Max, 3=Lerp
    float Opacity;        // slot 4 — layer/group opacity [0..1]
    uint StampCount;      // slot 5 — number of stamp instances
};

SamplerState sampLinear : register(s0);

// Per-stamp instance data (matches C# StampInstanceGPU)
struct StampData
{
    float2 Position;       // terrain-space UV [0..1]
    float Radius;          // UV-space radius
    float Strength;        // height multiplier
    float Falloff;         // falloff exponent
    float Rotation;        // degrees
    float2 _pad;
};

float Blend(float prev, float value, uint mode, float opacity)
{
    value *= opacity;
    switch (mode)
    {
        case 0: return value;                          // Set
        case 1: return prev + value;                   // Add
        case 2: return max(prev, value);               // Max
        case 3: return lerp(prev, value, opacity);     // Lerp
        default: return value;
    }
}

// ── CS_Clear: Zero the output heightmap ──
[numthreads(8, 8, 1)]
void CS_Clear(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;
    Output[dtid.xy] = 0;
}

// ── CS_ImportLayer: Copy/blend a full heightmap texture ──
[numthreads(8, 8, 1)]
void CS_ImportLayer(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float> Source = ResourceDescriptorHeap[SourceIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float src = Source.SampleLevel(sampLinear, uv, 0);

    float prev = Output[dtid.xy];
    Output[dtid.xy] = Blend(prev, src, BlendMode, Opacity);
}

// ── CS_StampGroup: Apply all stamps in a group (batched per brush texture) ──
[numthreads(8, 8, 1)]
void CS_StampGroup(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float> Brush = ResourceDescriptorHeap[SourceIdx];
    StructuredBuffer<StampData> Stamps = ResourceDescriptorHeap[StampBufIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float prev = Output[dtid.xy];
    float accumulated = 0;

    for (uint i = 0; i < StampCount; i++)
    {
        StampData s = Stamps[i];
        float2 delta = uv - s.Position;
        float dist = length(delta);

        // Outside stamp radius — skip
        if (dist > s.Radius) continue;

        // Falloff: 1 at center, 0 at edge
        float t = saturate(1.0 - dist / s.Radius);
        float falloff = pow(t, s.Falloff);

        // Rotate delta into brush UV space
        float rad = s.Rotation * 3.14159265 / 180.0;
        float cosR = cos(rad);
        float sinR = sin(rad);
        float2 rotDelta = float2(
            delta.x * cosR + delta.y * sinR,
           -delta.x * sinR + delta.y * cosR
        );

        // Sample brush at local UV
        float2 brushUV = (rotDelta / s.Radius) * 0.5 + 0.5;
        float brushValue = Brush.SampleLevel(sampLinear, brushUV, 0);

        accumulated += brushValue * s.Strength * falloff;
    }

    if (accumulated != 0)
        Output[dtid.xy] = Blend(prev, accumulated, BlendMode, Opacity);
}
