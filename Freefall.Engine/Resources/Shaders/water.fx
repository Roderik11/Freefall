// Lakes and rivers (WaterBody component): a flat or gently sloping mesh generated from a spline.
// No FFT displacement — ripples are normals only, borrowed from the ocean simulation's finest bands.
// All lighting comes from water_common.fx, shared with ocean.fx.

cbuffer PushConstants : register(b3)
{
    // Slots 0-1: set by the forward pass for every forward batch
    uint ShadowMapIdx;              // 0: Shadow cascade array SRV
    uint _reserved1;                // 1: CompositeSnapshot SRV (taken from WaterData instead)
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
    // Slots 9-15: Custom / Reserved
    uint WaterDataIdx;              // 9: Per-instance WaterData buffer
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    // Slot 16: Debug
    uint DebugMode;                 // 16: Debug visualization mode
    uint _reserved17;
    uint _reserved18;
    uint _reserved19;
    // Slots 20-21: Shadow pass
    uint ExpansionBufferIdx;        // 20: SRV: expansion buffer
    uint CascadeBufferSRVIdx;       // 21: SRV: StructuredBuffer<CascadeData>
};

#include "common.fx"
#include "sky_common.fx"
#include "water_common.fx"
// @RenderState(RenderTargets=1, CullMode=None)

// Must match WaterBody.WaterData (C#)
struct WaterData
{
    float Time;
    float FlowSpeed;            // m/s along the spline (rivers)
    float RippleStrength;
    float RippleSize;
    float3 WaterColor;          // deep water
    float Visibility;           // meters of water at which the bed is mostly gone
    float3 ShallowColor;
    float RefractionStrength;
    float3 SunDirection;
    float SunIntensity;
    float3 SunColor;
    float EdgeFoam;
    float3 CloudColor;          // SkyboxRenderer.CurrentCloudColor
    float RapidsFoam;
    uint SlopeSRV;              // ocean FFT slope maps (Texture2DArray), 0 = flat water
    uint NoiseSRV;              // ocean noise texture (Perlin + Worley)
    uint DepthGBufferSRV;
    uint CompositeSRV;
    float2 InvViewportSize;
    float TileScaleA;           // 1 / tile size (m) of the two ripple bands
    float TileScaleB;
    uint BandA;
    uint BandB;
    float Flowing;              // 1 = river (open spline), 0 = lake
    float _pad0;
};

SamplerState WaterSampler : register(s0);

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;
    float2 UV : TEXCOORD1;             // rivers: x = meters along the spline, y = meters across
    float Depth : TEXCOORD2;
    nointerpolation uint InstanceIdx : TEXCOORD3;
    float3 SkyAmbient : TEXCOORD4;     // hemisphere sky ambient (same as composition), constant per frame
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];

    uint dataPos = InstanceBaseOffset + instanceID;
    uint idx = sortedIndices[dataPos] & 0x7FFFFFFFu;

    InstanceDescriptor desc = descriptors[idx];
    row_major matrix World = globalTransforms[desc.TransformSlot];

    float3 worldPos = mul(float4(positions[vertexID], 1.0f), World).xyz;

    output.WorldPos = worldPos;
    output.UV = uvs[vertexID];
    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.InstanceIdx = idx;
    output.SkyAmbient = GetSkyColor(float3(0, 1, 0), FogSunDirection) * 0.45 * AmbientScale;
    return output;
}

// World-space (xz) gradient of a value from its screen-space derivatives
float2 WorldGradient(float2 dpx, float2 dpy, float dvx, float dvy)
{
    float det = dpx.x * dpy.y - dpx.y * dpy.x;
    if (abs(det) < 1e-10) return 0;
    return float2(dvx * dpy.y - dvy * dpx.y, dvy * dpx.x - dvx * dpy.x) / det;
}

struct PSOutput
{
    float4 Color : SV_Target0;
};

PSOutput PS(VSOutput input)
{
    PSOutput output;

    StructuredBuffer<WaterData> waterData = ResourceDescriptorHeap[WaterDataIdx];
    WaterData water = waterData[input.InstanceIdx];

    float3 camPos = ViewInverse[3].xyz;
    float3 worldPos = input.WorldPos;
    float3 V = normalize(camPos - worldPos);
    float dist = distance(worldPos, camPos);
    float t = water.Time;
    bool flowing = water.Flowing > 0.5;

    // ── Flow frame (rivers) ──
    // The strip's u coordinate counts meters along the spline, so its gradient is the current's
    // direction. The surface's own gradient tells how steep this stretch is (rapids).
    float2 dpx = ddx(worldPos.xz), dpy = ddy(worldPos.xz);
    float2 gradU = WorldGradient(dpx, dpy, ddx(input.UV.x), ddy(input.UV.x));
    float2 flowDir = dot(gradU, gradU) > 1e-8 ? normalize(gradU) : float2(1, 0);
    float2 acrossDir = float2(-flowDir.y, flowDir.x);
    float steep = flowing ? saturate(length(WorldGradient(dpx, dpy, ddx(worldPos.y), ddy(worldPos.y)))) : 0.0;

    // Pattern space: lakes sample in world space; rivers in strip space (u along the flow, v across),
    // scrolled downstream. Strip space bends with the river, so the scroll never shears.
    float2 patternPos = flowing ? float2(input.UV.x - water.FlowSpeed * t, input.UV.y) : worldPos.xz;

    // ── Ripple normal ──
    float2 slope = 0;
    if (water.SlopeSRV != 0)
    {
        Texture2DArray<float2> slopeTex = ResourceDescriptorHeap[water.SlopeSRV];
        float2 p = patternPos / max(0.05, water.RippleSize);
        slope = slopeTex.Sample(WaterSampler, float3(p * water.TileScaleA, water.BandA)).xy
              + slopeTex.Sample(WaterSampler, float3(p * water.TileScaleB + 0.37, water.BandB)).xy;
        slope *= water.RippleStrength * (1.0 + steep * 4.0);

        // Strip space → world
        if (flowing)
            slope = flowDir * slope.x + acrossDir * slope.y;
    }
    float3 N = normalize(float3(-slope.x, 1.0, -slope.y));
    float NdotV = saturate(dot(N, V));

    // ── Sun ──
    float3 L = normalize(-water.SunDirection);
    float NdotL = saturate(dot(N, L));
    float3 sunRadiance = water.SunColor * water.SunIntensity;
    sunRadiance *= GetCloudShadow(WaterSampler, worldPos - camPos, L);
    sunRadiance *= WaterSunShadow(ShadowMapIdx, worldPos - camPos, input.Position.xy);
    float3 skyAmbient = input.SkyAmbient;

    // ── Deep water color (same model as the ocean, without the wave-height term) ──
    float3 scatter = (0.1 * NdotV * NdotV + 0.08 * NdotL) * water.WaterColor * sunRadiance
                   + water.WaterColor * skyAmbient * 0.15;

    // ── Shared water shading ──
    float F = WaterFresnel(NdotV);
    float3 reflectColor = WaterSkyReflection(WaterSampler, V, N, worldPos - camPos, dist, water.CloudColor);
    float3 specular = WaterSunSpecular(N, V, L, sunRadiance, F, dist);

    WaterColumn column = GetWaterColumn(WaterSampler, water.DepthGBufferSRV, water.CompositeSRV,
        input.Position.xy * water.InvViewportSize, input.Depth, dist, abs(camPos.y - worldPos.y), N,
        water.RefractionStrength, scatter, water.ShallowColor, water.Visibility, 1.0);

    // Light scattered back from a bright shallow bed
    float shallow = exp(-max(column.bufDepth, 0.0) / max(0.01, water.Visibility)) * saturate(column.pathLen);
    float3 body = column.body
                + water.ShallowColor * (sunRadiance * saturate(L.y) + skyAmbient) * 0.06 * shallow;

    float3 color = (1.0 - F) * body + (specular + F * reflectColor) * column.edgeFade;
    color = max(0.0, color);

    // ── Foam: coverage eroded by a cellular texture (as on the ocean) ──
    float coverage = 0.0;

    // Along the banks and around anything standing in the water
    coverage += (1.0 - smoothstep(0.0, 0.12, column.bufDepth)) * water.EdgeFoam;

    // White water where the surface runs steeply downhill
    // (kept below full coverage so the cell texture always breaks it up into streaks of dark water)
    coverage += smoothstep(0.04, 0.3, steep) * 0.68 * water.RapidsFoam;
    coverage = saturate(coverage);

    float foamTex = 0.5;
    if (water.NoiseSRV != 0 && coverage > 0.001)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[water.NoiseSRV];

        // Foam rides the current, faster where it is steep (bounded extra speed keeps it from shearing)
        float2 foamPos = flowing
            ? float2(input.UV.x - water.FlowSpeed * 1.6 * t, input.UV.y)
            : worldPos.xz + t * float2(0.01, 0.006);
        float cells1 = noiseTex.SampleLevel(WaterSampler, foamPos * 0.23, 0).g;
        float cells2 = noiseTex.SampleLevel(WaterSampler, foamPos * 0.83 + 0.5, 0).g;
        foamTex = saturate((cells1 * 0.6 + cells2 * 0.4 - 0.15) / 0.6);
        foamTex = lerp(foamTex, 0.5, saturate(dist / 150.0));   // no mips on the noise texture
    }

    float foam = saturate((coverage * 1.25 - (1.0 - foamTex)) * 3.0);
    foam *= smoothstep(0.0, 0.025, column.pathLen);     // never end on the hard edge at the bank
    float3 foamLit = WaterFoamLit(float3(0.80, 0.80, 0.78) * lerp(0.7, 1.0, foamTex), NdotL, sunRadiance, skyAmbient);
    color = lerp(color, foamLit, foam * 0.9);

    // ── Aerial perspective ──
    if (FogEnabled > 0)
        color = ApplyAerialPerspective(color, worldPos, camPos, FogSunDirection);

    // HDR output — the finalize pass tonemaps the whole Composite
    output.Color = float4(color, 1.0);
    return output;
}

technique11 GBuffer
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
