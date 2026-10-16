// Universal effect for meshes that glow: beams, lasers, jets, flames, auras, spells. Unlit and additive,
// in the Forward pass onto the HDR composite, after opaque geometry and the sky (depth test on, depth
// write off, both faces). Everything is a material setting (the "//=" lines below, shown in the material
// inspector and saved with the .mat), so one effect serves every such look:
//
//   - a colour at an intensity, thick where the surface faces the eye and thinning towards its outline
//     (Body), with an optional rim (Rim, RimPower);
//   - noise that scrolls over the surface (NoiseScale, NoiseScroll, NoiseAmount, NoiseContrast), its
//     coordinates bent by a second noise (Warp...), or eating it away (Dissolve, DissolveEdge);
//   - the ends of the mesh fading (FadeEnds: share of V at the foot and the head);
//   - vertices pushed along their normal by noise (Push...), and kept wide enough to be seen from afar
//     (MinWidth, for meshes round their local Y axis: tubes, cones);
//   - a pulse travelling along the mesh from V = 0 to V = 1 (Pulse...): a band that swells the surface
//     and glows. A script starts it by setting PulseStart to Time.TotalTime (the shader's Time) on its
//     own instance of the material (Material.CreateInstance); the shader moves, grows and fades it, no work per frame.
//
// U runs round the mesh and V along it (Mesh.CreateTube: U 0..1 once round, V 0 at the foot). Noise is
// periodic in U, so a tube's seam does not show. Not pickable: it writes the composite only, so a beam
// that crosses the screen does not catch clicks meant for what is behind it.

cbuffer PushConstants : register(b3)
{
    uint ShadowMapIdx;              // 0 (set in the Forward pass; unused here)
    uint CompositeSRVIdx;           // 1 (set in the Forward pass; unused here)
    uint MeshPartId;                // 2: per draw (command signature)
    uint InstanceBaseOffset;        // 3: per draw (command signature)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint>, sorted draw order
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7
    uint GlobalTransformBufferIdx;  // 8
    uint _reserved9;
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    uint DebugMode;                 // 16
};

#include "common.fx"

// @RenderState(RenderTargets=1, DepthWrite=false, Blend=Additive, CullMode=None, RenderTargetFormat=R16G16B16A16_Float)

// The material's settings, one block per material, loaded by each instance's MaterialID (LoadSettings):
// draws are batched by effect, so a per-material cbuffer would only ever be the batch's first material's.
// Packed tightly in declaration order (the C# side writes them so, Material.WriteSetting): every member
// must carry its "//=" default, and the block must stay within Material.SettingsStride (256 bytes).
struct Settings
{
    float3 Color;            //= 1, 1, 1         -- Colour of the glow
    float  Intensity;        //= 1               -- Brightness (HDR: above 1 glows and blooms)
    float  Body;             //= 1               -- How it thins towards its outline: |N.V| to this power (0 = flat, higher = a thinner, softer body)
    float  Axial;            //= 0 [0, 1]        -- 1 for tubes and cones: thickness measured round the local Y axis, and brighter seen along it (more of it in the line of sight); 0 for spheres and blobs
    float  Rim;              //= 0               -- Extra glow at the outline
    float  RimPower;         //= 3               -- How tight that rim is
    float  MinWidth;         //= 0               -- Milliradians: never narrower on screen than this, so it stays seen from afar (meshes round their local Y axis)
    float2 FadeEnds;         //= 0, 0            -- Share of the length over which it fades in at the foot (x) and out at the head (y)
    float2 NoiseScale;       //= 4, 4            -- Noise cells round the mesh (U, rounded to whole cells) and along it (V)
    float2 NoiseScroll;      //= 0, 0            -- How fast the noise moves, in cells a second (positive V = towards the head)
    float  NoiseAmount;      //= 0 [0, 1]        -- 0 = no noise; 1 = the noise sets the brightness entirely
    float  NoiseContrast;    //= 1               -- Above 1 the noise has harder light and dark parts
    float2 WarpScale;        //= 2, 2            -- Cells of the second noise, which bends the first
    float2 WarpScroll;       //= 0, 0            -- Its speed, in cells a second
    float  Warp;             //= 0               -- How far it bends the first noise, in that noise's cells
    float  Dissolve;         //= 0 [0, 1]        -- 0..1: where the noise is below this the surface is gone
    float  DissolveEdge;     //= 0.05            -- Width of the glowing edge where it is eaten away
    float  Push;             //= 0               -- Metres: vertices pushed along their normal by noise
    float2 PushScale;        //= 3, 3            -- Cells of that noise round and along the mesh
    float2 PushScroll;       //= 0, 0            -- Its speed, in cells a second
    float3 PulseColor;       //= 1, 1, 1         -- Colour of the pulse
    float  PulseGlow;        //= 0               -- Brightness of the pulse, times Intensity (0 = no glow; it can still swell the surface)
    float  PulseSwell;       //= 0               -- Metres the pulse pushes the surface outwards at its middle
    float  PulseLength;      //= 0.05            -- Share of the length the pulse covers
    float  PulseStart;       //= -100000         -- Time (Time.TotalTime) the latest pulse set off; set by scripts
    float  PulseDuration;    //= 4               -- Seconds from the foot to the head
    float  PulseEase;        //= 1               -- How it travels: its place along the mesh is (t / duration) to this power (1 = even, higher = slow start)
    float  PulseGrow;        //= 0.15 [0, 1]     -- Share of the duration over which it fades in and swells
    float  PulseHold;        //= 0.6 [0, 1]      -- Share of the way it keeps full strength before it fades out by the head
    float  PulsePreview;     //= -1 [-1, 1]      -- 0..1: hold the pulse at this point of its run, to look at it while tuning; -1 = off
};

// This draw's settings, loaded at the start of VS and PS
static Settings S;

Settings LoadSettings(uint materialId)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    ByteAddressBuffer all = ResourceDescriptorHeap[materials[materialId].SettingsBufferIdx];
    return all.Load<Settings>(materialId * 256u);
}

// ── Noise ──

float VfxHash(float2 p)
{
    p = frac(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return frac(p.x * p.y);
}

// Value noise, periodic in x with the given period (whole cells), so it wraps round a tube's seam
float VfxNoise(float2 p, float period)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 s = f * f * (3.0 - 2.0 * f);
    float x0 = i.x - period * floor(i.x / period);
    float x1 = x0 + 1.0;
    x1 -= period * floor(x1 / period);
    float a = VfxHash(float2(x0, i.y));
    float b = VfxHash(float2(x1, i.y));
    float c = VfxHash(float2(x0, i.y + 1.0));
    float d = VfxHash(float2(x1, i.y + 1.0));
    return lerp(lerp(a, b, s.x), lerp(c, d, s.x), s.y);
}

// Three octaves, each twice as fine; stays periodic since the period doubles with the frequency
float VfxFbm(float2 p, float period)
{
    float sum = 0.0, weight = 0.5, total = 0.0;
    [unroll] for (int o = 0; o < 3; o++)
    {
        sum += weight * VfxNoise(p, period);
        total += weight;
        p *= 2.0;
        period *= 2.0;
        weight *= 0.5;
    }
    return sum / total;
}

// Noise at a point of the surface: cells round (whole, so it wraps) and along, moving at scroll cells a second
float SurfaceNoise(float2 uv, float2 scale, float2 scroll)
{
    float cellsU = max(round(scale.x), 1.0);
    float2 p = uv * float2(cellsU, scale.y) - scroll * Time;
    return VfxFbm(p, cellsU);
}

// ── The pulse ──

// Where the pulse is along the mesh (V) and how strong it is now (0 = none)
float PulseAt(out float strength)
{
    strength = 0.0;
    float t = S.PulsePreview >= 0.0 ? S.PulsePreview : (Time - S.PulseStart) / max(S.PulseDuration, 1e-3);
    if (t < 0.0 || t > 1.0) return -10.0;
    float along = pow(t, max(S.PulseEase, 0.01));
    strength = smoothstep(0.0, max(S.PulseGrow, 1e-3), t) * (1.0 - smoothstep(S.PulseHold, 1.0, along));
    return along;
}

// How much of the pulse a point at V gets: a soft band, its strength scaling it
float PulseBand(float v, float along, float strength)
{
    float x = (v - along) / max(S.PulseLength * 0.5, 1e-5);
    return strength * exp(-2.0 * x * x);
}

// ── Vertex shader ──

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 Normal   : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float3 ToEye    : TEXCOORD1;    // camera minus surface, world space
    float  Depth    : TEXCOORD2;    // linear view depth (fog)
    float3 Axis     : TEXCOORD3;    // the mesh's local Y in world space (Axial)
    nointerpolation uint MaterialId : TEXCOORD4;   // whose settings the pixel shader loads
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    uint idx = sortedIndices[InstanceBaseOffset + instanceID] & 0x7FFFFFFFu;
    InstanceDescriptor desc = descriptors[idx];
    S = LoadSettings(desc.MaterialId);
    output.MaterialId = desc.MaterialId;
    row_major matrix World = globalTransforms[desc.TransformSlot];
    float3x3 W3 = (float3x3)World;

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<float3> normals = ResourceDescriptorHeap[part.NormBufferIdx];
    StructuredBuffer<float2> uvs = ResourceDescriptorHeap[part.UVBufferIdx];
    float3 pos = positions[vertexID];
    float3 norm = normals[vertexID];
    float2 uv = uvs[vertexID];

    float3 worldPos = mul(float4(pos, 1.0), World).xyz;
    float3 N = normalize(mul(norm, W3));

    // Pushed outwards: by noise, and by the pulse where it passes
    float push = 0.0;
    if (S.Push != 0.0)
        push += S.Push * (SurfaceNoise(uv, S.PushScale, S.PushScroll) * 2.0 - 1.0);
    if (S.PulseSwell != 0.0)
    {
        float strength;
        float along = PulseAt(strength);
        push += S.PulseSwell * PulseBand(uv.y, along, strength);
    }

    // Never thinner on screen than MinWidth: widened where the camera is far (its distance from the
    // mesh's local Y axis is its radius here, in world units)
    if (S.MinWidth > 0.0)
    {
        float radius = length(mul(float3(pos.x, 0.0, pos.z), W3));
        float least = distance(worldPos, CamPos) * S.MinWidth * 0.0005;
        push += max(least - radius - push, 0.0);
    }

    worldPos += N * push;

    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.Normal = N;
    output.TexCoord = uv;
    output.ToEye = CamPos - worldPos;
    output.Depth = output.Position.w;
    output.Axis = mul(float3(0.0, 1.0, 0.0), W3);
    return output;
}

// ── Pixel shader ──

float4 PS(VSOutput input) : SV_Target0
{
    S = LoadSettings(input.MaterialId);
    float2 uv = input.TexCoord;
    float3 N = normalize(input.Normal);
    float3 V = normalize(input.ToEye);
    float facing = abs(dot(N, V));

    // Round an axis (tubes, cones): how central the point is in the tube's outline, whatever the angle it is
    // seen at, and more light the more nearly one looks along it (seen end-on, |N.V| alone would put it out)
    float gain = 1.0;
    if (S.Axial > 0.0)
    {
        float3 A = normalize(input.Axis);
        float3 across = V - A * dot(V, A);
        float side = length(across);
        float central = side > 1e-3 ? abs(dot(N, across / side)) : 1.0;
        facing = lerp(facing, central, S.Axial);
        gain = lerp(1.0, 1.0 / max(side, 0.3), S.Axial);
    }

    // Thick where it faces the eye, thinning towards the outline; the rim adds at the outline
    float body = S.Body > 0.0 ? pow(facing, S.Body) : 1.0;
    float rim = S.Rim > 0.0 ? S.Rim * pow(1.0 - facing, S.RimPower) : 0.0;

    // Noise, its coordinates bent by the warp noise
    float noise = 1.0;
    if (S.NoiseAmount > 0.0 || S.Dissolve > 0.0)
    {
        float2 p = uv;
        if (S.Warp != 0.0)
        {
            float wu = SurfaceNoise(uv, S.WarpScale, S.WarpScroll);
            float wv = SurfaceNoise(uv + 0.37, S.WarpScale, S.WarpScroll);
            float cellsU = max(round(S.NoiseScale.x), 1.0);
            p += (float2(wu, wv) * 2.0 - 1.0) * S.Warp / max(float2(cellsU, S.NoiseScale.y), 1e-3);
        }
        noise = SurfaceNoise(p, S.NoiseScale, S.NoiseScroll);
        noise = saturate((noise - 0.5) * S.NoiseContrast + 0.5);
    }
    float shade = lerp(1.0, noise, S.NoiseAmount);

    // Eaten away where the noise is below Dissolve, with a glowing edge
    float edge = 0.0;
    if (S.Dissolve > 0.0)
    {
        float keep = smoothstep(S.Dissolve, S.Dissolve + max(S.DissolveEdge, 1e-4), noise);
        edge = (1.0 - keep) * step(S.Dissolve - max(S.DissolveEdge, 1e-4), noise);
        shade *= keep;
    }

    // The ends fade in and out
    float ends = 1.0;
    if (S.FadeEnds.x > 0.0) ends *= smoothstep(0.0, S.FadeEnds.x, uv.y);
    if (S.FadeEnds.y > 0.0) ends *= 1.0 - smoothstep(1.0 - S.FadeEnds.y, 1.0, uv.y);

    float3 light = S.Color * S.Intensity * gain * ((body * shade + rim) * ends + edge * 2.0);

    // The pulse glows where it passes, with the same noise moving through it
    if (S.PulseGlow > 0.0)
    {
        float strength;
        float along = PulseAt(strength);
        float band = PulseBand(uv.y, along, strength);
        light += S.PulseColor * S.PulseGlow * S.Intensity * gain * band * lerp(1.0, noise, S.NoiseAmount) * body;   // thins to nothing at the outline, like the body
    }

    // Light seen through fog fades with it
    if (FogEnabled > 0)
        light *= exp(-pow(input.Depth * FogDensity, 2.0));

    return float4(light, 1.0);
}

technique11 Vfx
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
