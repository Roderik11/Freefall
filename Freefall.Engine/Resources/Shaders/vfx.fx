// Universal effect for meshes that glow: beams, lasers, jets, flames, auras, spells. Unlit and additive,
// in the Forward pass onto the HDR composite, after opaque geometry and the sky (depth test on, depth
// write off, both faces). Everything is a material setting (the "//=" lines below, shown in the material
// inspector and saved with the .mat), so one effect serves every such look:
//
//   - a colour at an intensity, thick where the surface faces the eye and thinning towards its outline
//     (Body), with an optional rim (Rim, RimPower);
//   - noise that scrolls over the surface (NoiseScale, NoiseScroll, NoiseAmount, NoiseContrast), as soft
//     clouds or thin strands (NoiseRidge), wound round the mesh (NoiseTwist), its coordinates bent by a second
//     noise (Warp...), or eating it away (Dissolve, DissolveEdge); its detail fades where its cells grow
//     too small on screen, so it does not sparkle from afar;
//   - the ends of the mesh fading (FadeEnds: share of V at the foot and the head);
//   - vertices pushed along their normal by noise (Push...), and kept wide enough to be seen from afar
//     (MinWidth).
//
// Something travelling (a pulse up a beam, a bolt) is a mesh of its own that a script moves, sized by its
// entity's scale: a sphere stretched into a capsule shades right (normals by the inverse transpose).
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
#include "sky_common.fx"

// @RenderState(RenderTargets=1, DepthWrite=false, Blend=Additive, CullMode=None, RenderTargetFormat=R16G16B16A16_Float)

// The material's settings, one block per material, loaded by each instance's MaterialID (LoadSettings):
// draws are batched by effect, so a per-material cbuffer would only ever be the batch's first material's.
// Packed tightly in declaration order (the C# side writes them so, Material.WriteSetting): every member
// must carry its "//=" default, and the block must stay within Material.SettingsStride (256 bytes).
struct Settings
{
    float3 Color;            //= 1, 1, 1         -- Colour of the glow
    float  Intensity;        //= 1               -- Brightness (HDR: above 1 glows and blooms)
    float  DayIntensity;     //= 1 [0, 20]       -- By day, Intensity times this: a glow that must stand out against a bright sky (night = 1; dusk in between, as the sky blends)
    float  Body;             //= 1               -- How it thins towards its outline: |N.V| to this power (0 = flat, higher = a thinner, softer body)
    float  Axial;            //= 0 [0, 1]        -- 1 for tubes and cones: thickness measured round the local Y axis, and brighter seen along it (more of it in the line of sight); 0 for spheres and blobs
    float  Rim;              //= 0               -- Extra glow at the outline
    float  RimPower;         //= 3               -- How tight that rim is
    float  MinWidth;         //= 0               -- Milliradians: never narrower on screen than this, so it stays seen from afar (its radius taken from the local Y axis if Axial, else from the centre)
    float2 FadeEnds;         //= 0, 0            -- Share of the length over which it fades in at the foot (x) and out at the head (y)
    float2 NoiseScale;       //= 4, 4            -- Noise cells round the mesh (U, rounded to whole cells) and along it (V)
    float2 NoiseScroll;      //= 0, 0            -- How fast the noise moves, in cells a second (positive V = towards the head)
    float  NoiseAmount;      //= 0 [0, 1]        -- 0 = no noise; 1 = the noise sets the brightness entirely
    float  NoiseContrast;    //= 1               -- Above 1 the noise has harder light and dark parts (with NoiseRidge: thinner strands)
    float  NoiseRidge;       //= 0 [0, 1]        -- 0 = soft clouds; 1 = thin bright strands where the noise crosses its middle (energy, plasma, lightning)
    float  NoiseTwist;       //= 0 [-1, 1]       -- Cells round the mesh the noise turns per cell along it: strands wind round a tube, and flow along their winding
    float2 WarpScale;       //= 2, 2            -- Cells of the second noise, which bends the first
    float2 WarpScroll;       //= 0, 0            -- Its speed, in cells a second
    float  Warp;             //= 0               -- How far it bends the first noise, in that noise's cells
    float  Dissolve;         //= 0 [0, 1]        -- 0..1: where the noise is below this the surface is gone
    float  DissolveEdge;     //= 0.05            -- Width of the glowing edge where it is eaten away
    float  Push;             //= 0               -- Metres: vertices pushed along their normal by noise
    float2 PushScale;        //= 3, 3            -- Cells of that noise round and along the mesh
    float2 PushScroll;       //= 0, 0            -- Its speed, in cells a second
};

// This draw's settings, loaded at the start of VS and PS
static Settings S;

Settings LoadSettings(uint materialId)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    ByteAddressBuffer all = ResourceDescriptorHeap[materials[materialId].SettingsBufferIdx];
    return all.Load<Settings>(materialId * 256u);
}

// What one object gives the effect on top of its material's settings: its renderer's values (the inspector's
// INSTANCE rows, PersistentRenderer.SetInstance from scripts), loaded by the draw's CustomDataIdx from the
// per-object data buffer (InstanceData: 128 bytes a slot). Packed tightly like Settings; every member carries
// its "//=" default, which objects that set nothing get.
struct Instance
{
    float3 Color;            //= 1, 1, 1         -- Multiplies the material's Color, for this object alone
    float  Intensity;        //= 1 [0, 4]        -- Multiplies the material's Intensity, for this object alone (fade it in and out, flash it)
};

static Instance I;

Instance LoadInstance(uint slot)
{
    ByteAddressBuffer all = ResourceDescriptorHeap[InstanceDataIdx];
    return all.Load<Instance>(slot * 128u);
}

// How much it is day, 0..1, from the sun's height (the old sky's day/sunset/night blend; DayIntensity goes with
// physical units in ATMOSPHERE_PLAN.md stage 3, when exposure makes a glow stand out at night by itself)
float DayFactor()
{
    float e = AtmSunDirection().y;
    float day = pow(saturate(e / 0.8), 0.7);
    float sunset = (e < 0.15 && e > -0.2) ? max(0.0, 1.0 - abs((e + 0.025) / 0.175)) : 0.0;
    float night = saturate((-e - 0.15) / 0.3);
    float total = day + sunset + night;
    return total > 0.0 ? day / total : 0.0;
}

// ── Noise ──

// A random direction for a lattice point. An integer hash: a float one (frac of a product) loses its
// precision as the scrolled coordinates grow, and goes blocky after a while.
float2 VfxGradient(float2 cell)
{
    uint2 q = uint2(int2(cell)) * uint2(1597334673u, 3812015801u);
    uint n = (q.x ^ q.y) * 1597334673u;
    float a = float(n) * (6.2831853 / 4294967296.0);
    return float2(cos(a), sin(a));
}

// Gradient noise (0..1, mostly near 0.5), periodic in x with the given period (whole cells), so it wraps round a
// tube's seam. Not value noise: that one's lattice shows as blocks and creases.
float VfxNoise(float2 p, float period)
{
    float2 i = floor(p);
    float2 f = p - i;
    float2 s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float x0 = i.x - period * floor(i.x / period);
    float x1 = x0 + 1.0;
    x1 -= period * floor(x1 / period);
    float a = dot(VfxGradient(float2(x0, i.y)), f);
    float b = dot(VfxGradient(float2(x1, i.y)), f - float2(1.0, 0.0));
    float c = dot(VfxGradient(float2(x0, i.y + 1.0)), f - float2(0.0, 1.0));
    float d = dot(VfxGradient(float2(x1, i.y + 1.0)), f - float2(1.0, 1.0));
    return lerp(lerp(a, b, s.x), lerp(c, d, s.x), s.y) * 0.7071 + 0.5;
}

// How far apart two cells of a noise are on screen fades its detail (FootprintOf): an octave whose cells
// shrink below a few pixels would only sparkle, so it gives way to its mean
float OctaveFade(float footprint)
{
    return smoothstep(0.25, 0.67, footprint);   // cells per pixel: full detail at 4 px a cell, none at 1.5
}

// Three octaves, each twice as fine and shifted so their lattices do not line up; stays periodic since the
// period doubles with the frequency (a shift keeps a periodic noise periodic). footprint is
// the first octave's cells per pixel (0 = no fading: the vertex shader); the finer two fade with it, the
// first is left to the caller (it fades the shaped noise, NoiseAt).
float VfxFbm(float2 p, float period, float footprint)
{
    float sum = 0.0, weight = 0.5, total = 0.0;
    [unroll] for (int o = 0; o < 3; o++)
    {
        float n = VfxNoise(p, period);
        if (o > 0) n = lerp(n, 0.5, OctaveFade(footprint));
        sum += weight * n;
        total += weight;
        p = p * 2.0 + float2(0.43, 0.71);
        period *= 2.0;
        weight *= 0.5;
        footprint *= 2.0;
    }
    return sum / total;
}

// Noise cells round the mesh: whole, so the noise wraps round a tube's seam
float CellsRound(float2 scale) { return max(round(scale.x), 1.0); }

// Noise at a point of the surface: cells round and along, wound round by twist (cells round per cell along),
// moving at scroll cells a second
float SurfaceNoise(float2 uv, float2 scale, float2 scroll, float twist, float footprint)
{
    float cellsU = CellsRound(scale);
    float2 p = uv * float2(cellsU, scale.y);
    p.x += p.y * twist;
    p -= scroll * Time;
    return VfxFbm(p, cellsU, footprint);
}

// Cells of a noise (scale, twist) per pixel at this pixel: the larger of round and along. U's derivative is taken
// both as is and shifted by half a turn, the smaller one counting, so the seam (U jumping from 1 to 0 between
// neighbouring pixels) does not read as a cell a pixel.
float FootprintOf(float2 uv, float2 scale, float twist)
{
    float du = min(fwidth(uv.x), fwidth(frac(uv.x + 0.5)));
    float dv = fwidth(uv.y) * scale.y;
    return max(du * CellsRound(scale) + dv * abs(twist), dv);
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
    nointerpolation uint InstanceSlot : TEXCOORD5; // and whose per-object values
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
    output.InstanceSlot = desc.CustomDataIdx;
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

    // The normal by the inverse transpose (the cofactors, the same up to scale): a sphere stretched into a
    // capsule keeps normals square to its surface, so its outline fades where it really is edge-on
    float3 N = normalize(norm.x * cross(W3[1], W3[2]) + norm.y * cross(W3[2], W3[0]) + norm.z * cross(W3[0], W3[1]));
    float3 axis = mul(float3(0.0, 1.0, 0.0), W3);

    // The radius here, in world units: the distance from the mesh's local Y axis (tubes, cones), else from its centre
    float radius = S.Axial > 0.0 ? length(mul(float3(pos.x, 0.0, pos.z), W3)) : length(mul(pos, W3));

    // Pushed outwards by noise
    float push = 0.0;
    if (S.Push != 0.0)
        push += S.Push * (SurfaceNoise(uv, S.PushScale, S.PushScroll, 0.0, 0.0) * 2.0 - 1.0);

    // Never thinner on screen than MinWidth: widened where the camera is far
    if (S.MinWidth > 0.0)
    {
        float least = distance(worldPos, CamPos) * S.MinWidth * 0.0005;
        push += max(least - radius - push, 0.0);
    }

    worldPos += N * push;

    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.Normal = N;
    output.TexCoord = uv;
    output.ToEye = CamPos - worldPos;
    output.Depth = output.Position.w;
    output.Axis = axis;
    return output;
}

// ── Pixel shader ──

float4 PS(VSOutput input) : SV_Target0
{
    S = LoadSettings(input.MaterialId);
    I = LoadInstance(input.InstanceSlot);
    float2 uv = input.TexCoord;
    float3 N = normalize(input.Normal);
    float3 V = normalize(input.ToEye);
    float facing = abs(dot(N, V));

    // Round an axis (tubes, cones): how central the point is in the tube's outline, whatever the angle it is
    // seen at, and more light the more nearly one looks along it (seen end-on, |N.V| alone would put it out).
    // On a straight wall that is |N.V| over the share of the view across the axis (|N.across|).
    float gain = 1.0;
    if (S.Axial > 0.0)
    {
        float3 A = normalize(input.Axis);
        float side = length(V - A * dot(V, A));
        float central = side > 1e-3 ? saturate(abs(dot(N, V)) / side) : 1.0;
        facing = lerp(facing, central, S.Axial);
        gain = lerp(1.0, 1.0 / max(side, 0.3), S.Axial);
    }

    // Thick where it faces the eye, thinning towards the outline; the rim adds at the outline
    float body = S.Body > 0.0 ? pow(facing, S.Body) : 1.0;
    float rim = S.Rim > 0.0 ? S.Rim * pow(1.0 - facing, S.RimPower) : 0.0;

    // Noise, its coordinates bent by the warp noise, shaped into strands (NoiseRidge), and giving way to its
    // mean where its cells are too small on screen to show: from afar it would only sparkle
    float noise = 1.0;
    if (S.NoiseAmount > 0.0 || S.Dissolve > 0.0)
    {
        float2 p = uv;
        if (S.Warp != 0.0)
        {
            // Its fine octaves fade with distance too: aliasing in the coordinates speckles what they bend
            float warpFootprint = FootprintOf(uv, S.WarpScale, 0.0);
            float wu = SurfaceNoise(uv, S.WarpScale, S.WarpScroll, 0.0, warpFootprint);
            float wv = SurfaceNoise(uv + 0.37, S.WarpScale, S.WarpScroll, 0.0, warpFootprint);
            p += (float2(wu, wv) * 2.0 - 1.0) * S.Warp / max(float2(CellsRound(S.NoiseScale), S.NoiseScale.y), 1e-3);
        }
        float footprint = FootprintOf(uv, S.NoiseScale, S.NoiseTwist);
        noise = SurfaceNoise(p, S.NoiseScale, S.NoiseScroll, S.NoiseTwist, footprint);
        float contrast = max(S.NoiseContrast, 0.0);
        noise = saturate((noise - 0.5) * contrast + 0.5);
        float mean = 0.5;
        if (S.NoiseRidge > 0.0)
        {
            // Bright where the noise crosses its middle: lines winding through it, thinner the higher the contrast
            float ridge = 1.0 - abs(noise * 2.0 - 1.0);
            noise = lerp(noise, ridge * ridge, S.NoiseRidge);
            mean = lerp(0.5, 0.65 / max(contrast, 1.0), S.NoiseRidge);   // roughly what the strands average to
        }
        noise = lerp(noise, mean, OctaveFade(footprint));
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

    float intensity = S.Intensity * lerp(1.0, S.DayIntensity, DayFactor()) * I.Intensity;
    float3 light = S.Color * I.Color * intensity * gain * ((body * shade + rim) * ends + edge * 2.0);

    // Light in the older units, pre-exposed like the lights (physical units: ATMOSPHERE_PLAN.md stage 3)
    light *= LegacyLightScale;

    // Additive, so it hides nothing: the air in front only dims it
    if (FogEnabled > 0 && AtmosphereIdx != 0)
        light *= AtmosphereAerialPerspective(LoadAtmosphere(), input.Position.xy / ViewportSize, input.Depth).a;

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
