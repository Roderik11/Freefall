// Universal effect for fire, smoke and glowing vapour as a volume: the ball a SphereMesh (radius 0.5) covers, stretched
// by its entity's scale, marched through per pixel. What is in it is billowing noise (the sky's 3D cloud noise, or
// gradient noise without one) carved into a ball or a tongue of flame (Taper), flowing through it (NoiseScroll) and
// bent by a slower noise (Churn). Its heart is hot and its outside cooler (Temperature, Cooling): the light it gives
// is the colour of a body that hot, falling steeply as it cools, so a flame is white-yellow inside, orange, then dark
// red where it thins, and its cool parts are smoke that hides what is behind it (Smoke, SmokeColor, lit by the sky
// and the lights). Every look is a material setting; per object (struct Instance) its colour, brightness, heat, fill
// and opacity, which a script animates: an explosion that cools into smoke, a flame that burns away.
//
// Drawn in the Forward pass onto the HDR composite with premultiplied alpha (what it gives plus what it lets through),
// its back faces always (DepthFunc=Always), so a camera inside still sees it; the march ends at the scene's depth
// (SceneDepthIdx), so it stands on the ground and wraps round what is in it. No depth written, not pickable, no
// shadows.

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
    uint SceneDepthIdx;             // 9: the scene's linear view depth (R32, 0 = sky), Forward pass only; 0 = none
    uint _reserved10;
    uint _reserved11;
    uint _reserved12;
    uint _reserved13;
    uint _reserved14;
    uint _reserved15;
    uint DebugMode;                 // 16
    uint IndexedDraw;               // 17: 1 = an indexed draw, the vertex index is SV_VertexID (common.fx MeshVertexIndex)
};

#include "common.fx"
#include "sky_common.fx"
#include "oit.hlsli"

SamplerState SamplerWrap : register(s0);                    // anisotropic, wrap
SamplerComparisonState ShadowCompare : register(s3);        // less-equal (ParticlePointLight)

// @RenderState(RenderTargets=1, DepthWrite=false, DepthFunc=Always, Blend=Premultiplied, CullMode=Front, RenderTargetFormat=R16G16B16A16_Float)

// The material's settings (Material.SettingsStride: 256 bytes, packed tightly in declaration order, every member with
// its "//=" default), loaded per draw by the instance's MaterialID
struct Settings
{
    float  Density;          //= 25 [0, 500]     -- How thick it is at its thickest: extinction per metre (25 = a hand's depth of it hides most of what is behind)
    float  Coverage;         //= 0.55 [0, 1]     -- How much of the ball the noise leaves standing: low = wisps and tongues, high = a full ball
    float  Softness;         //= 0.25 [0.01, 1]  -- How gradually its outside thins: low = a crisp billowing surface, high = haze
    float  Taper;            //= 0 [0, 1]        -- Narrower up its local Y: 0 = a ball, 1 = a tongue of flame coming to a point at the top
    float2 NoiseScale;       //= 2, 2            -- Billows across the ball (x) and up it (y): more up than across stretches them into licks
    float3 NoiseScroll;      //= 0, 1.5, 0       -- Billows a second the noise flows through it, in its own space (+y = up: rising fire and smoke)
    float  Detail;           //= 0.35 [0, 1]     -- How much a finer noise eats into its outside (curls, wisps)
    float  DetailScale;      //= 3.5             -- That noise's cells per billow
    float  Churn;            //= 0.5             -- How far a slow noise bends the billows (licking, rolling), in billows
    float  Temperature;      //= 1900 [0, 8000]  -- Kelvin at its heart: 1100 dull red, 1500 red-orange, 1900 orange, 2600 yellow, 5000+ white
    float  Cooling;          //= 0.5 [0, 1]      -- How much colder it is towards its outside and top than at its heart (a flame reddening as it thins)
    float  Emission;         //= 15000 [0, 500000] -- Nits of light at the heart's temperature; cooler parts give far less (as a hot body does)
    float  DayIntensity;     //= 1 [0, 20]       -- By day, Emission times this (blended by the sun's height)
    float3 GlowColor;        //= 1, 1, 1         -- Multiplies the fire's light: white = real fire, a colour for magic flames
    float  Blackbody;        //= 1 [0, 1]        -- 1 = the colour of a body that hot; 0 = the Glow Color alone, only its brightness following the heat
    float3 SmokeColor;       //= 0.04, 0.035, 0.03 -- What its cool parts are made of, lit by the sky and the lights (soot dark, steam pale)
    float  Smoke;            //= 0.6 [0, 1]      -- How much its cool parts hide what is behind: 0 = pure light (hides nothing), 1 = thick smoke
    float  Steps;            //= 24 [4, 64]      -- Samples along each ray through it: more is smoother and costs more
};

static Settings S;

Settings LoadSettings(uint materialId)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    ByteAddressBuffer all = ResourceDescriptorHeap[NonUniformResourceIndex(materials[materialId].SettingsBufferIdx)];
    return all.Load<Settings>(materialId * 256u);
}

// Per object (the renderer's INSTANCE rows, PersistentRenderer.SetInstance): 128 bytes a slot, packed tightly
struct Instance
{
    float3 Color;            //= 1, 1, 1         -- Multiplies the Glow Color, for this object alone
    float  Intensity;        //= 1 [0, 4]        -- Multiplies the light it gives (fade it, flash it)
    float  Heat;             //= 1 [0, 2]        -- Multiplies the temperature: an explosion cooling into smoke
    float  Fill;             //= 1 [0, 2]        -- Multiplies the coverage: billowing in, burning away
    float  Opacity;          //= 1 [0, 1]        -- Multiplies the density: the whole thinning away
};

static Instance I;

Instance LoadInstance(uint slot)
{
    ByteAddressBuffer all = ResourceDescriptorHeap[InstanceDataIdx];
    return all.Load<Instance>(slot * 128u);
}

// How much it is day, 0..1, from the sun's height (as vfx.fx has it)
float DayFactor()
{
    float e = AtmSunDirection().y;
    float day = pow(saturate(e / 0.8), 0.7);
    float sunset = (e < 0.15 && e > -0.2) ? max(0.0, 1.0 - abs((e + 0.025) / 0.175)) : 0.0;
    float night = saturate((-e - 0.15) / 0.3);
    float total = day + sunset + night;
    return total > 0.0 ? day / total : 0.0;
}

// ── Heat to light ──

// The colour of a body at t kelvin, in linear sRGB with its largest channel 1: the Planckian locus in CIE 1960 UCS
// (Kim et al. 2002, 1000–15000 K), to xyY, XYZ and sRGB
float3 BlackbodyColor(float t)
{
    t = clamp(t, 1000.0, 15000.0);
    float t2 = t * t;
    float u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t2) / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t2);
    float v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t2) / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t2);
    float d = 2.0 * u - 8.0 * v + 4.0;
    float x = 3.0 * u / d, y = 2.0 * v / d;
    float3 XYZ = float3(x / y, 1.0, (1.0 - x - y) / y);
    float3 rgb = float3( 3.2404542 * XYZ.x - 1.5371385 * XYZ.y - 0.4985314 * XYZ.z,
                        -0.9692660 * XYZ.x + 1.8760108 * XYZ.y + 0.0415560 * XYZ.z,
                         0.0556434 * XYZ.x - 0.2040259 * XYZ.y + 1.0572252 * XYZ.z);
    rgb = max(rgb, 0.0);
    return rgb / max(max(rgb.r, rgb.g), max(rgb.b, 1e-4));
}

// How bright a body at t is against one at its heart's temperature: the visible light of a hot body falls as Wien's law
// says, e^(−c/λT), here eased (4 at half the temperature: 1.8 %) so that the edges glow a dull red before they go dark
float HeatBrightness(float t, float heart)
{
    return exp(-4.0 * (heart / max(t, 1.0) - 1.0));
}

// ── Noise ──

uint VolHash3(int3 cell)
{
    uint3 q = uint3(cell) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (q.x ^ q.y ^ q.z) * 1597334673u;
}

float3 VolGradient3(int3 cell)
{
    uint n = VolHash3(cell);
    float z = float(n & 0xFFFFu) * (2.0 / 65535.0) - 1.0;
    float a = float(n >> 16) * (6.2831853 / 65536.0);
    float r = sqrt(saturate(1.0 - z * z));
    return float3(r * cos(a), r * sin(a), z);
}

// Gradient noise 0..1 (without the sky's noise texture)
float VolNoise3(float3 p)
{
    float3 i = floor(p);
    float3 f = p - i;
    float3 s = f * f * (3.0 - 2.0 * f);
    int3 c = int3(i);
    float x00 = lerp(dot(VolGradient3(c), f), dot(VolGradient3(c + int3(1, 0, 0)), f - float3(1, 0, 0)), s.x);
    float x10 = lerp(dot(VolGradient3(c + int3(0, 1, 0)), f - float3(0, 1, 0)), dot(VolGradient3(c + int3(1, 1, 0)), f - float3(1, 1, 0)), s.x);
    float x01 = lerp(dot(VolGradient3(c + int3(0, 0, 1)), f - float3(0, 0, 1)), dot(VolGradient3(c + int3(1, 0, 1)), f - float3(1, 0, 1)), s.x);
    float x11 = lerp(dot(VolGradient3(c + int3(0, 1, 1)), f - float3(0, 1, 1)), dot(VolGradient3(c + int3(1, 1, 1)), f - float3(1, 1, 1)), s.x);
    return lerp(lerp(x00, x10, s.y), lerp(x01, x11, s.y), s.z) * 0.8 + 0.5;
}

// The sky's cloud noise (SkyboxRenderer's LUT, 128³ tiling, 4 cells a repeat): r Perlin fBm, g and b Worley fBm at one
// and two times, a Perlin eroded by Worley. p is in cells.
float4 VolLut(float3 p)
{
    Texture3D<float4> lut = ResourceDescriptorHeap[CloudNoiseLUTIdx];
    return lut.SampleLevel(SamplerWrap, p * 0.25, 0.0);
}

// The billows at p (cells) and the finer noise that eats into them at q (cells): x billows, y detail, both 0..1
float2 VolNoise(float3 p, float3 q)
{
    if (CloudNoiseLUTIdx != 0u)
    {
        float4 big = VolLut(p);
        float4 fine = VolLut(q);
        return float2(lerp(big.r, big.a, 0.45), fine.b);
    }
    float billow = VolNoise3(p) * 0.67 + VolNoise3(p * 2.03 + 11.7) * 0.33;
    return float2(billow, VolNoise3(q));
}

// A slow noise that bends the billows (cells), centred on 0
float3 VolChurn(float3 p)
{
    if (CloudNoiseLUTIdx != 0u)
    {
        float3 a = VolLut(p * 0.37 + float3(0.0, 0.0, 0.5)).rgb;
        return (a - float3(0.5, 0.55, 0.55)) * 2.0;
    }
    return float3(VolNoise3(p * 0.37), VolNoise3(p * 0.37 + 23.1), VolNoise3(p * 0.37 - 7.9)) * 2.0 - 1.0;
}

// ── The ball ──

// What is at q (the ball's own space, radius 0.5): density 0..1 and heat 0..1 (1 = the heart's temperature)
float2 Field(float3 q, float fill, float3 seed)
{
    float up = saturate(q.y + 0.5);                          // 0 at the foot, 1 at the top
    float width = max(1.0 - S.Taper * up, 0.06);
    float3 shaped = float3(q.x / width, q.y, q.z / width);
    float body = 1.0 - length(shaped) * 2.17;                // 1 at the heart, 0 a little inside the mesh's outline

    float3 cells = float3(S.NoiseScale.x, S.NoiseScale.y, S.NoiseScale.x);
    float3 p = q * cells + seed - S.NoiseScroll * Time;
    if (S.Churn > 0.0)
        p += VolChurn(p + float3(0.0, Time * 0.2, 0.0)) * S.Churn;
    float3 fineAt = p * S.DetailScale - S.NoiseScroll * Time * (0.5 + 0.5 * S.Churn);
    float2 n = VolNoise(p, fineAt);

    // The billows raise and lower the ball's outline; the fine noise eats into what is left near it
    float coverage = saturate(S.Coverage * fill);
    float field = body + (n.x - 0.5) * 1.1 - (1.0 - coverage) * 0.9;
    field -= (1.0 - n.y) * S.Detail * 0.45 * saturate(1.0 - field * 2.0);
    float density = smoothstep(0.0, max(S.Softness, 0.01), field);

    // Hot inside, by how deep into the billows a point is: the visible outside is still warm, only its very edge cools,
    // and a tongue cools as it rises
    float heat = saturate(0.45 + field * 1.6 + (n.x - 0.5) * 0.4) * lerp(1.0, 1.0 - up * 0.55, S.Taper);
    return float2(density, heat);
}

// ── Vertex shader ──

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 ObjectPos : TEXCOORD0;                         // the back face's point in the ball's own space
    nointerpolation float3 ObjectEye : TEXCOORD1;          // the camera in the ball's own space
    float3 ToEye    : TEXCOORD2;                          // camera minus the back face's point, world
    float  Depth    : TEXCOORD3;                          // the back face's linear view depth
    nointerpolation uint MaterialId : TEXCOORD4;
    nointerpolation uint InstanceSlot : TEXCOORD5;
    nointerpolation float3 Centre : TEXCOORD6;            // the ball's centre, world
    nointerpolation float3 Seed : TEXCOORD7;              // where in the noise this ball is: its own, so two balls of one material do not boil alike
};

VSOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VSOutput output;

    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];

    uint idx = sortedIndices[InstanceBaseOffset + instanceID] & 0x7FFFFFFFu;
    InstanceDescriptor desc = descriptors[idx];
    output.MaterialId = desc.MaterialId;
    output.InstanceSlot = desc.CustomDataIdx;
    row_major matrix World = globalTransforms[desc.TransformSlot];
    float3x3 W3 = (float3x3)World;

    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    uint vertexID = MeshVertexIndex(part, primitiveVertexID, primitiveVertexID, IndexedDraw);
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    float3 pos = positions[vertexID];

    float3 worldPos = mul(float4(pos, 1.0), World).xyz;
    float3 origin = World[3].xyz;

    // The camera in the ball's own space: (world − origin) times the inverse of W3, its cofactors over its determinant
    float3 c0 = cross(W3[1], W3[2]), c1 = cross(W3[2], W3[0]), c2 = cross(W3[0], W3[1]);
    float det = dot(W3[0], c0);
    float3 rel = CamPos - origin;
    output.ObjectEye = abs(det) > 1e-12 ? float3(dot(rel, c0), dot(rel, c1), dot(rel, c2)) / det : float3(0, 0, -10);

    output.Position = mul(mul(float4(worldPos, 1.0), View), Projection);
    output.ObjectPos = pos;
    output.ToEye = CamPos - worldPos;
    output.Depth = output.Position.w;
    output.Centre = origin;
    uint h = desc.TransformSlot * 2654435761u;
    output.Seed = float3(h & 1023u, (h >> 10) & 1023u, (h >> 20) & 1023u) * 0.731;
    return output;
}

// ── Pixel shader ──

// The march: premultiplied light (rgb) and the share of the background it hides (a)
float4 March(VSOutput input)
{
    S = LoadSettings(input.MaterialId);
    I = LoadInstance(input.InstanceSlot);

    // The ray is the camera's through this pixel: s = 0 at the camera, 1 at the back face. An affine map keeps s the
    // same in the ball's space and in the world, so a length in metres is s times the back face's distance
    float3 o = input.ObjectEye;
    float3 d = input.ObjectPos - o;
    float a = dot(d, d), b = dot(o, d), c = dot(o, o) - 0.25;
    float disc = b * b - a * c;
    if (disc <= 0.0)
        return 0;
    float root = sqrt(disc);
    float s0 = max((-b - root) / a, 0.0);
    float s1 = (-b + root) / a;

    // It ends at what is behind it, or in it
    if (SceneDepthIdx != 0u && input.Depth > 1e-4)
    {
        Texture2D<float> sceneDepth = ResourceDescriptorHeap[SceneDepthIdx];
        float behind = sceneDepth.Load(int3(int2(input.Position.xy), 0));
        if (behind > 0.0)
            s1 = min(s1, behind / input.Depth);
    }
    if (s1 <= s0)
        return 0;

    float metres = length(input.ToEye);                      // metres per unit of s
    int steps = clamp((int)S.Steps, 4, 64);
    float ds = (s1 - s0) / steps;
    float stepMetres = ds * metres;

    // Started a blue-noise share of a step in, so the steps' banding becomes a fine grain
    float jitter = 0.5;
    if (BlueNoiseIdx != 0u)
    {
        Texture2D<float> blue = ResourceDescriptorHeap[BlueNoiseIdx];
        jitter = blue.Load(int3(int2(input.Position.xy) & 63, 0));
    }

    float day = lerp(1.0, S.DayIntensity, DayFactor());
    float heart = max(S.Temperature * I.Heat, 1.0);
    float3 tint = S.GlowColor * I.Color;
    float emission = S.Emission * I.Intensity * day * EmissiveExposure;
    float extinction = S.Density * I.Opacity;

    // The smoke is lit by the sky, the sun and moon, and the point lights near its centre (the fire's own among them),
    // taken once for the whole ball
    float3 relCentre = input.Centre - CamPos;
    float3 toEye = normalize(-relCentre);
    float3 smokeLight = ParticleLight(SamplerWrap, relCentre, toEye, 0.2)
                      + ParticlePointLight(relCentre, toEye, 0.2, 0.5, ShadowCompare);
    float3 smokeLit = S.SmokeColor * smokeLight;

    float3 light = 0;
    float seen = 1.0;           // what of the background still shows through
    float self = 1.0;           // what of its own light from here still reaches the eye through the fire in front
    float s = s0 + ds * jitter;
    [loop] for (int i = 0; i < steps; i++, s += ds)
    {
        float3 q = o + d * s;
        float2 f = Field(q, I.Fill, input.Seed);
        if (f.x > 0.001)
        {
            float sigma = extinction * f.x;
            float t = heart * (1.0 - S.Cooling * (1.0 - f.y));
            float bright = HeatBrightness(t, max(S.Temperature, 1.0));   // against the material's heat: a ball cooled by its Heat dims as it reddens
            float3 color = lerp(1.0.xxx, BlackbodyColor(t), S.Blackbody) * tint;

            // Through one step: the fire's own light (a thick fire shines at Emission, so the light per metre is the
            // extinction's), the smoke's (its cool share), and what both let through
            float smoke = S.Smoke * (1.0 - bright * 0.8);
            float through = exp(-sigma * stepMetres);
            float hidden = exp(-sigma * smoke * stepMetres);
            light += (color * (emission * bright) + smokeLit * smoke) * (1.0 - through) * self * seen;
            self *= through;
            seen *= hidden;
            if (self * seen < 0.003)
                break;
        }
    }

    float cover = 1.0 - seen;
    if (FogEnabled > 0 && AtmosphereIdx != 0)
    {
        float air = AtmosphereFog(input.Position.xy, input.Depth * s0, 1.0).a;
        light *= air;
        cover *= air;
    }
    return float4(light, cover);
}

float4 PS(VSOutput input) : SV_Target0
{
    return March(input);
}

// Order-independent transparency (EngineSettings.Transparency, oit.hlsli): what it hides at its depth, then its light
OITMomentTargets PS_OITMoments(VSOutput input)
{
    return OITMoments(March(input).a, input.Depth);
}

OITTargets PS_OIT(VSOutput input)
{
    return OITAccumulate(March(input), input.Depth, input.Position.xy);
}

technique11 VfxVolume
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
    pass OITMoments
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_OITMoments()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OITMoments, Blend=OIT, DepthWrite=false);
    }
    pass OIT
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_OIT()));
        SetRenderState(RenderTargets=2, RenderTargetFormat=OIT, Blend=OIT, DepthWrite=false);
    }
}
