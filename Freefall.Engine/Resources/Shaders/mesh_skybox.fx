cbuffer PushConstants : register(b3)
{
    // Slots 0-1: Reserved for light/composition passes
    uint _reserved0;
    uint _reserved1;
    // Slots 2-3: PER-DRAW (command signature writes these)
    uint MeshPartId;                // 2: Index into MeshRegistry
    uint InstanceBaseOffset;        // 3: Base offset for instance ID (per-command)
    // Slots 4-8: PER-BATCH (set before ExecuteIndirect)
    uint DescriptorBufIdx;          // 4: StructuredBuffer<InstanceDescriptor>
    uint SortedIndicesIdx;          // 5: StructuredBuffer<uint> - sorted draw order indices
    uint MeshRegistryIdx;           // 6: StructuredBuffer<MeshPartEntry>
    uint MaterialsIdx;              // 7: Index to materials buffer
    uint GlobalTransformBufferIdx;  // 8: Index to global TransformBuffer
};

#include "common.fx"
// @RenderState(RenderTargets=4, DepthWrite=false, DepthFunc=GreaterEqual, CullMode=None)

// SceneConstants (b0) from common.fx: Time, View, Projection, ViewProjection, ViewInverse

cbuffer ObjectConstants : register(b1)
{
    float4x4 World;
    float3 SunDirection; // direction to sun
    float TimeOfDay; // 0-24
    float SunIntensity;
    float StarDensity;
    float StarBrightness;
    float CloudBrightness;       // overall cloud brightness (0-3)
    float3 CloudShadowColor;     // color of cloud shade side
    float CloudNightBrightness;  // cloud brightness at night (0-3)
    float3 CloudSunlitColor;     // color of sun-facing cloud tops
    float _cloudPad0;
    float3 CloudSunsetTintColor; // warm tint mixed into clouds at sunset/sunrise
    float _cloudPad1;
    float3 CloudNightColor;      // moonlit cloud color at night
    float _cloudPad2;
    // Moons (SkyboxRenderer.Moons, at most SkyboxRenderer.MaxMoons)
    float4 MoonDirSize[4];       // xyz: direction to the moon, w: angular radius (radians)
    float4 MoonColor[4];         // rgb: tint x brightness of the lit side
    uint4 MoonTextures;          // bindless surface texture per moon, 0 = a plain disc
    uint MoonCount;
    // Star sphere (SkyboxRenderer): turns with the time of day round the sun path's pole
    row_major float4x4 StarRotation;   // world direction → direction on the star sphere
    float3 GalaxyPole;           // normal of the Milky Way's plane on the star sphere
    float MilkyWayBrightness;
    float3 GalaxyCenter;         // its bright core, a direction on that plane
    float _galaxyPad;
    // Cloud layer shape (coverage, time, altitude, noise LUT) lives in SceneConstants (common.fx):
    // the directional light needs it too for cloud shadows.
}

SamplerState linearWrap : register(s0); // Linear filter, wrap addressing (3D noise LUT)

struct VertexOutput
{
    float4 Position : SV_POSITION;
    float3 UV : TEXCOORD0;
    float3 ViewDir : TEXCOORD2;
};

struct FragmentOutput
{
    float4 Albedo : SV_TARGET0;
    float4 Normal : SV_TARGET1;
    float4 Data : SV_TARGET2;
    float4 Depth : SV_TARGET3;
    float fDepth : SV_DEPTH;
};

// ────────────────────────────────────────────────
// Noise utility (used by GetStars)
float hash13(float3 p3)
{
    p3 = frac(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

#include "sky_common.fx"

// ────────────────────────────────────────────────
// Main cloud function — LUT-based Nubis-style clouds
//
// LUT channels (from cloud_noise_gen.hlsl):
//   R: Perlin FBM     (smooth connected shapes)
//   G: Worley FBM     (low-freq billowy erosion)
//   B: Worley FBM     (high-freq detail erosion)
//   A: Perlin-Worley  (pre-combined cloud shape)
//
float GetCloudsAt(float3 viewDir, float altitude)
{
    if (viewDir.y <= 0.001)
        return 0.0;

    // Shared with the cloud shadows (sky_common.fx): same plane, same UVs, same base density
    float t = altitude / viewDir.y;
    float2 uv = CloudLayerUV(viewDir.xz * t);

    Texture3D<float4> noiseLUT = ResourceDescriptorHeap[CloudNoiseLUTIdx];
    float timeZ = CloudTime * 0.005;

    // Mip level based on viewing angle — blur out detail near the horizon
    // to hide tiling and create natural atmospheric softening
    float mip = saturate(1.0 - viewDir.y * 5.0) * 3.0;  // 0 at zenith, up to 3 at horizon

    // ── Base shape against the coverage threshold ──
    float cloudDensity = CloudBaseDensity(noiseLUT, linearWrap, uv, mip);

    // ── Detail erosion: Worley carves billowy edges ──
    // Use higher mip near horizon to blur out repetition
    float4 detailSample = noiseLUT.SampleLevel(linearWrap, float3(uv * 2.0 + 1.13, timeZ * 1.5), mip);
    float detailFBM = detailSample.g * 0.625 + detailSample.b * 0.375;

    // Key fix: erosion strength scales with (1 - density)
    // Thick cloud cores resist erosion; only thin edges get carved
    float erodeStrength = detailFBM * 0.2 * (1.0 - cloudDensity * 0.7);
    float eroded = remap(cloudDensity, erodeStrength, 1.0, 0.0, 1.0);

    // Softer smoothstep for fuller cloud bodies
    eroded = smoothstep(0.0, 0.45, eroded);

    // Horizon fade
    eroded *= smoothstep(0.0, 0.22, viewDir.y);

    return saturate(eroded);
}

// ────────────────────────────────────────────────
// Night sky: stars and the Milky Way on a sphere that turns with the time of day (StarRotation), so they wheel round
// the celestial pole the way the sun crosses the sky. Directions on that sphere are called s below.

// Star colours: mostly near white, some blue-white, a few yellow to orange. Kept pale (the game's muted palette).
float3 StarColor(float h)
{
    float3 c = lerp(float3(0.76, 0.84, 1.0), float3(1, 1, 1), smoothstep(0.0, 0.3, h));
    c = lerp(c, float3(1.0, 0.92, 0.78), smoothstep(0.6, 0.78, h));
    return lerp(c, float3(1.0, 0.76, 0.58), smoothstep(0.88, 0.98, h));
}

// One layer of stars: at most one per cell of a cube-map grid (2 × scale cells along a face), kept to the middle of its
// cell so it never straddles a cell edge. A star is a point about a pixel and a half wide at any resolution: its
// brightness, not its size, tells bright from faint; the brightest get a soft glow. They twinkle, more near the horizon.
float3 StarLayer(float3 s, float pixelAngle, float twinkle, float scale, float density, float minB, float maxB, float glow, float seed)
{
    float3 a = abs(s);
    float3 n, t, b;
    if (a.x >= a.y && a.x >= a.z) { n = float3(sign(s.x), 0, 0); t = float3(0, 1, 0); b = float3(0, 0, 1); }
    else if (a.y >= a.z)          { n = float3(0, sign(s.y), 0); t = float3(1, 0, 0); b = float3(0, 0, 1); }
    else                          { n = float3(0, 0, sign(s.z)); t = float3(1, 0, 0); b = float3(0, 1, 0); }
    float face = dot(n, float3(1, 2, 3));
    float2 g = float2(dot(s, t), dot(s, b)) / dot(s, n) * scale;
    float2 cell = floor(g);
    float3 key = float3(cell.x + face * 7.31, cell.y - face * 3.17, face * 13.7 + seed);
    if (hash13(key) > density)
        return 0;

    float2 uv = (cell + 0.2 + 0.6 * float2(hash13(key + 11.7), hash13(key + 23.1))) / scale;
    float3 starDir = normalize(n + uv.x * t + uv.y * b);
    float px = length(s - starDir) / pixelAngle;              // distance from the star in pixels

    float h = hash13(key + 37.9);
    float bright = lerp(minB, maxB, h * h * h);               // most stars are faint
    float flicker = 1.0 + twinkle * sin(Time * (1.7 + 2.9 * hash13(key + 61.3)) + h * 40.0);
    float shape = exp(-px * px * 0.9) + glow * smoothstep(0.4, 1.0, h) * 0.12 * exp(-px * 0.55);
    return StarColor(hash13(key + 51.3)) * bright * flicker * shape;
}

// The Milky Way: a band along the great circle round GalaxyPole, brightest towards GalaxyCenter, clumped into star
// clouds and split by dark dust lanes (noise from the cloud LUT, whose tiles the sphere spans once or a few times).
// Returns its glow; crowd (0..1) is how densely faint stars crowd it there.
float3 MilkyWay(float3 s, out float crowd)
{
    crowd = 0;
    float b = dot(s, GalaxyPole);                              // sine of the galactic latitude
    float wide = exp(-b * b / 0.09);
    if (wide < 0.01 || CloudNoiseLUTIdx == 0)
        return 0;

    // Perlin FBM from the LUT sits mostly in 0.3..0.7: stretch each octave to 0..1
    Texture3D<float4> lut = ResourceDescriptorHeap[CloudNoiseLUTIdx];
    float clumps = saturate((lut.SampleLevel(linearWrap, s * 0.5 + 0.5, 0).r - 0.3) * 2.5);
    float detail = saturate((lut.SampleLevel(linearWrap, s * 1.9 + 0.17, 0).r - 0.3) * 2.5);
    float fine = saturate((lut.SampleLevel(linearWrap, s * 5.3 + 0.71, 0).r - 0.3) * 2.5);
    float grain = saturate((lut.SampleLevel(linearWrap, s * 13.1 + 0.33, 0).r - 0.3) * 2.5);

    float bulge = exp((dot(s, GalaxyCenter) - 1.0) * 2.2);    // 1 at the core, ~0.01 opposite it
    float bw = lerp(0.08, 0.19, bulge) * lerp(0.7, 1.35, clumps);
    float band = exp(-b * b / (bw * bw));
    float structure = detail * 0.5 + fine * 0.33 + grain * 0.17;  // star clouds
    float lum = band * (0.25 + 0.75 * bulge) * smoothstep(0.15, 0.85, structure) * 2.2
              + wide * 0.12 * (0.4 + bulge);

    // Dust: a rift wandering along the middle, broken into pieces and widest at the core, and dark knots across
    // the band
    float wander = (clumps - 0.5) * 0.10 + (fine - 0.5) * 0.03;
    float rift = (b - wander) / (lerp(0.02, 0.06, bulge) * lerp(0.5, 1.5, detail));
    float riftOn = smoothstep(0.35, 0.65, clumps * 0.4 + fine * 0.4 + grain * 0.2);
    float dust = exp(-rift * rift) * riftOn + smoothstep(0.55, 0.95, detail * 0.55 + grain * 0.45) * band * 0.6;
    lum *= 1.0 - 0.8 * saturate(dust);

    crowd = saturate(band * (0.4 + bulge) * (1.0 - saturate(dust)));
    float3 color = lerp(float3(0.70, 0.77, 1.0), float3(1.0, 0.87, 0.72), saturate(bulge * 1.3 + band * 0.15));
    return color * lum;
}

// How much the moons above the horizon brighten the night (0 = none up or new, 1 = a full moon): it washes out the
// Milky Way and the faint stars. Also returns the brightest moon's direction (for the clouds).
float MoonLightOnSky(float3 sunDir, out float3 brightestDir)
{
    float total = 0, best = 0;
    brightestDir = -sunDir;
    for (uint m = 0; m < MoonCount; m++)
    {
        float3 md = MoonDirSize[m].xyz;
        float light = 0.5 * (1 - dot(md, sunDir)) * smoothstep(0.0, 0.1, md.y);
        total += light;
        if (light > best) { best = light; brightestDir = md; }
    }
    return saturate(total);
}

// Stars and the Milky Way seen in a world direction (moons are drawn over them by MoonsAndStars)
float3 NightSky(float3 viewDir, float pixelAngle, float nightFactor, float moonWash)
{
    float visible = saturate(nightFactor) * smoothstep(0.0, 0.25, viewDir.y);   // thicker air near the horizon
    if (visible <= 0)
        return 0;

    float3 s = mul(viewDir, (float3x3)StarRotation);
    float crowd;
    float3 milky = MilkyWay(s, crowd) * (0.03 * MilkyWayBrightness * (1 - 0.6 * moonWash));

    float twinkle = lerp(0.35, 0.08, saturate(viewDir.y * 2.0));
    float density = 0.4 + 1.2 * saturate(StarDensity);
    float3 stars = StarLayer(s, pixelAngle, twinkle, 8.0, 0.14 * density, 0.25, 1.6, 1.0, 1.0)
                 + StarLayer(s, pixelAngle, twinkle, 24.0, 0.40 * density, 0.04, 0.30, 0.0, 2.0) * (1 - 0.3 * moonWash)
                 + StarLayer(s, pixelAngle, twinkle, 70.0, (0.2 + 0.6 * crowd) * density, 0.008, 0.06, 0.0, 3.0) * (1 - 0.7 * moonWash);

    return (stars * StarBrightness + milky) * visible;
}

// The bright haze round each moon (its light scattered in the air near it), over the sky and under the clouds
float3 MoonGlow(float3 viewDir, float3 sunDir)
{
    float3 glow = 0;
    for (uint m = 0; m < MoonCount; m++)
    {
        float3 md = MoonDirSize[m].xyz;
        float light = 0.5 * (1 - dot(md, sunDir)) * smoothstep(-0.02, 0.05, md.y);
        float away = 1 - saturate(dot(viewDir, md));          // ≈ angle² / 2
        float3 tint = MoonColor[m].rgb * float3(0.8, 0.87, 1.0);
        glow += tint * light * (exp(-away * 900.0) * 0.035 + exp(-away * 40.0) * 0.008);
    }
    return glow * smoothstep(-0.01, 0.05, viewDir.y);
}

// The moons, each a sphere lit by the sun (so its phase shows) with an optional surface texture, and the stars
// wherever no moon covers them. Added over the sky colour: the atmosphere lies in front of the moons.
float3 MoonsAndStars(float3 viewDir, float3 sunDir, float3 stars)
{
    float3 moons = 0;
    float starsVisible = 1;
    for (uint m = 0; m < MoonCount; m++)
    {
        float3 md = MoonDirSize[m].xyz;
        float3 right = normalize(cross(abs(md.y) < 0.999 ? float3(0, 1, 0) : float3(0, 0, 1), md));
        float3 up = cross(md, right);
        float3 local = float3(dot(viewDir, right), dot(viewDir, up), dot(viewDir, md));

        // Place on the disc: 0 at its centre, 1 on its rim
        float2 p = local.xy / (max(local.z, 1e-4) * tan(MoonDirSize[m].w));
        float r = length(p);
        float2 uv = float2(p.x, -p.y) * 0.5 + 0.5;
        float2 uvDx = ddx(uv), uvDy = ddy(uv);
        float edge = max(fwidth(r), 1e-4);
        float cover = (1 - smoothstep(1 - edge, 1, r)) * step(0, local.z) * smoothstep(-0.002, 0.002, viewDir.y);
        if (cover <= 0)
            continue;

        // The sphere's normal here faces back towards the viewer
        float3 n = p.x * right + p.y * up - sqrt(saturate(1 - r * r)) * md;
        float lit = saturate(dot(n, sunDir));

        float3 albedo = 0.6;
        uint tex = MoonTextures[m];
        if (tex != 0)
        {
            Texture2D surface = ResourceDescriptorHeap[tex];
            albedo = surface.SampleGrad(linearWrap, saturate(uv), uvDx, uvDy).rgb;
        }
        float3 moon = albedo * MoonColor[m].rgb * (lit + 0.01);   // 0.01: earthshine, the dark side is not black

        moons = lerp(moons, moon, cover);
        starsVisible *= 1 - cover;
    }
    return moons + stars * starsVisible;
}

// The sun disc and its close glow, in the colour of the sunlight that reaches the ground (orange near the
// horizon). The disc is far brighter than the sky round it, so it blows out to yellow-white and blooms;
// at 1× it was barely brighter than the sky and read as a pale flat disc.
float3 GetSun(float3 viewDir, float3 sunDir)
{
    float sun = saturate(dot(viewDir, sunDir));
    // Disc ~0.55° across (the real sun: 0.53°); it was 2.3–3.6°, a big flat coin.
    float sunDisc = smoothstep(0.99996, 0.99999, sun);
    float sunGlow = pow(sun, 32.0) * 0.3;
    // Aureole: the bright haze within a few degrees of the sun (forward scattering by the haze, much tighter
    // than the sky's Mie glow): ~2 at 2°, 0.6 at 4°. With the small blown-out disc it reads as the glare of a
    // low sun. The wide part is mostly the sky's own low-sun glow (sky_common.fx).
    float aureole = pow(sun, 400.0) * 3.0 + pow(sun, 3000.0) * 20.0;
    float3 sunLight = SunlightThroughAir(sunDir.y) * float3(1.0, 0.95, 0.85);
    return sunLight * (sunDisc * 60.0 + sunGlow + aureole);
}

VertexOutput VS(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    VertexOutput output;

    // Look up mesh buffer indices from MeshRegistry
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];

    // Bindless index buffer - primitiveVertexID is 0 to N-1, add BaseIndex to offset into correct mesh part
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];

    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    StructuredBuffer<row_major matrix> globalTransforms = ResourceDescriptorHeap[GlobalTransformBufferIdx];
    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<InstanceDescriptor> descriptors = ResourceDescriptorHeap[DescriptorBufIdx];

    // Double-indirection: command signature → sorted index → original instance → descriptor
    uint dataPos = InstanceBaseOffset + instanceID;
    uint packedIdx = sortedIndices[dataPos];
    uint idx = packedIdx & 0x7FFFFFFFu;

    // Look up transform slot from descriptor buffer
    uint slot = descriptors[idx].TransformSlot;

    float3 rawPos = positions[vertexID];
    
    // Extract camera position from inverse view
    float3 cameraPos = ViewInverse[3].xyz;
    
    // Get world matrix from global transform buffer using the slot
    row_major float4x4 mat = globalTransforms[slot];
    mat._41 = cameraPos.x;
    mat._42 = cameraPos.y;
    mat._43 = cameraPos.z;
    mat._44 = 1;

    float4 worldPosition = mul(float4(rawPos, 1), mat);
    float4 viewPosition = mul(worldPosition, View);

    // Reverse-Z: far plane = 0, so set z=0 to place skybox at max distance
    float4 projPos = mul(viewPosition, Projection);
    output.Position = float4(projPos.xy, 0.0, projPos.w);
    output.UV = rawPos.xyz * 2;
    output.ViewDir = rawPos.xyz;

    return output;
}

FragmentOutput PS_Procedural(VertexOutput input)
{
    FragmentOutput output;

    float3 viewDir = normalize(input.ViewDir);
    float3 sunDir = normalize(SunDirection);
    float pixelAngle = max(length(fwidth(viewDir)) * 0.7071, 1e-5);   // radians per pixel, before any branch

    float3 skyColor = GetSkyColor(viewDir, sunDir);

    float dayFactor, sunsetFactor, nightFactor;
    GetDayNightFactors(sunDir.y, dayFactor, sunsetFactor, nightFactor);

    // Moons, their glow, stars and the Milky Way (not in shared GetSkyColor — ocean reflections don't need them).
    // The moons and stars go on after the clouds (below): they are far brighter than the sky behind a cloud.
    float3 moonDir;
    float moonLight = MoonLightOnSky(sunDir, moonDir);
    skyColor += MoonGlow(viewDir, sunDir);
    float3 behindClouds = MoonsAndStars(viewDir, sunDir, NightSky(viewDir, pixelAngle, nightFactor, moonLight));

    skyColor += GetSun(viewDir, sunDir) * SunIntensity;

    // Altitude scales the projected noise, so lerping it zooms the whole pattern. While a preset
    // transition runs we instead cross-fade the density of the old and new cloud layers.
    float density = GetCloudsAt(viewDir, CloudAltitude);
    if (CloudAltitudeBlend < 0.999)
    {
        float densityFrom = GetCloudsAt(viewDir, CloudAltitudeFrom);
        density = lerp(densityFrom, density, smoothstep(0.0, 1.0, saturate(CloudAltitudeBlend)));
    }

    if (density > 0.001)
    {
        // ── Opacity via Beer-Lambert ──
        float absorption = 1.0 - exp(-density * 4.5);

        // ── Core shading: density drives the shadow-sunlit gradient ──
        // This is the key: the cloud shape detail IS the shading detail.
        // Thin edges transmit light (bright), thick cores self-shadow (dark).
        float transmittance = exp(-density * 3.0);  // 1.0 at edges, ~0.05 at dense cores

        // ── Directional sun shading ──
        // At night the light comes from the brightest moon (none up: from opposite the sun, barely lit)
        float3 cloudLightDir = normalize(lerp(sunDir, moonDir, nightFactor));
        float sunDot = saturate(dot(viewDir, cloudLightDir));
        float sunGradient = sunDot * 0.5 + 0.5;  // 0.5 (shade side) to 1.0 (sun facing)

        // Combine: transmittance provides detail, sunGradient provides directionality
        float shadeFactor = lerp(transmittance * 0.7, transmittance, sunGradient);

        // ── Day palette: shadow → sunlit, plus silver lining and backlit powder glow ──
        float3 dayCloud = lerp(CloudShadowColor, CloudSunlitColor, shadeFactor);

        float edgeMask = smoothstep(0.0, 0.2, density) * smoothstep(0.5, 0.15, density);
        float silverLining = edgeMask * pow(sunDot, 2.0) * 0.4;
        dayCloud += float3(1.0, 1.0, 0.95) * silverLining;

        float powder = (1.0 - transmittance) * transmittance * 2.0;  // peaks at medium density
        dayCloud += float3(1.0, 0.9, 0.7) * powder * pow(sunDot, 3.0) * 0.3;
        dayCloud *= CloudBrightness * 1.3;

        // ── Sunset palette: day shading pulled toward the warm tint ──
        float3 sunsetCloud = lerp(dayCloud, CloudSunsetTintColor * CloudBrightness * 1.3, 0.5);

        // ── Night palette: lit by the moons — bright with a silver rim near a full moon, dark shapes against the
        // stars when none is up ──
        float3 nightCloud = lerp(CloudNightColor * 0.5, CloudNightColor, shadeFactor) * CloudNightBrightness * (0.25 + 0.75 * moonLight);
        float moonDot = saturate(dot(viewDir, moonDir));
        nightCloud += float3(0.8, 0.86, 1.0) * moonLight * (edgeMask * pow(moonDot, 40.0) * 0.5 + transmittance * pow(moonDot, 200.0) * 0.4);

        // ── Blend palettes with the same day/sunset/night weights as the sky ──
        float3 cloudColor = dayCloud * dayFactor
                          + sunsetCloud * sunsetFactor
                          + nightCloud * nightFactor;

        // Blend into sky
        skyColor = lerp(skyColor, cloudColor, absorption);
    }

    // Moons and stars through the clouds: hidden far sooner than the sky colour is. With the sky's absorption a thin
    // cloud let ~2/3 of a moon through, 50-100x brighter than the cloud round it, so it seemed to hang in front.
    skyColor += behindClouds * exp(-density * 15.0);

    output.Albedo = float4(skyColor, 1);
    output.Normal = float4(0, 1, 0, 1);
    output.Data = float4(0, 0, 0, 0);
    output.Depth = float4(0, 0, 0, 0);
    output.fDepth = 0; // Reverse-Z: far plane = 0

    return output;
}

RasterizerState DisableCull
{
    CullMode = None;
};

technique11 GBuffer
{
    pass Sky
    {
        SetRasterizerState(DisableCull);
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS_Procedural()));
    }
}
