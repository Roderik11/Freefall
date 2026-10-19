// The ocean's surface and its surf, shared by ocean.fx (the surface) and particle_compute.hlsl (spray and mist spawned
// where the surf breaks: ParticleEmitter shape Surf). OceanData is the per-ocean data C# OceanRenderer fills.
#ifndef OCEAN_SURF_HLSLI
#define OCEAN_SURF_HLSLI

struct OceanData
{
    float WaveTime;
    float Choppiness;
    float FoamDecayRate;
    float FoamThreshold;
    float3 OceanColor;
    uint SeaMaskSRV;        // where the open sea is (C# SeaMask), 0 = everywhere. Was padding: an older build writes 0
    float3 DeepColor;
    uint WindBands;         // bit per band carrying the wind sea (C# FollowWind). Was padding: an older build writes 0
    float3 SunDirection;
    float SunIntensity;
    float3 SunColor;
    uint WhitecapBand;      // the band whose foam marks the whitecaps + 1 (C# WhitecapBand); 0 = the coarsest wind band
    uint DisplacementSRV;
    uint SlopeSRV;
    uint NumBands;
    uint TileScalesSRV;      // bindless SRV to float[] buffer
    float DisplacementAtten;
    float NormalAtten;
    uint HeightmapSRV;      // bindless SRV to terrain heightmap
    float ShoreDepth;       // water depth (meters) for full wave strength
    float MaxTerrainHeight; // terrain MaxHeight scale
    float OceanPlaneY;      // ocean entity world Y
    float TerrainBaseY;     // terrain entity world Y
    float TerrainOriginX;   // terrain world origin X
    float TerrainOriginZ;   // terrain world origin Z
    float TerrainSizeX;     // terrain size X
    float TerrainSizeZ;     // terrain size Z
    float ShoreMinWave;     // minimum displacement at shore (0-1)
    uint DepthGBufferSRV;   // bindless SRV to linear depth gbuffer (PS shore effects)
    uint CompositeSRV;      // bindless SRV to composite buffer (terrain show-through)
    float ShoreFadeDepth;   // linear depth range for PS soft intersection (meters)
    float3 ShallowColor;    // shallow water tint color
    float RefractionStrength; // how much normal distorts terrain show-through
    uint NoiseSRV;              // bindless SRV for noise texture (Perlin+Worley)
    float2 InvViewportSize;     // 1.0 / viewport dimensions (replaces GetDimensions)
    float _shorePad;            // was ShoreFoamDepth (the clock-driven swash bands, gone with the surf 2026-10-19)
    float ShoreFoam;            // shore foam amount (0 = none)
    float ShoreSurge;           // the most the swash runs up the beach (m over the still sea, SwashLevel)
    uint HiZSRV;                // this view's Hi-Z pyramid of the opaque depth (0 = no occlusion test)
    uint ShoreWarpSRV;          // the surf (C# ShoreWarp): Texture2DArray RGBA32F per warped band: Δu.xz, amplitude, hollow (m, up the beach); 0 = none
    uint ShoreWarpJacobianSRV;  // its ∂u/∂x (RGBA16F: ∂ux/∂x, ∂ux/∂z, ∂uz/∂x, ∂uz/∂z)
    uint WarpSlots;             // band + 1 of warp slice 0 (bits 0-7) and slice 1 (bits 8-15); 0 = none
    float _warpPad;
    float4 WarpDirs;            // the deep heading (world xz) of slice 0 (xy) and slice 1 (zw)
    float2 WarpOmega;           // the angular frequency of slice 0 and slice 1's band at its peak (rad/s): the lip's wavelength
    float LipBarrels;           // OceanRenderer.Barrels: 0 lips where the physics puts them, 1 nearly every breaker (LipAmount)
    uint LipMapSRV;             // where the lips stand round the camera (C# LipMap, LipMapAt): amount, k, age; 0 = no lips
    float2 LipMapOrigin;        // its corner (world xz)
    uint LipSurfaceSRV;         // the smooth swell's displacement over the same texels (LipSurfaceAt): xyz, w 1 = written
    float _lipPad;
};

SamplerState OceanSampler : register(s0);

// The swash's cycle (SwashLevel): the share of a wave's period its bore runs up the beach, the rest it drains back
// (0.5 would be symmetric; 0.25 felt too abrupt up and too sluggish back)
static const float SurgeRise = 0.38;

// The surf's foam (PS): broken water breaks up along the sea's own finest crests (the bands the surf does not carry,
// their squeeze over their slopes, so it shows at any wind): its opacity where they are not squeezed and how much
// their crests raise it; and the bubbles all through the surf zone once its waves have broken
#define SURF_VEIL 0.15
#define SURF_CRESTS 1.2
#define SURF_BUBBLES 0.06
// The roller's opacity where the crests are not squeezed: nearly solid
#define SURF_ROLLER 0.6
// The still-water depth over which the bore's white water and its lean fade out toward the waterline (m)
static const float2 SURF_FOAM_END = float2(0.3, 1.2);

// The seabed under a point of the sea, from the terrain's heightmap
struct OceanShore
{
    float waveAtten;   // how far the shore flattens the longest waves (ShoreDepth, ShoreMinWave): 1 in the open sea
    float shoreDepth;  // still-water depth, at least 0, very large past the terrain (BandShoreAtten, the surf)
    float waterDepth;  // still-water depth, negative up the beach, very large past the terrain
    float2 surfUV;     // the point's terrain UV, clamped to the terrain (SampleBandWarp)
};

OceanShore ShoreAt(OceanData ocean, float2 worldXZ)
{
    OceanShore s;
    s.waveAtten = 1.0;
    s.shoreDepth = 1e4;
    s.waterDepth = 1e4;
    s.surfUV = 0.5;
    if (ocean.HeightmapSRV == 0) return s;

    float2 terrainSize = float2(ocean.TerrainSizeX, ocean.TerrainSizeZ);
    float2 terrainUV = (worldXZ - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ)) / terrainSize;

    // The seabed outside the terrain is unknown: extend the border's depth outward (clamped UV) and
    // fade to open-sea waves over the next 150 m. Stopping at the border instead makes the wave
    // height jump along it wherever the seabed there is shallower than ShoreDepth.
    float2 clampedUV = clamp(terrainUV, 0.002, 0.998);
    s.surfUV = clampedUV;
    float outside = length((terrainUV - clampedUV) * terrainSize);
    if (outside < 150.0)
    {
        Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
        float h = heightmap.SampleLevel(OceanSampler, clampedUV, 0);
        float terrainWorldY = ocean.TerrainBaseY + h * ocean.MaxTerrainHeight;

        // Water depth = ocean surface - terrain surface (positive = water exists)
        s.waterDepth = ocean.OceanPlaneY - terrainWorldY;
        float shoreAtten = lerp(ocean.ShoreMinWave, 1.0, smoothstep(0, ocean.ShoreDepth, s.waterDepth));

        // The terrain is the seabed: a sea built on a flat floor needs seabed stamps to deepen (Eldermoor since
        // 2026-10-19). For a day before that, terrain lying on its floor counted as open sea here, a stand-in
        float open = smoothstep(0.0, 150.0, outside);
        s.waveAtten = lerp(shoreAtten, 1.0, open);
        s.shoreDepth = max(s.waterDepth, 0.0) + open * 1e4;
    }
    return s;
}

// Per-band UV rotation to decorrelate cascade tiling patterns
static const float BandAngles[8] = { 0.0, 0.297, 0.645, 0.925, 1.17, 1.42, 1.73, 2.05 };

float2 RotateUV(float2 uv, float angle)
{
    float s, c;
    sincos(angle, s, c);
    return float2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
}

// How much of a band's waves the shore lets stand, for the bands the surf does not carry (below): waves feel the bottom
// where the water is shallower than about half their length, so the long waves flatten toward the beach while the
// ripples run on over a shallow shelf. They stand in full from an eighth of the band's tile deep (17 m band: 2 m, 64 m:
// 8 m, 250 m: 31 m, the 1000 m swell: 125 m), and everywhere from ShoreDepth. (A wind band holds from a quarter of the
// next finer wind band's tile up to its own: 250 m band 16-250 m, 64 m band 4.25-16 m.) Until 2026-10-18 the shore damped
// every band alike, ripples too, and only the heights: the PS shaded a flattened shelf as the full open-sea chop.
//   shoreAtten: the shore's damping of the longest waves (ShoreMinWave..1), depth: still-water depth (m)
float BandShoreAtten(float shoreAtten, float depth, float tileScale)
{
    return lerp(shoreAtten, 1.0, saturate(depth * tileScale * 8.0));
}

// ── The surf (C# ShoreWarp): the open sea's own waves carried to the beach ──
// Near the coast a warped band is sampled not where a point is but where its wave was out at sea, u = x + Δu, baked
// from the seabed by the wave's travel time: its crests turn toward the beach, crowd together and grow as they shoal,
// and a lee shore stays calm. What comes in is the sea's own field, its sets as sets.
struct BandWarp
{
    float2 u;          // where to sample the band (world xz, in the open sea's frame)
    float amp;         // the wave's height here over the open sea's: shoaling, the rays' convergence, the island's shadow
    float hollow;      // up the beach: how far below the ridge between it and the waterline this spot lies (m, 0 on a slope)
    float2x2 J;        // ∂u/∂x: rows ux, uz; columns ∂/∂x, ∂/∂z
    float2 turn;       // cos, sin of the turn from the deep heading to the local one (sideways displacement, breakers)
    float2 heading;    // the local heading (world xz unit vector)
};

// A band's slice in the warp, 0xFF if the surf does not carry it
uint WarpSlice(OceanData ocean, uint band)
{
    if (ocean.ShoreWarpSRV == 0) return 0xFF;
    if ((ocean.WarpSlots & 0xFFu) == band + 1) return 0;
    if (((ocean.WarpSlots >> 8) & 0xFFu) == band + 1) return 1;
    return 0xFF;
}

//   terrainUV: the point's terrain UV (outside the terrain the border's, where the warp is none)
BandWarp SampleBandWarp(OceanData ocean, uint slice, float2 worldXZ, float2 terrainUV)
{
    Texture2DArray<float4> warpTex = ResourceDescriptorHeap[ocean.ShoreWarpSRV];
    Texture2DArray<float4> jacobianTex = ResourceDescriptorHeap[ocean.ShoreWarpJacobianSRV];
    float w, h, slices;
    warpTex.GetDimensions(w, h, slices);
    // The bake's cell i lies at UV i / (n - 1), its texel's centre at (i + 0.5) / n
    float2 uv = (saturate(terrainUV) * (float2(w, h) - 1.0) + 0.5) / float2(w, h);
    float4 a = warpTex.SampleLevel(OceanSampler, float3(uv, slice), 0);
    float4 j = jacobianTex.SampleLevel(OceanSampler, float3(uv, slice), 0);

    BandWarp b;
    b.u = worldXZ + a.xy;
    b.amp = a.z;
    b.hollow = a.w;
    b.J = float2x2(j.x, j.y, j.z, j.w);
    // The local heading: the gradient of the deep phase d·u, Jᵀd
    float2 d = slice == 0 ? ocean.WarpDirs.xy : ocean.WarpDirs.zw;
    float2 r = mul(d, b.J);
    r *= rsqrt(max(dot(r, r), 1e-12));
    b.turn = float2(dot(d, r), d.x * r.y - d.y * r.x);
    b.heading = r;
    return b;
}

// The surf's wave at a point. A carried band's height and its sideways displacement along the local heading are in
// quadrature (η = A·cos φ, D = −λ·A·sin φ, λ the choppiness), which gives the wave's local amplitude and phase with no
// history: so each wave knows on its own whether it has broken, and where its crest and its trail are.
// Per carried band (a warp slice), so the two blend by their energy: following whichever was stronger (until
// 2026-10-19) jumped the phase where they were about equal, a zig-zag in the swash's edge on the sand
struct SurfWave
{
    float eta;             // the carried bands' height (m)
    float2 envelope;       // each carried band's local amplitude A (m)
    float2 phase;          // its phase: 0 on the crest, > 0 ahead of it (still to come), < 0 behind it (passed)
    float4 heading;        // its local heading (world xz): slice 0 in xy, slice 1 in zw
    float hollow;          // up the beach: how far below the ridge before it this spot lies (BandWarp.hollow, m)
};

//   d: a carried band's displacement in world xz (turned to the local heading) and height, both × its amplitude
void AddSurfWave(inout SurfWave s, OceanData ocean, float3 d, float2 heading, uint slice)
{
    float quadrature = -dot(d.xz, heading) / max(ocean.Choppiness, 0.05);
    float envelope = sqrt(d.y * d.y + quadrature * quadrature);
    float phase = atan2(quadrature, d.y);
    s.eta += d.y;
    if (slice == 0) { s.envelope.x = envelope; s.phase.x = phase; s.heading.xy = heading; }
    else            { s.envelope.y = envelope; s.phase.y = phase; s.heading.zw = heading; }
}

// The carried bands' amplitude together, and each band's share of their energy
float SurfEnvelope(SurfWave s) { return length(s.envelope); }
float2 SurfShare(SurfWave s) { return s.envelope * s.envelope / max(dot(s.envelope, s.envelope), 1e-8); }

// The local heading, weighted by the bands' energy
float2 SurfHeading(SurfWave s)
{
    float2 share = SurfShare(s);
    float2 h = s.heading.xy * share.x + s.heading.zw * share.y;
    return dot(h, h) > 1e-8 ? normalize(h) : float2(1.0, 0.0);
}

// Shoaling waves break when they stand about 0.8 × the depth from trough to crest: 0 before, 1 once broken (inside the
// surf zone every wave stays broken: as the water shallows its height keeps up with the depth)
float SurfBreaking(SurfWave s, float depth)
{
    return smoothstep(0.55, 0.85, 2.0 * SurfEnvelope(s) / max(depth, 0.05));
}

// Where in its period the arriving wave is, 0 as its crest arrives (rising with time), and how much the bore is running
// up the beach then (0..1, quick up for SurgeRise of the period, slow down)
float SwashRise(float phase)
{
    float cycle = frac(1.0 - phase / 6.2831853);
    return cycle < SurgeRise ? smoothstep(0.0, SurgeRise, cycle) : 1.0 - smoothstep(SurgeRise, 1.0, cycle);
}
float SwashAdvancing(float phase)
{
    float cycle = frac(1.0 - phase / 6.2831853);
    return max(1.0 - smoothstep(SurgeRise, SurgeRise + 0.2, cycle), smoothstep(0.92, 1.0, cycle));
}

// The swash: within this depth of the waterline (and up the beach) the sea's level is the run-up's
#define SWASH_DEPTH 0.6
// How high the bores run up the beach per metre of the arriving wave's height, and how far the backwash draws the
// water down below the still level, per metre of run-up
#define SWASH_RUNUP 0.35
#define SWASH_DRAWDOWN 0.3

// The water level of the swash (m over the still sea): each arriving crest's bore runs up quickly (SurgeRise of the
// wave's period) as far as its height carries it (at most ShoreSurge) and drains back slowly, in step with the wave
float SwashLevel(SurfWave s, float maxRunUp)
{
    float2 share = SurfShare(s);
    float rise = SwashRise(s.phase.x) * share.x + SwashRise(s.phase.y) * share.y;
    float runUp = min(SWASH_RUNUP * 2.0 * SurfEnvelope(s), maxRunUp);
    return lerp(-SWASH_DRAWDOWN * runUp, runUp, rise);
}

// The broken water a carried band's wave carries at its phase: the roller on its crest and face, the trail behind it
float2 SurfRollerTrail(float phase)
{
    float roller = exp(-(phase - 0.1) * (phase - 0.1) / 0.12);
    float trail = phase < 0.0 ? 0.7 * exp(phase / 0.8) : 0.0;
    return float2(roller, trail);
}

// How far into its backwash the swash is (0 at the top of the run-up, 1 as the next bore arrives), -1 while it runs up
float SwashBackwash(float phase)
{
    float cycle = frac(1.0 - phase / 6.2831853);
    return cycle < SurgeRise ? -1.0 : (cycle - SurgeRise) / (1.0 - SurgeRise);
}

float2 Hash22(float2 p)
{
    p = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3)));
    return frac(sin(p) * 43758.5453);
}

// Animated cellular noise for foam webs: the net of bubble films a thinning foam sheet breaks into. The look only: where
// and when it shows is the waves' (the backwash of each wave's swash, the thinning trail of the surf). Returns
// (F2 - F1, F1): x is 0 on the border between two cells and grows toward the cell centers; y is the distance to the
// nearest center, largest where several cells meet (used to thicken the knots). The feature points orbit with time, so
// the web shifts and reconnects in place. The cells are stretched along flowDir through the distance metric only — the
// grid itself stays world-aligned, so a flow direction that changes across the beach cannot shear the pattern. On its
// own this is straight-edged Voronoi: warp the input to bend it.
// drop (0..~0.65) is the fraction of points retired: raise it over the foam's life to make the cells grow.
float2 FoamWeb(float2 x, float2 flowDir, float stretch, float time, float drop)
{
    float2 cell = floor(x);
    float2 f = frac(x);
    float d1 = 64.0, d2 = 64.0;

    [unroll]
    for (int j = -2; j <= 2; ++j)
    {
        [unroll]
        for (int i = -2; i <= 2; ++i)
        {
            float2 g = float2(i, j);
            float2 h = Hash22(cell + g);

            float2 o = 0.5 + 0.48 * sin(time + 6.2831853 * h);
            float2 r = g + o - f;

            float along = dot(r, flowDir);
            float2 rr = (r - flowDir * along) + flowDir * (along / stretch);
            float d = dot(rr, rr);

            // Each point retires once 'drop' passes its own random threshold. Its distance is pushed
            // out gradually, so its cell shrinks away and the neighbours grow into the space — cells
            // of very different sizes, and a web that coarsens smoothly as drop rises (no popping,
            // and no zooming of the pattern, which would make it slide across the beach).
            float retire = frac(h.x * 13.7 + h.y * 91.3);
            float gone = smoothstep(retire - 0.1, retire + 0.1, drop);
            d += gone * gone * 6.0;

            if (d < d1) { d2 = d1; d1 = d; }
            else if (d < d2) { d2 = d; }
        }
    }
    d1 = sqrt(d1);
    return float2(sqrt(d2) - d1, d1);
}

// The water stands no higher over the seabed than a breaking wave allows (a wave breaks at about 0.78 × the depth from
// trough to crest, its crest some 0.6 × the depth over the still level) and sinks no lower than 0.8 × the depth. Soft,
// with what it leaves of the slopes (d limited / d height). The surf's shoaling waves would otherwise rise into walls.
float DepthLimit(float y, float depth, out float slopeScale)
{
    float lim = (y > 0.0 ? 0.6 : 0.8) * max(depth, 0.02);
    float t = tanh(y / lim);
    slopeScale = 1.0 - t * t;
    return lim * t;
}

//   shoreAtten, depth: as BandShoreAtten (1 and a large depth away from the shore)
//   terrainUV, surf: the point's terrain UV for the surf's warp (SampleBandWarp), and whether to warp at all
//   wave: the surf's wave here (its carried bands)
float3 SampleDisplacement(OceanData ocean, float2 worldXZ, float depthAtten, float camDist, float shoreAtten, float depth,
                          float2 terrainUV, bool surf, out SurfWave wave)
{
    wave = (SurfWave)0;
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float3 totalDisp = 0;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float2 p = worldXZ;
        float amp;
        float stretch = 1.0;
        float2 turn = float2(1.0, 0.0);
        float2 heading = float2(1.0, 0.0);
        uint slice = surf ? WarpSlice(ocean, i) : 0xFF;
        if (slice != 0xFF)
        {
            // The surf: where this wave was out at sea, as high as it has grown; the band's texture is squeezed by the
            // warp, so its mip goes by the largest stretch
            BandWarp w = SampleBandWarp(ocean, slice, worldXZ, terrainUV);
            p = w.u;
            amp = w.amp;
            turn = w.turn;
            heading = w.heading;
            wave.hollow = max(wave.hollow, w.hollow);
            stretch = sqrt(max(w.J[0][0] * w.J[0][0] + w.J[1][0] * w.J[1][0], w.J[0][1] * w.J[0][1] + w.J[1][1] * w.J[1][1]));
        }
        else
            amp = BandShoreAtten(shoreAtten, depth, tileScales[i]);

        float2 uv = RotateUV(p, BandAngles[i]) * tileScales[i];
        float mipLevel = log2(max(1.0, camDist * tileScales[i] * 0.15 * stretch));
        float3 d = dispTex.SampleLevel(OceanSampler, float3(uv, i), mipLevel).xyz;
        // The sideways displacement is in the band's turned frame: turn it back to the world, as the PS does the slopes.
        // Added as it came (until 2026-10-18), it pushed the surface off to one side of the band's waves instead of
        // toward their crests: choppiness sheared the pattern sideways and never sharpened a crest.
        d.xz = RotateUV(d.xz, -BandAngles[i]);
        // A carried wave runs along its local heading
        d.xz = float2(d.x * turn.x - d.z * turn.y, d.x * turn.y + d.z * turn.x);
        totalDisp += d * amp;
        if (slice != 0xFF)
            AddSurfWave(wave, ocean, d * amp, heading, slice);
    }

    return totalDisp * depthAtten;
}

// Where the ocean draws the point of its grid at worldXZ, as an offset from (worldXZ, OceanPlaneY): the bands'
// displacement, and near the shore the surf's limits (no higher than a breaking wave, no lower than the bed, nothing
// pushed sideways further than the water is deep), the breaking face's lean and the swash's run-up. The ocean's DS, and
// the lip (ocean_lip.fx) for the crest its sheet hangs from. shore, surf: as the DS needs them for the PS
float3 OceanSurfaceDisplacement(OceanData ocean, float2 worldXZ, float camDist, out OceanShore shore, out SurfWave surf)
{
    // How far the shore flattens the longest waves here and the depth that sets how much each band feels of it
    // (BandShoreAtten): the PS weighs the bands' slopes and foam alike
    shore = ShoreAt(ocean, worldXZ);
    float shoreDepth = shore.shoreDepth;
    float waterDepth = shore.waterDepth;

    // Sample FFT displacement from all bands (with distance-based mips), each as far as the shore lets it stand, the
    // surf's bands where their waves were out at sea
    float3 displacement = SampleDisplacement(ocean, worldXZ, ocean.DisplacementAtten, camDist, shore.waveAtten, shoreDepth,
                                             shore.surfUV, ocean.HeightmapSRV != 0, surf);

    if (ocean.ShoreWarpSRV != 0 && shoreDepth < 1e3)
    {
        // The surf's waves grow as they shoal: over shallow water the sea stands no higher than a breaking wave and sinks
        // no lower than the bed, and nothing is pushed sideways further than the water is deep
        float slopeScale;
        displacement.y = DepthLimit(displacement.y, shoreDepth, slopeScale);
        float sideways = length(displacement.xz);
        displacement.xz *= min(1.0, max(shoreDepth, 0.05) / max(sideways, 1e-4));
        // A breaking face used to lean forward here (SURF_LEAN 1.6 m per metre of crest height): the crest's points
        // and the white water riding them moved off the wave's bulge, the foam lagging behind it (user, 2026-10-19). The
        // lip (ocean_lip.fx) throws the crest forward instead

        // The swash: by the waterline and up the beach the level is the bores' run-up, each arriving wave's own, in step
        // with it. Until 2026-10-19 a clock (ShoreClock, noise-staggered) and a hash per wave (WaveStrength) ran it
        float swashZone = 1.0 - smoothstep(0.0, SWASH_DEPTH, waterDepth);
        float level = SwashLevel(surf, ocean.ShoreSurge);
        // Up the beach the run-up wets a spot only once it has topped everything between the waterline and there: a
        // hollow behind a ridge stays dry until the water pours over. One level over the whole beach filled every hollow
        // below it within the sea mask's reach, puddles far up the sand outlined by the lace (until 2026-10-19)
        // The ridge is the sand's own height (1 m heightmap) plus how far below it the spot lies (C# ShoreWarp, 0 on a
        // plain slope): baked as the ridge's height, its 4 m cells frayed the waterline into a sawtooth
        if (waterDepth < 0.0)
        {
            float sand = -waterDepth;                                          // the sand's height over the sea
            float ridge = sand + surf.hollow;
            float inHollow = smoothstep(0.02, 0.1, surf.hollow);
            float topped = smoothstep(ridge - 0.05, ridge + 0.05, level);
            level = lerp(level, sand - 0.3, inHollow * (1.0 - topped));
        }
        displacement.y = lerp(displacement.y, level, swashZone);
    }
    return displacement;
}

// ── The breaking lip (ocean_lip.fx, .agent/knowledge/ocean_system/surf_lip_plan.md) ──
// Where a lip stands the lip pass draws the strip ahead of the crest (the curl and a steep face under it) and the
// ocean leaves its own gentle face out (PS), both by LipAmount: the same test, so they meet with no gap.

// The wave's height over the depth (2A / d): the face starts to steepen (breaking begins at 0.55, SurfBreaking), the
// lip is fully thrown, and has collapsed into the roller's white water. At 0.45 (0.30 with Barrels) the lip threw while
// the face was still gentle, a crease on a sloping wave: "breaking a bit too early" (user, 2026-10-19)
static const float3 LIP_LIFE = float3(0.5, 0.8, 1.0);
// The share of its life (LipAmount's age: 0 at the onset, 0.5 fully thrown, 1 collapsed) over which the lip stands up
// (only the face steepening under a sharpening crest), and over which its curl is thrown (LipThrow). Driven by the
// amount, which rises and falls over the life, the curl threw the moment a lip formed and wound back up as it collapsed
static const float LIP_STAND = 0.2;
static const float2 LIP_THROW = float2(0.15, 0.55);
float LipThrow(float age) { return smoothstep(LIP_THROW.x, LIP_THROW.y, age); }
// Iribarren number (bed slope over the square root of the wave's steepness H / L0): spilling below, plunging above
static const float2 LIP_PLUNGE = float2(0.35, 0.7);
// The camera's distance over which the lip fades out: further out its strip has too few vertices
static const float2 LIP_FADE = float2(90.0, 120.0);
// The strip ahead of a crest the lip and its face take, in radians of the band's phase: nearly down to the trough (π),
// so the breaking face is the wave's front. At 1.2 the ocean's own gentle front ran on below the lip's face and the
// breakers sat behind the swell's front (user, 2026-10-19)
static const float LIP_STRIP = 2.4;
// Above this LipAmount the lip pass draws the strip and the ocean leaves it out. The lip covers down to 0 (its skirt):
// the margin must hold between the ocean's per-pixel amount and the lip's per-vertex one (sampled at another mip),
// which part where the amount changes fast along the crest. At 0.02 (the lip down to 0.004) the seabed showed through
// where a lip's edge forked near the beach
static const float LIP_MIN = 0.05;

// The bed under xz as the lip reads it, over its breaking zone: the still water's depth (a weighted cross of samples
// 6 m out) and the bed's steepness (rise over run, over 12 and 24 m). Read at the point and over 10 m, the seabed's
// 1 m detail made the lip's amount change every few metres along the crest: its edges tore into tongues and flaps,
// and where it faded the half-bent face stood as a shelf
void LipBed(OceanData ocean, float2 xz, out float depth, out float slope)
{
    Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
    float2 size = float2(ocean.TerrainSizeX, ocean.TerrainSizeZ);
    float2 uv = clamp((xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ)) / size, 0.002, 0.998);
    float2 near = 6.0 / size, far = 12.0 / size;
    float c = heightmap.SampleLevel(OceanSampler, uv, 0);
    float e1 = heightmap.SampleLevel(OceanSampler, uv + float2(near.x, 0), 0), w1 = heightmap.SampleLevel(OceanSampler, uv - float2(near.x, 0), 0);
    float n1 = heightmap.SampleLevel(OceanSampler, uv + float2(0, near.y), 0), s1 = heightmap.SampleLevel(OceanSampler, uv - float2(0, near.y), 0);
    float e2 = heightmap.SampleLevel(OceanSampler, uv + float2(far.x, 0), 0), w2 = heightmap.SampleLevel(OceanSampler, uv - float2(far.x, 0), 0);
    float n2 = heightmap.SampleLevel(OceanSampler, uv + float2(0, far.y), 0), s2 = heightmap.SampleLevel(OceanSampler, uv - float2(0, far.y), 0);
    float h = (2.0 * c + e1 + w1 + n1 + s1) / 6.0;
    depth = max(ocean.OceanPlaneY - (ocean.TerrainBaseY + h * ocean.MaxTerrainHeight), 0.0);
    float2 gradient = 0.5 * (float2(e1 - w1, n1 - s1) / 12.0 + float2(e2 - w2, n2 - s2) / 24.0);
    slope = length(gradient) * ocean.MaxTerrainHeight;
}

// The carried band a lip belongs to: the stronger one here; its phase (0 at the crest, positive ahead of it)
bool LipSecondBand(SurfWave wave) { return wave.envelope.y > wave.envelope.x; }
float LipPhase(SurfWave wave) { return LipSecondBand(wave) ? wave.phase.y : wave.phase.x; }

// How clearly one carried band leads (0 where the two are as high, 1 with one alone): a lip only where one does. Where
// the two crossed about evenly the stronger one flipped from point to point, each side bending round its own crests:
// the sheet tore into saw teeth along the line, and the ocean's cut-out and the lip chose different bands there
static const float2 LIP_LEAD = float2(0.15, 0.35);
float LipLead(SurfWave wave)
{
    return abs(wave.envelope.x - wave.envelope.y) / max(wave.envelope.x + wave.envelope.y, 1e-4);
}

// The carried bands' waves at xz as the lip reads them, both its pass (its strip, its crest, the crest's height and
// heading) and the ocean's PS (where it leaves its face out for the lip): blurred to texels of LIP_WAVE_BLUR metres.
// At full detail the swell band's shorter waves jittered each vertex's phase and phase rate, and a strip's length from
// the crest neighbouring vertices bent round crests tens of centimetres apart: the sheet crumpled like foil near the
// camera, where it is finest. The ocean decides by the same waves, or it cut itself out where the lip hung its skirt
// low, and the seabed showed through the open step between them
static const float LIP_WAVE_BLUR = 6.0;

SurfWave LipWave(OceanData ocean, float2 xz)
{
    SurfWave wave = (SurfWave)0;
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float texW, texH, slices;
    dispTex.GetDimensions(texW, texH, slices);
    float2 terrainUV = clamp((xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ))
                             / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ), 0.002, 0.998);
    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        uint slice = WarpSlice(ocean, i);
        if (slice == 0xFF) continue;
        // As SampleDisplacement samples a carried band, at the mip whose texels are LIP_WAVE_BLUR metres
        BandWarp bw = SampleBandWarp(ocean, slice, xz, terrainUV);
        float stretch = sqrt(max(bw.J[0][0] * bw.J[0][0] + bw.J[1][0] * bw.J[1][0], bw.J[0][1] * bw.J[0][1] + bw.J[1][1] * bw.J[1][1]));
        float2 uv = RotateUV(bw.u, BandAngles[i]) * tileScales[i];
        float mip = log2(max(1.0, LIP_WAVE_BLUR * tileScales[i] * texW * stretch));
        float3 d = dispTex.SampleLevel(OceanSampler, float3(uv, i), mip).xyz;
        d.xz = RotateUV(d.xz, -BandAngles[i]);
        d.xz = float2(d.x * bw.turn.x - d.z * bw.turn.y, d.x * bw.turn.y + d.z * bw.turn.x);
        AddSurfWave(wave, ocean, d * bw.amp, bw.heading, slice);
    }
    return wave;
}

// The curl (ocean_lip.fx draws it; the surf's spray leaves from its tip): the tube's radius per metre of wave height,
// where its centre lies from the crest top (in radii: ahead, below), how far the curl turns round it when fully thrown
// (radians), and how much its radius shrinks to the tip
static const float LIP_RADIUS = 0.3;
static const float2 LIP_CENTRE = float2(0.6, 0.8);
static const float LIP_SWEEP = 3.8;
static const float LIP_TIP_SHRINK = 0.35;

// The lip's tip, from its crest top C (fwd: the wave's heading, height: the wave's from trough to crest, thrown: how
// far its curl is thrown, LipThrow): where the curl's outer side ends. falling: 0 while it is thrown forward, 1 once
// it points down at the trough (the plunge, where it lands)
float3 LipTip(float3 C, float3 fwd, float height, float thrown, out float falling)
{
    float R = LIP_RADIUS * height;
    float3 O = C + fwd * (LIP_CENTRE.x * R) - float3(0.0, LIP_CENTRE.y * R, 0.0);
    float theta = atan2(-LIP_CENTRE.x, LIP_CENTRE.y) + LIP_SWEEP * thrown;
    float rho = R * length(LIP_CENTRE) * (1.0 - LIP_TIP_SHRINK * thrown);
    falling = smoothstep(1.6, 3.0, theta);
    return O + rho * (sin(theta) * fwd + float3(0.0, cos(theta), 0.0));
}

// How far a lip stands at xz, 0..1: the wave breaking but not yet collapsed, over a bed that makes it plunge, one band
// clearly leading, near the camera. 0 without lips (the C# leaves WarpOmega 0 when Engine.Settings.DisableSurfLip).
// age: how far through its life the breaker is, rising only as it runs into shallower water (0 at the onset, 0.5
// fully thrown, 1 collapsed): its shape goes by this (LipThrow), the amount only says how much of it stands
float LipAmount(OceanData ocean, float2 xz, SurfWave wave, float depth, float camDistXZ, out float age)
{
    age = 0.0;
    float omega = LipSecondBand(wave) ? ocean.WarpOmega.y : ocean.WarpOmega.x;
    if (omega <= 0.05 || ocean.HeightmapSRV == 0 || depth > 1e3) return 0.0;
    float bedDepth, bedSlope;
    LipBed(ocean, xz, bedDepth, bedSlope);
    float height = 2.0 * SurfEnvelope(wave);
    float r = height / max(bedDepth, 0.05);
    // OceanRenderer.Barrels (immersive over realistic): the lip lasts into shallower water, and the bed need hardly be
    // steeper than the wave for it to plunge (not earlier: an earlier onset threw it off a gentle face)
    float barrels = saturate(ocean.LipBarrels);
    float3 lifeWindow = float3(LIP_LIFE.x - 0.05 * barrels, LIP_LIFE.y, LIP_LIFE.z + 0.3 * barrels);
    age = r < lifeWindow.y ? 0.5 * saturate((r - lifeWindow.x) / (lifeWindow.y - lifeWindow.x))
                           : 0.5 + 0.5 * saturate((r - lifeWindow.y) / (lifeWindow.z - lifeWindow.y));
    // It stands up early in its life (the steepening face), and sinks into the white water as it collapses
    float life = smoothstep(0.0, LIP_STAND, age) * (1.0 - smoothstep(0.5, 1.0, age));
    if (life <= 0.0) return 0.0;
    float L0 = 6.2831853 * 9.81 / (omega * omega);
    float xi = bedSlope / sqrt(max(height, 0.05) / L0);
    float2 plunge = lerp(LIP_PLUNGE, float2(0.0, 0.1), barrels);
    return life * smoothstep(plunge.x, plunge.y, xi) * smoothstep(LIP_LEAD.x, LIP_LEAD.y, LipLead(wave))
         * (1.0 - smoothstep(LIP_FADE.x, LIP_FADE.y, camDistXZ));
}

// The local wavenumber of a carried band (its warp slice) at xz: its deep-water wavenumber at the band's peak, times how
// far the warp stretches the deep phase along the heading (|Jᵀd|: smooth, the warp's Jacobian is bilinear)
float LipWavenumber(OceanData ocean, uint slice, float2 xz)
{
    float2 terrainUV = clamp((xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ))
                             / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ), 0.002, 0.998);
    BandWarp bw = SampleBandWarp(ocean, slice, xz, terrainUV);
    float2 deep = slice == 0 ? ocean.WarpDirs.xy : ocean.WarpDirs.zw;
    float omega = slice == 0 ? ocean.WarpOmega.x : ocean.WarpOmega.y;
    return omega * omega / 9.81 * length(mul(deep, bw.J));
}

// A carried band's blurred phase at xz (as LipWave has it), and how fast it runs along the heading there, over 3 m
float LipBandPhase(OceanData ocean, uint slice, float2 xz)
{
    uint band = (slice == 0 ? (ocean.WarpSlots & 0xFFu) : ((ocean.WarpSlots >> 8) & 0xFFu)) - 1;
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];
    float texW, texH, slices;
    dispTex.GetDimensions(texW, texH, slices);
    float2 terrainUV = clamp((xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ))
                             / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ), 0.002, 0.998);
    BandWarp bw = SampleBandWarp(ocean, slice, xz, terrainUV);
    float stretch = sqrt(max(bw.J[0][0] * bw.J[0][0] + bw.J[1][0] * bw.J[1][0], bw.J[0][1] * bw.J[0][1] + bw.J[1][1] * bw.J[1][1]));
    float2 uv = RotateUV(bw.u, BandAngles[band]) * tileScales[band];
    float3 d = dispTex.SampleLevel(OceanSampler, float3(uv, band), log2(max(1.0, LIP_WAVE_BLUR * tileScales[band] * texW * stretch))).xyz;
    d.xz = RotateUV(d.xz, -BandAngles[band]);
    d.xz = float2(d.x * bw.turn.x - d.z * bw.turn.y, d.x * bw.turn.y + d.z * bw.turn.x);
    return atan2(-dot(d.xz, bw.heading) / max(ocean.Choppiness, 0.05), d.y);
}

float LipPhaseRate(OceanData ocean, uint slice, float2 xz, float2 heading)
{
    float dPhase = LipBandPhase(ocean, slice, xz + heading * 1.5) - LipBandPhase(ocean, slice, xz - heading * 1.5);
    dPhase -= 6.2831853 * round(dPhase / 6.2831853);           // across the ±π seam
    return dPhase / 3.0;
}

// The crest a point of a strip belongs to (wave: LipWave there): back along the heading where the blurred phase is 0.
// A first guess by the phase's rate at the point, then two secant steps: the phase drop from the point back to the
// guess is the rate over the whole span, and the guess moves to where that rate puts phase 0. The phase runs up to
// twice as fast near the crest as on the face: by the warp's own wavenumber (LipWavenumber, 1-1.2 x the dispersion
// relation's) the crests of the face's points lay far behind the drawn one (a strip judged by those came out as rings),
// and by local rates at two points they still wandered, the strip's far end in fingers. Bounded round the dispersion
// relation's wavenumber at the point's depth (depth: LipBed's), which is smooth and keeps the crest near in shallow
// water: bounded round the warp's, whose Jacobian goes erratic by the waterline, points in the shallows hung from
// crests tens of metres out and the sheet burst into giant shards there. At full detail the phase jittered and the
// lip's sheet crumpled like foil; stepping by the phase without bounds reached the next wave or stopped on the face,
// and the sheet tore into shards between vertices. k: the mean rate between the point and its crest
float2 LipCrest(OceanData ocean, float2 xz, SurfWave wave, float depth, out float k)
{
    bool second = LipSecondBand(wave);
    uint slice = second ? 1u : 0u;
    float phase = max(LipPhase(wave), 0.0);
    float2 heading = second ? wave.heading.zw : wave.heading.xy;
    float omega = second ? ocean.WarpOmega.y : ocean.WarpOmega.x;
    float k0 = omega * omega / 9.81;
    float kDepth = k0 / sqrt(tanh(k0 * max(depth, 0.3)));
    k = clamp(LipPhaseRate(ocean, slice, xz, heading), 0.7 * kDepth, 2.5 * kDepth);
    if (phase < 1e-3) return xz;
    [unroll]
    for (int i = 0; i < 2; ++i)
    {
        float back = phase / k;
        float drop = phase - LipBandPhase(ocean, slice, xz - heading * back);
        drop -= 6.2831853 * floor(drop / 6.2831853);        // the phase falls going back: into [0, 2π)
        k = clamp(drop / back, 0.7 * kDepth, 2.5 * kDepth);
    }
    return xz - heading * (phase / k);
}

// The lip amount a point of a strip goes by (wave: LipWave there): its crest's (LipCrest), where the lip stands, not
// its own. A strip's length ahead of its crest the water is shallower and the point's own amount had run out of the
// breaker's life: the strip was cut short in ragged tongues, and the half-bent face at the cut stood as a shelf. None
// in the swash's water (SWASH_DEPTH), where the ocean is the run-up's sheet. Behind the crest (phase below 0) the point
// is its own crest. Decided once per frame round the camera into the lip map (ocean_lip_map.hlsl), which the ocean's
// cut-out and the lip pass both read. crestXZ, crestWave: the crest and its waves; age: the crest's (LipAmount)
float LipStripAmount(OceanData ocean, float2 xz, SurfWave wave, float2 camXZ, out float2 crestXZ, out SurfWave crestWave,
                     out float k, out float age)
{
    float depth, slope;
    LipBed(ocean, xz, depth, slope);
    crestXZ = LipCrest(ocean, xz, wave, depth, k);
    crestWave = LipWave(ocean, crestXZ);
    OceanShore crestShore = ShoreAt(ocean, crestXZ);
    // The search checked: the crest found must lie where the phase is 0. Where the shallows' phase is too tangled for
    // it, the point is no strip's (by the beach such points hung from crests far off, the sheet in blocky shards)
    float found = LipPhase(wave) <= 0.0 ? 1.0 : 1.0 - smoothstep(0.3, 0.6, abs(LipPhase(crestWave)));
    return LipAmount(ocean, crestXZ, crestWave, crestShore.shoreDepth, distance(crestXZ, camXZ), age)
         * smoothstep(SWASH_DEPTH, 2.5 * SWASH_DEPTH, depth) * found;
}

// The lip map (C# LipMap): LIP_MAP_SIZE² texels of LIP_MAP_TEXEL m from ocean.LipMapOrigin
static const int LIP_MAP_SIZE = 256;
static const float LIP_MAP_TEXEL = 1.0;

// Where a lip stands at xz by the map: x the amount of the crest its strip belongs to (as LipStripAmount, smoothed),
// y the phase's mean rate back to that crest (0 outside a strip: the crest lies phase / y back along the heading), z
// the crest's age (LipAmount). Bilinear by hand: k and the age only over the texels in a strip, and 0 off the map
float3 LipMapAt(OceanData ocean, float2 xz)
{
    if (ocean.LipMapSRV == 0) return 0.0;
    Texture2D<float4> map = ResourceDescriptorHeap[ocean.LipMapSRV];
    float2 t = (xz - ocean.LipMapOrigin) / LIP_MAP_TEXEL - 0.5;
    int2 i = int2(floor(t));
    if (any(i < 0) || any(i >= LIP_MAP_SIZE - 1)) return 0.0;
    float2 f = t - float2(i);
    float3 a = map.Load(int3(i, 0)).xyz, b = map.Load(int3(i + int2(1, 0), 0)).xyz;
    float3 c = map.Load(int3(i + int2(0, 1), 0)).xyz, d = map.Load(int3(i + int2(1, 1), 0)).xyz;
    float4 w = float4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    float amount = dot(w, float4(a.x, b.x, c.x, d.x));
    float4 kw = w * float4(a.y > 0.0, b.y > 0.0, c.y > 0.0, d.y > 0.0);
    float kSum = dot(kw, 1.0);
    if (kSum <= 1e-4) return float3(amount, 0.0, 0.0);
    return float3(amount, dot(kw, float4(a.y, b.y, c.y, d.y)) / kSum, dot(kw, float4(a.z, b.z, c.z, d.z)) / kSum);
}

// The camera distance at which SampleDisplacement's mips have texels of LIP_WAVE_BLUR metres (its texel is camDist ×
// 0.15 / N, N = 512 the FFT's size): the ocean's surface with the small waves averaged out, the swell left. The lip's
// face is shaped from it (ocean_lip.fx)
static const float LIP_SMOOTH_CAM_DIST = LIP_WAVE_BLUR * 512.0 / 0.15;

// The smooth swell's displacement at xz (OceanSurfaceDisplacement at LIP_SMOOTH_CAM_DIST), from the lip map's surface
// texture (ocean_lip_map.hlsl writes it every frame where the coast is). valid: all four texels were written; where
// not, the caller samples the ocean itself
float3 LipSurfaceAt(OceanData ocean, float2 xz, out bool valid)
{
    valid = false;
    if (ocean.LipSurfaceSRV == 0) return 0.0;
    Texture2D<float4> map = ResourceDescriptorHeap[ocean.LipSurfaceSRV];
    float2 t = (xz - ocean.LipMapOrigin) / LIP_MAP_TEXEL - 0.5;
    int2 i = int2(floor(t));
    if (any(i < 0) || any(i >= LIP_MAP_SIZE - 1)) return 0.0;
    float2 f = t - float2(i);
    float4 a = map.Load(int3(i, 0)), b = map.Load(int3(i + int2(1, 0), 0));
    float4 c = map.Load(int3(i + int2(0, 1), 0)), d = map.Load(int3(i + int2(1, 1), 0));
    valid = min(min(a.w, b.w), min(c.w, d.w)) > 0.5;
    return lerp(lerp(a.xyz, b.xyz, f.x), lerp(c.xyz, d.xyz, f.x), f.y);
}

#endif
