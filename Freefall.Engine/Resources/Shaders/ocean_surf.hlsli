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

// How far a breaking face leans forward over its trough, per metre of crest height (m/m)
#define SURF_LEAN 1.6
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

#endif
