// What the ocean's pixel shader reads from its bands at a point of the water, and the surf's white water from it: shared
// by the ocean (ocean.fx) and the breaking lip (ocean_lip.fx), which shades the strip it takes over by the same state at
// the same water (its vertices keep the undisplaced xz of the ocean's surface they bent), so the two meet with no seam.
// Moved out of ocean.fx's PS for the lip (2026-10-19). After ocean_patch.hlsli
#ifndef OCEAN_SHADE_HLSLI
#define OCEAN_SHADE_HLSLI

// Whitecaps: the opacity of the coarsest wind band's foam where the finer bands' crests are not squeezed, and how much
// their squeeze (1 - J, summed) raises it
#define WHITECAP_VEIL 0.08
#define WHITECAP_CRESTS 3.0
#define WHITECAP_THIN 0.03
#define WHITECAP_RAMP 0.55
// Light through the crests: full at this height (m) of the wind sea above the mean, and the share of the light on a
// crest that scatters through it to the eye
#define CREST_GLOW_HEIGHT 3.0
#define CREST_GLOW 0.25
// Bubbles under a whitecap: the share of the light on the water that they send back up, per unit of whitecap foam
#define AERATION 0.1

struct OceanBands
{
    // Slope moments of every band (water_common.fx WaterSlopes): the mean slope makes the normal, the variance of the
    // slopes under this pixel its roughness. NormalAtten scales the slopes (and so the variance by its square)
    WaterSlopes slopes;
    float2 carriedMean;     // the carried bands' share of slopes.mean: the lip's geometry is their wave already
    float resolvedTotal;
    float totalFoam;
    float waveHeight;
    float windHeight;       // the wind sea's part of it (not the swell): what stands up in crests
    uint capBand;           // the whitecap band (0xFFFFFFFF: none)
    float capFoam;          // the whitecap band's foam: where the whitecaps are
    float smallCrests;      // the finer wind bands' crest squeeze now: the structure inside them
    float fineSqueeze;      // the surf's structure: the uncarried wind bands' crest squeeze...
    float fineNorm;         // ...and what squeeze their slopes make at a typical crest
    SurfWave surf;          // the carried bands' waves here
};

//   worldXZ: the water's undisplaced position. shoreAtten, shoreDepth: the shore's damping of the longest waves and the
//   depth that sets each band's share of it (the DS's OceanShore)
OceanBands SampleOceanBands(OceanData ocean, float2 worldXZ, float shoreAtten, float shoreDepth)
{
    OceanBands b = (OceanBands)0;
    Texture2DArray<float4> dispTex = ResourceDescriptorHeap[ocean.DisplacementSRV];
    Texture2DArray<float4> slopeTex = ResourceDescriptorHeap[ocean.SlopeSRV];
    StructuredBuffer<float> tileScales = ResourceDescriptorHeap[ocean.TileScalesSRV];

    // The surf's bands are sampled where their waves were out at sea (SampleBandWarp)
    float2 surfUV = saturate((worldXZ - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ))
                             / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ));

    // Whitecaps (C# WhitecapBand, NoWhitecapBands): only the wind band holding the wind sea's peak, the waves one sees
    // break (the finer the calmer the sea), makes foam; the other wind bands store how squeezed their crests are right
    // now (ocean_fft.hlsl), and the finer ones' light up inside its foam. An older build sends no band: the coarsest
    uint windBands = ocean.WindBands;
    uint capBand = 0xFFFFFFFF;
    float capScale = 1e30;
    [loop]
    for (uint wb = 0; wb < ocean.NumBands; ++wb)
    {
        if (((windBands >> wb) & 1u) && tileScales[wb] < capScale)
        {
            capScale = tileScales[wb];
            capBand = wb;
        }
    }
    if (ocean.WhitecapBand != 0 && ocean.WhitecapBand <= ocean.NumBands)
    {
        capBand = ocean.WhitecapBand - 1;
        capScale = tileScales[capBand];
    }
    b.capBand = capBand;

    [loop]
    for (uint i = 0; i < ocean.NumBands; ++i)
    {
        float angle = BandAngles[i];
        float scale = tileScales[i];

        // The surf's bands where their waves were out at sea, as high as they have grown (DS SampleDisplacement)
        uint slice = WarpSlice(ocean, i);
        float2 p = worldXZ;
        float waveAtten;
        float2x2 J = float2x2(1.0, 0.0, 0.0, 1.0);
        float2 turn = float2(1.0, 0.0), heading = float2(1.0, 0.0);
        if (slice != 0xFF)
        {
            BandWarp w = SampleBandWarp(ocean, slice, worldXZ, surfUV);
            p = w.u;
            waveAtten = w.amp;
            J = w.J;
            turn = w.turn;
            heading = w.heading;
        }
        else
            waveAtten = BandShoreAtten(shoreAtten, shoreDepth, scale);
        float2 uv = RotateUV(p, angle) * scale;

        // Filtered: the foam alpha is only a coverage mask, the visible detail comes from the foam texture
        float4 disp = dispTex.Sample(OceanSampler, float3(uv, i));
        float4 moments = slopeTex.Sample(OceanSampler, float3(uv, i));
        disp *= waveAtten;
        if (slice != 0xFF)
        {
            // The surf's wave: its height and its sideways displacement along the local heading (as the DS has them)
            float2 side = RotateUV(disp.xz, -angle);
            side = float2(side.x * turn.x - side.y * turn.y, side.x * turn.y + side.y * turn.x);
            AddSurfWave(b.surf, ocean, float3(side.x, disp.y, side.y), heading, slice);
        }

        // Counter-rotate the mean slope back to world space (the variance does not depend on the rotation)
        float s, c;
        sincos(-angle, s, c);
        float2 meanWorld = float2(moments.x * c - moments.y * s, moments.x * s + moments.y * c);
        if (slice != 0xFF)
        {
            // Through the warp the slopes are Jᵀ·(the open sea's): the crowded crests are steeper. The variance grows
            // with the mean square stretch
            float slopeScale = ocean.NormalAtten * waveAtten;
            float stretch2 = 0.5 * (J[0][0] * J[0][0] + J[0][1] * J[0][1] + J[1][0] * J[1][0] + J[1][1] * J[1][1]);
            float2 carried = mul(meanWorld, J) * slopeScale;
            b.slopes.mean += carried;
            b.carriedMean += carried;
            b.slopes.variance += max(moments.z + moments.w - dot(moments.xy, moments.xy), 0.0) * slopeScale * slopeScale * stretch2;
            b.resolvedTotal += WaterBandTotalVariance(slopeTex, OceanSampler, i, slopeScale) * stretch2;
        }
        else
        {
            AddWaterBand(b.slopes, moments, meanWorld, ocean.NormalAtten * waveAtten);
            float bandVariance = WaterBandTotalVariance(slopeTex, OceanSampler, i, ocean.NormalAtten * waveAtten);
            b.resolvedTotal += bandVariance;
            if (((windBands >> i) & 1u) && i != capBand)
            {
                b.fineSqueeze += disp.a;
                b.fineNorm += ocean.Choppiness * sqrt(bandVariance);
            }
        }
        b.totalFoam += disp.a;
        b.waveHeight += disp.y;
        if ((windBands >> i) & 1u)
        {
            b.windHeight += disp.y;
            // Weighted by the band's size against the whitecap band's: the 64 m band's crest lines give the structure,
            // the ripples (whose squeeze is the strongest, in round cells) only its finest detail. Unweighted they
            // spotted the whitecaps like a dalmatian
            if (i == capBand)
                b.capFoam = disp.a;
            else if (scale > capScale)
                b.smallCrests += disp.a * sqrt(capScale / scale);
        }
    }
    return b;
}

// ── The surf: foam from the waves themselves (C# ShoreWarp, SurfWave) ──
// Every carried wave knows its own height and phase: once it stands taller than ~0.8 × the depth it has broken, and
// stays broken toward the beach (big waves of a set break further out). Its white water rides its crest and face (the
// roller) and is left behind it, thinning over about a third of a wave, with the bubbles of the broken water all
// through the surf zone. Drawn like the whitecaps, through the finer bands' crests. Until 2026-10-19 the shore's foam
// ran on a clock: bands crossing the swash zone on a noise-staggered schedule (ShoreClock), a hash per wave
// (WaveStrength), a drifting Voronoi web (FoamWeb) and streaks, none of it a wave.
//   trailWeb: how much thinning trail there is, for the web
float SurfWhiteWater(OceanData ocean, OceanBands b, float shoreDepth, float dist, out float trailWeb)
{
    trailWeb = 0.0;
    if (ocean.ShoreWarpSRV == 0 || shoreDepth >= 1e3) return 0.0;

    float breaking = SurfBreaking(b.surf, shoreDepth);
    // Each carried band's roller and trail at its own phase, the weaker band's in proportion to its height
    float2 weight = b.surf.envelope / max(max(b.surf.envelope.x, b.surf.envelope.y), 1e-6);
    float2 rt0 = SurfRollerTrail(b.surf.phase.x) * weight.x;
    float2 rt1 = SurfRollerTrail(b.surf.phase.y) * weight.y;
    float roller = breaking * max(rt0.x, rt1.x);
    float trail = breaking * max(rt0.y, rt1.y) + SURF_BUBBLES * breaking * breaking;
    // The roller is a churning white mass, broken only at its edges; what it leaves behind breaks up along the crests
    float structure = saturate(b.fineSqueeze / max(b.fineNorm, 1e-4));
    float surfFoam = max(roller * saturate(SURF_ROLLER + SURF_CRESTS * structure),
                         trail * saturate(SURF_VEIL + SURF_CRESTS * structure));
    // Up the beach the warp holds the waterline's phase (C# ShoreWarp), lines across the shore that the swash runs on:
    // drawn as white water they made hooks along the waterline (user, 2026-10-19). The bore's white water ends where
    // the water gets that shallow (SURF_FOAM_END), and the swash's edge is the waterline's lace (ocean.fx)
    surfFoam *= smoothstep(SURF_FOAM_END.x, SURF_FOAM_END.y, shoreDepth);
    surfFoam = saturate((saturate(surfFoam) - WHITECAP_THIN) / WHITECAP_RAMP);
    surfFoam *= saturate(1.0 - dist / 3000.0);
    // The trail as it thins, away from the roller: the web takes it over
    trailWeb = saturate(trail * 1.4) * (1.0 - saturate(roller * 2.0))
             * smoothstep(SURF_FOAM_END.x, SURF_FOAM_END.y, shoreDepth);
    return surfFoam;
}

// The whitecaps' opacity. Foam is bubbles: there or not. Thin foam drawn at a fraction of its opacity greyed the water
// like smoke, so the wind sea's foam drops out below WHITECAP_THIN and is solid from WHITECAP_THIN + WHITECAP_RAMP: as
// it thins it breaks up along the small crests' pattern instead of fading as a whole
float OceanWhitecap(OceanBands b, float dist)
{
    float whitecap = b.capBand != 0xFFFFFFFF
        ? saturate((saturate(b.capFoam) * saturate(WHITECAP_VEIL + WHITECAP_CRESTS * b.smallCrests) - WHITECAP_THIN) / WHITECAP_RAMP)
        : saturate(b.totalFoam);
    return whitecap * saturate(1.0 - dist / 3000.0);
}

// The foam's cells: two drifting scales of cellular noise (noise texture has no mips: flatten it far away). Sampled
// where the surface's water rests (the undisplaced position): the bubbles ride the water they float on, out with a
// breaking lip and back with the swash. At the displaced position the white water slid over a fixed pattern.
//   cells2: the finer scale alone (the web's filaments use it)
float OceanFoamCells(Texture2D<float4> noiseTex, float2 foamPos, float t, out float cells2)
{
    float cells1 = noiseTex.SampleLevel(OceanSampler, foamPos * 0.23 + t * float2(0.010, 0.006), 0).g;
    cells2 = noiseTex.SampleLevel(OceanSampler, foamPos * 0.83 - t * float2(0.013, 0.004), 0).g;
    return saturate((cells1 * 0.6 + cells2 * 0.4 - 0.15) / 0.6);
}

#endif
