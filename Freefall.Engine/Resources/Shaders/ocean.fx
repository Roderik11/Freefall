#include "ocean_patch.hlsli"
#include "ocean_shade.hlsli"
// @RenderState(RenderTargets=1)


// SampleNormal + SampleFoam merged into PS loop below

// ═══════════════════════════════════════════════════════════════════════
// HULL SHADER — screen-space adaptive tessellation
// ═══════════════════════════════════════════════════════════════════════

HullConstantOutput PatchConstantFunc(InputPatch<ControlPoint, 3> patch, uint patchID : SV_PrimitiveID)
{
    HullConstantOutput output;

    float3 p0 = patch[0].WorldPos;
    float3 p1 = patch[1].WorldPos;
    float3 p2 = patch[2].WorldPos;

    // Off screen or hidden behind the opaque scene — skip
    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    if (CullTriangle(p0, p1, p2, oceanData[patch[0].InstanceIdx].HiZSRV))
    {
        output.EdgeTess[0] = output.EdgeTess[1] = output.EdgeTess[2] = 0;
        output.InsideTess = 0;
        return output;
    }

    // Screen-space edge length — tessellate proportional to pixel coverage
    // Edge indices: edge[0] = opposite vertex 0 = edge(1,2)
    //               edge[1] = opposite vertex 1 = edge(0,2)
    //               edge[2] = opposite vertex 2 = edge(0,1)
    output.EdgeTess[0] = ScreenSpaceEdgeFactor(p1, p2);
    output.EdgeTess[1] = ScreenSpaceEdgeFactor(p2, p0);
    output.EdgeTess[2] = ScreenSpaceEdgeFactor(p0, p1);

    output.InsideTess = (output.EdgeTess[0] + output.EdgeTess[1] + output.EdgeTess[2]) / 3.0;

    return output;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchConstantFunc")]
ControlPoint HS(InputPatch<ControlPoint, 3> patch, uint cpID : SV_OutputControlPointID)
{
    return patch[cpID];
}

// ═══════════════════════════════════════════════════════════════════════
// DOMAIN SHADER — FFT texture displacement
// ═══════════════════════════════════════════════════════════════════════

struct DSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;       // Displaced world position
    float4 UndisplacedXZ : TEXCOORD1;  // xy: pre-displacement XZ for PS texture sampling, z, w: the shore's damping of the
                                       // longest waves and the depth that sets each band's share of it (BandShoreAtten)
    float Depth : TEXCOORD2;
    nointerpolation uint InstanceIdx : TEXCOORD3;
    float4 SkyAmbient : TEXCOORD4;     // rgb: the sky's light on a level surface (SkyAmbient(up), the sky probe), constant per frame
};

[domain("tri")]
DSOutput DS(HullConstantOutput patchConstants,
            float3 bary : SV_DomainLocation,
            const OutputPatch<ControlPoint, 3> patch)
{
    DSOutput output;

    // Barycentric interpolation of world position
    float3 worldPos = patch[0].WorldPos * bary.x
                    + patch[1].WorldPos * bary.y
                    + patch[2].WorldPos * bary.z;

    uint idx = patch[0].InstanceIdx;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[idx];

    // Camera distance for mip selection
    float3 camPos = ViewInverse[3].xyz;
    float camDist = length(worldPos - camPos);

    // The bands' displacement and near the shore the surf's (ocean_surf.hlsli, shared with the lip)
    OceanShore shore;
    SurfWave surf;
    float3 displacement = OceanSurfaceDisplacement(ocean, worldPos.xz, camDist, shore, surf);
    float3 displaced = worldPos + displacement;

    output.UndisplacedXZ = float4(worldPos.xz, shore.waveAtten, shore.shoreDepth);
    output.WorldPos = displaced;
    output.Position = mul(mul(float4(displaced, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.InstanceIdx = idx;
    output.SkyAmbient = float4(SkyAmbient(float3(0, 1, 0)), 1.0);

    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// PIXEL SHADER — per-pixel normals from FFT slope maps
// ═══════════════════════════════════════════════════════════════════════

struct PSOutput
{
    float4 Color : SV_Target0;
};

// The sea plane is endless, but the sea is only where it can flow in from the open sea across ground below sea level
// (C# SeaMask: flooded from the terrain's border, reaching a little up the beaches). Enclosed low ground, a lake bed
// deeper than the sea or a hall under a hill, gets no sea. Past the terrain it is all sea.
bool OutsideSea(OceanData ocean, float2 xz)
{
    if (ocean.SeaMaskSRV == 0) return false;
    float2 uv = (xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ)) / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ);
    if (any(uv < 0.0) || any(uv > 1.0)) return false;

    Texture2D<float> mask = ResourceDescriptorHeap[ocean.SeaMaskSRV];
    uint w, h;
    mask.GetDimensions(w, h);
    uv = clamp(uv, 0.5 / float2(w, h), 1.0 - 0.5 / float2(w, h));   // no blending in the far border's texels
    return mask.SampleLevel(OceanSampler, uv, 0) < 0.5;
}

PSOutput PS(DSOutput input)
{
    PSOutput output;

    float3 camPos = ViewInverse[3].xyz;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[input.InstanceIdx];

    if (OutsideSea(ocean, input.UndisplacedXZ.xy))
        discard;

    float3 worldPos = input.WorldPos;
    // How much sky this pixel of sea sees past the terrain (per pixel, as water.fx: per vertex, corners on land darkened the coast).
    // The terrain only: the sky captures' 16 lookups per pixel cost 0.29 ms where the sea fills the view (the coast),
    // and over open water little but a pier stands in the sky's way
    float skyVisible = SkyVisibilityWith(worldPos, float3(0, 1, 0), 1.0);
    float3 V = normalize(camPos - worldPos);
    float dist = distance(worldPos, camPos);

    // ── The bands at this water (ocean_shade.hlsli, shared with the lip) ──
    float2 worldXZ = input.UndisplacedXZ.xy;
    // The shore flattens each band's waves by its length (DS, BandShoreAtten): their slopes, crests and foam shrink with
    // them. Until 2026-10-18 only the heights did, and over the shelf a nearly flat sea was shaded as the full open-sea chop
    float shoreAtten = input.UndisplacedXZ.z;
    float shoreDepth = input.UndisplacedXZ.w;
    OceanBands bands = SampleOceanBands(ocean, worldXZ, shoreAtten, shoreDepth);
    WaterSlopes slopes = bands.slopes;
    float waveHeight = bands.waveHeight;
    SurfWave surfWave = bands.surf;

    // The DS's depth limit flattens the shoaled crests (DepthLimit): their slopes with them
    if (ocean.ShoreWarpSRV != 0 && shoreDepth < 1e3)
    {
        float capSlope;
        waveHeight = DepthLimit(waveHeight, shoreDepth, capSlope);
        slopes.mean *= capSlope;
        slopes.variance *= capSlope * capSlope;

        // Where a breaking lip stands, the lip pass (ocean_lip.fx) draws the strip ahead of the crest, the curl and a
        // steep face under it: this gentle face would stand in front of the tube. By the same test on the same blurred
        // waves (LipWave, LipStripAmount: the crest's amount), so they meet; only where a wave stands anywhere near
        // breaking by the full detail (a strip's far end lies in shallower water than its crest), so the rest of the
        // coast pays nothing for it
        float nearBreaking = 2.0 * SurfEnvelope(surfWave) / max(shoreDepth, 0.05);
        if (ocean.WarpOmega.x + ocean.WarpOmega.y > 0.0 && nearBreaking > LIP_LIFE.x - 0.25 && nearBreaking < 2.0)
        {
            SurfWave lipWave = LipWave(ocean, worldXZ);
            float lipPhase = LipPhase(lipWave);
            [branch]
            if (lipPhase > 0.0 && lipPhase < LIP_STRIP)
            {
                float2 lipCrestXZ;
                SurfWave lipCrestWave;
                float lipK;
                if (LipStripAmount(ocean, worldXZ, lipWave, camPos.xz, lipCrestXZ, lipCrestWave, lipK) > LIP_MIN)
                    discard;
            }
        }
    }

    float3 N = normalize(float3(-slopes.mean.x, 1.0, -slopes.mean.y));
    float alpha = WaterAlpha(slopes, WaterUnresolvedVariance(bands.resolvedTotal));

    float NdotV = saturate(dot(N, V));

    // ── Sun vectors ──
    float3 sunDir = normalize(ocean.SunDirection);
    float3 L = normalize(-sunDir);
    float NdotL = saturate(dot(N, L));
    // The directional light (sun or moon) in the older units, pre-exposed: what a white Lambert surface facing it
    // sends back, as in the deferred light (E / π). The specular BRDF takes the illuminance itself (× π): until
    // 2026-10-17 it got this radiance, and the water's sun was π times weaker than the deferred one.
    float3 sunRadiance = ocean.SunColor * ocean.SunIntensity * SunLightScale;
    sunRadiance *= GetCloudShadow(OceanSampler, worldPos - camPos, L);
    // The cascades, and the terrain's shadow past them (a headland the sun stands behind), as the deferred light has them
    sunRadiance *= min(WaterSunShadow(ShadowMapIdx, worldPos - camPos, input.Position.xy), TerrainLightShadow(worldPos, L));
    sunRadiance *= TerrainOpen(worldPos, float3(0, 1, 0));   // a sea cave under a hill: no sun, whatever the shadow map holds

    float H = max(0.0, waveHeight);

    // ── Base color: dark deep ocean ──
    float depthBlend = saturate(dist / 4000.0);
    float3 baseColor = lerp(ocean.OceanColor, ocean.DeepColor, depthBlend);

    // ── Subsurface scattering ──
    float3 bubbleColor = float3(0.0, 0.008, 0.006);

    float k1 = 0.3 * H * pow(saturate(dot(L, -V)), 4.0)
             * pow(0.5 - 0.5 * dot(L, N), 3.0);
    float k2 = 0.1 * NdotV * NdotV;
    float k3 = 0.08 * NdotL;

    float3 scatter = ((k1 + k2) * baseColor + k3 * baseColor + 0.02 * bubbleColor) * sunRadiance
                   + baseColor * WaterSkyIrradiance(input.SkyAmbient.rgb * skyVisible) * 0.1;   // light from the sky above, too

    // ── Reflectance, sky reflection and sun glint: shared with lakes and rivers (water_common.fx) ──
    // All from the mean normal and the slope variance under the pixel: no distance ramps.
    float F = WaterSpecularReflectance(NdotV, alpha);
    float3 reflectColor = WaterSkyReflection(V, N, alpha) * skyVisible;   // no sky to mirror under a cliff
    float3 specular = WaterSunSpecular(N, V, L, sunRadiance * 3.14159, alpha);

    // ── Water column: what is seen through the surface ──
    // pathLen  = distance the view ray travels under water before it hits the scene (depth buffer)
    // bedDepth = vertical water depth right under this pixel (terrain heightmap, view independent)

    // Terrain seabed. Its geometry and heightmap stop at the terrain border while the sea goes on, so
    // everything derived from it is faded out across the border (seabedFade) to avoid a straight seam:
    // show-through over the last 60 m inside, the heightmap depth over the first 150 m outside.
    float terrainDepth = 1000.0;
    float stillDepth = 1000.0;      // depth below the undisturbed sea level (no waves, no surge)
    float seabedFade = 1.0;
    float shallowFade = 0.0;
    float2 terrainSize = float2(ocean.TerrainSizeX, ocean.TerrainSizeZ);
    float2 clampedUV = 0.5;
    if (ocean.HeightmapSRV != 0)
    {
        float2 terrainUV = (worldPos.xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ)) / terrainSize;
        clampedUV = clamp(terrainUV, 0.002, 0.998);
        float outside = length((terrainUV - clampedUV) * terrainSize);
        float2 toBorder = min(clampedUV, 1.0 - clampedUV) * terrainSize;

        seabedFade = smoothstep(0.0, 60.0, min(toBorder.x, toBorder.y));
        if (outside < 150.0)
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float h = heightmap.SampleLevel(OceanSampler, clampedUV, 0);
            float terrainY = ocean.TerrainBaseY + h * ocean.MaxTerrainHeight;
            terrainDepth = worldPos.y - terrainY;
            stillDepth = ocean.OceanPlaneY - terrainY;
            // The turquoise shelf is wherever the seabed is shallow (no floor rule since 2026-10-19, as in the DS)
            shallowFade = 1.0 - smoothstep(0.0, 150.0, outside);
            // Under the land: where the terrain stands more than 3 m above the still sea (far past any wave's run-up)
            // the sea shows only where the terrain does not cover what is under it, a sea cave under a hill or a
            // terrain hole. That is no beach: read as one, it was a sheet of "shore" foam. Open water there
            if (stillDepth < -3.0)
            {
                terrainDepth = 1000.0;
                stillDepth = 1000.0;
                shallowFade = 0.0;
            }
        }
    }

    WaterColumn column = GetWaterColumn(OceanSampler, ocean.DepthGBufferSRV, ocean.CompositeSRV,
        input.Position.xy * ocean.InvViewportSize, input.Depth, dist, abs(camPos.y - worldPos.y), N,
        ocean.RefractionStrength, scatter, ocean.ShallowColor, ocean.ShoreFadeDepth, seabedFade);
    float3 body = column.body;
    float edgeFade = column.edgeFade;
    float pathLen = column.pathLen;
    float bufDepth = column.bufDepth;

    // Vertical water depth under the pixel: heightmap where there is one, depth buffer otherwise
    float bedDepth = (terrainDepth < 999.0) ? terrainDepth : bufDepth;

    // Light scattered back from a bright shallow bed: the turquoise shelf along the coast
    float3 skyAmbient = input.SkyAmbient.rgb * skyVisible;
    float shallow = exp(-max(bedDepth, 0.0) / max(0.01, ocean.ShoreFadeDepth)) * saturate(pathLen);
    shallow *= (ocean.HeightmapSRV != 0) ? shallowFade : 1.0;
    body += ocean.ShallowColor * (sunRadiance * saturate(L.y) + WaterSkyIrradiance(skyAmbient)) * 0.08 * shallow;

    // Light through the crests: a wind wave's crest is a thin wall of water. Sky light (and the sun's, strongest when
    // looking toward it) entering its back and top scatters through it toward the eye, tinted by what ~1.5 m of water
    // lets through (ShallowColor is its per-metre transmittance): the teal glow of a storm sea's crests. It grows with the
    // crest's height above the mean (wind sea only: the long swell is thick) and on faces seen at a slant. Under a storm
    // sky the sun-only term above (k1) left the whole sea a dark sky mirror.
    float crestRise = smoothstep(CREST_GLOW_HEIGHT * 0.25, CREST_GLOW_HEIGHT, bands.windHeight);
    crestRise *= crestRise;
    float slant = 1.0 - NdotV;
    float3 throughCrest = pow(max(ocean.ShallowColor, 0.02), 2.5);
    throughCrest = lerp(dot(throughCrest, float3(0.2126, 0.7152, 0.0722)), throughCrest, 0.7);   // storm water is murky
    float3 crestLight = skyAmbient + sunRadiance * (0.2 + 3.0 * pow(saturate(dot(L, -V)), 4.0));
    body += crestLight * throughCrest * (CREST_GLOW * crestRise * lerp(0.3, 1.0, slant * slant)) * skyVisible;

    // Under a whitecap the water is full of the air the breaking crest carried down: the bubbles scatter the light
    // back up, pale turquoise water under the foam and behind it that lasts as long as the whitecap band's foam does
    // (its foam is the measure, as for the whitecaps below)
    body += (skyAmbient + sunRadiance * 0.3) * throughCrest * (AERATION * saturate(bands.capFoam)) * skyVisible;

    // ── Combine lighting ──
    float3 color = (1.0 - F) * body + (specular + F * reflectColor) * edgeFade;
    color = max(0.0, color);

    // ── Foam ──
    // Open water: foam's opacity comes straight from the bands' Jacobian foam, as in the reference implementation (the
    // GarrettGunnell port, 2026-02-28). From 2026-10-04 it went through the shore's coverage path below (eroded by the
    // cells, a soft fill past 150 m), which turned saturated foam into solid white sheets with holes and far foam into
    // blobs (user, 2026-10-18). With a wind sea the coarsest wind band's foam is the mask of where the whitecaps are,
    // and inside it the finer bands' crests, squeezed right now, raise the opacity: the small waves' crests light up
    // in the whitecap and move with it, and as the whitecap fades only they remain, a lace of the sea's own pattern
    // (the user's idea, 2026-10-18). Outside a whitecap they show nothing: on their own the finer bands' foam came up
    // as random blobs where no visible crest is. Without wind bands every band's foam is summed as in the reference.
    // The foam lives in mipmapped textures, so far away it averages out instead of aliasing.
    float t = ocean.WaveTime;
    // The whitecaps, and the surf's white water from the waves themselves (ocean_shade.hlsli, shared with the lip)
    float whitecap = OceanWhitecap(bands, dist);
    float surfTrailWeb;     // how much thinning trail there is, for the web (below)
    float surfFoam = SurfWhiteWater(ocean, bands, shoreDepth, dist, surfTrailWeb);

    // The waterline: foam gathers where the water meets the sand, a rock or a pier, a ragged band that swells and pinches
    // along the coast; on a beach the swash's edge, crisp as a bore runs up and thinner as it drains. Everything below
    // builds a COVERAGE value, eroded by a cellular foam texture so the edge breaks up into bubbles.
    float coverage = 0.0;
    float foamTex = 0.5;
    float laceTex = 0.5;
    float streakFoam = 0.0;
    if (ocean.NoiseSRV != 0)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[ocean.NoiseSRV];
        float frontBroad = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.045 + t * 0.004, 0).r;
        float frontFine = noiseTex.SampleLevel(OceanSampler, worldPos.xz * 0.21 - t * 0.006, 0).a;
        float edgeWidth = lerp(0.4, 2.4, saturate((frontBroad - 0.3) / 0.4)) * lerp(0.7, 1.3, frontFine);

        // The beach's steepness from the heightmap, over a 3 m baseline so it stays smooth: it converts the edge's width
        // along the beach into water depth, so the edge is as wide on a steep shore as on a flat one
        float beachSlope = 0.1;
        float2 downhill = float2(0.0, 1.0);
        float laceDepth = min(bedDepth, bufDepth);
        bool hasBeach = ocean.HeightmapSRV != 0 && terrainDepth < 999.0;
        if (hasBeach && laceDepth < 1.0)
        {
            Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
            float2 slopeStep = 3.0 / terrainSize;
            float2 uphill = float2(
                heightmap.SampleLevel(OceanSampler, clampedUV + float2(slopeStep.x, 0), 0)
              - heightmap.SampleLevel(OceanSampler, clampedUV - float2(slopeStep.x, 0), 0),
                heightmap.SampleLevel(OceanSampler, clampedUV + float2(0, slopeStep.y), 0)
              - heightmap.SampleLevel(OceanSampler, clampedUV - float2(0, slopeStep.y), 0));
            float steepness = length(uphill);
            downhill = -uphill / max(steepness, 1e-6);
            beachSlope = clamp(steepness * ocean.MaxTerrainHeight / 6.0, 0.03, 1.0);
        }
        static const float LaceWidth = 0.45;     // meters along the beach, before the variation (0.7 until 2026-10-19)
        float lace = 1.0 - smoothstep(0.0, min(LaceWidth * edgeWidth * beachSlope, 0.7), laceDepth);

        // On a beach the edge goes by the swash's cycle (the arriving wave's phase, as SwashLevel): crisp on the run-up,
        // thinner while the water drains. Objects standing in deeper water keep their lace.
        if (ocean.ShoreWarpSRV != 0 && hasBeach)
        {
            float2 share = SurfShare(surfWave);
            float advancing = SwashAdvancing(surfWave.phase.x) * share.x + SwashAdvancing(surfWave.phase.y) * share.y;
            float onBeach = 1.0 - smoothstep(0.0, 0.5, bedDepth);
            lace *= lerp(1.0, lerp(0.45, 1.0, advancing), onBeach);
        }

        // Stays below full coverage so the foam texture always breaks it up
        coverage = saturate(lace * 0.8 * ocean.ShoreFoam);

        // Foam texture: the foam's cells where the surface's water rests (OceanFoamCells)
        float2 foamPos = input.UndisplacedXZ.xy;
        float cells2;
        foamTex = OceanFoamCells(noiseTex, foamPos, t, cells2);

        // The waterline's foam is fine: a ragged line of small bubbles (0.3 to 0.6 m), not the coarse cells above. Cut by
        // those it broke into round clusters along the edge like paw prints (user, 2026-10-19)
        float fine1 = noiseTex.SampleLevel(OceanSampler, foamPos * 1.75 - t * float2(0.013, 0.004), 0).g;
        float fine2 = noiseTex.SampleLevel(OceanSampler, foamPos * 3.7 + t * float2(0.009, 0.011), 0).g;
        laceTex = lerp(saturate((fine1 * 0.55 + fine2 * 0.45 - 0.15) / 0.6), 0.5, saturate(dist / 45.0));

        // The web (FoamWeb): the net of bubble films foam breaks into as it thins, where and when the waves leave thin
        // foam: the film of each wave's backwash draining off the sand, strongest at the top of its run-up and gone by
        // the time the next bore arrives, and the thinning trail behind the surf's broken crests. Its look only is
        // made up; until 2026-10-19 a clock decided when it showed too, and it went with the clock that day. Only near
        // the camera: far away the cells would only alias
        float film = 0.0;
        float rise = 1.0;
        if (ocean.ShoreWarpSRV != 0 && hasBeach)
        {
            float2 share = SurfShare(surfWave);
            float age0 = SwashBackwash(surfWave.phase.x), age1 = SwashBackwash(surfWave.phase.y);
            float fresh = (age0 < 0.0 ? 0.0 : 1.0 - smoothstep(0.1, 0.9, age0)) * share.x
                        + (age1 < 0.0 ? 0.0 : 1.0 - smoothstep(0.1, 0.9, age1)) * share.y;
            film = fresh * (1.0 - smoothstep(0.0, max(0.15, ocean.ShoreSurge * 1.5), bedDepth));   // in thin water only
            rise = SwashRise(surfWave.phase.x) * share.x + SwashRise(surfWave.phase.y) * share.y;
        }
        float webAmount = max(film, surfTrailWeb) * (1.0 - saturate(dist / 150.0));
        if (webAmount > 0.01)
        {
            // It floats on the water: the backwash's film is carried down the beach as the water drains and back up with
            // the next bore (a bounded shift that follows the swash, so a flow that bends along the coast cannot shear
            // it), the trail's is stretched along the waves' heading
            bool backwash = film >= surfTrailWeb;
            float2 flowDir = backwash ? downhill : SurfHeading(surfWave);
            float swashZone = 1.0 - smoothstep(0.0, 2.0 * SWASH_DEPTH, stillDepth);
            float waterTravel = min(2.0 * ocean.ShoreSurge / beachSlope, 8.0) * 0.6 * swashZone;
            float2 webBase = foamPos - flowDir * (backwash ? waterTravel * (1.0 - rise) : 0.0);

            const float webCellSize = 0.95;    // meters across a cell (before stretching and merging)
            const float webStretch = 2.0;      // cells are this much longer along the flow
            const float webRate = 0.9;         // how fast the cell points drift (radians per second of wave time)
            const float webDrop = 0.33;        // a fixed third of the points retired, for uneven cell sizes

            // Bend the space first: a large slow warp squeezes some regions and opens others (uneven cell sizes), a
            // medium one curves the strands, a little fine wobble roughens them
            float2 warpUV = webBase * 0.19;
            float2 bigWarp = float2(noiseTex.SampleLevel(OceanSampler, warpUV * 0.31 + 0.27, 0).r,
                                    noiseTex.SampleLevel(OceanSampler, warpUV * 0.31 + 0.83, 0).r) - 0.5;
            float2 warp = float2(noiseTex.SampleLevel(OceanSampler, warpUV + t * 0.006, 0).r,
                                 noiseTex.SampleLevel(OceanSampler, warpUV + 0.43 - t * 0.005, 0).r) - 0.5;
            float2 wobble = float2(noiseTex.SampleLevel(OceanSampler, warpUV * 2.9 + 0.11, 0).a,
                                   noiseTex.SampleLevel(OceanSampler, warpUV * 2.9 + 0.71, 0).a) - 0.5;
            float2 webPos = webBase + bigWarp * 4.5 + warp * 1.8 + wobble * 0.12;

            // A soft mask, 1 on a strand's centre line, swelling into knots where cells meet (thin hard lines of even
            // brightness read as drawn cracks), with a finer, fainter web inside the cells
            float2 web = FoamWeb(webPos / webCellSize, flowDir, webStretch, t * webRate, webDrop);
            float webLace = saturate(1.0 - web.x / (0.09 + 0.17 * smoothstep(0.4, 0.9, web.y)));
            float2 fine = FoamWeb(webPos / (webCellSize * 0.4) + 17.3, flowDir, webStretch, t * webRate * 1.3, webDrop);
            webLace = max(webLace, saturate(1.0 - fine.x / 0.11) * 0.6 * smoothstep(0.2, 0.55, web.x));

            // Patchy: it thins out and tears open in places instead of covering the water evenly
            float patches = noiseTex.SampleLevel(OceanSampler, webBase * 0.06 - t * 0.004, 0).b;
            webLace *= lerp(0.25, 1.0, smoothstep(0.38, 0.62, patches));

            // Strands are made of bubbles: the fine cells erode the soft mask, so the cores stay solid and the edges
            // break into ragged clusters (cells ~7 cm; flattened before they start to shimmer)
            float bubbles = noiseTex.SampleLevel(OceanSampler, webBase * 1.75 - t * float2(0.013, 0.004), 0).g;
            bubbles = lerp(saturate((bubbles - 0.15) / 0.55), 0.5, saturate(dist / 45.0));
            float webFoam = saturate((webLace * 1.35 - (1.0 - bubbles) * 0.75) * 2.2);

            // Translucent filaments, lightly broken up by the fine cells
            streakFoam = webFoam * webAmount * 0.85 * lerp(0.75, 1.0, cells2) * ocean.ShoreFoam;
        }
    }

    // Thin foam is translucent, thick foam keeps the cell structure as shading.
    // The noise texture has no mips: far away the cells would only alias, so fade to a soft
    // version of the coverage itself.
    float foamFar = saturate(dist / 150.0);
    float foam = saturate((coverage * 1.25 - (1.0 - laceTex)) * 3.0);
    foam = lerp(foam, saturate((coverage - 0.35) * 1.5) * 0.7, foamFar);
    // The surf's white water: a mass of bubbles, eroded more gently than the waterline's thin lace (cut as hard, its
    // edges looked like paper), solid where it is thick; far away as it is (the lace's fade is for thin lines)
    float surfNear = saturate((surfFoam * 1.3 - (1.0 - foamTex)) * 2.0);
    foam = max(foam, lerp(surfNear, surfFoam, foamFar));
    foam = max(foam, streakFoam * (1.0 - foamFar));   // the web of thinning foam
    float thickness = saturate(max(coverage, surfFoam));

    // Always feather the last few centimeters of water: whatever the foam pattern, it must not end
    // in the hard pixel edge where the water surface cuts into the beach.
    float feather = smoothstep(0.0, 0.025, pathLen);
    foam *= feather;
    whitecap *= feather;
    foamTex = lerp(foamTex, 0.5, foamFar);
    if (foam > 0.001 || whitecap > 0.001)
    {
        float3 skyOnFoam = SkyAmbient(N) * skyVisible;
        color = lerp(color, WaterFoamLit(float3(0.80, 0.80, 0.78), NdotL, sunRadiance, skyOnFoam), whitecap);
        // Thick foam is bright, thinning foam greys as the water shows through its bubbles
        float3 foamAlbedo = float3(0.80, 0.80, 0.78) * lerp(0.65, 1.0, foamTex) * lerp(0.78, 1.05, thickness);
        color = lerp(color, WaterFoamLit(foamAlbedo, NdotL, sunRadiance, skyOnFoam), foam * 0.9);
    }

    // ── The air between the camera and the water (the atmosphere's aerial perspective) ──
    color = ApplyAerialPerspective(color, input.Position.xy, input.Depth, skyVisible);

    // HDR output — no tonemapping/gamma here.
    // The finalize pass applies ACES + gamma to the entire Composite.

    output.Color = float4(color, 1.0);

    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// SHADOW PASS — tessellated, FFT displacement only
// ═══════════════════════════════════════════════════════════════════════

struct ShadowDSOutput { float4 Position : SV_POSITION; };

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchConstantFunc")]
ControlPoint HS_Shadow(InputPatch<ControlPoint, 3> patch, uint cpID : SV_OutputControlPointID)
{
    return patch[cpID];
}

[domain("tri")]
ShadowDSOutput DS_Shadow(HullConstantOutput patchConstants,
                         float3 bary : SV_DomainLocation,
                         const OutputPatch<ControlPoint, 3> patch)
{
    ShadowDSOutput output;

    float3 worldPos = patch[0].WorldPos * bary.x
                    + patch[1].WorldPos * bary.y
                    + patch[2].WorldPos * bary.z;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[patch[0].InstanceIdx];

    // Without the surf's warp: the sun's shadow of the water near the coast is the open sea's
    SurfWave surf;
    float3 displacement = SampleDisplacement(ocean, worldPos.xz, 1.0, 0.0, 1.0, 1e4, 0.5, false, surf);
    float3 displaced = worldPos + displacement;

    output.Position = mul(mul(float4(displaced, 1.0), View), Projection);
    return output;
}

// ═══════════════════════════════════════════════════════════════════════
technique11 GBuffer
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Control()));
        SetHullShader(CompileShader(hs_6_6, HS()));
        SetDomainShader(CompileShader(ds_6_6, DS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
