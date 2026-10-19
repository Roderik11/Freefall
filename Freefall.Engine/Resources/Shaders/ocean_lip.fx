// The breaking lip and barrel of the surf (.agent/knowledge/ocean_system/surf_lip_plan.md, stage 1: the shape, one
// plain water colour). The ocean is a heightfield and cannot overhang, and at the depth the waves break in they are tens
// of metres long, their faces gentle (a quarter wave ahead of the crest the face has fallen only a quarter of the
// wave's height): any compact tube would lie inside the water. So this draws a small grid of its own round the camera
// and bends the strip ahead of each plunging crest into the breaker's cross-section: a curl thrown off the crest top,
// round a tube, and a steep face under it down to the trough ahead. The ocean leaves its own face out there (ocean.fx
// PS, LipAmount), and where the lip fades both fall back onto the ocean's own surface. From the waves at that point
// (ocean_surf.hlsli), as the surf: no history, no clock.
#include "ocean_patch.hlsli"
#include "ocean_shade.hlsli"
// @RenderState(RenderTargets=1, CullMode=None)

// The grid: 64 x 64 cells (OceanRenderer.LipGridSize) over twice this half size round the camera, snapped to a cell
static const float LipGridHalf = 128.0;
static const float LipCell = 4.0;
// The strip is the breaker's cross-section as one outline, so it joins the ocean's back of the wave at the crest and
// its trough ahead with no seam: from the crest over the curl's outer side to the tip (to this phase), back under the
// lip along its inner side to the root (to this one), and down the face to the trough (the rest of LIP_STRIP). With the
// tip at the strip's crest edge instead, the sheet tore there and the skirt spanned a web from the crest to the tip
static const float LipOuterEnd = 0.41 / LIP_STRIP;
static const float LipInnerEnd = 0.72 / LIP_STRIP;
// The curl's shape (ocean_surf.hlsli, shared with the spray thrown from its tip)
static const float LipRadius = LIP_RADIUS;
static const float2 LipCentre = LIP_CENTRE;
static const float LipSweep = LIP_SWEEP;
static const float LipTipShrink = LIP_TIP_SHRINK;
// The lip's thickness at its root, in radii (it thins to nothing at the tip)
static const float LipThickness = 0.4;
// The skirt round the lip: how far (radians of phase) it reaches past the strip's crest end and its trough end, down to
// which LipAmount beside it (LIP_MIN is the lip's own edge), how far under the ocean's surface it hangs (m) behind the
// crest and elsewhere
static const float2 LipSkirt = float2(0.2, 0.6);
static const float LipSkirtEdge = 0.0;
static const float2 LipSkirtDrop = float2(0.25, 0.08);
// Edges of this many pixels, at most 64 segments: the curl needs many vertices across a strip a few metres wide
static const float LipPixelsPerEdge = 7.0;

// ═══════════════════════════════════════════════════════════════════════
// CONTROL POINTS — the lip's own grid round the camera (the mesh's transform is the ocean's and is not used)
// ═══════════════════════════════════════════════════════════════════════

ControlPoint VS_Lip(uint primitiveVertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    ControlPoint output;

    StructuredBuffer<uint> sortedIndices = ResourceDescriptorHeap[SortedIndicesIdx];
    StructuredBuffer<MeshPartEntry> meshRegistry = ResourceDescriptorHeap[MeshRegistryIdx];
    MeshPartEntry part = meshRegistry[MeshPartId];
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    uint vertexID = indices[primitiveVertexID + part.BaseIndex];
    StructuredBuffer<float3> positions = ResourceDescriptorHeap[part.PosBufferIdx];
    uint idx = sortedIndices[InstanceBaseOffset + instanceID] & 0x7FFFFFFFu;

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    float2 centre = floor(ViewInverse[3].xz / LipCell) * LipCell;     // snapped, or the tessellation would swim
    float3 p = positions[vertexID];
    output.WorldPos = float3(centre.x + p.x * LipGridHalf, oceanData[idx].OceanPlaneY, centre.y + p.z * LipGridHalf);
    output.InstanceIdx = idx;
    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// HULL SHADER — only patches where a carried wave breaks, tessellated finely
// ═══════════════════════════════════════════════════════════════════════

// By the edge's length in the world against a pixel's size at its distance (WorldEdgeFactor). Measured on screen, low
// views got a few segments per patch, the sheet's edges ran in patch-sized stairs and missed the ocean's per-pixel
// cut-out (the seabed showed through along the seams)
float LipEdgeFactor(float3 p0, float3 p1)
{
    return WorldEdgeFactor(p0, p1, LipPixelsPerEdge, 64.0);
}

// Whether a lip stands at xz, its skirt included: by the lip map, as the ocean's cut-out decides (the map's smoothing
// spreads a strip's amount a few metres past it). The patch's corners, edge middles and middle are tested: with the
// corners and middle alone the ocean cut itself out inside a patch tested empty, and the seabed showed through
float LipBreakingAt(OceanData ocean, float2 xz)
{
    return LipMapAt(ocean, xz).x > 0.0 ? 1.0 : 0.0;
}

HullConstantOutput LipPatchConstantFunc(InputPatch<ControlPoint, 3> patch, uint patchID : SV_PrimitiveID)
{
    HullConstantOutput output = (HullConstantOutput)0;      // factors 0: the patch is left out

    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[patch[0].InstanceIdx];
    if (ocean.ShoreWarpSRV == 0 || ocean.HeightmapSRV == 0 || ocean.LipMapSRV == 0) return output;

    float3 p0 = patch[0].WorldPos, p1 = patch[1].WorldPos, p2 = patch[2].WorldPos;
    float3 mid = (p0 + p1 + p2) / 3.0;
    if (distance(mid.xz, ViewInverse[3].xz) > LIP_FADE.y + LipCell) return output;
    if (CullTriangle(p0, p1, p2, ocean.HiZSRV)) return output;

    float breaking = max(max(LipBreakingAt(ocean, p0.xz), LipBreakingAt(ocean, p1.xz)), LipBreakingAt(ocean, p2.xz));
    breaking = max(breaking, max(LipBreakingAt(ocean, 0.5 * (p0.xz + p1.xz)), LipBreakingAt(ocean, 0.5 * (p1.xz + p2.xz))));
    breaking = max(breaking, max(LipBreakingAt(ocean, 0.5 * (p2.xz + p0.xz)), LipBreakingAt(ocean, mid.xz)));
    if (breaking <= 0.0) return output;

    output.EdgeTess[0] = LipEdgeFactor(p1, p2);
    output.EdgeTess[1] = LipEdgeFactor(p2, p0);
    output.EdgeTess[2] = LipEdgeFactor(p0, p1);
    output.InsideTess = (output.EdgeTess[0] + output.EdgeTess[1] + output.EdgeTess[2]) / 3.0;
    return output;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("LipPatchConstantFunc")]
ControlPoint HS_Lip(InputPatch<ControlPoint, 3> patch, uint cpID : SV_OutputControlPointID)
{
    return patch[cpID];
}

// ═══════════════════════════════════════════════════════════════════════
// DOMAIN SHADER — the strip ahead of a crest bent into the curl
// ═══════════════════════════════════════════════════════════════════════

struct LipOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPos : TEXCOORD0;
    float3 Normal : TEXCOORD1;          // the water's outward side
    float2 Lip : TEXCOORD2;             // x: along the outline (0 at the crest, 1 in the trough ahead); y: how far it is thrown
    float Depth : TEXCOORD3;
    float4 SkyAmbient : TEXCOORD4;
    nointerpolation uint InstanceIdx : TEXCOORD5;
    float4 UndisplacedXZ : TEXCOORD6;   // the ocean's water this vertex bent, as the ocean's DS hands it to its PS: xz, the
                                        // shore's damping of the longest waves, the depth
    float4 Sheet : TEXCOORD7;           // xy: the lip's heading; z: the water a ray crosses through the sheet (m), thin
                                        // toward the tip; w: how far inside the tube (sky shut out by the lip)
    float4 Surf : TEXCOORD8;            // the carried bands' amplitude and phase at the vertex's own water (its foam)
    float Splash : TEXCOORD9;           // the white water thrown up where the lip lands, 0..1
    float Clip : SV_ClipDistance0;      // < 0 off the strip or where no lip is: the sheet's edges follow the wave
};

// The ocean's surface with the small waves averaged out, the swell left (ocean_surf.hlsli)
static const float LipSmoothCamDist = LIP_SMOOTH_CAM_DIST;
// 1: the smooth swell from the lip map's surface texture (LipSurfaceAt), written once per frame; 0: sampled from the
// ocean at every vertex, four points each (0.14 ms more with a breaker filling the view)
#define LIP_SURFACE_MAP 1

// The smooth swell's displacement at the four points the face is shaped from: the vertex's own water, the crest, the
// strip's end and 1 m past it. From the map where all four texel squares were written, else from the ocean
void LipSmoothSurfaces(OceanData ocean, float2 p0, float2 p1, float2 p2, float2 p3,
                       out float3 d0, out float3 d1, out float3 d2, out float3 d3)
{
    bool v0 = false, v1 = false, v2 = false, v3 = false;
    d0 = d1 = d2 = d3 = 0.0;
#if LIP_SURFACE_MAP
    d0 = LipSurfaceAt(ocean, p0, v0);
    d1 = LipSurfaceAt(ocean, p1, v1);
    d2 = LipSurfaceAt(ocean, p2, v2);
    d3 = LipSurfaceAt(ocean, p3, v3);
#endif
    [branch]
    if (!(v0 && v1 && v2 && v3))
    {
        // All four from the same source: the face's detail is the vertex's water less the swell there, and it meets
        // the ocean at the strip's end only where both swells are the same function
        OceanShore s;
        SurfWave w;
        d0 = OceanSurfaceDisplacement(ocean, p0, LipSmoothCamDist, s, w);
        d1 = OceanSurfaceDisplacement(ocean, p1, LipSmoothCamDist, s, w);
        d2 = OceanSurfaceDisplacement(ocean, p2, LipSmoothCamDist, s, w);
        d3 = OceanSurfaceDisplacement(ocean, p3, LipSmoothCamDist, s, w);
    }
}

// The face from its root (v = 0) down into the trough (v = 1): almost straight down at first, flattening as it runs
// forward, arriving at the ocean's own slope (the cubic term: nothing at the root, the slope's difference at the end)
float3 LipFaceCurve(float3 runFlat, float runY, float endRise, float v)
{
    return runFlat * pow(v, 2.5) + float3(0.0, runY * (1.0 - (1.0 - v) * (1.0 - v)) + endRise * v * v * (v - 1.0), 0.0);
}

// Its length from the root to v, in a few straight steps
float LipFaceLength(float3 runFlat, float runY, float endRise, float v)
{
    float len = 0.0;
    float3 prev = 0.0;
    [unroll]
    for (int i = 1; i <= 8; ++i)
    {
        float3 p = LipFaceCurve(runFlat, runY, endRise, v * i / 8.0);
        len += length(p - prev);
        prev = p;
    }
    return len;
}

[domain("tri")]
LipOutput DS_Lip(HullConstantOutput patchConstants, float3 bary : SV_DomainLocation, const OutputPatch<ControlPoint, 3> patch)
{
    LipOutput output = (LipOutput)0;
    float3 gridPos = patch[0].WorldPos * bary.x + patch[1].WorldPos * bary.y + patch[2].WorldPos * bary.z;
    uint idx = patch[0].InstanceIdx;
    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[idx];
    float3 camPos = ViewInverse[3].xyz;
    float camDist = distance(gridPos, camPos);

    OceanShore shore;
    SurfWave surf;
    float3 displacement = OceanSurfaceDisplacement(ocean, gridPos.xz, camDist, shore, surf);
    float3 surface = gridPos + displacement;    // the ocean's own surface here: where the lip fades the sheet lies on it
    float3 P = surface;
    float3 N = float3(0, 1, 0);
    float t = 0.0, thrown = 0.0;
    float clip = -1.0;
    float sheetThick = 10.0, tubeShade = 0.0, splash = 0.0;

    // The lip here, by the ocean's own test (its PS leaves its face out where the map's amount is above LIP_MIN): the
    // lip map's amount of the crest this strip belongs to and the phase's mean rate k back to it (LipMapAt), and the
    // band it is of, where on it the vertex lies and which way it runs: from the blurred waves (LipWave), as the
    // ocean's test. By the full detail the amount flickered from vertex to vertex, the sheet with it between the face
    // and the surface; by the crest search per vertex, it jumped where the search did (teeth, scallops, slivers of
    // seabed where the ocean's per-pixel search disagreed)
    SurfWave wave = LipWave(ocean, gridPos.xz);
    bool second = LipSecondBand(wave);
    float phase = LipPhase(wave);
    float2 heading = second ? wave.heading.zw : wave.heading.xy;
    float2 sheetHeading = heading;
    float2 texXZ = gridPos.xz;
    float local = 0.0, k = 1.0, age = 0.0;
    float2 crestXZ = gridPos.xz;
    SurfWave crest = wave;
    if (phase > -LipSkirt.x && phase < LIP_STRIP + LipSkirt.y)
    {
        float3 lipMap = LipMapAt(ocean, gridPos.xz);
        local = lipMap.y > 0.0 ? lipMap.x : 0.0;
        k = max(lipMap.y, 1e-3);
        age = lipMap.z;
        crestXZ = gridPos.xz - heading * (max(phase, 0.0) / k);
        crest = LipWave(ocean, crestXZ);
    }
    if (local > 0.0 && phase > -LipSkirt.x && phase < LIP_STRIP + LipSkirt.y)
    {
        // The sheet's shape comes from the swell alone (the ocean's surface with the small waves averaged out,
        // LipSmoothCamDist), and the vertex's own detail (the ocean here less the swell) rides on it: full at the crest
        // and in the trough, where the sheet meets the ocean, and none on the curl (detailWeight). Built from the full
        // surface, each vertex's crest, trough end and the slope past it fell on different ripples: the face jumped
        // from vertex to vertex, and the slope, carried over the face's whole run, raised it into spikes over the crest
        // At the vertex, the crest it hangs from, the strip's far end (where the face meets the ocean ahead of it) and
        // 1 m past it (how steeply the swell falls on from there: the face arrives at that slope, or a crease shows
        // where they meet)
        float2 h = second ? crest.heading.zw : crest.heading.xy;
        float2 endXZ = crestXZ + h * (LIP_STRIP / k);
        float2 aheadXZ = endXZ + h * 1.0;
        float3 smoothHere, smoothCrest, smoothEnd, smoothAhead;
        LipSmoothSurfaces(ocean, gridPos.xz, crestXZ, endXZ, aheadXZ, smoothHere, smoothCrest, smoothEnd, smoothAhead);
        float3 detail = displacement - smoothHere;
        float detailWeight;
        float3 C = float3(crestXZ.x, ocean.OceanPlaneY, crestXZ.y) + smoothCrest;
        float3 E = float3(endXZ.x, ocean.OceanPlaneY, endXZ.y) + smoothEnd;
        float3 ahead = float3(aheadXZ.x, ocean.OceanPlaneY, aheadXZ.y) + smoothAhead;
        // (bounded: on a choppy fold the two points barely part sideways, and the face shot up in shards)
        float endSlope = clamp((ahead.y - E.y) / max(length(ahead.xz - E.xz), 0.05), -0.6, 0.6);

        // How much of the crest's lip stands (the amount: risen as the face steepens, sunk as the roller takes over,
        // over a bed that makes it plunge) sets how far the sheet leaves the ocean's surface; how far through its life
        // the breaker is (the age, LipThrow) sets how far the curl is thrown. The same amount decides the strip, so
        // along the heading the whole strip agrees and along the crest the sheet's edges narrow back into the ocean
        // instead of being cut. Before the curl throws, the strip is a face hanging steeply from a sharp crest
        thrown = local;
        float blend = thrown * smoothstep(LIP_MIN, LIP_MIN + 0.15, local);
        float lipThrow = LipThrow(age);
        float height = 2.0 * SurfEnvelope(crest);

        float3 fwd = float3(h.x, 0.0, h.y);
        float3 up = float3(0.0, 1.0, 0.0);
        float3 target, tangent;         // tangent: along the outline, the way t runs
        t = saturate(phase / LIP_STRIP);
        // The curl turns round the tube's centre O from the crest top (u = 0) to its tip (u = 1): up and forward off the
        // crest, over O and down, its radius shrinking so the tip turns in under it. Its inner side keeps the lip's
        // thickness off the outer one, thinning to nothing at the tip, and its root C' is where the face hangs from.
        // Unthrown (lipThrow 0) the whole curl lies at the crest top, and the face hangs from it
        float R = LipRadius * height;
        float3 O = C + fwd * (LipCentre.x * R) - up * (LipCentre.y * R);
        float theta0 = atan2(-LipCentre.x, LipCentre.y);
        float rho0 = R * length(LipCentre);
        float thick0 = LipThickness * R * lipThrow;
        float shrink = LipTipShrink * lipThrow;

        // The ocean's textures over the sheet (its slopes' detail, the foam's cells) by the length along it, from the
        // crest: the curl's outer side carries the water from behind the crest (the texture mirrored back over it), its
        // inner side and the face the water in front. Laid on by the water each vertex bent, the texture ran through the
        // sheet as the wave passed in the outline's order: up the face, under the lip to the tip and back over the top, the
        // curl rolling against the wave (user), and a few metres of the ocean stretched down the whole face. This way both
        // sides of the lip run toward the tip, as the water thrown off the crest does, and the texture keeps its scale. It
        // meets the ocean's at the crest, at the tip (both sides) and in the trough; where the lip narrows out it goes back
        // to the vertex's own water
        float outerArc = length(float2(LipSweep * lipThrow * rho0 * (1.0 - 0.5 * shrink), rho0 * shrink));
        float innerArc = length(float2(LipSweep * lipThrow * (rho0 * (1.0 - 0.5 * shrink) - 0.5 * thick0),
                                       rho0 * shrink - thick0));
        float sheetOff;
        if (t < LipInnerEnd)
        {
            bool outer = t < LipOuterEnd;
            float u = outer ? t / LipOuterEnd : (LipInnerEnd - t) / (LipInnerEnd - LipOuterEnd);
            float dudt = outer ? 1.0 / LipOuterEnd : -1.0 / (LipInnerEnd - LipOuterEnd);
            float theta = theta0 + LipSweep * lipThrow * u;
            float rho = rho0 * (1.0 - shrink * u) - (outer ? 0.0 : thick0 * (1.0 - u));
            float drho = -rho0 * shrink + (outer ? 0.0 : thick0);
            float3 radial = sin(theta) * fwd + cos(theta) * up;
            target = O + rho * radial;
            tangent = (rho * LipSweep * lipThrow * (cos(theta) * fwd - sin(theta) * up) + drho * radial) * dudt;
            // Light crosses the lip's own thickness, toward its tip; at its root the crest, the whole wave under it.
            // Under the lip the tube's roof
            sheetThick = lerp(height, max(thick0 * (1.0 - u), 0.03), smoothstep(0.0, 0.4, u));
            splash = smoothstep(outer ? 0.6 : 0.5, 1.0, u);       // the tip's end of the lip, both sides
            tubeShade = outer ? 0.0 : 1.0;
            sheetOff = outer ? -outerArc * u : -outerArc + innerArc * (1.0 - u);
            detailWeight = outer ? 1.0 - smoothstep(0.0, 0.3, u) : 0.0;     // the thin curl carries none
        }
        else
        {
            // The face under it, the tube's back wall, from the lip's inner root to the trough at the strip's end, where
            // it meets the ocean again (LipFaceCurve)
            float v = (t - LipInnerEnd) / (1.0 - LipInnerEnd);
            float3 root = O + (rho0 - thick0) * (sin(theta0) * fwd + cos(theta0) * up);
            float3 run = E - root;
            float3 runFlat = float3(run.x, 0.0, run.z);
            // (bounded by the wave's height: the term bulges the face by up to 4/27 of it mid-way, and over a run of
            // 20-30 m, where the crest lay far back, a modest slope raised the face into a fin metres over the crest)
            float endRise = clamp(endSlope * 2.5 * length(runFlat), -0.75 * height, 0.75 * height);            target = root + LipFaceCurve(runFlat, run.y, endRise, v);
            tangent = runFlat * (2.5 * pow(max(v, 1e-4), 1.5)) + up * (run.y * 2.0 * (1.0 - v) + endRise * (3.0 * v * v - 2.0 * v));
            // The wave's body behind it; the tube's back wall under the lip, open to the sky a tube's width further down
            float faceLen = LipFaceLength(runFlat, run.y, endRise, v);
            splash = exp(-faceLen / (1.5 * R));                  // the face under the lip, where the tip comes down
            sheetThick = height;
            tubeShade = saturate(1.0 - faceLen / (3.0 * R));
            tubeShade *= tubeShade;
            sheetOff = innerArc - outerArc + (LIP_STRIP / k - innerArc + outerArc)
                     * faceLen / max(LipFaceLength(runFlat, run.y, endRise, 1.0), 1e-4);
            detailWeight = smoothstep(0.0, 1.0, v);
        }
        P = lerp(surface, target + detail * detailWeight, blend);
        sheetHeading = h;
        sheetThick = lerp(height, sheetThick, blend);
        tubeShade *= blend * lipThrow;        // no roof over the face before the lip is thrown
        // White water thrown up where the lip lands: once its tip points down at the trough (LipTip's falling, as the
        // spray splashes up there), over the lip's tip end and the face under it
        float falling;
        LipTip(C, fwd, height, lipThrow, falling);
        splash *= falling * blend;
        float gridOff = max(phase, 0.0) / k;
        texXZ = gridPos.xz + heading * ((sheetOff - gridOff) * blend);
        // The water's outward side: up on the curl's back, down into the tube under the lip, forward off the face
        float3 side = float3(-h.y, 0.0, h.x);
        float3 sheetN = cross(side, tangent);
        float3 n = lerp(up, sheetN / max(length(sheetN), 1e-6), blend);
        N = length(n) > 1e-4 ? normalize(n) : up;
        thrown = blend;
        // Round the strip, past both its ends and beside it where the lip fades out along the crest, a skirt of the ocean's
        // own surface hung a little under it: where the ocean is drawn it hides the skirt, where the two edges miss each
        // other the skirt fills the gap. The ocean cuts itself out per pixel, the lip's edge is interpolated between its
        // vertices, and the ocean's surface is only as exact as its coarse vertices: the seabed showed through the cracks
        // Behind the crest it hangs deep (the ocean's coarse crest sags under the true one, and a shallow skirt stood out
        // of it there); ahead and beside it close under the surface and further out, where a low view that slipped
        // through the ocean's cut under its coarse surface (above the true one in the trough) still finds it
        bool skirt = phase < 0.0 || phase > LIP_STRIP || local < LIP_MIN;
        P = skirt ? surface - up * (phase < 0.0 ? LipSkirtDrop.x : LipSkirtDrop.y) : P;
        N = skirt ? up : N;
        tubeShade = skirt ? 0.0 : tubeShade;
        splash = skirt ? 0.0 : splash;
        thrown = skirt ? 0.0 : thrown;
        texXZ = skirt ? gridPos.xz : texXZ;
        clip = min(min(phase + LipSkirt.x, LIP_STRIP + LipSkirt.y - phase), local - LipSkirtEdge);
    }

    output.WorldPos = P;
    output.Normal = N;
    output.Lip = float2(t, thrown);
    output.Position = mul(mul(float4(P, 1.0), View), Projection);
    output.Depth = output.Position.w;
    output.SkyAmbient = float4(SkyAmbient(float3(0, 1, 0)), 1.0);
    output.InstanceIdx = idx;
    output.Clip = clip;
    output.UndisplacedXZ = float4(texXZ, shore.waveAtten, shore.shoreDepth);
    output.Sheet = float4(sheetHeading, sheetThick, tubeShade);
    output.Surf = float4(surf.envelope, surf.phase);
    output.Splash = splash;
    return output;
}

// ═══════════════════════════════════════════════════════════════════════
// PIXEL SHADER — the ocean's own water (ocean.fx PS) from the bands at the water each vertex bent, so the lip meets the
// ocean with no seam: the finer bands' detail laid over the sheet, the sky's reflection and the sun's glint, the water
// seen through it, the ocean's surf foam where the ocean would have drawn it. The lip's own: the sky the tube's roof shuts
// out, and the light through the thin sheet toward its tip (the turquoise of a backlit lip)
// ═══════════════════════════════════════════════════════════════════════

// 1: matte clay instead, to judge the shape by (stage 1; the user could not read the form in one flat water colour)
#define LIP_CLAY 0

// The share of the sky the lip shuts out deepest in the tube
static const float LipTubeShade = 0.75;
// The light through the sheet: the share of the light on it that reaches the eye, and how fast the water it crosses
// tints it (ShallowColor is the water's per-metre transmittance, raised to metres × this); the sheet glows where it is
// thinner than LipThinSheet.y (m) and is the ocean's water body from LipThinSheet.x
static const float LipSheetGlow = 0.35;
static const float LipSheetDensity = 2.5;
static const float2 LipThinSheet = float2(1.0, 0.1);

struct LipPSOutput
{
    float4 Color : SV_Target0;
};

LipPSOutput PS_Lip(LipOutput input)
{
    LipPSOutput output;
    StructuredBuffer<OceanData> oceanData = ResourceDescriptorHeap[OceanDataIdx];
    OceanData ocean = oceanData[input.InstanceIdx];

    float3 camPos = ViewInverse[3].xyz;
    float3 worldPos = input.WorldPos;
    float3 V = normalize(camPos - worldPos);
    float dist = distance(worldPos, camPos);
    float3 Ng = normalize(input.Normal);
    float3 outward = Ng;                    // the water's outside, where its foam lies and is lit
    Ng = dot(Ng, V) < 0.0 ? -Ng : Ng;       // the water is closed round the tube: its back only through a crack
    float3 L = normalize(-ocean.SunDirection);
    float skyOpen = SkyVisibilityWith(worldPos, float3(0, 1, 0), 1.0);
    float skyVisible = skyOpen * (1.0 - LipTubeShade * input.Sheet.w);
    float3 sunRadiance = ocean.SunColor * ocean.SunIntensity * SunLightScale;
    sunRadiance *= GetCloudShadow(OceanSampler, worldPos - camPos, L);

#if LIP_CLAY
    // Lambert from the sun, the sky by how much the face looks up; a dark terracotta (neither sand, foam nor water,
    // and dark enough not to tone-map to one flat colour), darker where the surface turns away from the eye
    float clayNdotV = saturate(dot(Ng, V));
    float hemisphere = 0.15 + 0.85 * saturate(0.5 + 0.5 * Ng.y);
    float3 clay = float3(0.13, 0.055, 0.035)
                * (sunRadiance * saturate(dot(Ng, L)) + input.SkyAmbient.rgb * skyOpen * 1.15 * hemisphere);
    clay *= 0.35 + 0.65 * clayNdotV;
    output.Color = float4(ApplyAerialPerspective(max(clay, 0.0), input.Position.xy, input.Depth, skyOpen), 1.0);
    return output;
#endif

    sunRadiance *= min(WaterSunShadow(ShadowMapIdx, worldPos - camPos, input.Position.xy), TerrainLightShadow(worldPos, L));
    sunRadiance *= TerrainOpen(worldPos, float3(0, 1, 0));

    // The bands at the water this vertex bent, as the ocean's PS reads them there. As far as the lip is thrown its shape
    // is the carried bands' wave, so their slopes go: where it narrows back into the ocean the sheet lies on the ocean's
    // surface and takes them back, or it was lit level there. Flattened by the DS's depth limit as the ocean's are
    float2 worldXZ = input.UndisplacedXZ.xy;
    float shoreDepth = input.UndisplacedXZ.w;
    OceanBands bands = SampleOceanBands(ocean, worldXZ, input.UndisplacedXZ.z, shoreDepth);
    bands.slopes.mean -= bands.carriedMean * saturate(input.Lip.y);
    float capSlope;
    float waveHeight = DepthLimit(bands.waveHeight, shoreDepth, capSlope);
    bands.slopes.mean *= capSlope;
    bands.slopes.variance *= capSlope * capSlope;

    // Their slopes tilt the sheet in its own frame, along the outline (T) and along the crest, as the ocean tilts its
    // up by them: on the back of the wave the two are the same normal
    float2 h = dot(input.Sheet.xy, input.Sheet.xy) > 1e-8 ? normalize(input.Sheet.xy) : float2(1.0, 0.0);
    float3 side = float3(-h.y, 0.0, h.x);
    float3 T = normalize(cross(Ng, side));
    float3 N = normalize(Ng - dot(bands.slopes.mean, h) * T - dot(bands.slopes.mean, side.xz) * side);
    float alpha = WaterAlpha(bands.slopes, WaterUnresolvedVariance(bands.resolvedTotal));
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));

    // As the ocean: its colour, the light scattered in its body, the sky's reflection and the sun's glint
    float H = max(0.0, waveHeight);
    float3 baseColor = lerp(ocean.OceanColor, ocean.DeepColor, saturate(dist / 4000.0));
    float3 bubbleColor = float3(0.0, 0.008, 0.006);
    float k1 = 0.3 * H * pow(saturate(dot(L, -V)), 4.0) * pow(0.5 - 0.5 * dot(L, N), 3.0);
    float k2 = 0.1 * NdotV * NdotV;
    float k3 = 0.08 * NdotL;
    float3 skyAmbient = input.SkyAmbient.rgb * skyVisible;
    float3 scatter = ((k1 + k2) * baseColor + k3 * baseColor + 0.02 * bubbleColor) * sunRadiance
                   + baseColor * WaterSkyIrradiance(skyAmbient) * 0.1;
    float F = WaterSpecularReflectance(NdotV, alpha);
    float3 reflectColor = WaterSkyReflection(V, N, alpha) * skyVisible;
    float3 specular = WaterSunSpecular(N, V, L, sunRadiance * 3.14159, alpha);

    // The water seen through it down to the seabed (behind the tube's roof the tube's air counts as water)
    WaterColumn column = GetWaterColumn(OceanSampler, ocean.DepthGBufferSRV, ocean.CompositeSRV,
        input.Position.xy * ocean.InvViewportSize, input.Depth, dist, abs(camPos.y - worldPos.y), N,
        ocean.RefractionStrength, scatter, ocean.ShallowColor, ocean.ShoreFadeDepth, 1.0);
    float3 body = column.body;

    // Light back from a bright shallow bed: the turquoise shelf, by the depth of the bed under this point
    float bedDepth = column.bufDepth;
    if (ocean.HeightmapSRV != 0)
    {
        Texture2D<float> heightmap = ResourceDescriptorHeap[ocean.HeightmapSRV];
        float2 uv = clamp((worldPos.xz - float2(ocean.TerrainOriginX, ocean.TerrainOriginZ))
                          / float2(ocean.TerrainSizeX, ocean.TerrainSizeZ), 0.002, 0.998);
        bedDepth = worldPos.y - (ocean.TerrainBaseY + heightmap.SampleLevel(OceanSampler, uv, 0) * ocean.MaxTerrainHeight);
    }
    float shallow = exp(-max(bedDepth, 0.0) / max(0.01, ocean.ShoreFadeDepth)) * saturate(column.pathLen);
    body += ocean.ShallowColor * (sunRadiance * saturate(L.y) + WaterSkyIrradiance(skyAmbient)) * 0.08 * shallow;

    // Light through the wind sea's crests and the bubbles under its whitecaps, as the ocean's
    float crestRise = smoothstep(CREST_GLOW_HEIGHT * 0.25, CREST_GLOW_HEIGHT, bands.windHeight);
    crestRise *= crestRise;
    float slant = 1.0 - NdotV;
    float3 throughCrest = pow(max(ocean.ShallowColor, 0.02), 2.5);
    throughCrest = lerp(dot(throughCrest, float3(0.2126, 0.7152, 0.0722)), throughCrest, 0.7);   // storm water is murky
    float3 towardSun = sunRadiance * (0.2 + 3.0 * pow(saturate(dot(L, -V)), 4.0));
    body += (skyAmbient + towardSun) * throughCrest * (CREST_GLOW * crestRise * lerp(0.3, 1.0, slant * slant)) * skyVisible;
    body += (skyAmbient + sunRadiance * 0.3) * throughCrest * (AERATION * saturate(bands.capFoam)) * skyVisible;

    // The lip's own: where the sheet is thin, the light on its top (the open sky's, and the sun's, strongest looking
    // toward it) passes through it to the eye, tinted by the water it crosses: what is seen of it instead of the body
    float sheet = input.Sheet.z;
    float3 throughSheet = pow(max(ocean.ShallowColor, 0.02), LipSheetDensity * sheet);
    throughSheet = lerp(dot(throughSheet, float3(0.2126, 0.7152, 0.0722)), throughSheet, 0.7);
    float3 sheetLight = (input.SkyAmbient.rgb * skyOpen + towardSun) * throughSheet * LipSheetGlow;
    body = lerp(body, sheetLight, 1.0 - smoothstep(LipThinSheet.y, LipThinSheet.x, sheet));

    float3 color = (1.0 - F) * body + (specular + F * reflectColor) * column.edgeFade;
    color = max(0.0, color);

    // The ocean's foam at this water: the surf's white water (its roller rides the crest the curl is thrown from) and
    // the whitecaps, eroded by the same cells. The surf's by the vertex's own water (the DS): the texture's water on
    // the curl's outer side is from behind the crest, whose trail would whiten the lip
    float trailWeb;
    bands.surf.envelope = input.Surf.xy;
    bands.surf.phase = input.Surf.zw;
    float surfFoam = SurfWhiteWater(ocean, bands, shoreDepth, dist, trailWeb);
    float whitecap = OceanWhitecap(bands, dist);
    // With the white water the landing lip throws up (DS Splash): a churning mass over the tip's end and the face under
    // it. Both as the ocean's surf: a mass of lumps (SurfFoamLook) at the sheet's texture position
    float splash = saturate(input.Splash);
    float amount = max(surfFoam, splash * (1.4 / 1.3));
    float thickness = saturate(max(surfFoam, splash));
    SurfFoamMass mass = (SurfFoamMass)0;
    mass.height = 0.5;
    mass.cover = saturate((amount * 1.3 - 0.5) * 2.0);
    float footprint = SurfFoamFootprint(worldXZ);
    [branch]
    if (ocean.NoiseSRV != 0 && amount > 0.001)
    {
        Texture2D<float4> noiseTex = ResourceDescriptorHeap[ocean.NoiseSRV];
        // (no trail on the curl and the face: drawn by the vertex's own water, its lumps streaked down the face)
        mass = SurfFoamLook(noiseTex, worldXZ, ocean.WaveTime, amount, 0.0, h, dist, footprint);
    }
    float foamFar = saturate(dist / 150.0);
    float foam = lerp(mass.cover, saturate(amount), foamFar);
    if (foam > 0.001 || whitecap > 0.001)
    {
        float3 skyOnFoam = SkyAmbient(N) * skyVisible;
        color = lerp(color, WaterFoamLit(float3(0.80, 0.80, 0.78), NdotL, sunRadiance, skyOnFoam), whitecap);
        // The lumps' slope tilts the foam in the sheet's frame, as the bands' slopes tilt the water above. On the
        // water's outside: seen from behind (the curl's back over the crest, from in front of the wave), lit by the
        // normal turned to the eye the foam faced the ground and stood as a dark line along the crest
        float foamNear = 1.0 - foamFar;
        float3 outT = normalize(cross(outward, side));
        float2 foamSlope = bands.slopes.mean + mass.slope * foamNear;
        float3 foamN = normalize(outward - dot(foamSlope, h) * outT - dot(foamSlope, side.xz) * side);
        float3 foamAlbedo = float3(0.80, 0.80, 0.78) * lerp(1.0, SurfFoamShade(mass, thickness), foamNear)
                          * lerp(0.78, 1.05, thickness);
        color = lerp(color, WaterFoamLit(foamAlbedo, saturate(dot(foamN, L)), sunRadiance, SkyAmbient(foamN) * skyVisible),
                     foam * 0.9);
    }

    color = ApplyAerialPerspective(color, input.Position.xy, input.Depth, skyVisible);
    output.Color = float4(color, 1.0);
    return output;
}

// ═══════════════════════════════════════════════════════════════════════
technique11 Lip
{
    pass Forward
    {
        SetVertexShader(CompileShader(vs_6_6, VS_Lip()));
        SetHullShader(CompileShader(hs_6_6, HS_Lip()));
        SetDomainShader(CompileShader(ds_6_6, DS_Lip()));
        SetPixelShader(CompileShader(ps_6_6, PS_Lip()));
    }
}
