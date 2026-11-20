// Hulls keep the water out (Graphics/WaterHulls.cs): inside a ship's hold or an open boat the sea and the lakes are not
// drawn. Each hull is its own shape, its half-breadth at HULL_STATIONS places along it and HULL_LEVELS heights from its
// keel up, sliced from its mesh; a point is inside where |z| in the hull's frame is under the interpolated half-breadth.
// ocean.fx and water.fx use it, and the wakes (wake_draw.fx). Seen through the water, a hull fades within a metre of it.
// Docs: .agent/knowledge/engine_internals/ships.md.
#ifndef WATER_HULLS_HLSLI
#define WATER_HULLS_HLSLI

#define HULL_STATIONS 32
#define HULL_LEVELS 8

struct WaterHull
{
    float4 Row0, Row1, Row2;    // world to the hull's frame: local.x = dot(Row0, float4(world, 1))
    float4 Sphere;              // world centre, radius
    float X0, X1, Bottom, Top;  // the shape's extent in its frame
    float Breadth[HULL_STATIONS * HULL_LEVELS];
};

// Whether a world point is inside one of the hulls grown by margin metres on every side (srv: WaterHulls.Srv, 0 = none)
bool NearHull(uint srv, uint count, float3 p, float margin)
{
    if (srv == 0)
        return false;
    StructuredBuffer<WaterHull> hulls = ResourceDescriptorHeap[srv];
    for (uint i = 0; i < count; i++)
    {
        float4 sphere = hulls[i].Sphere;
        float3 d = p - sphere.xyz;
        float r = sphere.w + margin;
        if (dot(d, d) > r * r)
            continue;
        float4 w = float4(p, 1.0);
        float3 l = float3(dot(hulls[i].Row0, w), dot(hulls[i].Row1, w), dot(hulls[i].Row2, w));
        float x0 = hulls[i].X0, x1 = hulls[i].X1, y0 = hulls[i].Bottom, y1 = hulls[i].Top;
        if (l.x <= x0 - margin || l.x >= x1 + margin || l.y <= y0 - margin || l.y >= y1)
            continue;
        float fx = clamp((l.x - x0) / (x1 - x0), 0.0, 1.0) * (HULL_STATIONS - 1);
        float fy = clamp((l.y - y0) / (y1 - y0), 0.0, 1.0) * (HULL_LEVELS - 1);
        uint ix = min((uint)fx, HULL_STATIONS - 2), iy = min((uint)fy, HULL_LEVELS - 2);
        float tx = fx - ix, ty = fy - iy;
        uint at = iy * HULL_STATIONS + ix;
        float b0 = lerp(hulls[i].Breadth[at], hulls[i].Breadth[at + 1], tx);
        float b1 = lerp(hulls[i].Breadth[at + HULL_STATIONS], hulls[i].Breadth[at + HULL_STATIONS + 1], tx);
        float b = lerp(b0, b1, ty);
        if (b > 0.0 && abs(l.z) < b + margin)
            return true;
    }
    return false;
}

// Whether a world point is inside one of the hulls, where the water is kept out
bool InsideHull(uint srv, uint count, float3 p)
{
    return NearHull(srv, count, p, 0.0);
}

// How far one sees through the water to a ship's hull (m): the sea's clear shallows (ShoreFadeDepth 8 m) showed a
// hull's whole bottom through the surface, as if the ship sat on the water (2026-11-20). Round a hull the water is
// stirred and shadowed: what is seen of it fades within a metre of water
#define HULL_WATER_VISIBILITY 0.6

// The visibility to use for a pixel of water: the hull's where the scene seen through it lies on a ship's hull
// (depthSRV: the linear depth G-buffer; pixelDepth: the water's own; dist: the eye to the water; V: from the water to
// the eye)
float WaterVisibilityOverHulls(uint srv, uint count, uint depthSRV, SamplerState s, float2 screenUV, float pixelDepth,
                               float dist, float3 worldPos, float3 V, float visibility)
{
    if (srv == 0 || count == 0 || depthSRV == 0)
        return visibility;
    Texture2D<float> depthGB = ResourceDescriptorHeap[depthSRV];
    float sceneZ = depthGB.SampleLevel(s, screenUV, 0);
    if (sceneZ <= 0.0)
        return visibility;
    float behind = dist * (max(sceneZ / max(pixelDepth, 0.001), 1.0) - 1.0);
    return NearHull(srv, count, worldPos - V * behind, 0.4) ? min(visibility, HULL_WATER_VISIBILITY) : visibility;
}

#endif
