// How much of the sky a place sees past the meshes: walls, roofs, tree crowns, the cave kit (ATMOSPHERE_PLAN stage 4,
// part 2). SkyCapture.cs draws the scene's meshes into depth maps seen from directions spread over the sky, around the
// camera; a place sees the sky in a direction when nothing in that direction's map lies between it and the sky. The
// terrain is not in the maps (sky_visibility.hlsli does the terrain), nor are characters or grass.
// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §11.
#ifndef SKY_CAPTURE_HLSLI
#define SKY_CAPTURE_HLSLI

#define SKY_CAPTURE_VIEWS 16    // SkyCapture.ViewCount

// One direction of the sky. Must match SkyCapture.View (80 bytes).
struct SkyCaptureView
{
    row_major float4x4 VP;  // absolute world -> clip: orthographic, standard depth (0 toward the sky)
    float3 Direction;       // toward the sky, unit
    float Valid;            // 0 until this direction has been captured
};

// One direction's part: adds its weight to `total` and returns weight × how much of it is open. slice = the view's
// index (its slice of the depth maps), size = the maps' texels.
float SkyCaptureViewLight(SkyCaptureView v, uint slice, Texture2DArray<float> depths, float2 size, SamplerState clampSampler,
    float3 worldPos, float3 N, inout float total)
{
    // The directions are spread cosine-weighted about the zenith, so for a surface facing up each counts alike. A
    // surface facing elsewhere weighs them by its own cosine; a little for every one keeps a ceiling's average.
    // A direction not captured yet weighs nothing.
    float w = (max(dot(N, v.Direction), 0.0) / max(v.Direction.y, 0.05) + 0.05) * v.Valid;

    // Seen from 0.4 m out along the surface's normal, so that its own texels in the map are not in the way (the lift also
    // shifts the point across the map, as normal-offset shadows do). Nothing toward the sky: 0.3 m that way lifted the
    // underside of a roof and the top of a gable wall through the thin roof boards into daylight (bright bands along the
    // roof line, user 2026-10-18); along the normal a point moves away from its own surface and does not cross a roof or
    // wall that meets it at an angle.
    float3 p = worldPos + N * 0.4;
    float4 clip = mul(float4(p, 1.0), v.VP);
    float2 uv = clip.xy * float2(0.5, -0.5) + 0.5;
    float2 edge = min(uv, 1.0 - uv);
    float inside = saturate(min(edge.x, edge.y) * 20.0);    // the meshes fade out over the square's last 5 %

    float visible = 1.0;
    if (inside > 0.0 && clip.z < 1.0)
    {
        // Bilinear over the 2×2 texels round the point, each lit where the point is not behind it seen from the sky
        float2 texel = uv * size - 0.5;
        float2 f = frac(texel);
        float4 d = depths.GatherRed(clampSampler, float3((floor(texel) + 1.0) / size, slice));
        float4 open = step(clip.z, d);
        // Gather order: x = (0,1), y = (1,1), z = (1,0), w = (0,0)
        float seen = lerp(lerp(open.w, open.z, f.x), lerp(open.x, open.y, f.x), f.y);
        visible = lerp(1.0, seen, inside);
    }
    total += w;
    return w * visible;
}

// 0..1: the share of the sky's light that reaches a place past the meshes. worldPos is absolute, N the surface normal
// (straight up for a particle or a water surface). 1 where nothing has been captured (outside the captured square,
// above or below its heights, no captures at all). For the forward passes and particles; the opaque surfaces get it
// from sky_visibility_cs.hlsl, which holds the views in group shared memory (in the directional light's shader the
// views' loads and the 16 directions cost 0.6-0.8 ms at 2148×1249).
float MeshSkyVisibilityAt(uint viewsIdx, uint depthIdx, uint count, SamplerState clampSampler, float3 worldPos, float3 N)
{
    if (viewsIdx == 0 || depthIdx == 0 || count != SKY_CAPTURE_VIEWS)
        return 1;
    StructuredBuffer<SkyCaptureView> views = ResourceDescriptorHeap[viewsIdx];
    Texture2DArray<float> depths = ResourceDescriptorHeap[depthIdx];
    float width, height, slices;
    depths.GetDimensions(width, height, slices);

    float lit = 0, total = 0;
    [unroll] for (uint i = 0; i < SKY_CAPTURE_VIEWS; i++)
        lit += SkyCaptureViewLight(views[i], i, depths, float2(width, height), clampSampler, worldPos, N, total);
    return total > 0.0 ? lit / total : 1.0;
}

#endif
