// Point light shadows: the data LocalShadows.cs uploads each frame, shared by the culler
// (cull_instances.hlsl), the shadow vertex shaders (gbuffer*.fx via common.fx) and the point light
// shader. Every shadowed light renders up to six cube faces ("views") into tiles of the local shadow
// atlas (Texture2DArray, standard Z: cleared to 1, nearer is smaller).
#ifndef LOCAL_SHADOWS_HLSLI
#define LOCAL_SHADOWS_HLSLI

#define LOCAL_SHADOW_NO_VIEW        0xFFFFFFFFu
#define LOCAL_SHADOW_MAX_VIEWS      128u    // LocalShadows.MaxViews (the culler's view mask is a uint4)
#define LOCAL_SHADOW_PAGE_SIZE      2048.0  // LocalShadows.PageSize

// Expansion entries of the local shadow pass: view in bits 25-31, instance in bits 0-24
#define LOCAL_SHADOW_VIEW_SHIFT     25u
#define LOCAL_SHADOW_INSTANCE_MASK  0x01FFFFFFu

// One rendered cube face. Must match C# LocalShadows.View (256 bytes).
struct LocalShadowView
{
    float4 Planes[6];               // absolute-world frustum planes, normals pointing out (culling)
    row_major float4x4 VP;          // absolute world -> clip (rendering)
    row_major float4x4 SampleVP;    // camera-relative -> clip (lighting)
    float4 Tile;                    // the face's atlas rect in UV: xy = size, zw = offset
    uint Page;                      // atlas array slice
    uint Light;                     // index into LocalShadowLight[]
    float TexelPerMetre;            // a texel's width per metre of distance along the face axis
    uint _pad;
};

// A point light that has a shadow this frame. Must match C# LocalShadows.ShadowLight (48 bytes).
struct LocalShadowLight
{
    float3 Position;                // absolute world (culling)
    float Range;
    uint FaceView[6];               // view of each face (+X -X +Y -Y +Z -Z), LOCAL_SHADOW_NO_VIEW if not rendered
    float Strength;                 // 0..1: fades a shadow in when the light is picked and out when it is dropped
    float _pad;
};

// The cube face a direction from the light falls in: 0..5 = +X -X +Y -Y +Z -Z.
// Matches the face axes in LocalShadows.cs.
uint LocalShadowFace(float3 d)
{
    float3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0 ? 0u : 1u;
    if (a.y >= a.z) return d.y >= 0 ? 2u : 3u;
    return d.z >= 0 ? 4u : 5u;
}

#endif
