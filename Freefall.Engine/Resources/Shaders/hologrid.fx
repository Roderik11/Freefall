// @RenderState(DepthTest=false, DepthWrite=false)

// Push constants for bindless texture lookup
cbuffer PushConstants : register(b3)
{
    uint DepthTexIdx;
    uint CompositeTexIdx;
};

#include "common.fx"

// Grid settings from Hologrid component
cbuffer ObjectConstants : register(b1)
{
    float3 GridColor;
    float Opacity;
    float3 XAxisColor;
    float FadeRange;
    float3 ZAxisColor;
    float PlaneY;
};

SamplerState Sampler : register(s0);

struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

// Procedural fullscreen quad (TriangleStrip, 4 verts)
VSOutput VS(uint vertexID : SV_VertexID)
{
    VSOutput output;
    float2 pos[4] = {
        float2(-1,  1),
        float2( 1,  1),
        float2(-1, -1),
        float2( 1, -1)
    };
    float2 uv[4] = {
        float2(0, 0),
        float2(1, 0),
        float2(0, 1),
        float2(1, 1)
    };

    output.Position = float4(pos[vertexID], 0.0f, 1.0f);
    output.TexCoord = uv[vertexID];
    return output;
}

// Ray-plane intersection: returns true if ray hits the grid plane at Y = PlaneY
bool IntersectGridPlane(float3 rayOrigin, float3 rayDir, float planeY, out float3 hitPoint)
{
    float denom = rayDir.y;

    if (abs(denom) < 1e-7)
        return false;

    float t = (planeY - rayOrigin.y) / denom;

    if (t < 0)
        return false;

    hitPoint = rayOrigin + t * rayDir;
    return true;
}

float4 PS(VSOutput input) : SV_Target
{
    Texture2D DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    Texture2D CompositeTex = ResourceDescriptorHeap[CompositeTexIdx];

    float depth = DepthTex.Load(int3(input.Position.xy, 0)).r;

    // Build camera-relative ray direction from NDC
    float2 ndc = float2(input.TexCoord.x * 2.0 - 1.0, (1.0 - input.TexCoord.y) * 2.0 - 1.0);
    float4 farClip = mul(float4(ndc, 1.0, 1.0), CameraInverse);
    farClip /= farClip.w;
    float3 rayDir = normalize(farClip.xyz);

    // Ray-plane intersection in CAMERA-RELATIVE space
    // Camera is at origin, plane is at Y = PlaneY - CamPos.y
    float planeYRel = PlaneY - CamPos.y;
    float denom = rayDir.y;
    if (abs(denom) < 1e-7) discard;

    float t = planeYRel / denom;
    if (t < 0) discard;

    // Camera-relative hit point (for depth comparison)
    float3 hitPointRel = t * rayDir;
    float distToHit = length(hitPointRel);

    // Depth occlusion: compare linear depths along the view Z axis
    // gbuffer writes: Depth = Position.w = view-space Z (after mul(camRelPos, View))
    // Extract view Z-axis from View matrix (column 2, row-major) — no translation needed
    float3 viewZ = float3(View[0][2], View[1][2], View[2][2]);
    float gridLinearDepth = dot(hitPointRel, viewZ);

    // DEBUG: uncomment to visualize depths
    //float3 sceneColor0 = CompositeTex.Load(int3(input.Position.xy, 0)).rgb;
    //return float4(depth / 500.0, gridLinearDepth / 500.0, 0, 1);

    if (depth > 0 && gridLinearDepth > depth)
        discard;

    // World-space hit point (for stable grid coordinates)
    float3 hitPointWorld = hitPointRel + CamPos;

    // Camera height above grid plane
    float camHeight = abs(CamPos.y - PlaneY);

    // DEBUG: output depth value — white=geometry, black=sky
    //return float4(depth, depth, depth, 1.0);

    // LOD: smooth crossfade between 2 grid scales
    float logHeight = log10(max(camHeight, 1));
    int level = (int)floor(logHeight);
    float lodBlend = frac(logHeight);

    float fineScale = pow(10, level);
    float coarseScale = pow(10, level + 1);

    // Grid line intensity for each LOD level (world-space coordinates)
    float2 finePos = hitPointWorld.xz / fineScale;
    float2 fineGrid = abs(frac(finePos - 0.5) - 0.5) / fwidth(finePos);
    float fineLine = saturate(1.0 - min(fineGrid.x, fineGrid.y) * 0.9);

    float2 coarsePos = hitPointWorld.xz / coarseScale;
    float2 coarseGrid = abs(frac(coarsePos - 0.5) - 0.5) / fwidth(coarsePos);
    float coarseLine = saturate(1.0 - min(coarseGrid.x, coarseGrid.y) * 0.9);

    // Crossfade: fine fades out as lodBlend increases, coarse stays
    float value = fineLine * (1.0 - lodBlend) + coarseLine;
    value = saturate(value);

    // Distance fade — quadratic falloff
    float maxRange = max(camHeight, 1) * FadeRange;
    float distRatio = saturate(distToHit / maxRange);
    float distFade = (1.0 - distRatio) * (1.0 - distRatio);
    value *= distFade;

    // Grid magnitude for axis width
    float magnitude = fineScale;

    // No visible grid line — discard
    if (value < 0.01)
        discard;

    // Axis highlighting (world-space coordinates)
    float3 lineColor = GridColor;
    float axisWidth = magnitude * 0.01;

    if (abs(hitPointWorld.z) < axisWidth)
        lineColor = XAxisColor;
    else if (abs(hitPointWorld.x) < axisWidth)
        lineColor = ZAxisColor;

    // Origin highlight
    float originDist = length(hitPointWorld.xz);
    float originDot = saturate(1.0 - originDist / (axisWidth * 3));
    lineColor = lerp(lineColor, float3(1, 1, 0), originDot * 0.8);

    // Manual compositing: sample scene, lerp with grid, output opaque
    float alpha = value * Opacity;
    float3 sceneColor = CompositeTex.Load(int3(input.Position.xy, 0)).rgb;
    float3 result = lerp(sceneColor, lineColor, alpha);
    return float4(result, 1);
}

technique11 Standard
{
    pass PostProcess
    {
        SetVertexShader(CompileShader(vs_6_6, VS()));
        SetPixelShader(CompileShader(ps_6_6, PS()));
    }
}
