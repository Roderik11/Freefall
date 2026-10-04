// GTAO — Ground Truth Ambient Occlusion (Jimenez et al. 2016), structured after Intel XeGTAO.
// Spatial-only variant: no temporal accumulation. The per-pixel noise is a 4x4 interleaved
// pattern, which CSDenoise cancels exactly with a depth-aware 4x4 box filter.
//
// View space is left-handed (x right, y up, z forward); depth input is linear view-space Z.

#pragma kernel CSGTAO
#pragma kernel CSDenoise

cbuffer PushConstants : register(b3)
{
    uint DepthTexIdx;       // Linear view-space Z (R32_Float), 0 = sky
    uint NormalTexIdx;      // World-space normals (CSGTAO)
    uint AOInputIdx;        // Raw AO (CSDenoise)
    uint OutputUAVIdx;
    uint ScreenWidthIdx;
    uint ScreenHeightIdx;
};

cbuffer Params : register(b4)
{
    float4 ViewRight;       // Camera basis in world space (xyz)
    float4 ViewUp;
    float4 ViewForward;
    float2 ProjScale;       // Projection._11, Projection._22
    float Radius;           // World-space AO radius (meters)
    float Power;            // Final visibility exponent
    float Intensity;        // 0 = no AO, 1 = full
    uint SliceCount;
    uint StepCount;         // Per side of each slice
    float _gtaoPad;
};

static const float PI = 3.14159265f;
static const float HALF_PI = 1.57079633f;

// Fraction of the radius over which occluders fade out (XeGTAO default)
static const float FALLOFF_RANGE = 0.615f;

// Ordered 4x4 pattern — every value appears once per 4x4 block
static const uint Bayer4[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

float3 ReconstructVS(float2 pixelCenter, float viewZ)
{
    float2 uv = pixelCenter / float2(ScreenWidthIdx, ScreenHeightIdx);
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    return float3(ndc / ProjScale, 1.0f) * viewZ;
}

[numthreads(8, 8, 1)]
void CSGTAO(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadId.xy;
    if (px.x >= ScreenWidthIdx || px.y >= ScreenHeightIdx)
        return;

    Texture2D<float> DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    Texture2D NormalTex = ResourceDescriptorHeap[NormalTexIdx];
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputUAVIdx];

    float viewZ = DepthTex.Load(int3(px, 0));
    if (viewZ <= 0.0f)
    {
        Output[px] = 1.0f; // sky
        return;
    }

    // Projected radius in pixels. Too small = nothing to resolve; too large = cache thrash.
    float screenRadius = Radius * 0.5f * ProjScale.y * float(ScreenHeightIdx) / viewZ;
    float radiusFade = saturate((screenRadius - 2.0f) / 6.0f);
    if (radiusFade <= 0.0f)
    {
        Output[px] = 1.0f;
        return;
    }
    screenRadius = min(screenRadius, 0.25f * float(ScreenHeightIdx));

    // Nudge towards the camera to avoid self-occlusion from depth precision
    float2 pixelCenter = float2(px) + 0.5f;
    float3 P = ReconstructVS(pixelCenter, viewZ * 0.9999f);
    float3 V = normalize(-P);

    float3 worldN = NormalTex.Load(int3(px, 0)).xyz;
    float3 N = normalize(float3(dot(worldN, ViewRight.xyz), dot(worldN, ViewUp.xyz), dot(worldN, ViewForward.xyz)));

    uint noiseIdx = (px.x & 3) + ((px.y & 3) << 2);
    float noiseSlice = (float(Bayer4[noiseIdx]) + 0.5f) / 16.0f;
    float noiseStep = (float((Bayer4[noiseIdx] * 7 + 3) & 15) + 0.5f) / 16.0f;

    float falloffRange = FALLOFF_RANGE * Radius;
    float falloffMul = -1.0f / falloffRange;
    float falloffAdd = (Radius - falloffRange) / falloffRange + 1.0f;

    // Keep the first sample off the center pixel
    float minS = 1.3f / screenRadius;
    int2 maxCoord = int2(ScreenWidthIdx, ScreenHeightIdx) - 1;

    float visibility = 0.0f;
    for (uint slice = 0; slice < SliceCount; slice++)
    {
        float phi = (float(slice) + noiseSlice) / float(SliceCount) * PI;
        float cosPhi = cos(phi);
        float sinPhi = sin(phi);
        float2 omega = float2(cosPhi, -sinPhi) * screenRadius; // screen y is down

        float3 directionVec = float3(cosPhi, sinPhi, 0.0f);
        float3 orthoDirectionVec = directionVec - dot(directionVec, V) * V;
        float3 axisVec = normalize(cross(orthoDirectionVec, V));
        float3 projectedNormalVec = N - axisVec * dot(N, axisVec);

        float signNorm = sign(dot(orthoDirectionVec, projectedNormalVec));
        float projectedNormalVecLength = length(projectedNormalVec);
        float cosNorm = saturate(dot(projectedNormalVec, V) / max(projectedNormalVecLength, 1e-5f));
        float n = signNorm * acos(cosNorm);

        // Horizons start at the tangent plane, so the hemisphere is the upper bound
        float lowHorizonCos0 = cos(n + HALF_PI);
        float lowHorizonCos1 = cos(n - HALF_PI);
        float horizonCos0 = lowHorizonCos0;
        float horizonCos1 = lowHorizonCos1;

        for (uint step = 0; step < StepCount; step++)
        {
            float s = (float(step) + noiseStep) / float(StepCount);
            s = s * s + minS; // denser near the center
            float2 sampleOffset = round(s * omega);

            int2 coord0 = clamp(int2(pixelCenter + sampleOffset), int2(0, 0), maxCoord);
            float z0 = DepthTex.Load(int3(coord0, 0));
            if (z0 > 0.0f)
            {
                float3 delta = ReconstructVS(float2(coord0) + 0.5f, z0) - P;
                float dist = length(delta);
                float weight = saturate(dist * falloffMul + falloffAdd);
                float shc = lerp(lowHorizonCos0, dot(delta / dist, V), weight);
                horizonCos0 = max(horizonCos0, shc);
            }

            int2 coord1 = clamp(int2(pixelCenter - sampleOffset), int2(0, 0), maxCoord);
            float z1 = DepthTex.Load(int3(coord1, 0));
            if (z1 > 0.0f)
            {
                float3 delta = ReconstructVS(float2(coord1) + 0.5f, z1) - P;
                float dist = length(delta);
                float weight = saturate(dist * falloffMul + falloffAdd);
                float shc = lerp(lowHorizonCos1, dot(delta / dist, V), weight);
                horizonCos1 = max(horizonCos1, shc);
            }
        }

        // Slightly over-weight slices whose plane barely contains the normal (XeGTAO fudge)
        projectedNormalVecLength = lerp(projectedNormalVecLength, 1.0f, 0.05f);

        float h0 = -acos(clamp(horizonCos1, -1.0f, 1.0f));
        float h1 = acos(clamp(horizonCos0, -1.0f, 1.0f));
        h0 = n + clamp(h0 - n, -HALF_PI, HALF_PI);
        h1 = n + clamp(h1 - n, -HALF_PI, HALF_PI);

        // Cosine-weighted visible arc, integrated analytically
        float sinN = sin(n);
        float iarc0 = (cosNorm + 2.0f * h0 * sinN - cos(2.0f * h0 - n)) * 0.25f;
        float iarc1 = (cosNorm + 2.0f * h1 * sinN - cos(2.0f * h1 - n)) * 0.25f;
        visibility += projectedNormalVecLength * (iarc0 + iarc1);
    }

    visibility = saturate(visibility / float(SliceCount));
    visibility = max(0.03f, pow(visibility, Power));

    Output[px] = lerp(1.0f, visibility, radiusFade);
}

[numthreads(8, 8, 1)]
void CSDenoise(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 px = dispatchThreadId.xy;
    if (px.x >= ScreenWidthIdx || px.y >= ScreenHeightIdx)
        return;

    Texture2D<float> DepthTex = ResourceDescriptorHeap[DepthTexIdx];
    Texture2D<float> AOInput = ResourceDescriptorHeap[AOInputIdx];
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputUAVIdx];

    float centerZ = DepthTex.Load(int3(px, 0));
    if (centerZ <= 0.0f)
    {
        Output[px] = 1.0f;
        return;
    }

    int2 maxCoord = int2(ScreenWidthIdx, ScreenHeightIdx) - 1;
    float invDepthTolerance = 1.0f / (0.03f * centerZ);

    // 4x4 window covers each noise pattern value exactly once
    float sum = 0.0f;
    float weightSum = 0.0f;
    for (int y = -2; y <= 1; y++)
    {
        for (int x = -2; x <= 1; x++)
        {
            int3 coord = int3(clamp(int2(px) + int2(x, y), int2(0, 0), maxCoord), 0);
            float z = DepthTex.Load(coord);
            float w = saturate(1.0f - abs(z - centerZ) * invDepthTolerance);
            sum += AOInput.Load(coord) * w;
            weightSum += w;
        }
    }

    // Center tap always has weight 1, so weightSum >= 1
    float ao = sum / weightSum;
    Output[px] = lerp(1.0f, ao, Intensity);
}
