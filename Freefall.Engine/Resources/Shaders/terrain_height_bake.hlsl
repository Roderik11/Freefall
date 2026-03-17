// terrain_height_bake.hlsl — Composites terrain HeightLayers and StampGroups
// into a final R32_Float heightmap.
//
// HeightLayers: dispatched per-layer (CS_ImportLayer)
// StampGroups: dispatched per-group with a StructuredBuffer of instances (CS_StampGroup)
//
// Uses push constants (b3) matching engine convention.

#pragma kernel CS_ImportLayer
#pragma kernel CS_StampGroup
#pragma kernel CS_Clear
#pragma kernel CS_PaintBrush
#pragma kernel CS_ClearDelta
#pragma kernel CS_ImportChannel
#pragma kernel CS_PackChannels
#pragma kernel CS_BrushRaycast

// Push constants (root parameter 0, register b3) — bindless indices + params
cbuffer PushConstants : register(b3)
{
    uint SourceIdx;       // slot 0 — SRV: source heightmap texture (import) / brush texture (stamp)
    uint OutputIdx;       // slot 1 — UAV: output RWTexture2D<float>
    uint StampBufIdx;     // slot 2 — SRV: StructuredBuffer<StampData> (stamp group)
    uint BlendMode;       // slot 3 — 0=Set, 1=Add, 2=Max, 3=Lerp
    float Opacity;        // slot 4 — layer/group opacity [0..1]
    uint StampCount;      // slot 5 — number of stamp instances
    float BrushRadius;    // slot 6 — UV-space radius (brush only)
    float BrushFalloff;   // slot 7 — falloff exponent (brush only)
    float BrushTargetHeight; // slot 8 — normalized target height for flatten (brush only)

    // Raycast params (only used by CS_BrushRaycast)
    float RayOriginX;     // slot 9
    float RayOriginY;     // slot 10
    float RayOriginZ;     // slot 11
    float RayDirX;        // slot 12
    float RayDirY;        // slot 13
    float RayDirZ;        // slot 14
    float TerrainOriginX; // slot 15
    float TerrainOriginZ; // slot 16
    float TerrainSizeX;   // slot 17
    float TerrainSizeZ;   // slot 18
    float TerrainMaxHeight; // slot 19
};

SamplerState sampLinear : register(s0);

// Per-stamp instance data (matches C# StampInstanceGPU)
struct StampData
{
    float2 Position;       // terrain-space UV [0..1]
    float Radius;          // UV-space radius
    float Strength;        // height multiplier
    float Falloff;         // falloff exponent
    float Rotation;        // degrees
    float2 _pad;
};

float Blend(float prev, float value, uint mode, float opacity)
{
    value *= opacity;
    switch (mode)
    {
        case 0: return value;                          // Set
        case 1: return prev + value;                   // Add
        case 2: return max(prev, value);               // Max
        case 3: return lerp(prev, value, opacity);     // Lerp
        default: return value;
    }
}

// ── CS_Clear: Zero the output heightmap ──
[numthreads(8, 8, 1)]
void CS_Clear(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;
    Output[dtid.xy] = 0;
}

// ── CS_ImportLayer: Copy/blend a full heightmap texture ──
[numthreads(8, 8, 1)]
void CS_ImportLayer(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float> Source = ResourceDescriptorHeap[SourceIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float src = Source.SampleLevel(sampLinear, uv, 0);

    float prev = Output[dtid.xy];
    Output[dtid.xy] = Blend(prev, src, BlendMode, Opacity);
}

// ── CS_StampGroup: Apply all stamps in a group (batched per brush texture) ──
[numthreads(8, 8, 1)]
void CS_StampGroup(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float> Brush = ResourceDescriptorHeap[SourceIdx];
    StructuredBuffer<StampData> Stamps = ResourceDescriptorHeap[StampBufIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float prev = Output[dtid.xy];
    float accumulated = 0;

    for (uint i = 0; i < StampCount; i++)
    {
        StampData s = Stamps[i];
        float2 delta = uv - s.Position;
        float dist = length(delta);

        // Outside stamp radius — skip
        if (dist > s.Radius) continue;

        // Falloff: 1 at center, 0 at edge
        float t = saturate(1.0 - dist / s.Radius);
        float falloff = pow(t, s.Falloff);

        // Rotate delta into brush UV space
        float rad = s.Rotation * 3.14159265 / 180.0;
        float cosR = cos(rad);
        float sinR = sin(rad);
        float2 rotDelta = float2(
            delta.x * cosR + delta.y * sinR,
           -delta.x * sinR + delta.y * cosR
        );

        // Sample brush at local UV
        float2 brushUV = (rotDelta / s.Radius) * 0.5 + 0.5;
        float brushValue = Brush.SampleLevel(sampLinear, brushUV, 0);

        accumulated += brushValue * s.Strength * falloff;
    }

    if (accumulated != 0)
        Output[dtid.xy] = Blend(prev, accumulated, BlendMode, Opacity);
}

// ── CS_PaintBrush: Stroke-based terrain painting ──
// Paints along a polyline of points. Each pixel computes its minimum distance
// to any segment in the stroke, then applies the brush mode with smoothstep falloff.
//
// Push constants (reusing PushConstants cbuffer, different semantic per-kernel):
//   SourceIdx  → SRV: current BakedHeightmap (read-only, for flatten/smooth)
//   OutputIdx  → UAV: DeltaMap (R32_Float, written to)
//   StampBufIdx → SRV: StructuredBuffer<float2> stroke points (terrain UV)
//   BlendMode  → brush mode: 0=raise, 1=lower, 2=flatten, 3=smooth
//   Opacity    → brush strength [0..1]
//   StampCount → number of stroke points
//   BrushRadius, BrushFalloff, BrushTargetHeight — slots 6-8

// Distance from point P to line segment AB
float DistToSegment(float2 P, float2 A, float2 B)
{
    float2 AB = B - A;
    float lenSq = dot(AB, AB);
    if (lenSq < 1e-10)
        return length(P - A); // Degenerate segment (A == B)

    float t = saturate(dot(P - A, AB) / lenSq);
    float2 proj = A + t * AB;
    return length(P - proj);
}

[numthreads(8, 8, 1)]
void CS_PaintBrush(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> DeltaMap = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    DeltaMap.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    StructuredBuffer<float2> StrokePoints = ResourceDescriptorHeap[StampBufIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);

    // Find minimum distance to any segment in the stroke polyline
    float minDist = 1e10;
    uint pointCount = StampCount;

    if (pointCount == 1)
    {
        minDist = length(uv - StrokePoints[0]);
    }
    else
    {
        for (uint i = 0; i < pointCount - 1; i++)
        {
            float d = DistToSegment(uv, StrokePoints[i], StrokePoints[i + 1]);
            minDist = min(minDist, d);
        }
    }

    // Outside brush radius — skip
    if (minDist >= BrushRadius) return;

    // Sinusoidal falloff via smoothstep: 1 at center, smooth S-curve to 0 at edge
    float t = minDist / BrushRadius;
    float falloff = smoothstep(1.0, 0.0, t);
    falloff = pow(falloff, BrushFalloff); // Additional user-controlled shaping

    float weight = falloff * Opacity;
    float currentDelta = DeltaMap[dtid.xy];

    // BrushMode: 0=raise, 1=lower, 2=flatten, 3=smooth
    switch (BlendMode)
    {
        case 0: // Raise
            DeltaMap[dtid.xy] = currentDelta + weight;
            break;

        case 1: // Lower
            DeltaMap[dtid.xy] = currentDelta - weight;
            break;

        case 2: // Flatten
        {
            Texture2D<float> BakedHeight = ResourceDescriptorHeap[SourceIdx];
            float2 sampleUV = (float2(dtid.xy) + 0.5) / float2(w, h);
            float currentHeight = BakedHeight.SampleLevel(sampLinear, sampleUV, 0);

            // Target = total height at brush center
            float2 centerUV = StrokePoints[0];
            float centerBase = BakedHeight.SampleLevel(sampLinear, centerUV, 0);
            uint2 centerTexel = uint2(centerUV * float2(w, h));
            float centerDelta = DeltaMap[centerTexel];
            float targetH = centerBase + centerDelta;

            float desiredDelta = targetH - currentHeight;
            DeltaMap[dtid.xy] = lerp(currentDelta, desiredDelta, weight);
            break;
        }

        case 3: // Smooth
        {
            // Average current delta with neighbors
            float2 texel = 1.0 / float2(w, h);
            float sum = currentDelta;
            sum += DeltaMap[dtid.xy + int2(-1, 0)];
            sum += DeltaMap[dtid.xy + int2( 1, 0)];
            sum += DeltaMap[dtid.xy + int2( 0,-1)];
            sum += DeltaMap[dtid.xy + int2( 0, 1)];
            float avg = sum / 5.0;
            DeltaMap[dtid.xy] = lerp(currentDelta, avg, weight);
            break;
        }
    }
}

// ── CS_ClearDelta: Zero the delta map ──
[numthreads(8, 8, 1)]
void CS_ClearDelta(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> DeltaMap = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    DeltaMap.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;
    DeltaMap[dtid.xy] = 0;
}

// ── CS_ImportChannel: Extract a single channel from a source texture ──
// SourceIdx → any texture, OutputIdx → R16 UAV
// BlendMode repurposed as channel index: 0=R, 1=G, 2=B, 3=A
[numthreads(8, 8, 1)]
void CS_ImportChannel(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D Source = ResourceDescriptorHeap[SourceIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float4 sample = Source.SampleLevel(sampLinear, uv, 0);

    float value = sample[BlendMode]; // BlendMode = channel index (0-3)
    Output[dtid.xy] = value;
}

// ── CS_PackChannels: Pack up to 4 R16 sources into one RGBA pixel ──
// StampBufIdx → StructuredBuffer<uint> with up to 4 source SRV bindless indices
// OutputIdx → UAV: RWTexture2D<float4>
// BlendMode → number of valid channels (1-4)
[numthreads(8, 8, 1)]
void CS_PackChannels(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float4> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    StructuredBuffer<uint> SourceIndices = ResourceDescriptorHeap[StampBufIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float4 packed = float4(0, 0, 0, 0);

    uint channelCount = BlendMode;
    for (uint c = 0; c < channelCount && c < 4; c++)
    {
        uint srcIdx = SourceIndices[c];
        if (srcIdx != 0)
        {
            Texture2D<float> Src = ResourceDescriptorHeap[srcIdx];
            packed[c] = Src.SampleLevel(sampLinear, uv, 0);
        }
    }

    Output[dtid.xy] = packed;
}

// ── CS_BrushRaycast: GPU ray march against baked heightmap ──
// Single-thread dispatch [1,1,1]. Marches a ray against the heightmap SRV,
// converts hit to terrain UV, writes to stroke buffer for CS_PaintBrush.
//
// Push constants:
//   SourceIdx    → SRV: baked heightmap
//   StampBufIdx  → UAV: RWStructuredBuffer<float2> stroke buffer (raycast result)
//   Slots 9-19   → ray origin/dir, terrain transform
[numthreads(1, 1, 1)]
void CS_BrushRaycast(uint3 dtid : SV_DispatchThreadID)
{
    Texture2D<float> Heightmap = ResourceDescriptorHeap[SourceIdx];
    RWStructuredBuffer<float2> StrokeBuf = ResourceDescriptorHeap[StampBufIdx];

    float3 rayOrigin = float3(RayOriginX, RayOriginY, RayOriginZ);
    float3 rayDir    = float3(RayDirX, RayDirY, RayDirZ);
    float2 terrainSize = float2(TerrainSizeX, TerrainSizeZ);
    float maxHeight = TerrainMaxHeight;

    // Terrain AABB: (TerrainOriginX, 0, TerrainOriginZ) to (TerrainOriginX + SizeX, MaxHeight, TerrainOriginZ + SizeZ)
    float3 aabbMin = float3(TerrainOriginX, 0, TerrainOriginZ);
    float3 aabbMax = float3(TerrainOriginX + terrainSize.x, maxHeight, TerrainOriginZ + terrainSize.y);

    // Default: miss
    StrokeBuf[0] = float2(-1, -1);

    // Ray-AABB intersection (slab method)
    float3 invDir = 1.0 / rayDir;
    float3 t0 = (aabbMin - rayOrigin) * invDir;
    float3 t1 = (aabbMax - rayOrigin) * invDir;

    float3 tmin3 = min(t0, t1);
    float3 tmax3 = max(t0, t1);

    float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tExit  = min(min(tmax3.x, tmax3.y), tmax3.z);

    // No intersection with AABB at all
    if (tEnter > tExit || tExit < 0) return;

    // Clamp to forward direction
    tEnter = max(tEnter, 0);

    // Step size: proportional to terrain texel size for precision
    uint hW, hH;
    Heightmap.GetDimensions(hW, hH);
    float texelWorld = max(terrainSize.x / (float)hW, terrainSize.y / (float)hH);
    float stepSize = texelWorld; // one texel per step

    int maxSteps = min((int)((tExit - tEnter) / stepSize) + 1, 2048);

    float prevT = tEnter;
    float3 prevP = rayOrigin + rayDir * tEnter;
    float2 prevUV = (prevP.xz - aabbMin.xz) / terrainSize;
    float prevH = Heightmap.SampleLevel(sampLinear, saturate(prevUV), 0);
    float prevDelta = prevP.y - prevH * maxHeight;

    for (int i = 1; i <= maxSteps; i++)
    {
        float t = tEnter + i * stepSize;
        if (t > tExit) t = tExit;

        float3 p = rayOrigin + rayDir * t;
        float2 uv = (p.xz - aabbMin.xz) / terrainSize;

        // Clamp UV for safety
        uv = saturate(uv);

        float h = Heightmap.SampleLevel(sampLinear, uv, 0);
        float terrainY = h * maxHeight;
        float delta = p.y - terrainY;

        // Sign change: ray crossed the terrain surface
        if (delta < 0 && prevDelta >= 0)
        {
            // Binary refine between prevT and t
            float lo = prevT;
            float hi = t;
            for (int j = 0; j < 16; j++)
            {
                float mid = (lo + hi) * 0.5;
                float3 mp = rayOrigin + rayDir * mid;
                float2 muv = saturate((mp.xz - aabbMin.xz) / terrainSize);
                float mh = Heightmap.SampleLevel(sampLinear, muv, 0);
                if (mp.y - mh * maxHeight < 0)
                    hi = mid;
                else
                    lo = mid;
            }

            // Final hit position → UV
            float3 hitPos = rayOrigin + rayDir * ((lo + hi) * 0.5);
            float2 hitUV = (hitPos.xz - aabbMin.xz) / terrainSize;
            StrokeBuf[0] = hitUV;
            return;
        }

        prevT = t;
        prevDelta = delta;

        if (t >= tExit) break;
    }
}
