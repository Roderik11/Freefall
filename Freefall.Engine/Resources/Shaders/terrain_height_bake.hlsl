// terrain_height_bake.hlsl — Composites terrain HeightLayers and StampGroups
// into a final R32_Float heightmap.
//
// HeightLayers: dispatched per-layer (CS_ImportLayer, CS_NoiseLayer)
// StampGroups: dispatched per-group with a StructuredBuffer of instances (CS_StampGroup)
// Erosion: multi-pass iterative simulation (CS_ErosionInit, CS_ErosionStep)
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
#pragma kernel CS_NoiseLayer
#pragma kernel CS_ErosionFilter

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
    uint FlipV;             // slot 20 — flip V of stroke points for splatmap targets

    // ── Noise layer params (CS_NoiseLayer) ──
    uint NoiseType;       // slot 21 — 0=Simplex, 1=Perlin, 2=Ridged, 3=Billow
    uint Octaves;         // slot 22 — number of fBm octaves
    float Frequency;      // slot 23 — base frequency
    float Amplitude;      // slot 24 — output amplitude scale
    float Lacunarity;     // slot 25 — per-octave frequency multiplier
    float Persistence;    // slot 26 — per-octave amplitude decay
    float OffsetX;        // slot 27 — world-space noise offset X
    float OffsetY;        // slot 28 — world-space noise offset Y
    uint NoiseSeed;       // slot 29 — seed for noise permutation
    uint ErosionMode;     // slot 30 — 0=Hydraulic, 1=Thermal, 2=Both (reused as NoiseLUTIdx for CS_NoiseLayer)

    // ── Noise terrace + mask params (CS_NoiseLayer only) ──
    uint TerraceSteps;    // slot 31 — 0=disabled, N = number of terrace shelves
    float TerraceSmoothness; // slot 32 — 0=sharp, 1=fully rounded transitions
    float MaskCenterX;    // slot 33 — UV-space mask center X
    float MaskCenterY;    // slot 34 — UV-space mask center Y
    float MaskRadius;     // slot 35 — UV-space radius, 0=disabled (full terrain)
    float MaskFalloff;    // slot 36 — mask edge falloff exponent

    // ── Erosion params (CS_ErosionStep, reuses slots 23-28 since never concurrent) ──
    // RainRate      → slot 23 (Frequency)
    // SedimentCap   → slot 24 (Amplitude)
    // DepositionRate→ slot 25 (Lacunarity)
    // DissolutionRate→slot 26 (Persistence)
    // Evaporation   → slot 27 (OffsetX)
    // TalusAngle    → slot 28 (OffsetY)
    // ThermalRate   → slot 6  (BrushRadius)
    // Erosion aux UAVs: WaterIdx=slot 2 (StampBufIdx), SedimentIdx=slot 5 (StampCount)
    // PingPong source SRV: SourceIdx=slot 0
};

SamplerState sampLinear : register(s0);

// Per-stamp instance data (matches C# StampDataGPU)
struct StampData
{
    float2 Position;       // terrain-space UV [0..1]
    float Radius;          // UV-space radius
    float Strength;        // height multiplier
    float Falloff;         // plateau fraction before fade begins
    float Rotation;        // degrees
    uint BrushIdx;         // bindless SRV index for this stamp's brush texture
    uint BlendMode;        // 0=Set, 1=Add, 2=Max, 3=Lerp, 4=Min
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
        case 4: return min(prev, value);               // Min
        case 5: return lerp(prev, BrushTargetHeight, opacity); // Flatten toward target
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

// ── CS_StampGroup: Apply all stamps, each with its own brush and blend mode ──
// Per-stamp blend via target+lerp: compute what the ideal height should be,
// then blend toward it using falloff as spatial weight.
[numthreads(8, 8, 1)]
void CS_StampGroup(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    StructuredBuffer<StampData> Stamps = ResourceDescriptorHeap[StampBufIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float result = Output[dtid.xy];
    bool changed = false;

    for (uint i = 0; i < StampCount; i++)
    {
        StampData s = Stamps[i];
        float2 delta = uv - s.Position;
        float dist = length(delta);

        if (dist > s.Radius) continue;

        // Falloff: full strength inside s.Falloff fraction, smooth ramp to 0 at edge
        float t = dist / s.Radius;
        float w_falloff = 1.0 - smoothstep(s.Falloff, 1.0, t);

        // Rotate delta into brush UV space
        float rad = s.Rotation * 3.14159265 / 180.0;
        float cosR = cos(rad);
        float sinR = sin(rad);
        float2 rotDelta = float2(
            delta.x * cosR + delta.y * sinR,
           -delta.x * sinR + delta.y * cosR
        );

        // Sample per-stamp brush texture
        float2 brushUV = (rotDelta / s.Radius) * 0.5 + 0.5;
        Texture2D<float> Brush = ResourceDescriptorHeap[s.BrushIdx];
        float bv = Brush.SampleLevel(sampLinear, brushUV, 0) * s.Strength;

        // Compute target value for this blend mode, then lerp by falloff
        float target;
        switch (s.BlendMode)
        {
            case 0:  target = bv; break;                 // Set
            case 1:  target = result + bv; break;        // Add
            case 2:  target = max(result, bv); break;    // Max
            case 3:  target = bv; break;                 // Lerp (same as Set for stamps)
            case 4:  target = min(result, bv); break;    // Min
            default: target = result + bv; break;        // fallback: Add
        }

        result = lerp(result, target, w_falloff);
        changed = true;
    }

    if (changed)
        Output[dtid.xy] = result;
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
        float2 sp = StrokePoints[0];
        if (FlipV) sp.y = 1.0 - sp.y;
        minDist = length(uv - sp);
    }
    else
    {
        for (uint i = 0; i < pointCount - 1; i++)
        {
            float2 a = StrokePoints[i];
            float2 b = StrokePoints[i + 1];
            if (FlipV) { a.y = 1.0 - a.y; b.y = 1.0 - b.y; }
            float d = DistToSegment(uv, a, b);
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
            if (FlipV) // Splatmap: stamp max — falloff defines gradient, strength = peak
                DeltaMap[dtid.xy] = max(currentDelta, weight);
            else
                DeltaMap[dtid.xy] = currentDelta + weight;
            break;

        case 1: // Lower
            if (FlipV) // Splatmap/Density: subtractive erase, clamped to 0
                DeltaMap[dtid.xy] = max(0.0, currentDelta - weight);
            else
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
// OutputIdx → UAV: RWTexture2DArray<float4> (single-slice view)
// BlendMode → number of valid channels (1-4)
[numthreads(8, 8, 1)]
void CS_PackChannels(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2DArray<float4> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h, slices;
    Output.GetDimensions(w, h, slices);
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

    Output[uint3(dtid.xy, 0)] = packed;
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

// ═══════════════════════════════════════════════════════════════════════════
// CS_NoiseLayer — Texture-based fBm noise height generation
// Uses a pre-baked tileable noise LUT instead of GPU math noise to avoid
// precision artifacts (staircase/banding) on various GPU architectures.
// ═══════════════════════════════════════════════════════════════════════════

// NoiseLUTIdx reuses ErosionMode slot (30) — noise and erosion kernels never run concurrently
#define NoiseLUTIdx ErosionMode

// Sample the tileable noise LUT (R16G16_Float) at a given position.
// R and G hold two independent noise fields for per-octave decorrelation.
float sampleNoiseLUT(Texture2D<float2> lut, float2 p, uint octave)
{
    float2 s = lut.SampleLevel(sampLinear, p, 0);
    return (octave & 1) ? s.g : s.r;
}

// ── fBm accumulation with type-specific strategies ──
// Each noise type has a distinct accumulation to produce visually different terrain:
//   Simplex/Perlin: Standard fBm — smooth rolling hills
//   Ridged: Musgrave's ridged multifractal — sharp mountain ridges
//   Billow: Abs-folded fBm — puffy dome-like terrain
float fbmNoise(Texture2D<float2> lut, float2 p, uint type, uint octaves,
               float lacunarity, float persistence, float seed)
{
    // Seed-based initial rotation to make different seeds produce different patterns
    float seedAngle = seed * 1.9635;
    float cs = cos(seedAngle), sn = sin(seedAngle);
    float2x2 seedRot = float2x2(cs, -sn, sn, cs);
    p = mul(seedRot, p);

    // Per-octave UV transform helper
    #define OCTAVE_UV(i, freq) \
        float angle##i = 0.5 + (i) * 1.37; \
        float c##i = cos(angle##i), s##i = sin(angle##i); \
        float2 rp = float2(c##i * p.x - s##i * p.y, s##i * p.x + c##i * p.y) * (freq); \
        rp += float2((i) * 7.31, (i) * 11.17)

    float value = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float maxAmp = 0.0;

    if (type <= 1) // Simplex / Perlin — standard fBm
    {
        for (uint i = 0; i < octaves && i < 12; i++)
        {
            OCTAVE_UV(i, freq);
            value += amp * sampleNoiseLUT(lut, rp, i);
            maxAmp += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        return value / maxAmp;
    }
    else if (type == 2) // Ridged multifractal
    {
        float weight = 1.0;
        const float offset = 1.0; // ridge offset
        const float gain = 2.0;   // sharpness

        for (uint i = 0; i < octaves && i < 12; i++)
        {
            OCTAVE_UV(i, freq);
            float n = sampleNoiseLUT(lut, rp, i);
            // Fold: create sharp ridges where noise crosses 0.5
            float signal = offset - abs(n * 2.0 - 1.0);
            signal *= signal; // sharpen ridges
            signal *= weight; // detail concentrates in valleys
            value += signal * amp;
            maxAmp += amp;
            // Next octave weight depends on current signal
            weight = saturate(signal * gain);
            amp *= persistence;
            freq *= lacunarity;
        }
        return value / maxAmp;
    }
    else // Billow (type == 3) — abs-folded gives dome shapes
    {
        for (uint i = 0; i < octaves && i < 12; i++)
        {
            OCTAVE_UV(i, freq);
            float n = sampleNoiseLUT(lut, rp, i);
            // Abs fold: creates rounded dome/pillow shapes
            float signal = abs(n * 2.0 - 1.0);
            value += amp * signal;
            maxAmp += amp;
            amp *= persistence;
            freq *= lacunarity;
        }
        return value / maxAmp;
    }

    #undef OCTAVE_UV
}

[numthreads(8, 8, 1)]
void CS_NoiseLayer(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float2> NoiseLUT = ResourceDescriptorHeap[NoiseLUTIdx];

    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float2 p = (uv + float2(OffsetX, OffsetY)) * Frequency;

    float noise = fbmNoise(NoiseLUT, p, NoiseType, Octaves, Lacunarity, Persistence, (float)NoiseSeed);

    // ── Terracing post-process ──
    if (TerraceSteps > 0)
    {
        float steps = (float)TerraceSteps;
        float quantized = floor(noise * steps) / steps;
        float frac_part = frac(noise * steps);
        // Smooth-step between shelves for natural-looking ledges
        float smooth_frac = smoothstep(0.0, 1.0, frac_part) / steps;
        noise = lerp(quantized, quantized + smooth_frac, TerraceSmoothness);
    }

    float value = noise * Amplitude;

    // ── Spatial mask (radial falloff) ──
    if (MaskRadius > 0.0)
    {
        float2 delta = uv - float2(MaskCenterX, MaskCenterY);
        float dist = length(delta);
        float mask = 1.0 - saturate(pow(dist / MaskRadius, MaskFalloff));
        value *= mask;
    }

    float prev = Output[dtid.xy];
    Output[dtid.xy] = Blend(prev, value, BlendMode, Opacity);
}

// ═══════════════════════════════════════════════════════════════════════════
// CS_ErosionFilter — Single-pass noise-based erosion filter
// ═══════════════════════════════════════════════════════════════════════════
//
// Ported from "Advanced Terrain Erosion Filter" by Rune Skovbo Johansen
// (https://www.shadertoy.com/view/wXcfWn) — MPL-2.0 licensed.
//
// Uses Phacelle Noise to generate gradient-aligned gullies that produce
// crisp branching patterns in a single dispatch (no iterative simulation).
//
// Slot reuse (erosion filter kernels, never concurrent with noise/stamps):
//   SourceIdx          (0)  → SRV: current baked heightmap
//   OutputIdx          (1)  → UAV: output heightmap
//   BrushRadius        (6)  → EF Scale
//   BrushFalloff       (7)  → EF AssumedSlopeValue
//   BrushTargetHeight  (8)  → EF AssumedSlopeAmount
//   Frequency          (23) → EF Strength
//   Amplitude          (24) → EF GullyWeight
//   Lacunarity         (25) → EF Detail
//   Persistence        (26) → EF Lacunarity
//   OffsetX            (27) → EF Gain
//   OffsetY            (28) → EF CellScale
//   NoiseSeed          (29) → EF Octaves (uint)
//   ErosionMode        (30) → EF Normalization (asfloat)
//   TerraceSteps       (31) → EF RidgeRounding (asfloat)
//   TerraceSmoothness  (32) → EF CreaseRounding
//   MaskCenterX        (33) → EF RoundingInputMult
//   MaskCenterY        (34) → EF RoundingOctaveMult
//   MaskRadius         (35) → EF OnsetInput
//   MaskFalloff        (36) → EF OnsetOctave

// Aliases for readability
#define EFScale           BrushRadius
#define EFAssumedVal      BrushFalloff
#define EFAssumedAmt      BrushTargetHeight
#define EFStrength        Frequency
#define EFGullyWeight     Amplitude
#define EFDetail          Lacunarity
#define EFLacunarity      Persistence
#define EFGain            OffsetX
#define EFCellScale       OffsetY
#define EFOctaves         NoiseSeed
#define EFNormalization   asfloat(ErosionMode)
#define EFRidgeRounding   asfloat(TerraceSteps)
#define EFCreaseRounding  TerraceSmoothness
#define EFRoundInputMul   MaskCenterX
#define EFRoundOctMul     MaskCenterY
#define EFOnsetInput      MaskRadius
#define EFOnsetOctave     MaskFalloff

#define EF_TAU 6.28318530717959

// ── Hash function (algebraic, no texture lookups) ──
float2 ef_hash(float2 x) {
    const float2 k = float2(0.3183099, 0.3678794);
    x = x * k + k.yx;
    return -1.0 + 2.0 * frac(16.0 * k * frac(x.x * x.y * (x.x + x.y)));
}

// ── Phacelle Noise: directional stripe noise aligned with input vector ──
// Produces cosine/sine wave pairs blended across Worley-like cells.
// Copyright (c) 2025 Rune Skovbo Johansen — MPL-2.0
float4 EF_PhacelleNoise(float2 p, float2 normDir, float freq, float offset, float normalization) {
    float2 sideDir = normDir.yx * float2(-1.0, 1.0) * freq * EF_TAU;
    offset *= EF_TAU;

    float2 pInt = floor(p);
    float2 pFrac = frac(p);
    float2 phaseDir = 0;
    float weightSum = 0;

    [unroll]
    for (int i = -1; i <= 2; i++) {
        [unroll]
        for (int j = -1; j <= 2; j++) {
            float2 gridOffset = float2(i, j);
            float2 gridPoint = pInt + gridOffset;
            float2 randomOffset = ef_hash(gridPoint) * 0.5;
            float2 vectorFromCellPoint = pFrac - gridOffset - randomOffset;

            float sqrDist = dot(vectorFromCellPoint, vectorFromCellPoint);
            float weight = exp(-sqrDist * 2.0);
            weight = max(0.0, weight - 0.01111);
            weightSum += weight;

            float waveInput = dot(vectorFromCellPoint, sideDir) + offset;
            phaseDir += float2(cos(waveInput), sin(waveInput)) * weight;
        }
    }

    float2 interpolated = phaseDir / weightSum;
    float magnitude = sqrt(dot(interpolated, interpolated));
    magnitude = max(1.0 - normalization, magnitude);
    return float4(interpolated / magnitude, sideDir);
}

// ── Helper functions ──

float ef_pow_inv(float t, float power) {
    return 1.0 - pow(1.0 - saturate(t), power);
}

float ef_ease_out(float t) {
    float v = 1.0 - saturate(t);
    return 1.0 - v * v;
}

float ef_smooth_start(float t, float smoothing) {
    if (t >= smoothing)
        return t - 0.5 * smoothing;
    return 0.5 * t * t / smoothing;
}

float2 ef_safe_normalize(float2 n) {
    float l = length(n);
    return (abs(l) > 1e-10) ? (n / l) : n;
}

// ── Advanced Terrain Erosion Filter ──
// Copyright (c) 2025 Rune Skovbo Johansen — MPL-2.0
//
// Returns float4(heightDelta, slopeDelta.xy, magnitude).
float4 EF_ErosionFilter(
    float2 p, float3 heightAndSlope, float fadeTarget,
    float strength, float gullyWeight, float detail,
    float4 rounding, float2 onset, float2 assumedSlope,
    float scale, uint octaves, float lacunarity,
    float gain, float cellScale, float normalization
) {
    strength *= scale;
    fadeTarget = clamp(fadeTarget, -1.0, 1.0);

    float3 inputHeightAndSlope = heightAndSlope;
    float freq = 1.0 / (scale * cellScale);
    float slopeLength = max(length(heightAndSlope.yz), 1e-10);
    float magnitude = 0.0;
    float roundingMult = 1.0;

    float roundingForInput = lerp(rounding.y, rounding.x, saturate(fadeTarget + 0.5)) * rounding.z;
    float combiMask = ef_ease_out(ef_smooth_start(slopeLength * onset.x, roundingForInput * onset.x));

    // Gully slope: mix of actual slope and assumed slope
    float2 gullySlope = lerp(heightAndSlope.yz,
        heightAndSlope.yz / slopeLength * assumedSlope.x, assumedSlope.y);

    for (uint i = 0; i < octaves && i < 8; i++) {
        float4 phacelle = EF_PhacelleNoise(p * freq, ef_safe_normalize(gullySlope),
            cellScale, 0.25, normalization);
        phacelle.zw *= -freq;
        float sloping = abs(phacelle.y);

        // Add normalized slope for gully direction (straight gullies technique)
        gullySlope += sign(phacelle.y) * phacelle.zw * strength * gullyWeight;

        // Gullies: height offset (x) and derivative (yz)
        float3 gullies = float3(phacelle.x, phacelle.y * phacelle.zw);
        // Fade towards fadeTarget based on combiMask
        float3 fadedGullies = lerp(float3(fadeTarget, 0, 0), gullies * gullyWeight, combiMask);
        heightAndSlope += fadedGullies * strength;
        magnitude += strength;

        // Update fadeTarget for next octave (stacked fading)
        fadeTarget = fadedGullies.x;

        // Update mask with this octave's ridge/crease contribution
        float roundingForOctave = lerp(rounding.y, rounding.x,
            saturate(phacelle.x + 0.5)) * roundingMult;
        float newMask = ef_ease_out(ef_smooth_start(sloping * onset.y,
            roundingForOctave * onset.y));
        combiMask = ef_pow_inv(combiMask, detail) * newMask;

        // Prepare next octave
        strength *= gain;
        freq *= lacunarity;
        roundingMult *= rounding.w;
    }

    float3 heightAndSlopeDelta = heightAndSlope - inputHeightAndSlope;
    return float4(heightAndSlopeDelta, magnitude);
}

// ── CS_ErosionFilter: Single-dispatch erosion filter applied to baked heightmap ──
[numthreads(8, 8, 1)]
void CS_ErosionFilter(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Output = ResourceDescriptorHeap[OutputIdx];
    uint w, h;
    Output.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    Texture2D<float> Source = ResourceDescriptorHeap[SourceIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    float2 texel = 1.0 / float2(w, h);

    // Sample height and compute gradient via finite differences
    float hC = Source.SampleLevel(sampLinear, uv, 0);
    float hL = Source.SampleLevel(sampLinear, uv - float2(texel.x, 0), 0);
    float hR = Source.SampleLevel(sampLinear, uv + float2(texel.x, 0), 0);
    float hD = Source.SampleLevel(sampLinear, uv - float2(0, texel.y), 0);
    float hU = Source.SampleLevel(sampLinear, uv + float2(0, texel.y), 0);
    float2 gradient = float2(hR - hL, hU - hD) * 0.5 / texel;

    // heightAndSlope: (height, dh/dx, dh/dy)
    float3 heightAndSlope = float3(hC, gradient);

    // Fade target: map height [0,1] → [-1,1] (valleys→peaks)
    float fadeTarget = clamp(hC * 2.0 - 1.0, -1.0, 1.0);

    // Rounding: (ridgeRounding, creaseRounding, inputMult, octaveMult)
    float4 rounding = float4(EFRidgeRounding, EFCreaseRounding, EFRoundInputMul, EFRoundOctMul);
    float2 onset = float2(EFOnsetInput, EFOnsetOctave);
    float2 assumedSlope = float2(EFAssumedVal, EFAssumedAmt);

    // Run the erosion filter
    float4 erosion = EF_ErosionFilter(
        uv, heightAndSlope, fadeTarget,
        EFStrength, EFGullyWeight, EFDetail,
        rounding, onset, assumedSlope,
        EFScale, EFOctaves, EFLacunarity,
        EFGain, EFCellScale, EFNormalization
    );

    // erosion.x = height delta, erosion.w = magnitude
    // Offset to preserve peaks/valleys: raise valleys, lower peaks
    float offset = -fadeTarget * erosion.w;
    float eroded = hC + erosion.x + offset;

    float prev = Output[dtid.xy];
    Output[dtid.xy] = Blend(prev, eroded, BlendMode, Opacity);
}
