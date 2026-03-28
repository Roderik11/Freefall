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
#pragma kernel CS_ErosionInit
#pragma kernel CS_ErosionStep

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
// CS_ErosionInit / CS_ErosionStep — Grid-based hydraulic + thermal erosion
// ═══════════════════════════════════════════════════════════════════════════
//
// Slot reuse (erosion kernels):
//   SourceIdx    (0)  → SRV: current baked heightmap
//   OutputIdx    (1)  → UAV: erosion working height
//   StampBufIdx  (2)  → UAV: water map (R32_Float)
//   StampCount   (5)  → UAV: sediment map (R32_Float)
//   Frequency    (23) → RainRate
//   Amplitude    (24) → SedimentCapacity
//   Lacunarity   (25) → DepositionRate
//   Persistence  (26) → DissolutionRate
//   OffsetX      (27) → Evaporation
//   OffsetY      (28) → TalusAngle
//   BrushRadius  (6)  → ThermalRate
//   ErosionMode  (30) → 0=Hydraulic, 1=Thermal, 2=Both
//   NoiseSeed    (29) → Seed for rain variation

// Aliases for erosion readability
#define RainRate     Frequency
#define SedimentCap  Amplitude
#define DepositRate  Lacunarity
#define DissolveRate Persistence
#define Evaporation  OffsetX
#define TalusAngleDeg OffsetY
#define ThermalRate  BrushRadius
#define WaterIdx     StampBufIdx
#define SedimentIdx  StampCount

// ── CS_ErosionInit: Copy input heightmap to working buffer, clear aux maps ──
[numthreads(8, 8, 1)]
void CS_ErosionInit(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Height   = ResourceDescriptorHeap[OutputIdx];
    RWTexture2D<float> Water    = ResourceDescriptorHeap[WaterIdx];
    RWTexture2D<float> Sediment = ResourceDescriptorHeap[SedimentIdx];

    uint w, h;
    Height.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    // Copy current accumulated height into erosion working buffer
    Texture2D<float> Source = ResourceDescriptorHeap[SourceIdx];
    float2 uv = (float2(dtid.xy) + 0.5) / float2(w, h);
    Height[dtid.xy] = Source.SampleLevel(sampLinear, uv, 0);

    Water[dtid.xy] = 0;
    Sediment[dtid.xy] = 0;
}

// ── CS_ErosionStep: One iteration of hydraulic + thermal erosion ──
[numthreads(8, 8, 1)]
void CS_ErosionStep(uint3 dtid : SV_DispatchThreadID)
{
    RWTexture2D<float> Height   = ResourceDescriptorHeap[OutputIdx];
    RWTexture2D<float> Water    = ResourceDescriptorHeap[WaterIdx];
    RWTexture2D<float> Sediment = ResourceDescriptorHeap[SedimentIdx];

    uint w, h;
    Height.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;

    int2 coord = int2(dtid.xy);
    float myH = Height[coord];
    float myW = Water[coord];
    float myS = Sediment[coord];

    // ── Hydraulic erosion (mode 0 or 2) ──
    if (ErosionMode == 0 || ErosionMode == 2)
    {
        // Rain with spatial variation
        float2 hashP = float2(coord) * 0.1 + float2((float)NoiseSeed * 0.37, (float)NoiseSeed * 0.71);
        float rainVar = frac(sin(dot(hashP, float2(127.1, 311.7))) * 43758.5453);
        myW += RainRate * (0.5 + rainVar);

        // 4-neighbor heights
        float hL = (coord.x > 0)         ? Height[coord + int2(-1,  0)] : myH;
        float hR = (coord.x < (int)w - 1) ? Height[coord + int2( 1,  0)] : myH;
        float hD = (coord.y > 0)         ? Height[coord + int2( 0, -1)] : myH;
        float hU = (coord.y < (int)h - 1) ? Height[coord + int2( 0,  1)] : myH;

        // Height difference with average neighbor
        float avgNeighborH = (hL + hR + hD + hU) * 0.25;
        float deltaH = (myH + myW) - avgNeighborH;

        if (deltaH > 0)
        {
            // Flowing out: dissolve terrain
            float erosion = min(deltaH, DissolveRate) * myW;
            myH -= erosion;
            myS += erosion;

            // Outflow reduces local water
            float outflow = min(myW, deltaH * 0.25);
            myW -= outflow;
        }
        else
        {
            // Valley: deposit sediment
            float deposit = min(myS, DepositRate * -deltaH);
            myH += deposit;
            myS -= deposit;
        }

        // Sediment capacity check
        float capacity = SedimentCap * myW * max(0.01, abs(deltaH));
        if (myS > capacity)
        {
            float excess = (myS - capacity) * DepositRate;
            myH += excess;
            myS -= excess;
        }

        // Evaporate
        myW *= (1.0 - Evaporation);
    }

    // ── Thermal erosion (mode 1 or 2) ──
    if (ErosionMode == 1 || ErosionMode == 2)
    {
        // Talus threshold: tan(angle) / resolution
        float talusRad = TalusAngleDeg * 3.14159265 / 180.0;
        float talusThreshold = tan(talusRad) / (float)w;

        float hL = (coord.x > 0)         ? Height[coord + int2(-1,  0)] : myH;
        float hR = (coord.x < (int)w - 1) ? Height[coord + int2( 1,  0)] : myH;
        float hD = (coord.y > 0)         ? Height[coord + int2( 0, -1)] : myH;
        float hU = (coord.y < (int)h - 1) ? Height[coord + int2( 0,  1)] : myH;

        float dL = myH - hL - talusThreshold;
        float dR = myH - hR - talusThreshold;
        float dD = myH - hD - talusThreshold;
        float dU = myH - hU - talusThreshold;

        float totalExcess = max(0, dL) + max(0, dR) + max(0, dD) + max(0, dU);

        if (totalExcess > 0)
        {
            float transfer = totalExcess * ThermalRate * 0.25;
            myH -= transfer;
        }
    }

    // Write back
    Height[coord] = myH;
    Water[coord] = max(0, myW);
    Sediment[coord] = max(0, myS);
}
