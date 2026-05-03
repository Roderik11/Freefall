// Screen-Space Shadow Projection
// Based on Bend Studio's technique (Days Gone / PS5)
// Original: Copyright 2023 Sony Interactive Entertainment, Apache 2.0
// Adapted for Freefall Engine bindless D3D12 architecture.
//
// Dispatch parameters are computed on CPU via ScreenSpaceShadows.BuildDispatchList().
// Typically 4-8 dispatches per light (one per screen quadrant around the projected light position).
// No GPU sync required between dispatches.

#pragma kernel CSScreenSpaceShadow

#define WAVE_SIZE 64
#define SAMPLE_COUNT 60
#define HARD_SHADOW_SAMPLES 4
#define FADE_OUT_SAMPLES 8
#define READ_COUNT (SAMPLE_COUNT / WAVE_SIZE + 2)

cbuffer PushConstants : register(b3)
{
    uint DepthTexIdx;       // Hardware depth buffer SRV (reverse-Z: near=1, far=0)
    uint OutputUAVIdx;      // Output R8_UNORM shadow texture UAV
    uint WaveOffsetXIdx;    // Per-dispatch wave offset X (reinterpret as float)
    uint WaveOffsetYIdx;    // Per-dispatch wave offset Y (reinterpret as float)
};

cbuffer Params : register(b4)
{
    float4 LightCoordinate; // XY = screen pixel pos of light, Z = projected depth, W = sign (+1/-1)
    float2 InvDepthTextureSize; // 1/width, 1/height
    float  _pad_align0;
    float  _pad_align1;

    float  SurfaceThickness;    // Assumed pixel thickness (0.005 = 0.5%)
    float  BilinearThreshold;   // Edge detection threshold (0.02 = 2%)
    float  ShadowContrast;      // Contrast boost (4.0 recommended)
    float  FarDepthValue;       // Depth buffer far plane value (0 for reverse-Z)

    float  NearDepthValue;      // Depth buffer near plane value (1 for reverse-Z)
    uint   TexWidth;            // Depth texture width
    uint   TexHeight;           // Depth texture height
    uint   _pad0;
};

// ============================================================
// Border-safe depth read (replaces PointBorderSampler)
// ============================================================
float ReadDepth(int2 coord)
{
    if (any(coord < 0) || coord.x >= (int)TexWidth || coord.y >= (int)TexHeight)
        return FarDepthValue;
    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
    return depthTex.Load(int3(coord, 0));
}

// ============================================================
// Compute wavefront ray extents
// ============================================================
void ComputeWavefrontExtents(
    int3 inGroupID, uint inGroupThreadID,
    out float2 outDeltaXY, out float2 outPixelXY,
    out float outPixelDistance, out bool outMajorAxisX)
{
    int2 xy = inGroupID.yz * WAVE_SIZE + int2(
        (int)asfloat(WaveOffsetXIdx),
        (int)asfloat(WaveOffsetYIdx));

    float2 light_xy = floor(LightCoordinate.xy) + 0.5;
    float2 light_xy_fraction = LightCoordinate.xy - light_xy;
    bool reverse_direction = LightCoordinate.w > 0.0f;

    int2 sign_xy = sign(xy);
    bool horizontal = abs(xy.x + sign_xy.y) < abs(xy.y - sign_xy.x);

    int2 axis;
    axis.x = horizontal ? (+sign_xy.y) : (0);
    axis.y = horizontal ? (0) : (-sign_xy.x);

    xy = axis * (int)inGroupID.x + xy;
    float2 xy_f = (float2)xy;

    bool x_axis_major = abs(xy_f.x) > abs(xy_f.y);
    float major_axis = x_axis_major ? xy_f.x : xy_f.y;

    float major_axis_start = abs(major_axis);
    float major_axis_end = abs(major_axis) - (float)WAVE_SIZE;

    float ma_light_frac = x_axis_major ? light_xy_fraction.x : light_xy_fraction.y;
    ma_light_frac = major_axis > 0 ? -ma_light_frac : ma_light_frac;

    float2 start_xy = xy_f + light_xy;
    float2 end_xy = lerp(LightCoordinate.xy, start_xy,
        (major_axis_end + ma_light_frac) / (major_axis_start + ma_light_frac));

    float2 xy_delta = (start_xy - end_xy);

    float thread_step = (float)(inGroupThreadID ^ (reverse_direction ? 0 : (WAVE_SIZE - 1)));

    outPixelXY = lerp(start_xy, end_xy, thread_step / (float)WAVE_SIZE);
    outPixelDistance = major_axis_start - thread_step + ma_light_frac;
    outDeltaXY = xy_delta;
    outMajorAxisX = x_axis_major;
}

// ============================================================
// Groupshared memory for depth data exchange within wavefront
// ============================================================
groupshared float DepthData[READ_COUNT * WAVE_SIZE];

// ============================================================
// Main kernel
// ============================================================
[numthreads(WAVE_SIZE, 1, 1)]
void CSScreenSpaceShadow(uint3 groupID : SV_GroupID, uint3 threadID : SV_GroupThreadID)
{
    float2 xy_delta;
    float2 pixel_xy;
    float pixel_distance;
    bool x_axis_major;

    ComputeWavefrontExtents(groupID, threadID.x, xy_delta, pixel_xy, pixel_distance, x_axis_major);

    const float direction = -LightCoordinate.w;
    const float z_sign = NearDepthValue > FarDepthValue ? -1 : +1;

    float sampling_depth[READ_COUNT];
    float shadowing_depth[READ_COUNT];
    float depth_thickness_scale[READ_COUNT];
    float sample_distance[READ_COUNT];

    int i;
    float2 write_xy = floor(pixel_xy);

    [unroll] for (i = 0; i < READ_COUNT; i++)
    {
        int2 read_coord = (int2)floor(pixel_xy);
        float minor_axis = x_axis_major ? pixel_xy.y : pixel_xy.x;

        const float edge_skip = 1e20;

        float2 depths;
        float bilinear = frac(minor_axis) - 0.5;

        float bias = bilinear > 0 ? 1 : -1;
        int2 offset_xy = int2(x_axis_major ? 0 : (int)bias, x_axis_major ? (int)bias : 0);

        depths.x = ReadDepth(read_coord);
        depths.y = ReadDepth(read_coord + offset_xy);

        depth_thickness_scale[i] = abs(FarDepthValue - depths.x);

        bool use_point_filter = abs(depths.x - depths.y) > depth_thickness_scale[i] * BilinearThreshold;

        // BilinearSamplingOffsetMode = true: both depths use the same interpolation
        bilinear = use_point_filter ? 0 : bilinear;
        sampling_depth[i] = lerp(depths.x, depths.y, abs(bilinear));
        // IgnoreEdgePixels: edge pixels don't cast shadows (eliminates striations)
        shadowing_depth[i] = use_point_filter ? edge_skip : sampling_depth[i];

        sample_distance[i] = pixel_distance + (WAVE_SIZE * i) * direction;

        pixel_xy += xy_delta * direction;
    }

    // Write shadow depths to LDS
    [unroll] for (i = 0; i < READ_COUNT; i++)
    {
        float stored_depth = (shadowing_depth[i] - LightCoordinate.z) / sample_distance[i];

        if (i != 0)
        {
            stored_depth = sample_distance[i] > 0 ? stored_depth : 1e10;
        }

        int idx = (i * WAVE_SIZE) + threadID.x;
        DepthData[idx] = stored_depth;
    }

    // Early-out: skip sky pixels (depth == FarDepthValue in reverse-Z)
    // and pixels outside depth bounds. Uses wave intrinsics to skip entire
    // wavefronts when all 64 threads can early-out.
    bool skip_pixel = (sampling_depth[0] <= FarDepthValue);
    bool wave_all_skip = WaveActiveAnyTrue(!skip_pixel) == false;
    if (wave_all_skip)
        return;

    GroupMemoryBarrierWithGroupSync();

    // Individual pixel skip after LDS is populated
    if (skip_pixel)
        return;

    float start_depth = sampling_depth[0];
    start_depth = (start_depth - LightCoordinate.z) / sample_distance[0];

    int sample_index = threadID.x + 1;

    float4 shadow_value = 1;
    float hard_shadow = 1;

    float depth_scale = min(sample_distance[0] + direction, 1.0 / SurfaceThickness)
                      * sample_distance[0] / depth_thickness_scale[0];

    start_depth = start_depth * depth_scale - z_sign;

    // Hard shadow samples (near contact)
    [unroll] for (i = 0; i < HARD_SHADOW_SAMPLES; i++)
    {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
        hard_shadow = min(hard_shadow, depth_delta);
    }

    // Main averaged shadow samples
    [unroll] for (i = HARD_SHADOW_SAMPLES; i < SAMPLE_COUNT - FADE_OUT_SAMPLES; i++)
    {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta);
    }

    // Fade-out samples
    [unroll] for (i = SAMPLE_COUNT - FADE_OUT_SAMPLES; i < SAMPLE_COUNT; i++)
    {
        float depth_delta = abs(start_depth - DepthData[sample_index + i] * depth_scale);
        const float fade_out = (float)(i + 1 - (SAMPLE_COUNT - FADE_OUT_SAMPLES))
                             / (float)(FADE_OUT_SAMPLES + 1) * 0.75;
        shadow_value[i & 3] = min(shadow_value[i & 3], depth_delta + fade_out);
    }

    // Apply contrast
    shadow_value = saturate(shadow_value * ShadowContrast + (1 - ShadowContrast));
    hard_shadow = saturate(hard_shadow * ShadowContrast + (1 - ShadowContrast));

    // Average 4 sub-samples + hard shadow minimum
    float result = dot(shadow_value, 0.25);
    result = min(hard_shadow, result);

    // Write output
    RWTexture2D<float> outputTex = ResourceDescriptorHeap[OutputUAVIdx];
    outputTex[(int2)write_xy] = result;
}
