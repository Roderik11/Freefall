// Motion vectors (DeferredRenderer.CameraMotion, UPSCALING_PLAN.md stage 2): where each pixel's surface was on screen
// last frame, as last frame's UV minus this frame's, with neither frame's jitter in it (what DLSS and FSR take).
//
// The G-buffer's Motion target (RG16F, render size) starts each frame at MOTION_UNWRITTEN. Effects that know how their
// surface moved on its own (a moved entity, a posed character, a plant in the wind) write their motion into it; every
// pixel nobody wrote is a surface that stood still, and CSCameraMotion gives it the camera's motion: the depth's point
// projected by last frame's camera and by this frame's. Run after the forward pass, so the water's depth is in.
//
// CSMotionView paints the field over the finished picture for DebugVizMode.Motion: the hue is the direction a pixel
// came from, the brightness how far (MotionViewPixels for full), grey where nothing moved.

#pragma kernel CSCameraMotion
#pragma kernel CSMotionView

cbuffer PushConstants : register(b3)
{
    uint DepthIdx;          // the hardware depth (reverse Z), render size
    uint MotionUAVIdx;      // Motion (RG16F), render size: read and written
    uint MotionSrvIdx;      // the same, read (CSMotionView)
    uint OutputUAVIdx;      // LdrOutput (CSMotionView), display size
    uint RenderWidth;
    uint RenderHeight;
    uint DisplayWidth;
    uint DisplayHeight;
};

cbuffer Params : register(b4)
{
    row_major float4x4 JitteredInverse;     // this frame's view-projection without translation, jittered as drawn, inverted
    row_major float4x4 CurrentVP;           // this frame's view-projection without translation or jitter
    row_major float4x4 PreviousVP;          // last frame's, the same way
    float3 CameraMove;                      // this frame's camera position minus last frame's
    float MotionViewPixels;                 // CSMotionView: a move this many pixels long shows at full brightness
};

// What the G-buffer's Motion target is cleared to: no effect writes it (a motion of 32768 screens)
#define MOTION_UNWRITTEN 32768.0

float2 ClipToUV(float4 clip)
{
    float2 ndc = clip.xy / clip.w;
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

[numthreads(8, 8, 1)]
void CSCameraMotion(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= RenderWidth || id.y >= RenderHeight)
        return;

    RWTexture2D<float2> motion = ResourceDescriptorHeap[MotionUAVIdx];
    if (motion[id.xy].x < MOTION_UNWRITTEN * 0.5)
        return;     // the surface's own motion

    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthIdx];
    float depth = depthTex.Load(int3(id.xy, 0));

    // The point the pixel shows, relative to this frame's camera: through the pixel's centre as it was drawn (jittered).
    // The sky (depth 0) is the far plane, a point 16 km out: the camera's own move hardly shifts it.
    float2 uv = (float2(id.xy) + 0.5) / float2(RenderWidth, RenderHeight);
    float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    float4 rel = mul(clip, JitteredInverse);
    rel.xyz /= rel.w;

    // Where it is on this frame's screen and where it was on last frame's, both without jitter
    float2 now = ClipToUV(mul(float4(rel.xyz, 1.0), CurrentVP));
    float4 prevClip = mul(float4(rel.xyz + CameraMove, 1.0), PreviousVP);
    float2 before = prevClip.w > 0.0 ? ClipToUV(prevClip) : now;   // behind last frame's camera: no history there anyway

    motion[id.xy] = before - now;
}

float3 Hue(float h)
{
    float3 c = saturate(abs(frac(h + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return c;
}

[numthreads(8, 8, 1)]
void CSMotionView(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= DisplayWidth || id.y >= DisplayHeight)
        return;

    Texture2D<float2> motion = ResourceDescriptorHeap[MotionSrvIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];

    uint2 at = uint2((float2(id.xy) + 0.5) * float2(RenderWidth, RenderHeight) / float2(DisplayWidth, DisplayHeight));
    float2 m = motion.Load(int3(at, 0));
    if (m.x >= MOTION_UNWRITTEN * 0.5)
    {
        output[id.xy] = float4(1.0, 0.0, 1.0, 1.0);    // never resolved: a bug
        return;
    }

    // In this frame's pixels (the display's, which the picture is shown in)
    float2 pixels = m * float2(DisplayWidth, DisplayHeight);
    float len = length(pixels);
    float amount = saturate(len / max(MotionViewPixels, 0.001));
    float3 colour = len > 0.001 ? Hue(atan2(pixels.y, pixels.x) / 6.2831853 + 0.5) : 0.5;
    output[id.xy] = float4(lerp(float3(0.5, 0.5, 0.5), colour, amount), 1.0);
}
