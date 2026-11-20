// The temporal upscaler's reactive mask (Graphics/OrderIndependentTransparency.cs ComposeReactive), render size.
//
// The OIT accumulation writes each frame's mask where particles, trails and vfx are now (oit.hlsli OITReactive). That
// alone leaves a ghost where they were: the pixel a bubble has left carries the motion of the water behind it, the
// upscaler reads its history there, finds the bubble and lets it fade over many frames (a fading trail behind every
// bubble and speck, worst over murky water, which rejects nothing). CSReactive therefore also takes last frame's mask
// at the place this pixel's history is read from (the pixel plus its motion), the largest of the 2 × 2 texels round
// it: the upscaler is told not to trust the history wherever a transparent was in it.
//
// CSReactiveView paints the mask over the finished picture for DebugVizMode.Reactive: red as far as the history is
// distrusted.

#pragma kernel CSReactive
#pragma kernel CSReactiveView

cbuffer PushConstants : register(b3)
{
    uint NowIdx;            // Texture2D<float>: this frame's mask from the accumulation
    uint PrevIdx;           // Texture2D<float>: last frame's, 0 = none (a cut, a resize, no transparents then)
    uint MotionIdx;         // Texture2D<float2>: last frame's UV minus this frame's (after the camera pass)
    uint OutUAVIdx;         // RWTexture2D<unorm float>: the mask the upscaler reads
    uint RenderWidth;
    uint RenderHeight;
    uint MaskIdx;           // CSReactiveView: the composed mask
    uint OutputUAVIdx;      // CSReactiveView: LdrOutput, display size
    uint DisplayWidth;
    uint DisplayHeight;
};

[numthreads(8, 8, 1)]
void CSReactive(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= RenderWidth || id.y >= RenderHeight)
        return;

    Texture2D<float> nowTex = ResourceDescriptorHeap[NowIdx];
    float r = nowTex.Load(int3(id.xy, 0));

    if (PrevIdx != 0)
    {
        Texture2D<float2> motion = ResourceDescriptorHeap[MotionIdx];
        Texture2D<float> prevTex = ResourceDescriptorHeap[PrevIdx];
        float2 size = float2(RenderWidth, RenderHeight);
        // Where the history this pixel reads was on last frame's screen, and the four texels a filter takes there
        float2 at = float2(id.xy) + 0.5 + motion.Load(int3(id.xy, 0)) * size;
        int2 base = int2(floor(at - 0.5));
        int2 hi = int2(RenderWidth, RenderHeight) - 1;
        [unroll] for (int y = 0; y < 2; y++)
            [unroll] for (int x = 0; x < 2; x++)
            {
                int2 p = base + int2(x, y);
                if (all(p >= 0) && all(p <= hi))
                    r = max(r, prevTex.Load(int3(p, 0)));
            }
    }

    RWTexture2D<unorm float> output = ResourceDescriptorHeap[OutUAVIdx];
    output[id.xy] = r;
}

[numthreads(8, 8, 1)]
void CSReactiveView(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= DisplayWidth || id.y >= DisplayHeight)
        return;

    Texture2D<float> mask = ResourceDescriptorHeap[MaskIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    uint2 at = uint2((float2(id.xy) + 0.5) * float2(RenderWidth, RenderHeight) / float2(DisplayWidth, DisplayHeight));
    float m = mask.Load(int3(at, 0));
    float4 c = output[id.xy];
    output[id.xy] = float4(lerp(c.rgb * 0.5, float3(1.0, 0.1, 0.05), m), c.a);
}
