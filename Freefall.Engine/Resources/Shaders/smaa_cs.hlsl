// SMAA 1x — Enhanced Subpixel Morphological Anti-Aliasing (Compute)
// Simplified for SMAA 1x: integer searches, analytical blend weights (no LUT).

#pragma kernel CSEdgeDetection
#pragma kernel CSBlendWeights
#pragma kernel CSNeighborhoodBlend

cbuffer PushConstants : register(b3)
{
    uint InputTexIdx;
    uint EdgesUAVIdx;
    uint BlendUAVIdx;
    uint OutputUAVIdx;
    uint Unused0;
    uint Unused1;
    uint EdgesSRVIdx;
    uint BlendSRVIdx;
    uint ScreenWidth;
    uint ScreenHeight;
};

SamplerState linearClamp : register(s2);

#define SMAA_THRESHOLD 0.05
#define SMAA_MAX_SEARCH_STEPS 32

float2 PixelToUV(int2 px)
{
    return (float2(px) + 0.5) / float2(ScreenWidth, ScreenHeight);
}

float Luma(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// ═══════════════════════════════════════════════════════════════════
// Pass 1: Luma Edge Detection
// ═══════════════════════════════════════════════════════════════════

[numthreads(8, 8, 1)]
void CSEdgeDetection(uint3 dtid : SV_DispatchThreadID)
{
    int2 px = int2(dtid.xy);
    if ((uint)px.x >= ScreenWidth || (uint)px.y >= ScreenHeight)
        return;

    RWTexture2D<float2> edgesOut = ResourceDescriptorHeap[EdgesUAVIdx];
    Texture2D<float4> inputTex = ResourceDescriptorHeap[InputTexIdx];

    float L      = Luma(inputTex.Load(int3(px, 0)).rgb);
    float Lleft  = Luma(inputTex.Load(int3(px + int2(-1,  0), 0)).rgb);
    float Ltop   = Luma(inputTex.Load(int3(px + int2( 0, -1), 0)).rgb);
    float Lright = Luma(inputTex.Load(int3(px + int2( 1,  0), 0)).rgb);
    float Lbot   = Luma(inputTex.Load(int3(px + int2( 0,  1), 0)).rgb);

    float4 delta = abs(L - float4(Lleft, Ltop, Lright, Lbot));
    float2 edges = step(SMAA_THRESHOLD, delta.xy);

    // Local contrast adaptation
    if (dot(edges, 1.0) > 0)
    {
        float2 maxDelta = max(delta.xy, delta.zw);

        float Ltopleft  = Luma(inputTex.Load(int3(px + int2(-1, -1), 0)).rgb);
        float Ltopright = Luma(inputTex.Load(int3(px + int2( 1, -1), 0)).rgb);
        float Lbotleft  = Luma(inputTex.Load(int3(px + int2(-1,  1), 0)).rgb);

        maxDelta.x = max(maxDelta.x, max(abs(Lleft - Ltopleft), abs(Lleft - Lbotleft)));
        maxDelta.y = max(maxDelta.y, max(abs(Ltop - Ltopleft), abs(Ltop - Ltopright)));

        edges *= step(0.5 * maxDelta, delta.xy);
    }

    edgesOut[px] = edges;
}

// ═══════════════════════════════════════════════════════════════════
// Pass 2: Blend Weight Calculation (analytical — no LUT)
// ═══════════════════════════════════════════════════════════════════

// Walk left along a horizontal edge (.g channel), return pixel distance
int SearchLeft(Texture2D<float2> edges, int2 px)
{
    int d = 0;
    [loop] for (d = 1; d <= SMAA_MAX_SEARCH_STEPS; d++)
    {
        if (edges.Load(int3(px.x - d, px.y, 0)).g < 0.5) break;
    }
    return d - 1; // distance to last pixel that still had the edge
}

int SearchRight(Texture2D<float2> edges, int2 px)
{
    int d = 0;
    [loop] for (d = 1; d <= SMAA_MAX_SEARCH_STEPS; d++)
    {
        if (edges.Load(int3(px.x + d, px.y, 0)).g < 0.5) break;
    }
    return d - 1;
}

int SearchUp(Texture2D<float2> edges, int2 px)
{
    int d = 0;
    [loop] for (d = 1; d <= SMAA_MAX_SEARCH_STEPS; d++)
    {
        if (edges.Load(int3(px.x, px.y - d, 0)).r < 0.5) break;
    }
    return d - 1;
}

int SearchDown(Texture2D<float2> edges, int2 px)
{
    int d = 0;
    [loop] for (d = 1; d <= SMAA_MAX_SEARCH_STEPS; d++)
    {
        if (edges.Load(int3(px.x, px.y + d, 0)).r < 0.5) break;
    }
    return d - 1;
}

// d1, d2 = sqrt(pixel distances) to left/top and right/bottom endpoints
// e1 = crossing edge at left/top end, e2 = crossing edge at right/bottom end
// Returns blend weights for the two sides of the edge
//
// The smoothed edge line displaces from the pixel center near crossing points
// and returns to center along straight segments. The displacement (= blend weight)
// falls off as 1/(d+1) from each crossing, ensuring blending only at stair corners.
float2 ComputeArea(float d1, float d2, float e1, float e2)
{
    if (e1 < 0.5 && e2 < 0.5)
        return 0;

    float2 result = 0;

    // Each crossing contributes blending that decays with distance
    if (e1 > 0.5)
        result.x += 0.5 / (d1 + 1.0);

    if (e2 > 0.5)
        result.y += 0.5 / (d2 + 1.0);

    return result;
}

[numthreads(8, 8, 1)]
void CSBlendWeights(uint3 dtid : SV_DispatchThreadID)
{
    int2 px = int2(dtid.xy);
    if ((uint)px.x >= ScreenWidth || (uint)px.y >= ScreenHeight)
        return;

    RWTexture2D<float4> blendOut = ResourceDescriptorHeap[BlendUAVIdx];
    Texture2D<float2> edgesTex   = ResourceDescriptorHeap[EdgesSRVIdx];

    float4 weights = 0;
    float2 e = edgesTex.Load(int3(px, 0));

    [branch]
    if (e.g > 0.5) // Horizontal edge (top)
    {
        // Search left and right along the edge
        int dLeft  = SearchLeft(edgesTex, px);
        int dRight = SearchRight(edgesTex, px);

        // Check for crossing edges (vertical edges) at the endpoints
        // .r at pixel X = vertical edge between X and X-1
        // Left endpoint: the span's first pixel has .r encoding the left boundary
        float e1 = edgesTex.Load(int3(px.x - dLeft, px.y, 0)).r;
        // Right endpoint: first pixel past the span has .r encoding the right boundary
        float e2 = edgesTex.Load(int3(px.x + dRight + 1, px.y, 0)).r;

        weights.rg = ComputeArea(sqrt((float)dLeft), sqrt((float)dRight), e1, e2);
    }

    [branch]
    if (e.r > 0.5) // Vertical edge (left)
    {
        int dUp   = SearchUp(edgesTex, px);
        int dDown = SearchDown(edgesTex, px);

        // .g at pixel Y = horizontal edge between Y and Y-1
        float e1 = edgesTex.Load(int3(px.x, px.y - dUp, 0)).g;
        float e2 = edgesTex.Load(int3(px.x, px.y + dDown + 1, 0)).g;

        weights.ba = ComputeArea(sqrt((float)dUp), sqrt((float)dDown), e1, e2);
    }

    blendOut[px] = weights;
}
// ═══════════════════════════════════════════════════════════════════
// Pass 3: Neighborhood Blending
// ═══════════════════════════════════════════════════════════════════
//
// Blend weights encode sub-pixel coverage at each edge:
//   .r = blend towards pixel ABOVE  (from horizontal edge processing)
//   .g = blend towards pixel BELOW
//   .b = blend towards pixel LEFT   (from vertical edge processing)
//   .a = blend towards pixel RIGHT
//
// Each pixel reads its own weights AND checks neighbors' weights
// to ensure both sides of every edge get smoothed symmetrically.

[numthreads(8, 8, 1)]
void CSNeighborhoodBlend(uint3 dtid : SV_DispatchThreadID)
{
    int2 px = int2(dtid.xy);
    if ((uint)px.x >= ScreenWidth || (uint)px.y >= ScreenHeight)
        return;

    Texture2D<float4> inputTex = ResourceDescriptorHeap[InputTexIdx];
    Texture2D<float4> blendTex = ResourceDescriptorHeap[BlendSRVIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];

    // Our own blend weights
    float4 w = blendTex.Load(int3(px, 0));

    // Neighbor weights that affect us:
    // Bottom neighbor's .r (it wants to blend UP = towards us)
    float fromBelow = blendTex.Load(int3(px + int2(0, 1), 0)).r;
    // Right neighbor's .b (it wants to blend LEFT = towards us)
    float fromRight = blendTex.Load(int3(px + int2(1, 0), 0)).b;
    // Top neighbor's .g (it wants to blend DOWN = towards us)
    float fromAbove = blendTex.Load(int3(px + int2(0, -1), 0)).g;
    // Left neighbor's .a (it wants to blend RIGHT = towards us)
    float fromLeft = blendTex.Load(int3(px + int2(-1, 0), 0)).a;

    // Combine: max of own weight and neighbor's matching weight
    float up    = max(w.r, fromAbove);
    float down  = max(w.g, fromBelow);
    float left  = max(w.b, fromLeft);
    float right = max(w.a, fromRight);

    float total = up + down + left + right;
    if (total < 0.01)
    {
        output[px] = inputTex.Load(int3(px, 0));
        return;
    }

    float4 self = inputTex.Load(int3(px, 0));
    float4 result = self;

    if (up > 0.01)
        result = lerp(result, inputTex.Load(int3(px + int2(0, -1), 0)), up);
    if (down > 0.01)
        result = lerp(result, inputTex.Load(int3(px + int2(0, 1), 0)), down);
    if (left > 0.01)
        result = lerp(result, inputTex.Load(int3(px + int2(-1, 0), 0)), left);
    if (right > 0.01)
        result = lerp(result, inputTex.Load(int3(px + int2(1, 0), 0)), right);

    output[px] = result;
}
