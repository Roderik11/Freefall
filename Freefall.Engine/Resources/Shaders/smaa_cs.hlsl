// SMAA 1x — Enhanced Subpixel Morphological Antialiasing (Jimenez et al. 2012), High preset, as compute.
// A port of the reference shader: the searches, the pattern and corner logic and the two precomputed tables
// (Resources/smaa_area.rg8, smaa_search.r8) are the reference's, so keep it that way: what blurred before was a
// formula of our own in place of the area table, see .agent/knowledge/rendering_engine/features/smaa.md.
//
// Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
// Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
// Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
// Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
// Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to
// do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. As clarification, there
// is no requirement that the copyright notice and permission be included in
// binary distributions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma kernel CSEdgeDetection
#pragma kernel CSBlendWeights
#pragma kernel CSNeighborhoodBlend

cbuffer PushConstants : register(b3)
{
    uint InputTexIdx;
    uint EdgesUAVIdx;
    uint BlendUAVIdx;
    uint OutputUAVIdx;
    uint EdgesSRVIdx;
    uint BlendSRVIdx;
    uint AreaTexIdx;
    uint SearchTexIdx;
    uint ScreenWidth;
    uint ScreenHeight;
};

SamplerState LinearClamp : register(s2);

// The High preset
#define SMAA_THRESHOLD 0.1
#define SMAA_MAX_SEARCH_STEPS 16
#define SMAA_MAX_SEARCH_STEPS_DIAG 8
#define SMAA_CORNER_ROUNDING_NORM 0.25
#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR 2.0

#define SMAA_AREATEX_MAX_DISTANCE 16
#define SMAA_AREATEX_MAX_DISTANCE_DIAG 20
#define SMAA_AREATEX_PIXEL_SIZE (1.0 / float2(160.0, 560.0))
#define SMAA_SEARCHTEX_SIZE float2(66.0, 33.0)
#define SMAA_SEARCHTEX_PACKED_SIZE float2(64.0, 16.0)
#define SMAA_RT_METRICS float4(1.0 / float2(ScreenWidth, ScreenHeight), float2(ScreenWidth, ScreenHeight))

// finalize_cs.hlsl's output curve
#define DISPLAY_GAMMA 2.2

void Movc(bool2 cond, inout float2 variable, float2 value)
{
    if (cond.x) variable.x = value.x;
    if (cond.y) variable.y = value.y;
}

// Clamped like the reference's sampler: no edges along the border of the screen
int3 Texel(int2 px)
{
    return int3(clamp(px, int2(0, 0), int2(ScreenWidth, ScreenHeight) - 1), 0);
}

// ═══════════════════════════════════════════════════════════════════
// Pass 1: Luma Edge Detection, on the gamma encoded picture
// ═══════════════════════════════════════════════════════════════════

float Luma(Texture2D<float4> tex, int2 px)
{
    return dot(tex.Load(Texel(px)).rgb, float3(0.2126, 0.7152, 0.0722));
}

[numthreads(8, 8, 1)]
void CSEdgeDetection(uint3 dtid : SV_DispatchThreadID)
{
    int2 px = int2(dtid.xy);
    if ((uint)px.x >= ScreenWidth || (uint)px.y >= ScreenHeight)
        return;

    RWTexture2D<float2> edgesOut = ResourceDescriptorHeap[EdgesUAVIdx];
    Texture2D<float4> colorTex = ResourceDescriptorHeap[InputTexIdx];

    float L = Luma(colorTex, px);
    float Lleft = Luma(colorTex, px + int2(-1, 0));
    float Ltop = Luma(colorTex, px + int2(0, -1));

    float4 delta;
    delta.xy = abs(L - float2(Lleft, Ltop));
    float2 edges = step(SMAA_THRESHOLD, delta.xy);

    if (dot(edges, float2(1.0, 1.0)) == 0.0)
    {
        edgesOut[px] = float2(0.0, 0.0);
        return;
    }

    // The largest delta around, the direct neighbourhood first
    float Lright = Luma(colorTex, px + int2(1, 0));
    float Lbottom = Luma(colorTex, px + int2(0, 1));
    delta.zw = abs(L - float2(Lright, Lbottom));
    float2 maxDelta = max(delta.xy, delta.zw);

    float Lleftleft = Luma(colorTex, px + int2(-2, 0));
    float Ltoptop = Luma(colorTex, px + int2(0, -2));
    delta.zw = abs(float2(Lleft, Ltop) - float2(Lleftleft, Ltoptop));
    maxDelta = max(maxDelta.xy, delta.zw);
    float finalDelta = max(maxDelta.x, maxDelta.y);

    // Local contrast adaptation: a much stronger edge nearby hides this one
    edges *= step(finalDelta, SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);

    edgesOut[px] = edges;
}

// ═══════════════════════════════════════════════════════════════════
// Pass 2: Blend Weight Calculation
// ═══════════════════════════════════════════════════════════════════

// ── Diagonal searches ──

// Two binary values out of a bilinear fetch with a 0.25 offset: red 0.25 or 1, green 0.75 or 1
float2 DecodeDiagBilinearAccess(float2 e)
{
    e.r = e.r * abs(5.0 * e.r - 5.0 * 0.75);
    return round(e);
}

float4 DecodeDiagBilinearAccess(float4 e)
{
    e.rb = e.rb * abs(5.0 * e.rb - 5.0 * 0.75);
    return round(e);
}

float2 SearchDiag1(Texture2D<float2> edgesTex, float2 texcoord, float2 dir, out float2 e)
{
    e = float2(0.0, 0.0);
    float4 coord = float4(texcoord, -1.0, 1.0);
    float3 t = float3(SMAA_RT_METRICS.xy, 1.0);
    [loop] while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) && coord.w > 0.9)
    {
        coord.xyz = t * float3(dir, 1.0) + coord.xyz;
        e = edgesTex.SampleLevel(LinearClamp, coord.xy, 0);
        coord.w = dot(e, float2(0.5, 0.5));
    }
    return coord.zw;
}

float2 SearchDiag2(Texture2D<float2> edgesTex, float2 texcoord, float2 dir, out float2 e)
{
    e = float2(0.0, 0.0);
    float4 coord = float4(texcoord, -1.0, 1.0);
    coord.x += 0.25 * SMAA_RT_METRICS.x;
    float3 t = float3(SMAA_RT_METRICS.xy, 1.0);
    [loop] while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) && coord.w > 0.9)
    {
        coord.xyz = t * float3(dir, 1.0) + coord.xyz;
        // Both edges at once through the bilinear filter
        e = DecodeDiagBilinearAccess(edgesTex.SampleLevel(LinearClamp, coord.xy, 0));
        coord.w = dot(e, float2(0.5, 0.5));
    }
    return coord.zw;
}

// The area of a diagonal distance and crossing edges: the second half of the table
float2 AreaDiag(Texture2D<float2> areaTex, float2 dist, float2 e)
{
    float2 texcoord = float2(SMAA_AREATEX_MAX_DISTANCE_DIAG, SMAA_AREATEX_MAX_DISTANCE_DIAG) * e + dist;
    texcoord = SMAA_AREATEX_PIXEL_SIZE * texcoord + 0.5 * SMAA_AREATEX_PIXEL_SIZE;
    texcoord.x += 0.5;
    return areaTex.SampleLevel(LinearClamp, texcoord, 0);
}

float2 CalculateDiagWeights(Texture2D<float2> edgesTex, Texture2D<float2> areaTex, float2 texcoord, float2 e)
{
    float2 rt = SMAA_RT_METRICS.xy;
    float2 weights = float2(0.0, 0.0);

    // The line ends
    float4 d;
    float2 end;
    if (e.r > 0.0)
    {
        d.xz = SearchDiag1(edgesTex, texcoord, float2(-1.0, 1.0), end);
        d.x += float(end.y > 0.9);
    }
    else
        d.xz = float2(0.0, 0.0);
    d.yw = SearchDiag1(edgesTex, texcoord, float2(1.0, -1.0), end);

    [branch]
    if (d.x + d.y > 2.0)
    {
        // The crossing edges
        float4 coords = float4(-d.x + 0.25, d.x, d.y, -d.y - 0.25) * rt.xyxy + texcoord.xyxy;
        float4 c;
        c.xy = edgesTex.SampleLevel(LinearClamp, coords.xy, 0, int2(-1, 0));
        c.zw = edgesTex.SampleLevel(LinearClamp, coords.zw, 0, int2(1, 0));
        c.yxwz = DecodeDiagBilinearAccess(c.xyzw);

        // Merged into one value per side, none where the end of the line was not found
        float2 cc = float2(2.0, 2.0) * c.xz + c.yw;
        Movc(d.zw >= float2(0.9, 0.9), cc, float2(0.0, 0.0));

        weights += AreaDiag(areaTex, d.xy, cc);
    }

    // The other diagonal
    d.xz = SearchDiag2(edgesTex, texcoord, float2(-1.0, -1.0), end);
    if (edgesTex.SampleLevel(LinearClamp, texcoord, 0, int2(1, 0)).r > 0.0)
    {
        d.yw = SearchDiag2(edgesTex, texcoord, float2(1.0, 1.0), end);
        d.y += float(end.y > 0.9);
    }
    else
        d.yw = float2(0.0, 0.0);

    [branch]
    if (d.x + d.y > 2.0)
    {
        float4 coords = float4(-d.x, -d.x, d.y, d.y) * rt.xyxy + texcoord.xyxy;
        float4 c;
        c.x = edgesTex.SampleLevel(LinearClamp, coords.xy, 0, int2(-1, 0)).g;
        c.y = edgesTex.SampleLevel(LinearClamp, coords.xy, 0, int2(0, -1)).r;
        c.zw = edgesTex.SampleLevel(LinearClamp, coords.zw, 0, int2(1, 0)).gr;

        float2 cc = float2(2.0, 2.0) * c.xz + c.yw;
        Movc(d.zw >= float2(0.9, 0.9), cc, float2(0.0, 0.0));

        weights += AreaDiag(areaTex, d.xy, cc).gr;
    }

    return weights;
}

// ── Horizontal and vertical searches ──

// How much length the last step adds, from the bilinearly fetched edges
float SearchLength(Texture2D<float> searchTex, float2 e, float offset)
{
    // The texture is flipped vertically, left and right cases taking half of the space horizontally
    float2 scale = SMAA_SEARCHTEX_SIZE * float2(0.5, -1.0);
    float2 bias = SMAA_SEARCHTEX_SIZE * float2(offset, 1.0);

    // To texel centers
    scale += float2(-1.0, 1.0);
    bias += float2(0.5, -0.5);

    // To texcoords, of the cropped texture
    scale *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;
    bias *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;

    return searchTex.SampleLevel(LinearClamp, scale * e + bias, 0);
}

// The coordinates come offset by a quarter of a pixel and an eighth, between edges, so that a bilinear fetch reads
// four edges in a row and tells them apart by their weights
float SearchXLeft(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
{
    float2 e = float2(0.0, 1.0);
    [loop] while (texcoord.x > end && e.g > 0.8281 && e.r == 0.0)
    {
        e = edgesTex.SampleLevel(LinearClamp, texcoord, 0);
        texcoord = -float2(2.0, 0.0) * SMAA_RT_METRICS.xy + texcoord;
    }
    float offset = -(255.0 / 127.0) * SearchLength(searchTex, e, 0.0) + 3.25;
    return SMAA_RT_METRICS.x * offset + texcoord.x;
}

float SearchXRight(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
{
    float2 e = float2(0.0, 1.0);
    [loop] while (texcoord.x < end && e.g > 0.8281 && e.r == 0.0)
    {
        e = edgesTex.SampleLevel(LinearClamp, texcoord, 0);
        texcoord = float2(2.0, 0.0) * SMAA_RT_METRICS.xy + texcoord;
    }
    float offset = -(255.0 / 127.0) * SearchLength(searchTex, e, 0.5) + 3.25;
    return -SMAA_RT_METRICS.x * offset + texcoord.x;
}

float SearchYUp(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
{
    float2 e = float2(1.0, 0.0);
    [loop] while (texcoord.y > end && e.r > 0.8281 && e.g == 0.0)
    {
        e = edgesTex.SampleLevel(LinearClamp, texcoord, 0);
        texcoord = -float2(0.0, 2.0) * SMAA_RT_METRICS.xy + texcoord;
    }
    float offset = -(255.0 / 127.0) * SearchLength(searchTex, e.gr, 0.0) + 3.25;
    return SMAA_RT_METRICS.y * offset + texcoord.y;
}

float SearchYDown(Texture2D<float2> edgesTex, Texture2D<float> searchTex, float2 texcoord, float end)
{
    float2 e = float2(1.0, 0.0);
    [loop] while (texcoord.y < end && e.r > 0.8281 && e.g == 0.0)
    {
        e = edgesTex.SampleLevel(LinearClamp, texcoord, 0);
        texcoord = float2(0.0, 2.0) * SMAA_RT_METRICS.xy + texcoord;
    }
    float offset = -(255.0 / 127.0) * SearchLength(searchTex, e.gr, 0.5) + 3.25;
    return -SMAA_RT_METRICS.y * offset + texcoord.y;
}

// The areas at each side of an edge, from the distance and both crossing edges
float2 Area(Texture2D<float2> areaTex, float2 dist, float e1, float e2)
{
    // Rounding: the precision of the bilinear filter
    float2 texcoord = float2(SMAA_AREATEX_MAX_DISTANCE, SMAA_AREATEX_MAX_DISTANCE) * round(4.0 * float2(e1, e2)) + dist;
    texcoord = SMAA_AREATEX_PIXEL_SIZE * texcoord + 0.5 * SMAA_AREATEX_PIXEL_SIZE;
    return areaTex.SampleLevel(LinearClamp, texcoord, 0);
}

// ── Corners ──

void DetectHorizontalCornerPattern(Texture2D<float2> edgesTex, inout float2 weights, float4 texcoord, float2 d)
{
    float2 leftRight = step(d.xy, d.yx);
    float2 rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * leftRight;

    rounding /= leftRight.x + leftRight.y;   // less blending for pixels in the middle of a line

    float2 factor = float2(1.0, 1.0);
    factor.x -= rounding.x * edgesTex.SampleLevel(LinearClamp, texcoord.xy, 0, int2(0, 1)).r;
    factor.x -= rounding.y * edgesTex.SampleLevel(LinearClamp, texcoord.zw, 0, int2(1, 1)).r;
    factor.y -= rounding.x * edgesTex.SampleLevel(LinearClamp, texcoord.xy, 0, int2(0, -2)).r;
    factor.y -= rounding.y * edgesTex.SampleLevel(LinearClamp, texcoord.zw, 0, int2(1, -2)).r;

    weights *= saturate(factor);
}

void DetectVerticalCornerPattern(Texture2D<float2> edgesTex, inout float2 weights, float4 texcoord, float2 d)
{
    float2 leftRight = step(d.xy, d.yx);
    float2 rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * leftRight;

    rounding /= leftRight.x + leftRight.y;

    float2 factor = float2(1.0, 1.0);
    factor.x -= rounding.x * edgesTex.SampleLevel(LinearClamp, texcoord.xy, 0, int2(1, 0)).g;
    factor.x -= rounding.y * edgesTex.SampleLevel(LinearClamp, texcoord.zw, 0, int2(1, 1)).g;
    factor.y -= rounding.x * edgesTex.SampleLevel(LinearClamp, texcoord.xy, 0, int2(-2, 0)).g;
    factor.y -= rounding.y * edgesTex.SampleLevel(LinearClamp, texcoord.zw, 0, int2(-2, 1)).g;

    weights *= saturate(factor);
}

[numthreads(8, 8, 1)]
void CSBlendWeights(uint3 dtid : SV_DispatchThreadID)
{
    int2 px = int2(dtid.xy);
    if ((uint)px.x >= ScreenWidth || (uint)px.y >= ScreenHeight)
        return;

    RWTexture2D<float4> blendOut = ResourceDescriptorHeap[BlendUAVIdx];
    Texture2D<float2> edgesTex = ResourceDescriptorHeap[EdgesSRVIdx];
    Texture2D<float2> areaTex = ResourceDescriptorHeap[AreaTexIdx];
    Texture2D<float> searchTex = ResourceDescriptorHeap[SearchTexIdx];

    float4 weights = float4(0.0, 0.0, 0.0, 0.0);
    float2 e = edgesTex.Load(int3(px, 0));

    if (e.g == 0.0 && e.r == 0.0)
    {
        blendOut[px] = weights;
        return;
    }

    float4 rt = SMAA_RT_METRICS;
    float2 pixcoord = float2(px) + 0.5;
    float2 texcoord = pixcoord * rt.xy;

    // The offsets of the searches, and where they end
    float4 offset0 = rt.xyxy * float4(-0.25, -0.125, 1.25, -0.125) + texcoord.xyxy;
    float4 offset1 = rt.xyxy * float4(-0.125, -0.25, -0.125, 1.25) + texcoord.xyxy;
    float4 offset2 = rt.xxyy * float4(-2.0, 2.0, -2.0, 2.0) * float(SMAA_MAX_SEARCH_STEPS) + float4(offset0.xz, offset1.yw);

    [branch]
    if (e.g > 0.0) // Edge at north
    {
        // Diagonals have both north and west edges: one of the two is enough to search them from
        weights.rg = CalculateDiagWeights(edgesTex, areaTex, texcoord, e);

        // Diagonals come first: one found, the horizontal and vertical processing is skipped
        [branch]
        if (weights.r == -weights.g)
        {
            float2 d;

            // The distance to the left, and the left crossing edges two at a time through the bilinear filter
            float3 coords;
            coords.x = SearchXLeft(edgesTex, searchTex, offset0.xy, offset2.x);
            coords.y = offset1.y;
            d.x = coords.x;
            float e1 = edgesTex.SampleLevel(LinearClamp, coords.xy, 0).r;

            // The distance to the right
            coords.z = SearchXRight(edgesTex, searchTex, offset0.zw, offset2.y);
            d.y = coords.z;

            // In pixels, square rooted since the area table is compressed quadratically
            d = abs(round(rt.zz * d - pixcoord.xx));
            float2 sqrtD = sqrt(d);

            float e2 = edgesTex.SampleLevel(LinearClamp, coords.zy, 0, int2(1, 0)).r;

            weights.rg = Area(areaTex, sqrtD, e1, e2);

            coords.y = texcoord.y;
            DetectHorizontalCornerPattern(edgesTex, weights.rg, coords.xyzy, d);
        }
        else
            e.r = 0.0;
    }

    [branch]
    if (e.r > 0.0) // Edge at west
    {
        float2 d;

        // The distance to the top
        float3 coords;
        coords.y = SearchYUp(edgesTex, searchTex, offset1.xy, offset2.z);
        coords.x = offset0.x;
        d.x = coords.y;
        float e1 = edgesTex.SampleLevel(LinearClamp, coords.xy, 0).g;

        // The distance to the bottom
        coords.z = SearchYDown(edgesTex, searchTex, offset1.zw, offset2.w);
        d.y = coords.z;

        d = abs(round(rt.ww * d - pixcoord.yy));
        float2 sqrtD = sqrt(d);

        float e2 = edgesTex.SampleLevel(LinearClamp, coords.xz, 0, int2(0, 1)).g;

        weights.ba = Area(areaTex, sqrtD, e1, e2);

        coords.x = texcoord.x;
        DetectVerticalCornerPattern(edgesTex, weights.ba, coords.xyxz, d);
    }

    blendOut[px] = weights;
}

// ═══════════════════════════════════════════════════════════════════
// Pass 3: Neighborhood Blending
// ═══════════════════════════════════════════════════════════════════

// The reference blends through an sRGB view and the bilinear filter. LdrOutput is plain UNORM: decoded by hand.
float3 Decoded(Texture2D<float4> tex, int2 px)
{
    return pow(tex.Load(Texel(px)).rgb, DISPLAY_GAMMA);
}

[numthreads(8, 8, 1)]
void CSNeighborhoodBlend(uint3 dtid : SV_DispatchThreadID)
{
    int2 px = int2(dtid.xy);
    if ((uint)px.x >= ScreenWidth || (uint)px.y >= ScreenHeight)
        return;

    Texture2D<float4> colorTex = ResourceDescriptorHeap[InputTexIdx];
    Texture2D<float4> blendTex = ResourceDescriptorHeap[BlendSRVIdx];
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];

    // The weights of this pixel, some kept by the neighbours to the right and below
    float4 a;
    a.x = blendTex.Load(int3(px + int2(1, 0), 0)).a;
    a.y = blendTex.Load(int3(px + int2(0, 1), 0)).g;
    a.wz = blendTex.Load(int3(px, 0)).xz;

    if (dot(a, float4(1.0, 1.0, 1.0, 1.0)) < 1e-5)
    {
        output[px] = colorTex.Load(int3(px, 0));
        return;
    }

    // Along one axis only, whichever has the larger weight
    bool h = max(a.x, a.z) > max(a.y, a.w);
    int2 dir = h ? int2(1, 0) : int2(0, 1);
    float2 offset = h ? a.xz : a.yw;
    float2 weight = offset / dot(offset, float2(1.0, 1.0));

    float3 self = Decoded(colorTex, px);
    float3 color = weight.x * lerp(self, Decoded(colorTex, px + dir), offset.x)
                 + weight.y * lerp(self, Decoded(colorTex, px - dir), offset.y);

    output[px] = float4(pow(color, 1.0 / DISPLAY_GAMMA), 1.0);
}
