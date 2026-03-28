// SpriteBatch - Bindless DX12 Sprite Renderer + Slug GPU Text
// No geometry shader, no vertex buffer, no atlas.
// Sprites are read from a StructuredBuffer via SV_VertexID.
// Each sprite carries its own texture index into the bindless heap.
// Type 0 = Quad, Type 1 = Line, Type 2 = Slug Glyph (GPU Bézier text)

cbuffer PushConstants : register(b3)
{
    uint SpriteBufferIdx;   // 0
    float ScreenWidth;      // 1
    float ScreenHeight;     // 2
    uint GlyphBufferIdx;    // 3
};

struct Sprite
{
    float2 Position;    // pixel coords (top-left)
    float2 Size;        // pixel size (width, height)
    float4 Color;       // tint RGBA
    float4 UVs;         // minU, minV, maxU, maxV  (for Type=2: em-space bounds)
    uint   TextureIndex; // SRV heap index (bindless) — for Type=2: index into SlugGlyphData
    uint   Type;         // 0 = Quad, 1 = Line, 2 = SlugGlyph
    float2 EndPosition;  // For lines: pixel coords of end point
};

// Per-glyph data for Slug text rendering (must match C# SlugGlyphData)
struct SlugGlyphData
{
    uint   CurveTextureIndex;   // Bindless SRV for curve texture
    uint   BandTextureIndex;    // Bindless SRV for band texture
    int2   GlyphLoc;           // Band texture origin (x, y)
    int    BandMaxX;            // Max horizontal band index
    int    BandMaxY;            // Max vertical band index
    float  BandScaleX;          // Em-to-band scale X
    float  BandScaleY;          // Em-to-band scale Y
    float  BandOffsetX;         // Em-to-band offset X
    float  BandOffsetY;         // Em-to-band offset Y
    float  PixelsPerEmX;        // CPU-computed pixels per em-unit X
    float  PixelsPerEmY;        // CPU-computed pixels per em-unit Y
};

struct PS_Input
{
    float4 Position    : SV_POSITION;
    float2 UV          : TEXCOORD0;
    float4 Color       : COLOR0;
    nointerpolation uint TextureIndex : TEXCOORD1;
    nointerpolation uint Type         : TEXCOORD2;
};

SamplerState PointSampler : register(s1); // s1 = point clamp (static sampler)

// ── Exact port of FormeShader.fx evaluation logic ──

static const int kBandTexWidth = 4096;

float4 TexelLoadCurve(Texture2D tex, int2 loc)
{
    return tex.Load(int3(loc, 0));
}

float2 TexelLoadBandAbs(Texture2D tex, float absIndex)
{
    int ix = int(absIndex) % kBandTexWidth;
    int iy = int(absIndex) / kBandTexWidth;
    return tex.Load(int3(ix, iy, 0)).xy;
}

// Float-based root eligibility (FormeShader.fx CalcRootEligibility)
// NO asuint() — uses float comparisons only
float2 CalcRootEligibility(float y1, float y2, float y3)
{
    float s0 = (y1 > 0.0) ? 1.0 : 0.0;
    float s1 = (y2 > 0.0) ? 1.0 : 0.0;
    float s2 = (y3 > 0.0) ? 1.0 : 0.0;
    float ns0 = 1.0 - s0, ns1 = 1.0 - s1, ns2 = 1.0 - s2;

    float root1 = saturate(
        s0 * ns1 * ns2 +
        ns0 * s1 * ns2 +
        s0 * s1 * ns2  +
        s0 * ns1 * s2);

    float root2 = saturate(
        ns0 * s1 * ns2 +
        ns0 * ns1 * s2 +
        s0 * ns1 * s2  +
        ns0 * s1 * s2);

    return float2(root1, root2);
}

// Solve for X where Bézier curve crosses Y=0 (FormeShader.fx SolveHorizPoly)
float2 SolveHorizPoly(float4 p12, float2 p3)
{
    float2 a = p12.xy - p12.zw * 2.0 + p3;
    float2 b = p12.xy - p12.zw;
    float ra = 1.0 / a.y;
    float rb = 0.5 / b.y;

    float d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    float t1 = (b.y - d) * ra;
    float t2 = (b.y + d) * ra;

    if (abs(a.y) < 0.0001) { t1 = p12.y * rb; t2 = t1; }

    return float2(
        (a.x * t1 - b.x * 2.0) * t1 + p12.x,
        (a.x * t2 - b.x * 2.0) * t2 + p12.x);
}

// Exact port of FormeRender from FormeShader.fx
float FormeRender(Texture2D curveData, Texture2D bandData,
    float2 renderCoord, float4 bandTransform, int4 glyphData, float2 pixelsPerEm)
{
    int2 glyphLoc = glyphData.xy;
    int bandMaxX = glyphData.z & 0x00FF;
    int bandMaxY = glyphData.w & 0x00FF;
    float bandCount = float(bandMaxY + 1);

    float2 bandPos = renderCoord * bandTransform.xy + bandTransform.zw;
    float bandIndexY = clamp(floor(bandPos.y), 0.0, float(bandMaxY));
    float bandIndexX = clamp(floor(bandPos.x), 0.0, float(bandMaxX));

    float glyphBaseTexel = float(glyphLoc.y) * float(kBandTexWidth) + float(glyphLoc.x);

    // ── Horizontal bands ──
    float xcov = 0.0;
    float xwgt = 0.0;

    float2 hBandHeader = TexelLoadBandAbs(bandData, glyphBaseTexel + bandIndexY);
    float hCurveCount = hBandHeader.r;
    float hCurveOffset = hBandHeader.g;

    [loop]
    for (float ci = 0.0; ci < hCurveCount; ci += 1.0)
    {
        float2 curveLoc = TexelLoadBandAbs(bandData, hCurveOffset + ci);
        float4 p12 = TexelLoadCurve(curveData, int2(curveLoc)) - float4(renderCoord, renderCoord);
        float2 p3  = TexelLoadCurve(curveData, int2(curveLoc.x + 1, curveLoc.y)).xy - renderCoord;

        // Curves sorted by max-x descending; early exit
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.x < -0.5) break;

        float2 elig = CalcRootEligibility(p12.y, p12.w, p3.y);
        if (elig.x + elig.y > 0.0)
        {
            float2 r = SolveHorizPoly(p12, p3) * pixelsPerEm.x;

            if (elig.x > 0.5) { xcov += saturate(r.x + 0.5); xwgt = max(xwgt, saturate(1.0 - abs(r.x) * 2.0)); }
            if (elig.y > 0.5) { xcov -= saturate(r.y + 0.5); xwgt = max(xwgt, saturate(1.0 - abs(r.y) * 2.0)); }
        }
    }

    // ── Vertical bands ──
    // Swap X↔Y and reuse SolveHorizPoly (matching FormeShader.fx exactly)
    float ycov = 0.0;
    float ywgt = 0.0;

    float2 vBandHeader = TexelLoadBandAbs(bandData, glyphBaseTexel + bandCount + bandIndexX);
    float vCurveCount = vBandHeader.r;
    float vCurveOffset = vBandHeader.g;

    [loop]
    for (float vi = 0.0; vi < vCurveCount; vi += 1.0)
    {
        float2 curveLoc = TexelLoadBandAbs(bandData, vCurveOffset + vi);
        float4 rawP12 = TexelLoadCurve(curveData, int2(curveLoc));
        float2 rawP3  = TexelLoadCurve(curveData, int2(curveLoc.x + 1, curveLoc.y)).xy;

        // Swap X↔Y to reuse horizontal solver for vertical crossings
        float4 p12 = float4(rawP12.y - renderCoord.y, rawP12.x - renderCoord.x,
                             rawP12.w - renderCoord.y, rawP12.z - renderCoord.x);
        float2 p3  = float2(rawP3.y - renderCoord.y, rawP3.x - renderCoord.x);

        // Curves sorted by max-y descending; early exit (check swapped .x = original .y)
        if (max(max(p12.x, p12.z), p3.x) * pixelsPerEm.y < -0.5) break;

        float2 elig = CalcRootEligibility(p12.y, p12.w, p3.y);
        if (elig.x + elig.y > 0.0)
        {
            float2 r = SolveHorizPoly(p12, p3) * pixelsPerEm.y;

            if (elig.x > 0.5) { ycov += saturate(r.x + 0.5); ywgt = max(ywgt, saturate(1.0 - abs(r.x) * 2.0)); }
            if (elig.y > 0.5) { ycov -= saturate(r.y + 0.5); ywgt = max(ywgt, saturate(1.0 - abs(r.y) * 2.0)); }
        }
    }

    // min() properly anti-aliases edges in both directions:
    // inside: min(1,1)=1, vert edge: min(0.3,1)=0.3, horiz edge: min(1,0.5)=0.5
    float coverage = min(abs(xcov), abs(ycov));

    return saturate(coverage);
}

// ═══════════════════════════════════════════════════════════════
// Vertex Shader
// ═══════════════════════════════════════════════════════════════

PS_Input VS(uint vertexID : SV_VertexID)
{
    StructuredBuffer<Sprite> Sprites = ResourceDescriptorHeap[SpriteBufferIdx];

    uint spriteIndex = vertexID / 6;
    uint cornerIndex = vertexID % 6;

    Sprite s = Sprites[spriteIndex];

    PS_Input o;
    o.TextureIndex = s.TextureIndex;
    o.Color = s.Color;
    o.Type = s.Type;

    if (s.Type == 1)
    {
        float2 p0 = s.Position;
        float2 p1 = s.EndPosition;
        float2 dir = p1 - p0;
        float len = length(dir);
        if (len < 0.001) dir = float2(1, 0);
        else dir /= len;

        float2 perp = float2(-dir.y, dir.x) * s.Size.x * 0.5;

        float2 corners[6] = {
            p0 - perp, p0 + perp, p1 - perp,
            p1 - perp, p0 + perp, p1 + perp
        };

        float2 uvCorners[6] = {
            float2(0, 0), float2(0, 1), float2(1, 0),
            float2(1, 0), float2(0, 1), float2(1, 1)
        };

        float2 posPixel = corners[cornerIndex];
        o.Position = float4(
            posPixel.x / ScreenWidth * 2.0 - 1.0,
            -(posPixel.y / ScreenHeight * 2.0 - 1.0),
            0, 1
        );
        o.UV = uvCorners[cornerIndex];
    }
    else if (s.Type == 2)
    {
        static const float2 quadCorners[6] = {
            float2(0, 0), float2(1, 0), float2(0, 1),
            float2(0, 1), float2(1, 0), float2(1, 1)
        };
        float2 corner = quadCorners[cornerIndex];

        float emW = s.UVs.z - s.UVs.x;
        float emH = s.UVs.w - s.UVs.y;

        float2 posPixel = s.Position + s.Size * corner;
        o.UV = lerp(s.UVs.xy, s.UVs.zw, corner);

        o.Position = float4(
            posPixel.x / ScreenWidth * 2.0 - 1.0,
            -(posPixel.y / ScreenHeight * 2.0 - 1.0),
            0, 1
        );
    }
    else
    {
        static const float2 quadCorners[6] = {
            float2(0, 0), float2(1, 0), float2(0, 1),
            float2(0, 1), float2(1, 0), float2(1, 1)
        };

        float2 corner = quadCorners[cornerIndex];
        float2 posPixel = s.Position + s.Size * corner;

        o.Position = float4(
            posPixel.x / ScreenWidth * 2.0 - 1.0,
            -(posPixel.y / ScreenHeight * 2.0 - 1.0),
            0, 1
        );
        o.UV = lerp(s.UVs.xy, s.UVs.zw, corner);
    }

    return o;
}

// ═══════════════════════════════════════════════════════════════
// Pixel Shader
// ═══════════════════════════════════════════════════════════════

float4 PS(PS_Input input) : SV_Target
{
    if (input.Type == 2)
    {
        StructuredBuffer<SlugGlyphData> GlyphData = ResourceDescriptorHeap[GlyphBufferIdx];
        SlugGlyphData g = GlyphData[input.TextureIndex];

        Texture2D curveTexture = ResourceDescriptorHeap[g.CurveTextureIndex];
        Texture2D bandTexture = ResourceDescriptorHeap[g.BandTextureIndex];

        float2 renderCoord = input.UV;
        float4 bandTransform = float4(g.BandScaleX, g.BandScaleY, g.BandOffsetX, g.BandOffsetY);
        int4 glyphData = int4(g.GlyphLoc, g.BandMaxX, g.BandMaxY);

        float coverage = FormeRender(curveTexture, bandTexture, renderCoord, bandTransform, glyphData,
            float2(g.PixelsPerEmX, g.PixelsPerEmY));

        // Standard alpha output (SrcAlpha/InvSrcAlpha blend state)
        return float4(input.Color.rgb, input.Color.a * coverage);
    }
    else
    {
        Texture2D tex = ResourceDescriptorHeap[input.TextureIndex];
        float4 texColor = tex.Sample(PointSampler, input.UV);
        return texColor * input.Color;
    }
}
