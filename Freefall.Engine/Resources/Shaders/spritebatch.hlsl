// SpriteBatch - Bindless DX12 Sprite Renderer
// No geometry shader, no vertex buffer, no atlas.
// Sprites are read from a StructuredBuffer via SV_VertexID.
// Each sprite carries its own texture index into the bindless heap.

struct PushConstantsData
{
    uint4 indices[8]; // 32 uints tightly packed as 8 vectors of 4
};
ConstantBuffer<PushConstantsData> PushConstants : register(b3);
#define GET_INDEX(i) PushConstants.indices[i/4][i%4]

// Push constant slots
#define SpriteBufferIdx GET_INDEX(0)
#define ScreenWidth     asfloat(GET_INDEX(1))
#define ScreenHeight    asfloat(GET_INDEX(2))

struct Sprite
{
    float2 Position;    // pixel coords (top-left)
    float2 Size;        // pixel size (width, height)
    float4 Color;       // tint RGBA
    float4 UVs;         // minU, minV, maxU, maxV
    uint   TextureIndex; // SRV heap index (bindless)
    uint   Type;         // 0 = Quad, 1 = Line
    float2 EndPosition;  // For lines: pixel coords of end point
};

struct PS_Input
{
    float4 Position    : SV_POSITION;
    float2 UV          : TEXCOORD0;
    float4 Color       : COLOR0;
    uint   TextureIndex : TEXCOORD1;
};

SamplerState PointSampler : register(s1); // s1 = point clamp (static sampler)

PS_Input VS(uint vertexID : SV_VertexID)
{
    StructuredBuffer<Sprite> Sprites = ResourceDescriptorHeap[SpriteBufferIdx];

    uint spriteIndex = vertexID / 6;
    uint cornerIndex = vertexID % 6;

    Sprite s = Sprites[spriteIndex];

    PS_Input o;
    o.TextureIndex = s.TextureIndex;
    o.Color = s.Color;

    if (s.Type == 1)
    {
        // Line rendering: extrude a thin quad along the line direction
        float2 p0 = s.Position;
        float2 p1 = s.EndPosition;
        float2 dir = p1 - p0;
        float len = length(dir);
        if (len < 0.001) dir = float2(1, 0);
        else dir /= len;

        float2 perp = float2(-dir.y, dir.x) * s.Size.x * 0.5; // Size.x = line width

        // 6 vertices for 2 triangles
        float2 corners[6] = {
            p0 - perp, p0 + perp, p1 - perp,
            p1 - perp, p0 + perp, p1 + perp
        };

        float2 uvCorners[6] = {
            float2(0, 0), float2(0, 1), float2(1, 0),
            float2(1, 0), float2(0, 1), float2(1, 1)
        };

        float2 posPixel = corners[cornerIndex];
        // Convert pixel coords to NDC
        o.Position = float4(
            posPixel.x / ScreenWidth * 2.0 - 1.0,
            -(posPixel.y / ScreenHeight * 2.0 - 1.0),
            0, 1
        );
        o.UV = uvCorners[cornerIndex];
    }
    else
    {
        // Quad rendering: expand from Position + Size
        static const float2 quadCorners[6] = {
            float2(0, 0), float2(1, 0), float2(0, 1),
            float2(0, 1), float2(1, 0), float2(1, 1)
        };

        float2 corner = quadCorners[cornerIndex];
        float2 posPixel = s.Position + s.Size * corner;

        // Convert pixel coords to NDC
        o.Position = float4(
            posPixel.x / ScreenWidth * 2.0 - 1.0,
            -(posPixel.y / ScreenHeight * 2.0 - 1.0),
            0, 1
        );
        o.UV = lerp(s.UVs.xy, s.UVs.zw, corner);
    }

    return o;
}

float4 PS(PS_Input input) : SV_Target
{
    Texture2D tex = ResourceDescriptorHeap[input.TextureIndex];
    float4 texColor = tex.Sample(PointSampler, input.UV);
    return texColor * input.Color;
}
