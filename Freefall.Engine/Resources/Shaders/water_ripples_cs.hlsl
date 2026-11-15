// Ripples on the water round the camera (Graphics/WaterRipples.cs): one step of the wave equation over a height field,
// h(t+1) = (2 h(t) − h(t−1) + c² Δt² / Δx² ∇²h) × damping, the frame's stirs pushed into its first step, and the waves
// fading out toward the grid's edge so that none comes back off it. CSShift moves a height field by whole cells as the grid
// follows the camera: both h(t) and h(t−1), before the step. Shifting only what a step reads, h(t−1) of the next step was
// still in the old grid: every shift read as motion wherever the water sloped, and the grid filled with steep waves within
// a second (2026-11-15). Docs: .agent/knowledge/ocean_system/underwater.md "Ripples".

#pragma kernel CSStep
#pragma kernel CSClear
#pragma kernel CSShift

cbuffer PushConstants : register(b3)
{
    uint PrevSrvIdx;        // Texture2D<float>: h(t−1)
    uint CurSrvIdx;         // Texture2D<float>: h(t)
    uint OutUAVIdx;         // RWTexture2D<float>: h(t+1)
    uint DropsSrvIdx;       // StructuredBuffer<RippleDrop>
    uint DropCount;
    int ShiftX;             // CSShift: the new grid's cell (i, j) was the old one's (i + ShiftX, j + ShiftZ)
    int ShiftZ;
    uint OriginXBits;       // world xz of the new grid's corner
    uint OriginZBits;
    uint Courant2Bits;      // (c Δt / Δx)²
    uint DampingBits;       // per step
};

#define SIZE 512
#define CELL (48.0 / SIZE)
#define EDGE_CELLS 24.0     // the waves fade out over this many cells at the grid's edge

struct RippleDrop
{
    float2 XZ;
    float Radius;
    float Strength;
};

float Height(Texture2D<float> t, int2 c)
{
    return any(c < 0) || any(c >= SIZE) ? 0.0 : t.Load(int3(c, 0));
}

[numthreads(8, 8, 1)]
void CSStep(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float> prev = ResourceDescriptorHeap[PrevSrvIdx];
    Texture2D<float> cur = ResourceDescriptorHeap[CurSrvIdx];
    RWTexture2D<float> output = ResourceDescriptorHeap[OutUAVIdx];

    int2 p = int2(id.xy);
    int2 q = p;
    float h = Height(cur, q);
    float laplacian = Height(cur, q + int2(1, 0)) + Height(cur, q - int2(1, 0))
                    + Height(cur, q + int2(0, 1)) + Height(cur, q - int2(0, 1)) - 4.0 * h;
    float next = (2.0 * h - Height(prev, q) + asfloat(Courant2Bits) * laplacian) * asfloat(DampingBits);

    // The frame's stirs: a smooth push over each one's radius
    if (DropCount > 0)
    {
        StructuredBuffer<RippleDrop> drops = ResourceDescriptorHeap[DropsSrvIdx];
        float2 world = float2(asfloat(OriginXBits), asfloat(OriginZBits)) + (float2(p) + 0.5) * CELL;
        for (uint i = 0; i < DropCount; i++)
        {
            RippleDrop d = drops[i];
            float r = length(world - d.XZ) / max(d.Radius, CELL);
            if (r < 1.0)
            {
                float bump = 1.0 - r * r;
                next += d.Strength * bump * bump;
            }
        }
    }

    // Nothing comes back off the edge
    float2 edge = min(float2(p), float(SIZE - 1) - float2(p));
    next *= saturate(min(edge.x, edge.y) / EDGE_CELLS);
    output[p] = clamp(next, -0.5, 0.5);
}

[numthreads(8, 8, 1)]
void CSClear(uint3 id : SV_DispatchThreadID)
{
    RWTexture2D<float> output = ResourceDescriptorHeap[OutUAVIdx];
    output[id.xy] = 0.0;
}

// A height field into the grid moved by (ShiftX, ShiftZ) cells: what comes in at the far side is still water
[numthreads(8, 8, 1)]
void CSShift(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float> source = ResourceDescriptorHeap[CurSrvIdx];
    RWTexture2D<float> output = ResourceDescriptorHeap[OutUAVIdx];
    output[id.xy] = Height(source, int2(id.xy) + int2(ShiftX, ShiftZ));
}
