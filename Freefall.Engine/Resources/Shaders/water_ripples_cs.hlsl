// Ripples on the water round the camera (Graphics/WaterRipples.cs): one step of the wave equation over a height field,
// h(t+1) = (2 h(t) − h(t−1) + c² Δt² / Δx² ∇²h) × damping, the frame's stirs pushed into its first step, and the waves
// fading out toward the grid's edge so that none comes back off it. The second channel is foam a splash or a wader left
// on the surface: it fades (FoamKept a step), spreads a little to its neighbours, and the stirs add to it.
// CSShift moves the field by whole cells as the grid follows the camera: both h(t) and h(t−1), before the step. Shifting
// only what a step reads, h(t−1) of the next step was still in the old grid: every shift read as motion wherever the
// water sloped, and the grid filled with steep waves within a second (2026-11-15).
// Docs: .agent/knowledge/ocean_system/underwater.md "Ripples".

#pragma kernel CSStep
#pragma kernel CSClear
#pragma kernel CSShift

cbuffer PushConstants : register(b3)
{
    uint PrevSrvIdx;        // Texture2D<float2>: h(t−1), foam
    uint CurSrvIdx;         // Texture2D<float2>: h(t), foam
    uint OutUAVIdx;         // RWTexture2D<float2>: h(t+1), foam
    uint DropsSrvIdx;       // StructuredBuffer<RippleDrop>
    uint DropCount;
    int ShiftX;             // CSShift: the new grid's cell (i, j) was the old one's (i + ShiftX, j + ShiftZ)
    int ShiftZ;
    uint OriginXBits;       // world xz of the new grid's corner
    uint OriginZBits;
    uint Courant2Bits;      // (c Δt / Δx)²
    uint DampingBits;       // per step
    uint FoamKeptBits;      // foam left after a step
};

#define SIZE 512
#define CELL (48.0 / SIZE)
#define EDGE_CELLS 24.0     // the waves fade out over this many cells at the grid's edge
#define FOAM_SPREAD 0.02    // share of the neighbours' mean the foam takes a step: a patch widens ~0.3 m over seconds
#define FOAM_MOST 1.5

struct RippleDrop
{
    float2 XZ;
    float Radius;
    float Strength;
    float Foam;
    float3 Pad;
};

float2 Cell(Texture2D<float2> t, int2 c)
{
    return any(c < 0) || any(c >= SIZE) ? 0.0 : t.Load(int3(c, 0));
}

[numthreads(8, 8, 1)]
void CSStep(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float2> prev = ResourceDescriptorHeap[PrevSrvIdx];
    Texture2D<float2> cur = ResourceDescriptorHeap[CurSrvIdx];
    RWTexture2D<float2> output = ResourceDescriptorHeap[OutUAVIdx];

    int2 p = int2(id.xy);
    float2 here = Cell(cur, p);
    float2 r = Cell(cur, p + int2(1, 0)), l = Cell(cur, p - int2(1, 0));
    float2 u = Cell(cur, p + int2(0, 1)), d = Cell(cur, p - int2(0, 1));
    float laplacian = r.x + l.x + u.x + d.x - 4.0 * here.x;
    float next = (2.0 * here.x - Cell(prev, p).x + asfloat(Courant2Bits) * laplacian) * asfloat(DampingBits);

    // The foam fades and spreads toward its neighbours' mean
    float foamMean = 0.25 * (r.y + l.y + u.y + d.y);
    float foam = lerp(here.y, foamMean, FOAM_SPREAD) * asfloat(FoamKeptBits);

    // The frame's stirs: a smooth push over each one's radius, and the foam it leaves
    if (DropCount > 0)
    {
        StructuredBuffer<RippleDrop> drops = ResourceDescriptorHeap[DropsSrvIdx];
        float2 world = float2(asfloat(OriginXBits), asfloat(OriginZBits)) + (float2(p) + 0.5) * CELL;
        for (uint i = 0; i < DropCount; i++)
        {
            RippleDrop drop = drops[i];
            float rr = length(world - drop.XZ) / max(drop.Radius, CELL);
            if (rr < 1.0)
            {
                float bump = 1.0 - rr * rr;
                next += drop.Strength * bump * bump;
                foam += drop.Foam * bump;
            }
        }
    }

    // Nothing comes back off the edge
    float2 edge = min(float2(p), float(SIZE - 1) - float2(p));
    float fade = saturate(min(edge.x, edge.y) / EDGE_CELLS);
    output[p] = float2(clamp(next * fade, -0.5, 0.5), clamp(foam * fade, 0.0, FOAM_MOST));
}

[numthreads(8, 8, 1)]
void CSClear(uint3 id : SV_DispatchThreadID)
{
    RWTexture2D<float2> output = ResourceDescriptorHeap[OutUAVIdx];
    output[id.xy] = 0.0;
}

// The field into the grid moved by (ShiftX, ShiftZ) cells: what comes in at the far side is still water without foam
[numthreads(8, 8, 1)]
void CSShift(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float2> source = ResourceDescriptorHeap[CurSrvIdx];
    RWTexture2D<float2> output = ResourceDescriptorHeap[OutUAVIdx];
    output[id.xy] = Cell(source, int2(id.xy) + int2(ShiftX, ShiftZ));
}
