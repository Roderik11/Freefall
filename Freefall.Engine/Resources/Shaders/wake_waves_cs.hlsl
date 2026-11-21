// The ships' waves round the camera (Graphics/WakeWaves.cs): linear deep-water gravity waves over a grid SIZE² across
// SPAN metres, kept centred on the camera (moved by whole cells), stepped exactly in Fourier space. The state is the
// surface's height h and its velocity potential φ, packed as one complex field h + iφ; a step is
//   FFT (rows, twice: each pass writes transposed, so two passes give the 2D transform back in place)
//   CSEvolve: every wave turned by its own frequency ω = √(g k): dh/dt = k φ, dφ/dt = −g h, solved exactly, damped
//             (a little overall, more the shorter the wave), so long waves run fast and short ones slow: dispersion,
//             which is what draws a Kelvin wake behind a moving pressure (a wave equation's one speed draws a cone)
//   inverse FFT (twice)
//   CSForce: moved by whole cells as the grid follows the camera, damped toward its edge so nothing comes back off it,
//            and every hull (WaterHulls) pressing the water down by its depth under the still sea there: dφ/dt −= g κ d
// CSSurface writes what the water reads: height and its slopes. Docs: .agent/knowledge/engine_internals/ships.md.

// CSFoam keeps the white water on the same grid: made where a hull goes into the water and where it leaves it (its
// depth there changing) and where the simulated waves grow steep enough to break, carried by the water's own surface
// velocity (∇φ) and fading. Foam ages where the water took it, so nothing is drawn as a shape round the ship.
// Beside it the calm a hull leaves in its track (2026-11-21): made behind the stern as a moving hull leaves the water,
// carried the same way and fading over CALM_LIFE; the ocean turns the sea's short waves down by it (ocean_surf.hlsli
// CalmShare). The foam texture holds foam, how near a hull's waterline the cell was a step ago, and the calm.

#pragma kernel CSFft
#pragma kernel CSEvolve
#pragma kernel CSForce
#pragma kernel CSFoam
#pragma kernel CSSurface
#pragma kernel CSClear
#pragma kernel CSClear4

#include "water_hulls.hlsli"

cbuffer PushConstants : register(b3)
{
    uint InSrvIdx;          // Texture2D<float2> (CSSurface: the state)
    uint OutUAVIdx;         // RWTexture2D<float2> (CSSurface: RWTexture2D<float4>)
    uint Inverse;           // CSFft: 1 = inverse (scaled by 1/SIZE a pass)
    uint DtBits;            // CSEvolve: the step (s)
    uint DampBits;          // CSEvolve: overall damping (1/s)
    uint ViscBits;          // CSEvolve: damping by k² (m²/s): short waves die sooner
    int ShiftX;             // CSForce: the new grid's cell (i, j) was the old one's (i + ShiftX, j + ShiftZ)
    int ShiftZ;
    uint OriginXBits;       // CSForce: world xz of the grid's corner
    uint OriginZBits;
    uint HullsSrvIdx;       // CSForce: WaterHulls.Srv (0 = none)
    uint HullCount;
    uint SeaLevelBits;      // CSForce: the still sea's height (world y)
    uint PushBits;          // CSForce: g κ Δt: how hard a metre of hull presses φ a step
    uint FoamSrvIdx;        // CSFoam: Texture2D<float4> foam, nearness a step ago, calm; CSSurface: the foam now
    uint CalmUAVIdx;        // CSSurface: RWTexture2D<float> the calm the ocean reads (published with the surface)
    uint NoCalm;            // CSFoam: 1 = no calm is made (EngineSettings.DisableWakeCalm): white water at FOAM_LIFE everywhere
};

#define SIZE 512
#define HALF (SIZE / 2)
#define SPAN 256.0
#define CELL (SPAN / SIZE)
#define GRAVITY 9.81
#define EDGE_CELLS 32.0     // the waves fade out over this many cells at the grid's edge
#define EDGE_KEEP 0.9       // what is left of them a step at the very edge
#define END_SOFT 2.0        // m over which a hull's push fades out at its stern and its stem (HullDepth)

// ── FFT: one row per group, radix-2 Stockham in shared memory, written transposed ──
groupshared float2 Line[SIZE];

[numthreads(HALF, 1, 1)]
void CSFft(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    Texture2D<float2> input = ResourceDescriptorHeap[InSrvIdx];
    RWTexture2D<float2> output = ResourceDescriptorHeap[OutUAVIdx];
    uint row = gid.x;
    uint t = tid.x;
    Line[t] = input.Load(int3(t, row, 0));
    Line[t + HALF] = input.Load(int3(t + HALF, row, 0));
    GroupMemoryBarrierWithGroupSync();

    float dir = Inverse != 0 ? 1.0 : -1.0;
    [unroll]
    for (uint ns = 1; ns < SIZE; ns <<= 1)
    {
        float2 v0 = Line[t];
        float2 v1 = Line[t + HALF];
        uint k = t & (ns - 1);
        float s, c;
        sincos(dir * 3.14159265 * (float)k / (float)ns, s, c);
        v1 = float2(v1.x * c - v1.y * s, v1.x * s + v1.y * c);
        uint d = (t - k) * 2 + k;
        GroupMemoryBarrierWithGroupSync();
        Line[d] = v0 + v1;
        Line[d + ns] = v0 - v1;
        GroupMemoryBarrierWithGroupSync();
    }

    float scale = Inverse != 0 ? 1.0 / SIZE : 1.0;
    output[uint2(row, t)] = Line[t] * scale;
    output[uint2(row, t + HALF)] = Line[t + HALF] * scale;
}

// ── One step of every wave, exactly ──
[numthreads(8, 8, 1)]
void CSEvolve(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float2> spectrum = ResourceDescriptorHeap[InSrvIdx];
    RWTexture2D<float2> output = ResourceDescriptorHeap[OutUAVIdx];
    uint2 i = id.xy;
    uint2 m = (SIZE - i) & (SIZE - 1);                  // the wave going the other way, −k
    float2 z = spectrum.Load(int3(i, 0));
    float2 zm = spectrum.Load(int3(m, 0));
    float2 zmc = float2(zm.x, -zm.y);
    // h and φ are real, so their transforms are Hermitian: unpacked from Z = H + iΦ
    float2 H = 0.5 * (z + zmc);
    float2 dd = 0.5 * (z - zmc);
    float2 P = float2(dd.y, -dd.x);                     // dd / i

    float2 f = float2(i.x < HALF ? (float)i.x : (float)i.x - SIZE, i.y < HALF ? (float)i.y : (float)i.y - SIZE);
    float k = length(f) * (6.2831853 / SPAN);
    if (k < 1e-6)
    {
        output[i] = 0.0;                                // no mean level, no drifting potential
        return;
    }
    float w = sqrt(GRAVITY * k);
    float dt = asfloat(DtBits);
    float s, c;
    sincos(w * dt, s, c);
    float2 H2 = H * c + P * (k / w * s);
    float2 P2 = P * c - H * (w / k * s);
    float keep = exp(-(asfloat(DampBits) + asfloat(ViscBits) * k * k) * dt);
    H2 *= keep;
    P2 *= keep;
    output[i] = float2(H2.x - P2.y, H2.y + P2.x);       // H2 + i P2
}

// How deep a hull lies under the water at a point (m): the lowest of its levels whose half-breadth there reaches past
// the point's distance from its middle, under its own waterline (y 0 in its frame); soft over the last metre inside its
// waterline, so the push has no hard edge to ring at the grid's own scale.
// The water is the hull's waterline, not the still sea: a ship rides the swell, and the waves it makes are the ones its
// draft raises as it goes through the water, whatever the swell does under it. Measured against the still sea, a caravel
// down 2 m in a trough pressed 2 m deeper, up on a crest hardly at all, and pumped a white ridge 2–3 m high round its
// stern with every swell; its waterline swept the water as it pitched and heaved and churned white water all round it
// (user, 2026-11-21: the wake "seems to do the opposite" of calming the sea). The point is taken at the hull's own height,
// so its pitch and roll do not slide the footprint about either.
//   near: 1 inside a hull's waterline, fading out 2 m outside it (the hull's own dent is no breaking wave)
//   edge: 1 in the band along the waterline where the push ramps up (where the hull meets the water), 0 elsewhere
float HullDepth(float2 xz, float seaLevel, out float near, out float edge)
{
    near = 0.0;
    edge = 0.0;
    if (HullsSrvIdx == 0)
        return 0.0;
    StructuredBuffer<WaterHull> hulls = ResourceDescriptorHeap[HullsSrvIdx];
    float depth = 0.0;
    for (uint i = 0; i < HullCount; i++)
    {
        float4 sphere = hulls[i].Sphere;
        float2 dxz = xz - sphere.xz;
        float reach = sphere.w + 2.0;
        if (dot(dxz, dxz) > reach * reach)
            continue;
        float x0 = hulls[i].X0, x1 = hulls[i].X1, y0 = hulls[i].Bottom, y1 = hulls[i].Top;
        // At the height of the hull's waterline (its sphere's centre is the middle of its extent, y (y0 + y1) / 2)
        float4 w = float4(xz.x, sphere.y - 0.5 * (y0 + y1), xz.y, 1.0);
        float3 l = float3(dot(hulls[i].Row0, w), dot(hulls[i].Row1, w), dot(hulls[i].Row2, w));
        if (l.x <= x0 || l.x >= x1 || y1 <= y0)
            continue;
        float fx = (l.x - x0) / (x1 - x0) * (HULL_STATIONS - 1);
        uint ix = min((uint)fx, HULL_STATIONS - 2);
        float tx = fx - ix;
        float az = abs(l.z);
        float water = clamp(0.0, y0, y1);
        // The half-breadth at the water's height, and the lowest level reaching past |z|
        float fyW = saturate((water - y0) / (y1 - y0)) * (HULL_LEVELS - 1);
        uint iyW = min((uint)fyW, HULL_LEVELS - 2);
        float bw0 = lerp(hulls[i].Breadth[iyW * HULL_STATIONS + ix], hulls[i].Breadth[iyW * HULL_STATIONS + ix + 1], tx);
        float bw1 = lerp(hulls[i].Breadth[(iyW + 1) * HULL_STATIONS + ix], hulls[i].Breadth[(iyW + 1) * HULL_STATIONS + ix + 1], tx);
        float bWater = lerp(bw0, bw1, fyW - iyW);
        // Only where the hull reaches the water: its extent runs on over the beakhead and the bowsprit, and a band round
        // the middle line out there laid a white tongue 6 m ahead of the stem (user: "its counting that wood beam in
        // the front as part of the bow", 2026-11-21)
        if (bWater <= 0.05)
            continue;
        near = max(near, 1.0 - smoothstep(bWater, bWater + 2.0, az));
        if (bWater <= az)
            continue;
        float bottom = water;
        float bPrev = 0.0;
        float yPrev = y0;
        [loop]
        for (uint level = 0; level < HULL_LEVELS; level++)
        {
            float y = y0 + (y1 - y0) * level / (HULL_LEVELS - 1);
            float b = lerp(hulls[i].Breadth[level * HULL_STATIONS + ix], hulls[i].Breadth[level * HULL_STATIONS + ix + 1], tx);
            if (b > az)
            {
                // Between the last level that did not reach |z| and this one
                bottom = level == 0 ? y : lerp(yPrev, y, saturate((az - bPrev) / max(b - bPrev, 1e-4)));
                break;
            }
            bPrev = b;
            yPrev = y;
        }
        // Soft across the waterline, and along it at the hull's ends: a transom ends the hull square, and there the push
        // stopped in one cell, rang at the grid's own scale into short steep waves behind the stern, and those counted as
        // breaking (BREAK_SLOPE) all down the track: a white sheet behind every ship
        float soft = smoothstep(0.0, 1.0, bWater - az) * smoothstep(0.0, END_SOFT, l.x - x0) * smoothstep(0.0, END_SOFT, x1 - l.x);
        depth = max(depth, max(water - bottom, 0.0) * soft);
        edge = max(edge, 4.0 * soft * (1.0 - soft));
    }
    return depth;
}

// ── After the inverse transform: moved with the grid, damped at the edge, pressed by the hulls ──
[numthreads(8, 8, 1)]
void CSForce(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float2> field = ResourceDescriptorHeap[InSrvIdx];
    RWTexture2D<float2> state = ResourceDescriptorHeap[OutUAVIdx];
    int2 p = int2(id.xy);
    int2 src = p + int2(ShiftX, ShiftZ);
    float2 hp = any(src < 0) || any(src >= SIZE) ? 0.0 : field.Load(int3(src, 0));

    float edge = (float)min(min(p.x, p.y), min(SIZE - 1 - p.x, SIZE - 1 - p.y));
    float inner = smoothstep(0.0, EDGE_CELLS, edge);
    hp *= lerp(EDGE_KEEP, 1.0, inner);

    // A hull coming into the grid at its edge presses in gradually: a sudden press rings out
    float2 world = float2(asfloat(OriginXBits), asfloat(OriginZBits)) + (float2(p) + 0.5) * CELL;
    float near, rim;
    float depth = HullDepth(world, asfloat(SeaLevelBits), near, rim) * smoothstep(EDGE_CELLS, 3.0 * EDGE_CELLS, edge);
    hp.y -= asfloat(PushBits) * depth;
    state[p] = hp;
}

// ── The white water ──
#define FOAM_LIFE 3.5       // s for it to fade to a third
#define FOAM_MOST 1.5
#define FOAM_SPREAD 4.0     // per second, toward the mean of its four neighbours
#define BREAK_SLOPE 0.4     // the waves' slope where they start to break (a deep-water wave breaks at ~0.45: steepness
                            // 1/7); at 0.15 the whole wake's ordinary waves foamed into a white V
#define BREAK_RANGE 0.25    // and over which they break fully
#define BREAK_RATE 0.8      // foam a second where they break fully (4 laid solid white sheets; 1.5 white blotches)
#define ADVANCE 1.6         // foam a cell gets as a hull's outline comes over it (the stem, the bow widening past it),
                            // at full sweep (below)
#define LEAVE 0.6           // and as the outline leaves it behind (the stern): a short wash into the calm track behind
                            // it (1.1 laid ~20 m of solid white behind a caravel at 9 m/s once the track had no chop)
#define SWEEP_FROM 0.4      // how fast the outline must sweep over the water (nearness a second, ~0.8 m/s of hull at the
                            // stem) before it churns anything: a ship lying still churned as it rocked (user, 2026-11-21:
                            // "the churn should be minimal at standstill and ramp up with speed")
#define SWEEP_FULL 2.0      // and by which it churns fully (~4 m/s)
#define CARRY 0.3           // share of the simulated surface velocity that carries the foam: the hull as a pressure patch
                            // drives a stronger flow round it than a hull does, and all of it blew the foam into a sheet

// ── The calm in a hull's track ──
#define CALM_LIFE 20.0      // s for it to fade to a third: the wind rebuilds the short waves behind the ship
#define CALM_SPREAD 1.5     // per second, toward the mean of its four neighbours: the track's edges soften and it widens
#define CALM_FROM 0.2       // how fast the stern's outline must leave the water before it leaves calm behind (nearness a
                            // second, ~0.4 m/s of hull): a ship lying still and rocking leaves none
#define CALM_FULL 0.6       // and by which the calm is full (~1.2 m/s)
#define CALM_FOAM_LIFE 0.5  // s for the white water in a full calm to fade to a third (FOAM_LIFE elsewhere)

float2 LoadClamped(Texture2D<float2> t, int2 c)
{
    return t.Load(int3(clamp(c, 0, SIZE - 1), 0));
}

float4 LoadClamped4(Texture2D<float4> t, int2 c)
{
    return t.Load(int3(clamp(c, 0, SIZE - 1), 0));
}

float4 LoadInside(Texture2D<float4> t, int2 c)
{
    return all(c >= 0) && all(c < SIZE) ? t.Load(int3(c, 0)) : 0.0;
}

[numthreads(8, 8, 1)]
void CSFoam(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float2> state = ResourceDescriptorHeap[InSrvIdx];
    Texture2D<float4> foamIn = ResourceDescriptorHeap[FoamSrvIdx];
    RWTexture2D<float4> foamOut = ResourceDescriptorHeap[OutUAVIdx];
    int2 p = int2(id.xy);
    float dt = asfloat(DtBits);

    // The water's velocity at its surface (∇φ) and its slope
    float2 r = LoadClamped(state, p + int2(1, 0)), l = LoadClamped(state, p - int2(1, 0));
    float2 u = LoadClamped(state, p + int2(0, 1)), d = LoadClamped(state, p - int2(0, 1));
    float2 velocity = float2(r.y - l.y, u.y - d.y) / (2.0 * CELL) * CARRY;
    float steep = length(float2(r.x - l.x, u.x - d.x) / (2.0 * CELL));

    // Where this water was a step ago, in the old grid (moved by the shift): bilinear from four cells
    float2 from = float2(p) + float2(ShiftX, ShiftZ) - velocity * dt / CELL;
    int2 c0 = int2(floor(from));
    float2 t = from - c0;
    float4 f00 = LoadInside(foamIn, c0);
    float4 f10 = LoadInside(foamIn, c0 + int2(1, 0));
    float4 f01 = LoadInside(foamIn, c0 + int2(0, 1));
    float4 f11 = LoadInside(foamIn, c0 + 1);
    float4 carried = lerp(lerp(f00, f10, t.x), lerp(f01, f11, t.x), t.y);
    float foam = carried.x;
    float calm = carried.z;
    // Spreading a little into its neighbours (about a cell in a quarter second), so no cell stands out of its field
    float4 neighbours = 0.25 * (LoadClamped4(foamIn, p + int2(ShiftX + 1, ShiftZ)) + LoadClamped4(foamIn, p + int2(ShiftX - 1, ShiftZ))
                              + LoadClamped4(foamIn, p + int2(ShiftX, ShiftZ + 1)) + LoadClamped4(foamIn, p + int2(ShiftX, ShiftZ - 1)));
    foam = lerp(foam, neighbours.x, saturate(FOAM_SPREAD * dt));
    // In the track the white water goes within a second (CALM_FOAM_LIFE) instead of FOAM_LIFE: what the stern churned
    // and the track's own breaking waves turn into the smooth band behind it at once (in a storm the hull plunging in
    // the swell churned a white sheet over the stern that lay there for seconds: "it seems to do the opposite")
    foam *= exp(-dt / lerp(FOAM_LIFE, CALM_FOAM_LIFE, saturate(calm)));
    calm = lerp(calm, neighbours.z, saturate(CALM_SPREAD * dt));
    calm *= exp(-dt / CALM_LIFE);

    // How near a hull's waterline this water was a step ago (moved with the grid) and is now: 1 inside, falling to 0
    // 2 m outside it (and 2 m past the stem and the stern)
    int2 src = p + int2(ShiftX, ShiftZ);
    float before = any(src < 0) || any(src >= SIZE) ? 0.0 : foamIn.Load(int3(src, 0)).y;
    float2 world = float2(asfloat(OriginXBits), asfloat(OriginZBits)) + (float2(p) + 0.5) * CELL;
    float near, rim;
    HullDepth(world, asfloat(SeaLevelBits), near, rim);

    // White water where the hull's outline comes over the water (ahead of the stem, beside the bow as it widens past)
    // and where it leaves it (behind the stern): outside the waterline, where it shows, and as the hull goes on the foam
    // stays in the water, so the bow's lines the hull's sides aft and trails off behind. Made just inside the waterline
    // (from the change of the hull's depth there) it was under the hull, where the sea is not drawn, and showed only
    // when the hull pitched, all round at once in blotches (user, 2026-11-21: "big white blotches but pretty much
    // invisible most of the time")
    // Only as the hull goes through the water, more the faster: the outline sweeping over a cell at a rate past
    // SWEEP_FROM, the foam growing with the rate up to SWEEP_FULL
    float rise = near - before;
    if (dt > 0.0)
    {
        float sweep = abs(rise) / dt;
        float churn = max(sweep - SWEEP_FROM, 0.0) * dt * saturate(sweep / SWEEP_FULL);
        foam += churn * (rise > 0.0 ? ADVANCE : LEAVE);
        // The calm: where the outline leaves the water behind it (the stern going on), as much as the water was inside the
        // hull's reach a step ago, by how fast it leaves. Not at the bow (it comes over the water there) nor along the
        // sides (their outline stands still as the hull glides past), so the track starts at the stern, as wide as it
        if (rise < 0.0)
            calm = max(calm, before * saturate((sweep - CALM_FROM) / (CALM_FULL - CALM_FROM)));
    }
    if (NoCalm != 0)
        calm = 0.0;
    // Where the waves break: on crests, away from the hull's own dent (its sides are steep but nothing breaks there;
    // until 2026-11-21 its whole outline foamed and the path it swept was solid white)
    float height = state.Load(int3(p, 0)).x;
    foam += saturate((steep - BREAK_SLOPE) / BREAK_RANGE) * BREAK_RATE * dt * (1.0 - near) * step(0.0, height);
    foamOut[p] = float4(min(foam, FOAM_MOST), near, saturate(calm), 0.0);
}

// ── What the water reads: height, slopes (m, m/m) and foam, fading out toward the edge ──
[numthreads(8, 8, 1)]
void CSSurface(uint3 id : SV_DispatchThreadID)
{
    Texture2D<float2> state = ResourceDescriptorHeap[InSrvIdx];
    Texture2D<float4> foam = ResourceDescriptorHeap[FoamSrvIdx];
    RWTexture2D<float4> surface = ResourceDescriptorHeap[OutUAVIdx];
    RWTexture2D<float> calmOut = ResourceDescriptorHeap[CalmUAVIdx];
    int2 p = int2(id.xy);
    int2 r = min(p + int2(1, 0), SIZE - 1), l = max(p - int2(1, 0), 0);
    int2 u = min(p + int2(0, 1), SIZE - 1), d = max(p - int2(0, 1), 0);
    float h = state.Load(int3(p, 0)).x;
    float hx = (state.Load(int3(r, 0)).x - state.Load(int3(l, 0)).x) / (2.0 * CELL);
    float hz = (state.Load(int3(u, 0)).x - state.Load(int3(d, 0)).x) / (2.0 * CELL);
    float edge = (float)min(min(p.x, p.y), min(SIZE - 1 - p.x, SIZE - 1 - p.y));
    float fade = smoothstep(0.0, EDGE_CELLS, edge);
    float4 f = foam.Load(int3(p, 0));
    surface[p] = float4(h, hx, hz, f.x) * fade;
    calmOut[p] = f.z * fade;
}

[numthreads(8, 8, 1)]
void CSClear(uint3 id : SV_DispatchThreadID)
{
    RWTexture2D<float2> output = ResourceDescriptorHeap[OutUAVIdx];
    output[id.xy] = 0.0;
}

// The foam textures (four channels)
[numthreads(8, 8, 1)]
void CSClear4(uint3 id : SV_DispatchThreadID)
{
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutUAVIdx];
    output[id.xy] = 0.0;
}
