// A broken moon's picture (Moon.Broken): something passed it in a glancing blow and took one side of it apart. The crust
// there broke into thick curved plates; the middle ones were carried off along the blow, turning as they went, the ones
// at the edge only lifted and cracked open. Below the crust lies a dark core, scooped where the blow struck. Cracks run
// out across the rest of the face, small chunks, grit and dust trail after the plates. The sun lights all of it, so the
// phases show on every piece, and the pieces shadow each other.
//
// Made by Graphics/BrokenMoons.cs into a texture over a rectangle of the moon's disc frame that holds the moon, its
// pieces and its dust, again only when the sun has moved across it (a band of rows a frame); the sky (mesh_skybox.fx
// MoonsAndStars) puts the picture where the moon stands. Docs: .agent/knowledge/atmospheric_effects/sky_and_environment.md
// §8 "Broken moons".
//
// Frame: x = right on the disc, y = up, z = away from the viewer (the moon's direction; the visible half of the whole
// moon is z < 0); the moon's radius is 1. The view is orthographic: every ray runs along +z.
//
// The crust is cut into plates by a Voronoi pattern on the sphere: seeds in the broken cap round BreakDir (the plates
// that moved, or stayed and cracked), a ring of seeds just outside it that stay put, and one on the far side. A plate is
// its cell (bounded by the great-circle bisectors to the other seeds) of a shell CrustThickness thick, moved and turned
// as the blow carried it. The layout comes from the CPU (BrokenMoons.Layout), so a texel only marches.

#pragma kernel CSBakeMoon

cbuffer PushConstants : register(b3)
{
    uint DataIdx;           // StructuredBuffer<float4>: the header, the seeds, the chips (BrokenMoons.cs)
    uint TargetUAVIdx;      // RWTexture2D<float4>: rgb the light it sends (× the moon's colour, premultiplied), a its cover
    uint RowStart;          // the first row this dispatch makes
    uint TargetWidth;
    uint TargetHeight;
};

SamplerState linearWrap : register(s0);

// The header, a float4 each (BrokenMoons.cs writes the same)
#define H_SUN 0             // xyz: the sun's direction in the disc frame, w: earthshine
#define H_RECT 1            // xy: the rectangle's corner (y up), zw: its size, in moon radii
#define H_AMOUNTS 2         // x: crust thickness, y: cracks, z: dust, w: grit
#define H_INNER 3           // rgb: the broken rock's albedo (relative: the surface texture averages 1), w: dust light
#define H_CORE 4            // rgb: the core's
#define H_DUST 5            // rgb: the dust's, w: the scoop in the core where the blow struck (its radius)
#define H_BREAK 6           // xyz: the middle of the broken side, w: the broken cap's angle (radians)
#define H_BLOW 7            // xyz: the way the blow carried what it broke, w: how far the dust reaches along it
#define H_COUNTS 8          // x: seeds, y: of them in the cap, z: chips, w: asuint the surface texture
#define H_TEX 9             // x: asuint the normal map, y: their mip for a texel of this picture
#define HEADER 10

#define MAX_SEEDS 64
#define MAX_CHIPS 48

groupshared float4 gSeed[MAX_SEEDS];    // xyz: the seed's direction, w: how far the blow carried its plate (0 = stayed)
groupshared float4 gMove[MAX_SEEDS];    // xyz: the plate's offset, w: the angle it turned by
groupshared float4 gAxis[MAX_SEEDS];    // xyz: the axis it turned round, w: its cell's reach (radians from the seed)
groupshared float4 gChip[MAX_CHIPS];    // xyz: a small chunk's centre, w: its size

static float3 SunL;
static float Earthshine;
static float4 Rect;
static float Thick, CracksAmount, DustAmount, GritAmount;
static float3 InnerColor, CoreColor, DustColor;
static float DustLight, ScoopRadius;
static float3 BreakDir, BlowDir;
static float BreakCap, DustReach;
static uint SeedCount, CapCount, ChipCount;
static uint SurfaceTex, NormalTex;
static float TexMip;

void LoadHeader(StructuredBuffer<float4> data)
{
    float4 h;
    h = data[H_SUN];      SunL = h.xyz; Earthshine = h.w;
    Rect = data[H_RECT];
    h = data[H_AMOUNTS];  Thick = h.x; CracksAmount = h.y; DustAmount = h.z; GritAmount = h.w;
    h = data[H_INNER];    InnerColor = h.rgb; DustLight = h.w;
    h = data[H_CORE];     CoreColor = h.rgb;
    h = data[H_DUST];     DustColor = h.rgb; ScoopRadius = h.w;
    h = data[H_BREAK];    BreakDir = h.xyz; BreakCap = h.w;
    h = data[H_BLOW];     BlowDir = h.xyz; DustReach = h.w;
    h = data[H_COUNTS];   SeedCount = min((uint)h.x, MAX_SEEDS); CapCount = min((uint)h.y, SeedCount);
                          ChipCount = min((uint)h.z, MAX_CHIPS); SurfaceTex = asuint(h.w);
    h = data[H_TEX];      NormalTex = asuint(h.x); TexMip = h.y;
}

// ── Noise ──────────────────────────────────────────

float BHash(float3 p)
{
    p = frac(p * 0.3183099 + 0.1);
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float BNoise(float3 x)
{
    float3 i = floor(x);
    float3 f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(lerp(BHash(i + float3(0, 0, 0)), BHash(i + float3(1, 0, 0)), f.x),
                     lerp(BHash(i + float3(0, 1, 0)), BHash(i + float3(1, 1, 0)), f.x), f.y),
                lerp(lerp(BHash(i + float3(0, 0, 1)), BHash(i + float3(1, 0, 1)), f.x),
                     lerp(BHash(i + float3(0, 1, 1)), BHash(i + float3(1, 1, 1)), f.x), f.y), f.z);
}

float BFbm(float3 p)
{
    float s = 0.0, a = 0.5;
    [unroll] for (int o = 0; o < 4; o++)
    {
        s += a * BNoise(p);
        p = p * 2.03 + 11.7;
        a *= 0.5;
    }
    return s / 0.9375;
}

// Rough fracture: ridged octaves (creases rather than bumps), about −1..1
float BFracture(float3 p)
{
    float s = 0.0, a = 0.5;
    [unroll] for (int o = 0; o < 3; o++)
    {
        s += a * (1.0 - 4.0 * abs(BNoise(p) - 0.5));
        p = p * 2.07 + 5.3;
        a *= 0.5;
    }
    return s;
}

// Distance to the nearest cell wall of a 3D Worley pattern (F2 − F1, ≈ 0 on a wall)
float WorleyEdge(float3 x)
{
    float3 i = floor(x);
    float f1 = 1e9, f2 = 1e9;
    [unroll] for (int a = -1; a <= 1; a++)
    [unroll] for (int b = -1; b <= 1; b++)
    [unroll] for (int c = -1; c <= 1; c++)
    {
        float3 cell = i + float3(a, b, c);
        float3 pt = cell + float3(BHash(cell), BHash(cell + 17.1), BHash(cell + 41.7)) * 0.8 + 0.1;
        float d = length(x - pt);
        if (d < f1) { f2 = f1; f1 = d; }
        else if (d < f2) f2 = d;
    }
    return f2 - f1;
}

float3 RotateAxis(float3 v, float3 axis, float angle)
{
    float s, c;
    sincos(angle, s, c);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// ── The pieces ─────────────────────────────────────

// How far inside seed k's cell a direction lies (≈ an angle; < 0 outside)
float CellInside(float3 dir, uint k)
{
    float d = 1e9;
    float3 a = gSeed[k].xyz;
    [loop] for (uint j = 0; j < SeedCount; j++)
    {
        if (j == k) continue;
        float3 ab = a - gSeed[j].xyz;
        d = min(d, dot(dir, ab) * rsqrt(dot(ab, ab)));
    }
    return d;
}

uint NearestSeed(float3 dir)
{
    uint best = 0;
    float bestDot = -2.0;
    [loop] for (uint j = 0; j < SeedCount; j++)
    {
        float d = dot(dir, gSeed[j].xyz);
        if (d > bestDot) { bestDot = d; best = j; }
    }
    return best;
}

// A crack's half width between two plates, in radii: wider where the crust broke more
float CrackWidth(float carried)
{
    return 0.002 + 0.012 * saturate(carried * 4.0);
}

// The shell of the crust (signed distance from its outer and inner surface)
float ShellDist(float3 q)
{
    return abs(length(q) - (1.0 - Thick * 0.5)) - Thick * 0.5;
}

// The plate of seed k, at a point already brought back to where it lay
float PlateDist(float3 q, uint k)
{
    // Off the shell, its distance is a lower bound of the plate's: the cell walls need not be looked at
    float shell = ShellDist(q);
    if (shell > 0.03)
        return shell;
    float len = length(q);
    float3 dir = q / max(len, 1e-5);
    float inside = CellInside(dir, k) * len;
    float edge = -inside + CrackWidth(gSeed[k].w) + 0.004 * BFracture(q * 22.0);
    return max(ShellDist(q), edge);
}

float3 PlateHomeCentre(uint k)
{
    return gSeed[k].xyz * (1.0 - Thick * 0.5);
}

// Where a point of plate k lay before the blow moved it
float3 PlateHome(float3 p, uint k)
{
    float3 home = PlateHomeCentre(k);
    return RotateAxis(p - home - gMove[k].xyz, gAxis[k].xyz, -gMove[k].w) + home;
}

// A chunk of broken rock: a ball with caps knocked off it by four fracture planes, rough
float RockSdf(float3 q, float r, float seed)
{
    float d = length(q) - r;
    [unroll] for (int i = 0; i < 4; i++)
    {
        float3 n = normalize(float3(BHash(float3(seed, i * 3.1, 0.3)), BHash(float3(seed, i * 3.1, 1.7)),
                                    BHash(float3(seed, i * 3.1, 2.9))) - 0.5);
        d = max(d, dot(q, n) - r * (0.40 + 0.3 * BHash(float3(seed, i * 3.1, 4.4))));
    }
    return d + r * 0.10 * BFracture(q / r * 1.6 + seed);
}

bool Moved(uint k)
{
    return k < CapCount && gSeed[k].w >= 0.02;
}

// The whole broken moon: x = distance, y = what: 0 the core, 1 the crust in place, 2 + k plate k, 2 + MAX_SEEDS + c chip c
float2 BrokenSdf(float3 p)
{
    float len = length(p);
    float2 best = float2(len - 1.0, 1);

    if (len < 1.15)
    {
        // The core, scooped where the blow struck hardest
        float core = len - (1.0 - Thick) + 0.03 * BFracture(p * 5.0);
        core = max(core, -(length(p - BreakDir * 1.05) - ScoopRadius));
        best = float2(core, 0);

        // The crust where its plate stayed: cracked along the cells, missing where the plate has gone. Off the shell
        // its distance is a lower bound, and nothing nearer than what was found already needs the cells
        float shell = ShellDist(p);
        if (shell > 0.03 || shell >= best.x)
        {
            if (shell < best.x)
                best = float2(shell, 1);
        }
        else
        {
        float3 dir = p / max(len, 1e-5);
        uint k = NearestSeed(dir);
        float inside = CellInside(dir, k) * len;
        float crust;
        if (!Moved(k))
        {
            float nearBreak = smoothstep(BreakCap + 0.6, BreakCap, acos(clamp(dot(dir, BreakDir), -1.0, 1.0)));
            crust = max(ShellDist(p), -inside + CrackWidth(0.0) * nearBreak);
        }
        else
            crust = max(ShellDist(p), inside);      // a lower bound: this cell's crust is gone
        if (crust < best.x)
            best = float2(crust, 1);
        }
    }

    // The plates the blow moved
    [loop] for (uint k = 0; k < CapCount; k++)
    {
        if (!Moved(k))
            continue;
        float bound = length(p - PlateHomeCentre(k) - gMove[k].xyz) - (gAxis[k].w + Thick);
        float d = bound;
        if (bound < 0.05)
            d = PlateDist(PlateHome(p, k), k);
        if (d < best.x)
            best = float2(d, 2 + k);
    }

    // The small chunks
    [loop] for (uint c = 0; c < ChipCount; c++)
    {
        float3 q = p - gChip[c].xyz;
        float d = length(q) - gChip[c].w;
        if (d < 0.03)
            d = RockSdf(q, gChip[c].w, c * 7.3 + 2.1);
        if (d < best.x)
            best = float2(d, 2 + MAX_SEEDS + c);
    }
    return best;
}

bool IsPlate(uint id)
{
    return id >= 2 && id < 2 + MAX_SEEDS;
}

// Whether anything can stand between a point of the crust in place and the sun: the broken edge and the core near
// the break, or a moved plate or a chunk the sun's ray passes (their bounding balls). Elsewhere the sphere is convex
// and nothing else lies over it, so the shadow march is left out.
bool ShadowPossible(float3 o, float3 dir)
{
    if (dot(normalize(o), BreakDir) > cos(min(BreakCap + 0.35, 3.1)))
        return true;
    [loop] for (uint k = 0; k < CapCount; k++)
    {
        if (!Moved(k)) continue;
        float3 oc = PlateHomeCentre(k) + gMove[k].xyz - o;
        float r = gAxis[k].w + Thick;
        float along = dot(oc, dir);
        if (along > 0.0 && dot(oc, oc) - along * along < r * r)
            return true;
    }
    [loop] for (uint c = 0; c < ChipCount; c++)
    {
        float3 oc = gChip[c].xyz - o;
        float along = dot(oc, dir);
        if (along > 0.0 && dot(oc, oc) - along * along < gChip[c].w * gChip[c].w)
            return true;
    }
    return false;
}

// ── Surface ────────────────────────────────────────

// Cracks across the old surface (0..1) at a direction on the moon: thick and close round the break, a few reaching out
// across the face. pixel: a texel of the picture in moon radii, so no crack is drawn thinner than it can show.
float BrokenCracks(float3 dir, float pixel)
{
    if (CracksAmount <= 0.0)
        return 0;
    float away = acos(clamp(dot(dir, BreakDir), -1.0, 1.0)) - BreakCap;     // < 0 inside the break
    float near = saturate(1.0 - away / (0.6 + 0.8 * CracksAmount));
    if (near <= 0)
        return 0;
    float3 warp = dir + 0.14 * (float3(BNoise(dir * 4.0), BNoise(dir * 4.0 + 13.1), BNoise(dir * 4.0 + 27.3)) - 0.5)
                      + 0.04 * (float3(BNoise(dir * 13.0), BNoise(dir * 13.0 + 5.1), BNoise(dir * 13.0 + 9.3)) - 0.5);
    float wobble = BNoise(dir * 3.0 + 7.7) - 0.5;

    // Big cracks: the walls of a coarse cell pattern, kept near the break and a few further out
    float big = WorleyEdge(warp * 3.2);
    float bigOn = smoothstep(0.45, 0.75, near + wobble * 0.6);
    float bigW = (0.010 + 0.045 * near * near) * saturate(CracksAmount);
    float c1 = (1.0 - smoothstep(0.0, max(bigW, pixel * 3.2), big)) * bigOn * saturate(bigW / (pixel * 3.2));

    // Fine cracks, only right round the break
    float fine = WorleyEdge(warp * 9.0 + 3.7);
    float fineOn = smoothstep(0.75, 0.95, near + wobble * 0.3);
    float fineW = 0.03 * near * saturate(CracksAmount);
    float c2 = (1.0 - smoothstep(0.0, max(fineW, pixel * 9.0), fine)) * fineOn * saturate(fineW / (pixel * 9.0));
    return saturate(max(c1, c2 * 0.8));
}

// Dust and grit thrown out of the break along the blow, flat on the disc: rgb is the light it sends (premultiplied),
// a how much it hides of what lies behind. pixel: a texel in moon radii.
float4 BrokenDust(float2 p, float pixel)
{
    float2 b = BreakDir.xy * 0.92;                  // the break as seen
    float2 a = normalize(BlowDir.xy + float2(1e-5, 0));
    float2 d = p - b;
    float s = dot(d, a), t = dot(d, float2(-a.y, a.x));
    float reach = smoothstep(DustReach, DustReach * 0.6, s);
    float w = 0.30 + 0.40 * max(s, 0.0);
    float plume = exp(-(t * t) / (w * w)) * smoothstep(-0.6, 0.05, s) * exp(-max(s, 0.0) / 1.6);
    plume = max(plume, exp(-dot(d, d) / 0.16));     // a burst round the break itself
    plume *= reach;
    float4 dust = 0;
    if (plume > 0.01 && DustAmount > 0.0)
    {
        float3 x = float3(p * 1.8, 0.0);
        float2 warp = float2(BFbm(x + 3.1), BFbm(x + 7.4)) - 0.5;
        float billow = BFbm(float3((p + warp * 0.7) * 2.6, 1.7));
        float density = plume * smoothstep(0.25, 0.85, billow) * 3.0 * DustAmount;
        dust.a = min(1.0 - exp(-density), 0.9);
        dust.rgb = DustColor * DustLight * dust.a;
    }

    // Grit: a speck at most per cell, more where the plume is thick, scattered wider than it
    float wg = 0.45 + 0.55 * max(s, 0.0);
    float grit = exp(-(t * t) / (wg * wg)) * smoothstep(-0.4, 0.1, s) * exp(-max(s, 0.0) / 1.4) * reach * GritAmount;
    if (grit > 0.01)
    {
        const float cellSize = 0.045;
        float2 g = float2(s, t) / cellSize;
        float2 cell0 = floor(g);
        float4 specks = 0;
        [unroll] for (int i = -1; i <= 1; i++)
        [unroll] for (int j = -1; j <= 1; j++)
        {
            float2 cell = cell0 + float2(i, j);
            float3 key = float3(cell, 19.3);
            if (BHash(key) > grit * 0.9)
                continue;
            float2 at = (cell + 0.15 + 0.7 * float2(BHash(key + 3.7), BHash(key + 9.1))) * cellSize;
            float h = BHash(key + 5.3);
            float radius = 0.002 + 0.022 * pow(h, 5.0);
            float dist = length(float2(s, t) - at);
            float cov = saturate((radius - dist) / pixel + 0.5) * saturate(radius * radius / (pixel * pixel) * 4.0);
            float3 rgb = InnerColor * DustLight * (0.1 + 1.1 * pow(BHash(key + 1.3), 2.0));
            specks.rgb = specks.rgb * (1.0 - cov) + rgb * cov;
            specks.a = 1.0 - (1.0 - specks.a) * (1.0 - cov);
        }
        dust.rgb = dust.rgb * (1.0 - specks.a) + specks.rgb;
        dust.a = 1.0 - (1.0 - dust.a) * (1.0 - specks.a);
    }
    return dust;
}

// ── One texel ──────────────────────────────────────

float4 BrokenMoonAt(float2 p, float pixel)
{
    float4 result = 0;

    // The stretch of the ray (along +z) that can meet anything: the moon's ball, the moved plates' and the chips' balls
    float pad = 0.02 + pixel;
    float z0 = 1e9, z1 = -1e9;
    if (dot(p, p) < (1.0 + pad) * (1.0 + pad)) { z0 = -1.0 - pad; z1 = 1.0 + pad; }
    [loop] for (uint k = 0; k < CapCount; k++)
    {
        if (!Moved(k)) continue;
        float3 c = PlateHomeCentre(k) + gMove[k].xyz;
        float2 dxy = p - c.xy;
        float rr = gAxis[k].w + Thick + pad;
        if (dot(dxy, dxy) < rr * rr)
        {
            float hz = sqrt(rr * rr - dot(dxy, dxy));
            z0 = min(z0, c.z - hz);
            z1 = max(z1, c.z + hz);
        }
    }
    [loop] for (uint ci = 0; ci < ChipCount; ci++)
    {
        float2 dxy = p - gChip[ci].xy;
        float rr = gChip[ci].w + pad;
        if (dot(dxy, dxy) < rr * rr)
        {
            float hz = sqrt(rr * rr - dot(dxy, dxy));
            z0 = min(z0, gChip[ci].z - hz);
            z1 = max(z1, gChip[ci].z + hz);
        }
    }

    if (z0 <= z1)
    {
        float3 q = float3(p, z0);
        float minF = 1e9;
        float3 qMin = q;
        bool hit = false;
        [loop] for (int i = 0; i < 128; i++)
        {
            float f = BrokenSdf(q).x;
            if (f < minF) { minF = f; qMin = q; }
            if (f < 0.0005) { hit = true; break; }
            q.z += max(f * 0.8, pixel * 0.2);
            if (q.z > z1) break;
        }
        float cover = hit ? 1.0 : saturate(1.0 - minF / pixel);
        if (cover > 0)
        {
            if (!hit)
                q = qMin;
            uint piece = (uint)BrokenSdf(q).y;
            float e = 0.0012;
            float3 g = float3(BrokenSdf(q + float3(e, 0, 0)).x - BrokenSdf(q - float3(e, 0, 0)).x,
                              BrokenSdf(q + float3(0, e, 0)).x - BrokenSdf(q - float3(0, e, 0)).x,
                              BrokenSdf(q + float3(0, 0, e)).x - BrokenSdf(q - float3(0, 0, e)).x);
            float3 gl = normalize(g);

            // Where this point lay before the blow (a moved plate's), and whether it is the old outer surface
            float3 home = IsPlate(piece) ? PlateHome(q, piece - 2) : q;
            bool outer = (piece == 1 || IsPlate(piece)) && length(home) > 1.0 - 0.004;
            float3 n = gl;
            float3 albedo;
            if (outer)
            {
                // The old surface: the sphere's normal where it lay with its relief, turned with its plate
                float3 hn = normalize(home);
                if (NormalTex != 0)
                {
                    Texture2D relief = ResourceDescriptorHeap[NormalTex];
                    float2 uv = float2(home.x, -home.y) * 0.5 + 0.5;
                    float2 t = relief.SampleLevel(linearWrap, saturate(uv), TexMip).rg * 2.0 - 1.0;   // BC5: blue rebuilt
                    float3 east = normalize(float3(1, 0, 0) - hn.x * hn);
                    float3 north = cross(east, hn);
                    hn = normalize(east * t.x + north * t.y + hn * max(sqrt(saturate(1 - dot(t, t))), 0.05));
                }
                n = IsPlate(piece) ? RotateAxis(hn, gAxis[piece - 2].xyz, gMove[piece - 2].w) : hn;

                // Relative albedo: the texture against a mean of 0.5 (a plain disc is 1)
                albedo = 1.0;
                if (SurfaceTex != 0)
                {
                    Texture2D surface = ResourceDescriptorHeap[SurfaceTex];
                    float2 uv = float2(home.x, -home.y) * 0.5 + 0.5;
                    albedo = surface.SampleLevel(linearWrap, saturate(uv), TexMip).rgb * 2.0;
                }
                albedo *= 1.0 - 0.9 * BrokenCracks(normalize(home), pixel);
            }
            else
                albedo = (piece == 0 ? CoreColor : InnerColor) * (0.7 + 0.6 * BNoise(home * 19.0));

            // Shadows the plates and the body cast on each other (soft, from the sun's side); none to look for where the
            // sun does not reach. Clear of everything the march strides (the bounds keep the distance honest).
            float lit = saturate(dot(n, SunL));
            float shade = 1.0;
            float t = 0.01;
            bool shadowed = lit > 0.0 && (piece != 1 || ShadowPossible(q, SunL));
            [loop] for (int s = 0; s < 32 && shadowed; s++)
            {
                float h = BrokenSdf(q + gl * 0.003 + SunL * t).x;
                shade = min(shade, saturate(10.0 * h / t));
                if (shade < 0.01) break;
                t += clamp(h * 0.8, 0.005, 1.0);
                if (t > 8.0) break;
            }
            lit *= shade;
            result = float4(albedo * (lit + Earthshine) * cover, cover);
        }
    }

    float4 dust = BrokenDust(p, pixel);
    result.rgb = result.rgb * (1.0 - dust.a) + dust.rgb;
    result.a = 1.0 - (1.0 - result.a) * (1.0 - dust.a);

    // Nothing should reach the rectangle's edges; if something does, it fades out there rather than ending in a line
    float2 inRect = min(p - Rect.xy, Rect.xy + Rect.zw - p);
    result *= saturate(min(inRect.x, inRect.y) / (pixel * 4.0));
    return result;
}

[numthreads(8, 8, 1)]
void CSBakeMoon(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    StructuredBuffer<float4> data = ResourceDescriptorHeap[DataIdx];
    LoadHeader(data);
    for (uint i = gi; i < SeedCount; i += 64)
    {
        gSeed[i] = data[HEADER + 3 * i];
        gMove[i] = data[HEADER + 3 * i + 1];
        gAxis[i] = data[HEADER + 3 * i + 2];
    }
    for (uint c = gi; c < ChipCount; c += 64)
        gChip[c] = data[HEADER + 3 * SeedCount + c];
    GroupMemoryBarrierWithGroupSync();

    uint2 px = uint2(id.x, RowStart + id.y);
    if (px.x >= TargetWidth || px.y >= TargetHeight)
        return;

    // Texel centre on the disc (row 0 at the top)
    float pixel = Rect.z / TargetWidth;
    float2 p = float2(Rect.x + (px.x + 0.5) * pixel, Rect.y + Rect.w - (px.y + 0.5) * (Rect.w / TargetHeight));

    RWTexture2D<float4> target = ResourceDescriptorHeap[TargetUAVIdx];
    target[px] = BrokenMoonAt(p, pixel);
}
