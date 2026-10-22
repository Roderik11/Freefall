// Screen-space reflections (Graphics/ScreenSpaceReflections.cs, WEATHER_PLAN.md: wet streets and puddles must mirror the
// houses and the lanterns, not only the sky). At half resolution, after the lighting, before the composition, which
// lays the result over the sky probe's reflection by its confidence.
//
// For every smooth enough surface (roughness under SSR_ROUGH_END): from the surface (the hardware depth's place, the
// G-buffer normal) along the mirror direction, steps that grow with the distance, each one projected to the screen and
// compared with the linear depth there; a hit within the step's thickness is refined by halving and lit as the
// composition would light that pixel (its direct light, which holds the lanterns', the sky's ambient, emissive glass,
// the air in front of it). Misses (the ray leaves the screen, goes behind the camera, or reaches the sky) give
// confidence 0 and the composition keeps the sky probe. One sharp ray: rougher surfaces than SSR_ROUGH_END keep the
// probe's blurred reflection, between SSR_ROUGH_START and the end the two blend.
// Output RGBA16F: rgb the reflected light × the confidence (premultiplied, so the composition's bilinear upsampling
// blends right at the edges), a the confidence.

#pragma kernel CSReflect

cbuffer PushConstants : register(b3)
{
    uint AlbedoTexIdx;
    uint NormalTexIdx;
    uint DataTexIdx;
    uint LightTexIdx;           // direct light (rgb) and the sky the surface sees (a)
    uint DepthTexIdx;           // the hardware depth (reverse Z, 0 = sky)
    uint LinearDepthTexIdx;     // the linear view depth (0 = sky): what rays are compared with
    uint OutputUAVIdx;
    uint OutWidth;              // the half-size target
    uint OutHeight;
    uint FullWidth;             // the G-buffer's size
    uint FullHeight;
};

#include "common.fx"
#include "sky_common.fx"

#define SSR_ROUGH_START 0.25    // fully the ray's reflection under this roughness
#define SSR_ROUGH_END 0.45      // the sky probe's alone over this
#define SSR_STEPS 48
#define SSR_FIRST_STEP 0.06     // m, growing by SSR_STEP_GROWTH a step (48 steps reach ~140 m)
#define SSR_STEP_GROWTH 1.12
#define SSR_REFINE 5

float4 LoadTex(uint idx, int2 px) { Texture2D t = ResourceDescriptorHeap[idx]; return t.Load(int3(px, 0)); }

// The light a pixel sends toward the camera, as the composition makes it (without its own reflection)
float3 HitRadiance(int2 px)
{
    float4 albedo = LoadTex(AlbedoTexIdx, px);
    float4 light = LoadTex(LightTexIdx, px);
    float4 data = LoadTex(DataTexIdx, px);
    float3 n = normalize(LoadTex(NormalTexIdx, px).xyz);
    float ao = data.b;
    float3 ambient = SkyAmbient(n) * ao * saturate(light.a);
    float3 lit = ambient * albedo.rgb * (1.0 - saturate(data.g)) + light.rgb;
    float isLit = step(0.1, data.a);
    float3 colour = lerp(albedo.rgb, lerp(lit, albedo.rgb, albedo.a), isLit);
    Texture2D<float> linearDepth = ResourceDescriptorHeap[LinearDepthTexIdx];
    return ApplyAerialPerspective(colour, float2(px) + 0.5, linearDepth.Load(int3(px, 0)), saturate(light.a));
}

// The screen position (full-size pixels) and view depth of a camera-relative place; false behind the camera
bool Project(float3 p, out float2 px, out float viewZ)
{
    float4 clip = mul(float4(p, 1.0), CameraRelativeVP);
    viewZ = clip.w;
    px = 0;
    if (clip.w <= 0.05)
        return false;
    float2 ndc = clip.xy / clip.w;
    px = (ndc * float2(0.5, -0.5) + 0.5) * float2(FullWidth, FullHeight);
    return true;
}

[numthreads(8, 8, 1)]
void CSReflect(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= OutWidth || id.y >= OutHeight)
        return;
    RWTexture2D<float4> output = ResourceDescriptorHeap[OutputUAVIdx];
    int2 px = int2(min(id.xy * 2u, uint2(FullWidth, FullHeight) - 1u));

    Texture2D<float> depthTex = ResourceDescriptorHeap[DepthTexIdx];
    float depth = depthTex.Load(int3(px, 0));
    float4 data = LoadTex(DataTexIdx, px);
    float roughness = data.r;
    if (depth <= 0.0 || data.a < 0.1 || roughness >= SSR_ROUGH_END)
    {
        output[id.xy] = 0;
        return;
    }

    float2 uv = (float2(px) + 0.5) / float2(FullWidth, FullHeight);
    float3 rel = posFromDepth(uv, depth, CameraInverse).xyz;
    float3 n = normalize(LoadTex(NormalTexIdx, px).xyz);
    float3 v = normalize(rel);
    float3 r = reflect(v, n);
    if (dot(r, n) <= 0.0)
    {
        output[id.xy] = 0;
        return;
    }

    // From just off the surface (its own depth must not stop the ray), the steps' start dithered per pixel so their
    // bands do not line up
    Texture2D<float> linearDepth = ResourceDescriptorHeap[LinearDepthTexIdx];
    float dither = frac(52.9829189 * frac(dot(float2(px), float2(0.06711056, 0.00583715))));
    float3 origin = rel + n * (0.01 + 0.002 * length(rel));
    float stepLen = SSR_FIRST_STEP * (1.0 + 0.002 * length(rel)) * (0.5 + dither);
    float t = stepLen, lastT = 0.0;
    bool hit = false;
    float2 hitPx = 0;
    [loop]
    for (uint i = 0; i < SSR_STEPS; i++)
    {
        float2 sp;
        float rayZ;
        if (!Project(origin + r * t, sp, rayZ) || any(sp < 0.0) || any(sp >= float2(FullWidth, FullHeight)))
            break;
        float sceneZ = linearDepth.Load(int3(int2(sp), 0));
        if (sceneZ > 0.0 && rayZ > sceneZ)
        {
            // In front of the surface by less than a step's thickness: a hit (else the ray passed behind something)
            if (rayZ - sceneZ < max(0.2, stepLen * 2.0))
            {
                // Halve between the last step in front and this one
                float lo = lastT, hi = t;
                [unroll]
                for (uint k = 0; k < SSR_REFINE; k++)
                {
                    float mid = 0.5 * (lo + hi);
                    float2 mp;
                    float midZ;
                    Project(origin + r * mid, mp, midZ);
                    float midScene = linearDepth.Load(int3(clamp(int2(mp), 0, int2(FullWidth, FullHeight) - 1), 0));
                    if (midScene > 0.0 && midZ > midScene) hi = mid; else lo = mid;
                }
                float hz;
                Project(origin + r * hi, hitPx, hz);
                hit = true;
                t = hi;
                break;
            }
        }
        lastT = t;
        stepLen *= SSR_STEP_GROWTH;
        t += stepLen;
    }
    if (!hit)
    {
        output[id.xy] = 0;
        return;
    }
    int2 hp = clamp(int2(hitPx), 0, int2(FullWidth, FullHeight) - 1);

    // A hit on a face turned away from the ray is the back of something: nothing true to show
    float3 hitN = normalize(LoadTex(NormalTexIdx, hp).xyz);
    float facing = saturate(-dot(hitN, r) * 4.0);
    // Faded toward the screen's edges (the ray would have found more past them), with roughness
    float2 huv = (float2(hp) + 0.5) / float2(FullWidth, FullHeight);
    float edge = saturate(min(min(huv.x, 1.0 - huv.x), min(huv.y, 1.0 - huv.y)) / 0.08);
    float rough = 1.0 - smoothstep(SSR_ROUGH_START, SSR_ROUGH_END, roughness);
    float confidence = facing * edge * rough;
    output[id.xy] = float4(HitRadiance(hp) * confidence, confidence);
}
