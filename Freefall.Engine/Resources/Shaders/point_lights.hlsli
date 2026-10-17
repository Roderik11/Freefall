// Point lights on the GPU (PointLightBuffer.cs): the persistent registry the CPU writes only when a light
// changes, and the visible list the cull builds from it every frame (point_light_cull.hlsl), which the tiled
// lighting (light_point_cs.hlsl) and the shadow selection (local_shadow_select.hlsl) read.
#ifndef POINT_LIGHTS_HLSLI
#define POINT_LIGHTS_HLSLI

#define POINT_LIGHT_ENABLED     1u      // PointLightRecord.Flags
#define POINT_LIGHT_WHEN_DARK   2u      // PointLight.OnlyWhenDark: intensity × PointLightFrame.Lamps
#define MAX_VISIBLE_LIGHTS      4096u   // PointLightBuffer.MaxVisible
#define VISIBLE_LIGHT_OCCLUDED  1u      // PointLightData.Flags: behind the Hi-Z (CSOccludeLights)
#define POINT_LIGHT_NO_SHADOW   0xFFFFFFFFu

// One per PointLight, at its registry slot. Must match C# PointLightBuffer.Record (48 bytes).
struct PointLightRecord
{
    float3 Color;
    float Intensity;        // luminous intensity in candela (PointLight.Lumens / 4π): lux at 1 m
    uint TransformSlot;     // the light entity's world matrix in the global transform buffer
    float Range;
    float Flicker;
    float FlickerSpeed;
    float FlickerPhase;     // per light, so neighbouring lights do not waver in step
    uint Flags;             // POINT_LIGHT_ENABLED
    uint2 _pad;
};

// A light in the camera's view this frame. Must match C# PointLightBuffer.VisibleLight (48 bytes).
struct PointLightData
{
    float3 Color;
    float Intensity;        // candela, flicker applied
    float3 Position;        // camera-relative, steady (flicker changes only the intensity)
    float Range;
    uint ShadowLight;       // index into LocalShadowLight[], POINT_LIGHT_NO_SHADOW
    uint Slot;              // registry slot
    uint Flags;             // VISIBLE_LIGHT_OCCLUDED
    float Score;            // the cull writes luminance × base candela; the shadow selection its rank score
};

// The camera and frame, per renderer, for the light cull and the shadow selection (not SceneConstants: those
// are set for lighting later in the frame, and by whichever renderer ran last). Must match C#
// PointLightBuffer.FrameParams (192 bytes).
struct PointLightFrame
{
    row_major float4x4 CameraRelativeVP;    // camera-relative world -> clip (view without translation)
    float4 Planes[5];                       // camera-relative left, right, bottom, top, near; normals in
    float3 CameraPosition;
    float Time;                             // flicker clock (Time.TotalTime)
    float PixelsPerMetre;                   // on-screen pixels of 1 m seen from 1 m (height / 2 × cot(fovY / 2))
    float FadeStep;                         // shadow fade per frame (Δt / fade time)
    float SunLight;                         // lux on a surface facing the sun (or moon), clouds' average shade included
    uint ShadowBudget;                      // MaxShadowedPointLights
    uint Frame;                             // this renderer's frame count, from 1
    float Ambient;                          // lux from the whole sky on level ground
    float Lamps;                            // Daylight.Lamps: 0 = street lamps out, 1 = lit
    uint _pad;
};

// ── Flicker: value noise over an integer hash, no trig ───────────────────────
// (was PointLight.ApplyFlicker on the CPU: same hash, same constants)
float FlickerHash(int i)
{
    uint state = (uint)i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return float((word >> 22) ^ word) * (1.0 / 4294967295.0);
}

float FlickerNoise(float x)
{
    float fl = floor(x);
    int i = (int)fl;
    float f = x - fl;
    f = f * f * (3.0 - 2.0 * f);
    float a = FlickerHash(i), b = FlickerHash(i + 1);
    return a + (b - a) * f;
}

// This frame's intensity of a flickering light: only the brightness wavers. Until 2026-10-16 the light also
// moved a few centimetres; bars and grilles near the flame (brazier cages) magnified that into shadows
// swinging tens of centimetres across a whole plaza.
void ApplyFlicker(PointLightRecord r, float time, inout float intensity)
{
    float t = time * r.FlickerSpeed + r.FlickerPhase;
    float wave = 0.65 * FlickerNoise(t) + 0.35 * FlickerNoise(t * 2.7 + 11.3);
    intensity *= 1.0 + r.Flicker * (wave * 1.4 - 0.7);
}

#endif
