// Included once however often it is named (a project shader may include it and mesh_common.fx, which includes it)
#ifndef COMMON_FX
#define COMMON_FX

cbuffer SceneConstants : register(b0)
{
    float Time;
    row_major float4x4 View;
    row_major float4x4 Projection;
    row_major float4x4 ViewProjection;
    row_major float4x4 ViewInverse;
    row_major float4x4 CameraInverse; // ViewProjection inverse for depth reconstruction
    row_major float4x4 CameraRelativeVP; // Zero-translation VP (inverse of CameraInverse)
    float ShadowTexelSize;             // 1.0 / shadowMapResolution (precomputed on CPU)
    float FogEnabled;                  // 0 or 1: aerial perspective (the atmosphere's haze and ground fog) on surfaces
    float Exposure;                    // pre-exposure: physical light (cd/m²) × Exposure is what the HDR targets store
    float EmissiveExposure;            // light sources seen directly (emissive, flames, VFX glow), nits × this = pre-exposed (Atmosphere.EmissiveExposure)
    float SunLightScale;               // the directional light's intensity (sun, moon: lux × ReferenceExposure / π) × this = pre-exposed
    float3 CamPos;                     // camera world position
    uint InstanceDataIdx;              // bindless raw buffer of per-object effect data (InstanceData), 128 bytes a slot at InstanceDescriptor.CustomDataIdx
    float2 ViewportSize;               // pixels of the target being drawn (aerial perspective lookups by SV_Position)
    uint AtmosphereIdx;                // bindless StructuredBuffer<AtmosphereParams> (atmosphere_common.hlsli); 0 = not made yet
    float CloudAlbedo;                 // share of the light a cloud sends back (SkyboxRenderer)

    // ── Cloud layer (set by SkyboxRenderer) — shared by the sky dome and the cloud shadows ──
    float CloudCoverage;               // 0-1
    float CloudTime;                   // integral of CloudSpeed over time (drift + evolution)
    float CloudAltitude;               // cloud layer height above the camera in world units
    float CloudAltitudeFrom;           // altitude of the previous preset while a transition runs
    float CloudAltitudeBlend;          // 0 = show CloudAltitudeFrom, 1 = show CloudAltitude (cross-fade weight)
    uint CloudNoiseLUTIdx;             // bindless index of the 3D cloud noise LUT (0 = none)

    // ── Pick hole (set per view by DeferredRenderer from RequestPick's seeThrough) ──
    float2 PickPixel;                  // centre of the pixel the pick reads this frame
    float2 _pickPad;
    uint4 PickThroughBones;            // bone buffers (InstanceDescriptor.BoneBufferIdx) of the one character that leaves that pixel open; 0 = unused

    // ── Wind (Graphics.Wind: the environment preset's, else a light breeze) — sways grass, flowers, leaves ──
    float3 WindDirection;              // horizontal unit vector the wind blows towards; zero in a calm
    float WindStrength;                // m/s
    float WindPhase;                   // sway and flutter: integral of their speed, so a change of strength never jumps
    float2 WindOffset;                 // how far the air has moved (m), wrapped every 200 m: scrolls the gust pattern

    // ── Near fog volume (Graphics.VolumetricFog, set per view by DeferredRenderer) — sky_common.fx AtmosphereFog ──
    uint FogVolumeIdx;                 // bindless Texture3D: the air's light (rgb × Exposure) and transmittance (a) up to each slice; 0 = off
    float FogVolumeRange;              // metres of view depth it covers; the aerial perspective volume takes over past it
    uint AerialPerspectiveIdx;         // this view's aerial perspective volume (Graphics.AerialPerspectiveVolume, sized to it); 0 = AtmosphereParams.AerialPerspective

    // ── Volumetric clouds (Graphics.Clouds, cloud_common.hlsli) ──
    uint CloudParamsIdx;               // StructuredBuffer<CloudParams>: the cloud field this frame, always a real buffer (cloud_common.hlsli LoadClouds)
    uint CloudTargetIdx;               // this view's clouds (Graphics.CloudView, half size): rgb their light × Exposure, a what they let through; 0 = none
    uint CloudDepthIdx;                // the same view's cloud depth (R16F, km along the ray to where their light comes from); 0 = none
    uint CloudSceneDepthIdx;           // the scene's view depth (m, 0 = sky) each of its texels was marched to: sky_common.fx CloudsAt
    uint CloudsOn;                     // 1: the volumetric clouds this frame; 0: the 2D layer (CloudParamsIdx then holds an empty frame)

    // ── Light clusters (Graphics.LightClusters, set per view by DeferredRenderer) — light_clusters.hlsli ──
    uint LightClustersIdx;             // ByteAddressBuffer: header, z-bins and tile masks of this view's point lights; 0 = none
    uint ClusterLightsIdx;             // StructuredBuffer<PointLightData>: the visible lights in view-depth order

    uint BlueNoiseIdx;                 // Texture2D<float> 64 × 64 blue noise, tiled by pixel (InternalAssets.BlueNoise); 0 = none

    // ── Order-independent transparency (Graphics.OrderIndependentTransparency, set per view by DeferredRenderer) — oit.hlsli ──
    uint OITZerothIdx;                 // Texture2D<float>: Σ absorbance from the moments pass; 0 = no moments this view
    uint OITMomentsIdx;                // Texture2D<float4>: Σ absorbance × (z, z², z³, z⁴) of the warped depth
    float OITMomentsFrom;              // pixel columns from here ...
    float OITMomentsTo;                // ... up to here are weighed by the moments; elsewhere by WBOIT's depth weight
    float OITLogNear;                  // ln of the near plane: the warped depth's −1
    float OITLogScale;                 // 2 / (ln far − ln near): the warp to [−1, 1]

    // ── The scene's terrain (Graphics.Atmosphere.TerrainIdx) — terrain_pages.hlsli LoadTerrain ──
    uint TerrainIdx;                   // StructuredBuffer<TerrainPagesParams> of this frame; 0 = no terrain

    // ── The far tier (Graphics.FarTier) — FarTierShows / FarTierCasts ──
    float FarTierFrom;                 // metres from the camera where merged far meshes start to take over from instances
    float FarTierBand;                 // over how many metres they dither into each other; 0 = no far tier
    float FarTierCell;                 // metres a side of a cell: the unit a far mesh's vertices and the instances fade by
    uint FarTierProfilesIdx;           // StructuredBuffer<float2> (from, band) per profile (FarTier.ProfilesSrv); 0 = profile 0 only

    // ── Motion vectors (DeferredRenderer, set per view) — MotionClip, PrevMotionClip, MotionVector below ──
    row_major float4x4 MotionVP;       // this frame's view-projection without translation or jitter
    row_major float4x4 PrevMotionVP;   // last frame's, the same way (this frame's when there is no last frame)
    float3 PrevCamPos;                 // where the camera was last frame
    uint PrevTransformsIdx;            // StructuredBuffer<float4x4>: last frame's transforms (TransformBuffer.PreviousSrvIndex); 0 = none
    uint FrameTick;                    // Engine.TickCount: tells a bone pose written this frame (PreviousBones)
    float3 PrevWindDirection;          // last frame's WindDirection, WindStrength, WindPhase, WindOffset (WindBefore)
    float PrevWindStrength;
    float2 PrevWindOffset;
    float PrevWindPhase;
}

// ── Motion vectors: an effect with MotionVectors=true writes its surface's own motion into the G-buffer's SV_Target6 ──

// This frame's and last frame's clip position of a vertex for the motion vectors, without jitter. Relative to each
// frame's camera: world coordinates kilometres out would lose the millimetres the difference is made of.
float4 MotionClip(float3 worldPos)
{
    return mul(float4(worldPos - CamPos, 1.0), MotionVP);
}

float4 PrevMotionClip(float3 prevWorldPos)
{
    return mul(float4(prevWorldPos - PrevCamPos, 1.0), PrevMotionVP);
}

// What the motion target holds (motion_cs.hlsl): last frame's UV minus this frame's, from the two clip positions
// interpolated to the pixel. Behind last frame's camera there was no picture of it: no motion then.
float2 MotionVector(float4 clipNow, float4 clipBefore)
{
    float2 now = clipNow.xy / clipNow.w;
    float2 before = clipBefore.w > 0.0 ? clipBefore.xy / clipBefore.w : now;
    return (before - now) * float2(0.5, -0.5);
}

// Last frame's world matrix of a transform slot, or this frame's where there is none: a slot taken since holds
// identity (a slot's value while free) or zeros there, and so does a slot whose object really sits at the origin
// unrotated, where taking this frame's matrix loses at most one frame of its motion.
row_major float4x4 PrevWorldMatrix(uint slot, row_major float4x4 world)
{
    if (PrevTransformsIdx == 0)
        return world;
    StructuredBuffer<row_major float4x4> previous = ResourceDescriptorHeap[PrevTransformsIdx];
    row_major float4x4 m = previous[slot];
    bool identity = all(m[0] == float4(1, 0, 0, 0)) && all(m[1] == float4(0, 1, 0, 0))
                 && all(m[2] == float4(0, 0, 1, 0)) && all(m[3] == float4(0, 0, 0, 1));
    return m._44 == 0.0 || identity ? world : m;
}

// The bone buffer of a character's pose last frame. A pose writes a stamp after the bones (Animator.Pose, the buffer's
// last element): the tick, and the slot the pose before went to. Posed this tick: that slot. Not posed this tick
// (posed by turns, or still): the same pose showed last frame. 0 (no pose) stays 0.
uint PreviousBones(uint bonesIdx)
{
    if (bonesIdx == 0u)
        return 0u;
    StructuredBuffer<row_major float4x4> bones = ResourceDescriptorHeap[NonUniformResourceIndex(bonesIdx)];
    uint count, stride;
    bones.GetDimensions(count, stride);
    row_major float4x4 stamp = bones[count - 1u];
    uint previous = asuint(stamp._12);
    return asuint(stamp._11) == FrameTick && previous != 0u ? previous : bonesIdx;
}

// ── Large-scale variation of the ground and what grows on it (gputerrain.fx, grass cards, blades, flowers) ──

float MacroHash(float2 p)
{
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float MacroNoise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = MacroHash(i), b = MacroHash(i + float2(1, 0)), c = MacroHash(i + float2(0, 1)), d = MacroHash(i + float2(1, 1));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// Colour multiplier at a place: patches tens to hundreds of metres across that are a little lighter or darker and
// drier (yellower) or lusher. The terrain and the plants on it take the same field, so a dry patch carries on from
// the grass into the bare ground past the decoration radius, and far meadows are no longer one flat colour.
float3 GroundVariation(float2 xz)
{
    float n1 = MacroNoise(xz / 18.0);
    float n2 = MacroNoise(xz / 55.0 + 17.3);
    float n3 = MacroNoise(xz / 240.0 + 41.7);
    // Value noise seldom leaves the middle: summed as is, the field moved only ~±8% (invisible on Eldermoor).
    // Stretched into patches instead, carried mostly by the smaller scales: with 100–300 m patches a whole view
    // sat inside one or two of them and still read as one colour.
    float patches = smoothstep(0.3, 0.7, n1) * 0.35 + smoothstep(0.3, 0.7, n2) * 0.4 + smoothstep(0.3, 0.7, n3) * 0.25;
    float brightness = 1.0 + (patches - 0.5) * 0.6;                                             // ~0.7 .. 1.3
    float dry = smoothstep(0.35, 0.65, n2 * 0.6 + n3 * 0.4);
    // Dry is paler and a little warmer, not a more saturated yellow: the blue channel stays. This tone stacks with the
    // dry grass layer's PlantTint, the crest tone and the sward grain; with each cutting blue (0.72 here) the far
    // meadow came out lime (sRGB ~62,77,17, saturation ~0.8, 2026-10-15)
    return brightness * lerp(float3(0.92, 1.0, 0.97), float3(1.12, 1.04, 0.94), dry);
}

// The colour of one tree's crown, from the tree's origin (its world matrix' translation), for every LOD of it
// (gbuffer_foliage.fx near, gbuffer_crossmesh.fx far) so they hand over without a jump: a brightness and a hue of its
// own (yellower .. bluer), and half of the land's GroundVariation, so groves follow the same dry and lush patches as
// the ground and the grass under them. Every tree of a species had the one colour of its leaf atlas (2026-10-15).
float3 TreeTint(float3 origin)
{
    float h1 = MacroHash(origin.xz * 0.731 + 17.3);
    float h2 = MacroHash(origin.xz * 1.137 - 41.9);
    float brightness = lerp(0.86, 1.14, h1);
    float3 hue = lerp(float3(1.08, 1.0, 0.82), float3(0.92, 1.0, 1.12), h2);
    return brightness * hue * lerp(1.0.xxx, GroundVariation(origin.xz), 0.5);
}

// A far tree's cross-mesh card hands the decal pass the share of the falling snow its pixel takes (decals_cs.hlsl
// SurfaceOpenness) in the G-buffer's vegetation flag, Data.a: the card LODs write 0.5, the cross mesh 0.32..0.48 by its
// exposure. The lighting reads all of it as vegetation (0.3..0.7). The cover map holds no top for vertical cards, so
// its crown depth (weather_cover.hlsli CrownExposure) cannot be had for them
#define FAR_TREE_FLAG_BASE  0.32
#define FAR_TREE_FLAG_RANGE 0.16

float FarTreeFlag(float exposure)
{
    return FAR_TREE_FLAG_BASE + FAR_TREE_FLAG_RANGE * saturate(exposure);
}

// True for a far tree's pixel (a vegetation flag under the card LODs' 0.5), with the exposure it brought
bool FarTreeExposure(float flag, out float exposure)
{
    exposure = saturate((flag - FAR_TREE_FLAG_BASE) / FAR_TREE_FLAG_RANGE);
    return flag > 0.3 && flag < 0.49;
}

// ── Wind (grass cards, blades, flowers, trees) ──
// Every oscillation runs on WindPhase / WindOffset. Never Time x a wind-dependent frequency (it jumps by tens of
// radians while the wind changes and scrubs the animation), never a wave over dot(position, WindDirection) (far from
// the origin it races through hundreds of radians whenever a transition turns the wind).

// The wind of one frame. The functions below take one, so the motion vectors can put a vertex where last frame's wind
// had it (WindBefore); their plain forms take this frame's (WindNow), as every caller did before.
struct WindFrame
{
    float3 Direction;
    float Strength;
    float Phase;
    float2 Offset;
};

WindFrame WindNow()
{
    WindFrame w;
    w.Direction = WindDirection;
    w.Strength = WindStrength;
    w.Phase = WindPhase;
    w.Offset = WindOffset;
    return w;
}

WindFrame WindBefore()
{
    WindFrame w;
    w.Direction = PrevWindDirection;
    w.Strength = PrevWindStrength;
    w.Phase = PrevWindPhase;
    w.Offset = PrevWindOffset;
    return w;
}

// Value noise whose lattice repeats every 'period' cells, for patterns scrolled by WindOffset (which wraps every 200 m)
float PeriodicNoise(float2 p, float period)
{
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float2 i1 = i + 1.0;
    i -= period * floor(i / period);
    i1 -= period * floor(i1 / period);
    float a = MacroHash(i), b = MacroHash(float2(i1.x, i.y)), c = MacroHash(float2(i.x, i1.y)), d = MacroHash(i1);
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// Gusts at a place, 0..1: irregular fronts (~20 m) with smaller eddies (~8 m) in them, scrolled downwind by
// WindOffset and slowly changing shape on WindPhase. The lattices repeat every 200 m, the offset's wrap. (Two sine
// waves did this first: the field rolled in even bands and everything wiggled to one rhythm.)
float GustPatternIn(float2 p, WindFrame w)
{
    float2 q = p - w.Offset;
    float fronts = PeriodicNoise(q / 20.0, 10.0);
    float eddies = PeriodicNoise(q / 8.0 + float2(w.Phase * 0.11, -w.Phase * 0.07), 25.0);
    return smoothstep(0.2, 0.8, fronts * 0.65 + eddies * 0.35);
}

float GustPattern(float2 p)
{
    return GustPatternIn(p, WindNow());
}

// How hard the wind pushes at a place, gusts and a little springiness: where a front is arriving the push runs ahead
// of it, where one has passed it swings back below the base. The air that passed here 0.6 s ago is at p + wind x 0.6 s.
float WindPushIn(float2 p, WindFrame w)
{
    float gust = GustPatternIn(p, w);
    float before = GustPatternIn(p + w.Direction.xz * w.Strength * 0.6, w);
    return clamp(gust + 0.7 * (gust - before), -0.2, 1.4);
}

float WindPush(float2 p)
{
    return WindPushIn(p, WindNow());
}

// GrassWind for parts rooted close together (a clump's blades, a plant's stems), with the push looked up once at
// their common root (two gust noises per lookup, which per vertex were most of the procedural grass' cost): the sway
// at the tip per metre of height, so GrassWind = GrassSway x t² x height. For the motion vectors last frame's sway takes
// this frame's push: the gust pattern drifts a few centimetres a frame across fronts tens of metres wide, while the
// flutter's phase is what moves the tip from frame to frame.
float3 GrassSwayIn(float push, float phase, WindFrame w)
{
    float s = saturate(w.Strength / 12.0);                                      // calm .. gale
    float2 dir = w.Direction.xz;
    float2 across = float2(-dir.y, dir.x);
    float lean = s * (0.25 + 0.75 * push) * 0.6;
    float rate = 0.75 + 0.5 * frac(phase * 7.13);
    float wave = sin(w.Phase * 4.0 * rate + phase) * 0.65 + sin(w.Phase * 6.7 * rate + phase * 2.3) * 0.35;
    float flutter = wave * (0.02 + 0.05 * s) * (0.3 + 0.7 * saturate(push));
    float calm = saturate(1.0 - w.Strength / 0.5);                             // no direction to flutter across
    float2 offset = dir * lean + across * flutter * (1.0 - calm) + float2(flutter, flutter * 0.5) * calm;
    return float3(offset.x, -0.3 * lean * lean, offset.y);   // bending over costs a little height
}

float3 GrassSway(float push, float phase)
{
    return GrassSwayIn(push, phase, WindNow());
}

// How a grass blade, card or flower stem sways at height fraction t of a part 'height' tall rooted at 'root': a lean
// downwind with the push (WindPush) and a flutter across it on two out-of-step rhythms, each part at its own rate,
// harder in the gusts. 'phase' (0..2π) keeps neighbours apart. Same in the shadow passes.
float3 GrassWind(float3 root, float t, float height, float phase)
{
    return GrassSway(WindPush(root.xz), phase) * (t * t * height);
}

// A whole tree bending about its base 'origin' (the world matrix' translation) — trunks, far crossmesh LODs and
// foliage alike, so a crown rides with its trunk. The bend grows with the square of the height above the base
// (15 m up: 1, capped at 4); downwind with the push at the tree, plus a slow swing on two rhythms with a per-tree
// phase. A breeze moves a 15 m point ~0.15 m, a gale ~0.9 m. Bending keeps the trunk's length (it drops a little).
float3 TreeSwayIn(float3 worldPos, float3 origin, float push, WindFrame w)
{
    float h = max(0.0, worldPos.y - origin.y);
    float bend = min((h / 15.0) * (h / 15.0), 4.0);
    float s = saturate(w.Strength / 12.0);
    float treePhase = MacroHash(origin.xz) * 6.2831853;
    float swing = sin(w.Phase * 0.9 + treePhase) * 0.6 + sin(w.Phase * 1.53 + treePhase * 1.7) * 0.4;
    float calm = saturate(1.0 - w.Strength / 0.5);
    float2 dir = w.Direction.xz;
    float2 across = float2(-dir.y, dir.x);
    float2 d = (dir * (s * (0.3 + 0.5 * push)) + (dir * 0.6 + across * 0.4) * swing * (0.05 + 0.25 * s)) * (1.0 - calm)
             + float2(swing, swing * 0.5) * 0.03 * calm;
    float3 offset = float3(d.x, 0, d.y) * bend;
    offset.y = -0.5 * dot(offset.xz, offset.xz) / max(h, 1.0);
    return offset;
}

float3 TreeSway(float3 worldPos, float3 origin)
{
    return TreeSwayIn(worldPos, origin, WindPush(origin.xz), WindNow());
}

// A tree's vertex as last frame's wind had it, for the motion vectors: 'swayed' is this frame's place with this frame's
// TreeSway added, 'rest' the place without it (last frame's transform), 'sway' what was added. The push is this
// frame's (see GrassSwayIn).
float3 TreeSwayedBefore(float3 rest, float3 origin, float push)
{
    return rest + TreeSwayIn(rest, origin, push, WindBefore());
}

// Per-vertex screen motion (last frame's UV minus this frame's) for parts too small for the motion to change across a
// triangle: grass blades, flower stems. Larger surfaces pass both clip positions to the pixel (MotionClip).
float2 MotionOfVertex(float3 worldNow, float3 worldBefore)
{
    return MotionVector(MotionClip(worldNow), PrevMotionClip(worldBefore));
}
// Material data for bindless texture lookup via Material ID indirection
// 80 bytes — matches C# MaterialData layout exactly
// One draw holds every instance of a mesh part, whatever its material: a wave mixes materials, so fetch these
// indices with NonUniformResourceIndex (AMD otherwise uses one lane's descriptor for the whole wave)
struct MaterialData
{
    uint AlbedoIdx;
    uint NormalIdx;
    uint RoughnessIdx;
    uint MetallicIdx;
    uint EmissiveIdx;
    uint AOIdx;
    uint DetailNormalIdx;
    uint DetailMaskIdx;
    float2 DetailTiling;
    float2 DetailOffset;
    float3 EmissiveColor;
    float EmissiveIntensity;
    float3 Tint;                // multiplies the albedo map (Material "Tint", default white)
    uint SettingsBufferIdx;     // the raw buffer of every material's effect settings (Material.SettingsStride bytes at its MaterialID); was padding
};

// A texture in an effect's material settings or instance data: its bindless index. Declared "TextureRef Noise; //= Gray"
// (the default an engine texture: White, Black, Gray, FlatNormal), shown as a texture slot in the material inspector
// and saved under the .mat's Textures. Read it with: Texture2D noise = ResourceDescriptorHeap[NonUniformResourceIndex(S.Noise)];
// (materials and instances differ within one draw, see MaterialData)
typedef uint TextureRef;

// Shadow cascade data — shared by all shadow-related shaders
// Stride must match C# GPUCuller.CascadeData exactly (240 bytes)
struct CascadeData
{
    float4 Planes[6];           // frustum planes (96 bytes)
    row_major float4x4 VP;      // current frame VP (64 bytes)
    row_major float4x4 PrevVP;  // previous frame VP for Hi-Z (64 bytes)
    float4 SplitDistances;      // X=near, Y=far (16 bytes)
};

#include "local_shadows.hlsli"
#include "shadow_cascades.hlsli"

// Shadow vertex shaders serve two passes: Shadow (sun cascades, one slice each) and LocalShadow (point light
// cube faces, a tile each in the local shadow atlas). These decode the pass's expansion entry and project.
void DecodeShadowEntry(uint entry, bool local, out uint view, out uint instance)
{
    if (local)
    {
        view = entry >> LOCAL_SHADOW_VIEW_SHIFT;
        instance = entry & LOCAL_SHADOW_INSTANCE_MASK;
    }
    else
    {
        view = entry >> CASCADE_ENTRY_SHIFT;
        instance = entry & CASCADE_ENTRY_INSTANCE_MASK;
    }
}

// Absolute world position -> the view's clip position and array slice. A cube face is squeezed into its
// atlas tile; the clip distances (taken before the squeeze) cut what lies outside the face, which would
// otherwise spill into the neighbouring tiles.
float4 ShadowClipPosition(float4 worldPos, uint view, bool local, uint viewBufferIdx, out uint slice, out float4 clipDistance)
{
    if (!local)
    {
        StructuredBuffer<CascadeData> cascades = ResourceDescriptorHeap[viewBufferIdx];
        slice = view;
        clipDistance = 1;
        return mul(worldPos, cascades[view].VP);
    }

    StructuredBuffer<LocalShadowView> views = ResourceDescriptorHeap[viewBufferIdx];
    LocalShadowView v = views[view];
    float4 clip = mul(worldPos, v.VP);
    clipDistance = float4(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);

    // Tile rect (UV) -> NDC scale and bias. UV y runs down and NDC y up, so y's bias is mirrored.
    float2 size = v.Tile.xy;
    float2 bias = float2(size.x + 2.0 * v.Tile.z - 1.0, 1.0 - size.y - 2.0 * v.Tile.w);
    clip.xy = clip.xy * size + bias * clip.w;
    slice = v.Page;
    return clip;
}

// Per-instance descriptor (matches C# InstanceDescriptor: 20 bytes)
// Single source of truth — do NOT redeclare in individual shaders.
struct InstanceDescriptor
{
    uint TransformSlot;   // index into GlobalTransformBuffer
    uint MaterialId;      // index into MaterialsBuffer
    uint CustomDataIdx;   // the object's slot of the per-object effect data (InstanceData, SceneConstants.InstanceDataIdx); its effect's defaults slot if it has none
    uint MeshPartIdx;     // meshpart index within the mesh (for GPU picking)
    uint BoneBufferIdx;   // per-Animator bone buffer SRV (0 = static mesh)
};

// Per mesh/part metadata (matches C# MeshPartEntry: 72 bytes = 18 uints)
// Single source of truth — VS looks up buffer indices from MeshRegistry using MeshPartId.
struct MeshPartEntry
{
    uint PosBufferIdx;
    uint NormBufferIdx;
    uint UVBufferIdx;
    uint IndexBufferIdx;
    uint BaseIndex;
    uint VertexCount;
    uint BoneWeightsBufferIdx;
    uint PushNormBufferIdx;     // Mesh.PushNormBufferIndex: what gbuffer.fx's displacement pushes along (0 = the normals)
    float4 LocalBounds;        // center.xyz + radius.w
    uint TanBufferIdx;
    uint IndexBufferAddressLo;  // the mesh's index buffer for indexed indirect draws (cull_instances.hlsl IndexedDrawCommand)
    uint IndexBufferAddressHi;
    uint IndexBufferBytes;
    uint CullBone;              // a skinned part's bone with most of its vertices + 1, 0 = none (cull_instances.hlsl)
    uint Flags;                 // MESH_* bits (MeshRegistry.MeshFlags)
};

// Merged far cards (Mesh.CardOrigins): each vertex's normal holds its tree's origin and its tangent the card's centre,
// both mesh-local, where a tree's own cards take them from its world matrix and its part's bounds (gbuffer_crossmesh.fx)
#define MESH_CARD_ORIGINS 1u

// A merged far mesh of the far tier (Mesh.CellFade): drawn past FarTierFrom, each vertex fading with its cell. Its cell
// is the one its object stands in: a card's from its tree's origin (MESH_CARD_ORIGINS), any other vertex's UV holds the
// world XZ of its cell's centre. Every triangle lies in one cell.
#define MESH_CELL_FADE 2u

// InstanceDescriptor.MeshPartIdx: the part's index in its mesh is the low bits (only the low 8 are read, for picking);
// this bit marks an instance of the far tier's near side (CommandBuffer.AddPersistent farTier), which gives way to the
// merged far meshes past its profile's distance (bits 8-13, FarTierProfileOfDraw)
#define DRAW_FAR_TIER (1u << 31)

// ── The far tier: instances near, merged far meshes past FarTierFrom, dithered into each other over FarTierBand ──
// A draw's profile (Graphics.FarTier profiles) moves both: 0 = FarTierFrom and FarTierBand themselves.

// The far tier profile of an instance (InstanceDescriptor.MeshPartIdx) and of a merged far mesh's part (MeshPartEntry.Flags)
uint FarTierProfileOfDraw(uint meshPartIdx) { return (meshPartIdx >> 8u) & 63u; }
uint FarTierProfileOfPart(uint flags) { return (flags >> 8u) & 63u; }

// A profile's distance and band
float2 FarTierProfile(uint profile)
{
    if (profile == 0u || FarTierProfilesIdx == 0u)
        return float2(FarTierFrom, FarTierBand);
    StructuredBuffer<float2> profiles = ResourceDescriptorHeap[FarTierProfilesIdx];
    return profiles[profile];
}

// The centre of the cell a world point stands in
float2 FarTierCellOf(float3 p)
{
    return (floor(p.xz / FarTierCell) + 0.5) * FarTierCell;
}

// The far side's share of a cell, 0 (instances only) to 1 (the far mesh only), by the camera's level distance to it
float FarTierShare(float2 cellCentre, uint profile)
{
    float2 p = FarTierProfile(profile);
    if (p.y <= 0.0)
        return 0.0;
    return saturate((length(cellCentre - CamPos.xz) - p.x) / p.y);
}

float FarTierShare(float2 cellCentre) { return FarTierShare(cellCentre, 0u); }

// Whether a draw of the far tier shows in this cell (false: collapse the vertex), and the LodFadeClip word it draws
// with: the far side on a share of the pixels, the near side on the rest of the same blue noise, so together they
// cover each pixel once. 0 = every pixel.
bool FarTierShows(float2 cellCentre, bool farSide, uint profile, out uint fade)
{
    uint share = (uint)(FarTierShare(cellCentre, profile) * 127.0 + 0.5);
    fade = 0u;
    if (farSide)
    {
        if (share == 0u) return false;
        if (share < 127u) fade = share;
        return true;
    }
    if (share >= 127u) return false;
    if (share > 0u) fade = 128u | share;
    return true;
}

bool FarTierShows(float2 cellCentre, bool farSide, out uint fade) { return FarTierShows(cellCentre, farSide, 0u, fade); }

// Shadows switch at half way, with no dither (as the LOD crossfade's)
bool FarTierCasts(float2 cellCentre, bool farSide, uint profile)
{
    return (FarTierShare(cellCentre, profile) >= 0.5) == farSide;
}

bool FarTierCasts(float2 cellCentre, bool farSide) { return FarTierCasts(cellCentre, farSide, 0u); }

// The mesh vertex a mesh draw's vertex shader invocation shades. An indexed draw (InstanceBatch, the effect's push
// constant IndexedDraw in slot 17 = 1) hands it over in SV_VertexID: the input assembler read the part's indices, and a
// vertex the neighbouring triangles share is shaded once. A pulled draw counts the corners 0..N-1 and reads the index
// of 'corner' (vertexId, or the corner a mirrored instance swaps it with; the camera draws the parts that have a mirrored
// instance in view pulled, cull_instances.hlsl CSMain). Effect.DrawsIndexed: every vertex shader of the effect must
// read IndexedDraw, or none of its batches draws indexed.
uint MeshVertexIndex(MeshPartEntry part, uint vertexId, uint corner, uint indexedDraw)
{
    if (indexedDraw != 0)
        return vertexId;
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[part.IndexBufferIdx];
    return indices[corner + part.BaseIndex];
}

// Interleaved gradient noise — deterministic per-pixel, no banding
float InterleavedGradientNoise(float2 screenPos)
{
    float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
    return frac(magic.z * frac(dot(screenPos, magic.xy)));
}

#include "lod_fade.hlsli"

// The LOD crossfade's dither (lod_fade.hlsli), from a pixel shader at its SV_Position with the vertex shader's
// DrawEntryFade: the new part draws where the blue noise is under its share, the part faded from everywhere else, so
// together they cover each pixel once. The pattern stays put on the screen.
void LodFadeClip(float2 pixel, uint fade)
{
    if (fade == 0)
        return;
    uint n;
    if (BlueNoiseIdx != 0)
    {
        Texture2D<float> noise = ResourceDescriptorHeap[BlueNoiseIdx];
        n = (uint)(noise.Load(int3(uint2(pixel) & 63u, 0)) * 255.0 + 0.5);
    }
    else
        n = (uint)(InterleavedGradientNoise(pixel) * 255.0);
    bool newPart = n < (fade & 127u) * 2u;
    if (newPart == ((fade & 128u) != 0))
        discard;
}

// Vogel disk: golden-angle spiral produces well-distributed 2D samples
float2 VogelDiskSample(int index, int count, float rotation)
{
    float goldenAngle = 2.4f; // ~137.5° in radians
    float r = sqrt((float(index) + 0.5f) / float(count));
    float theta = float(index) * goldenAngle + rotation;
    return float2(r * cos(theta), r * sin(theta));
}

static const int SHADOW_TAP_COUNT = 8;
static const float SHADOW_DISK_RADIUS = 2.5f; // in texels

float GetShadowFactor(in Texture2DArray tex, in SamplerComparisonState cmpSampler, in float2 uv, float depth, int sliceIndex, float3 normal, float3 lightDir, float zScale, float2 screenPos)
{
    // Precomputed on CPU — avoids per-pixel GetDimensions() and hardcoded resolution
    float texelSize = ShadowTexelSize;
	
    // Slope-scaled bias: surfaces at grazing angles need more bias.
    // NOTE: The hardware rasterizer already applies SlopeScaledDepthBias (3.0),
    // so this shader-side bias should be minimal — just enough for PCF softening.
    float cosTheta = saturate(dot(normal, -lightDir));
    float slopeBias = 0.002f * sqrt(1.0f - cosTheta * cosTheta) / max(cosTheta, 0.05f);
	
    // World-space bias scaled by zScale (projection._33) to get NDC bias
    // Per-cascade minimum ensures larger cascades (lower zScale) still get adequate bias
    float bias = (0.001f + slopeBias) * zScale;
    float minBias = 0.0001f;  // floor: prevents self-shadowing on terrain in wider cascades
    bias = clamp(bias, minBias, 0.005f);
    
    float biasedDepth = depth - bias;
    
    // Per-pixel rotation from interleaved gradient noise
    float rotation = InterleavedGradientNoise(screenPos) * 6.2831853f; // 0..2π
    // Per-cascade filter radius: near cascades use tighter filter to preserve
    // thin shadow caster detail (grass blades), far cascades use wider filter
    // for smoother large shadows.
    float diskRadius = (sliceIndex <= 1) ? 1.2f : SHADOW_DISK_RADIUS;
    float radius = diskRadius * texelSize;
    
    float result = 0.0f;
    [unroll]
    for (int i = 0; i < SHADOW_TAP_COUNT; i++)
    {
        float2 offset = VogelDiskSample(i, SHADOW_TAP_COUNT, rotation) * radius;
        result += tex.SampleCmpLevelZero(cmpSampler, float3(uv + offset, sliceIndex), biasedDepth);
    }
    
    return result / float(SHADOW_TAP_COUNT);
}

float4 posFromDepth(in float2 uv, in float depth, in float4x4 cameraInverse)
{
    float x = uv.x * 2 - 1;
    float y = (1 - uv.y) * 2 - 1;
    float4 wpos = float4(x, y, depth, 1.0f);

    wpos = mul(wpos, cameraInverse);
    wpos /= wpos.w;

    return wpos;
}

float view_depth(in float2 uv, in float depth, in float4x4 projInverse)
{
    float x = uv.x * 2 - 1;
    float y = (1 - uv.y) * 2 - 1;
    float4 wpos = float4(x, y, depth, 1.0f);

    wpos = mul(wpos, projInverse);
    wpos /= wpos.w;

    return wpos.z;
}

float LinearizeDepth(float depth, in float near, in float far)
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));
}

float linearize_depth(in float d, in float zNear, in float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

float lineardepth(in float d, in float4x4 proj)
{
    return proj._43 / (d - proj._33);
}

float3 blend_linear(in float3 n1, in float3 n2)
{
	float3 r = (n1 + n2) * 0.5f;
	return normalize(r);
}

float3 blend_overlay(float4 n1, float4 n2)
{
	n1 = n1 * 4 - 2;
	float4 a = select(n1 >= 0, (float4)-1, (float4)1);
	float4 b = select(n1 >= 0, (float4)1, (float4)0);
	n1 = 2 * a + n1;
	n2 = n2 * a + b;
	float3 r = (n1 * n2 - a).xyz;
	return normalize(r);
}

float3 blend_pd(float4 n1, float4 n2)
{
	n1 = n1 * 2 - 1;
	n2 = n2.xyzz*float4(2, 2, 2, 0) + float4(-1, -1, -1, 0);
	float3 r = n1.xyz*n2.z + n2.xyw*n1.z;
	return normalize(r);
}

float3 blend_whiteout(float4 n1, float4 n2)
{
	n1 = n1 * 2 - 1;
	n2 = n2 * 2 - 1;
	float3 r = float3(n1.xy + n2.xy, n1.z*n2.z);
	return normalize(r);
}

float3 blend_udn(float3 n1, float3 n2)
{
	float3 c = float3(2, 1, 0);
	float3 r;
	r = n2 * c.yyz + n1.xyz;
	r = r * c.xxx - c.xxy;
	return normalize(r);
}

float3 blend_rnm(float4 n1, float4 n2)
{
	float3 t = n1.xyz*float3(2, 2, 2) + float3(-1, -1, 0);
	float3 u = n2.xyz*float3(-2, -2, 2) + float3(1, 1, -1);
	float3 r = t * dot(t, u) - u * t.z;
	return normalize(r);
}

float3 blend_unity(float4 n1, float4 n2)
{
	n1 = n1.xyzz*float4(2, 2, 2, -2) + float4(-1, -1, -1, 1);
	n2 = n2 * 2 - 1;
	float3 r;
	r.x = dot(n1.zxx, n2.xyz);
	r.y = dot(n1.yzy, n2.xyz);
	r.z = dot(n1.xyw, -n2.xyz);
	return normalize(r);
}

float noise(float2 uv)
{
	return 2.0 * frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453) - 1.0;
}

float sum(float3 v) { return v.x + v.y + v.z; }

half3 blend(half4 texture1, float a1, half4 texture2, float a2)
{
	float depth = 0.2;
	float ma = max(texture1.a + a1, texture2.a + a2) - depth;

	float b1 = max(texture1.a + a1 - ma, 0);
	float b2 = max(texture2.a + a2 - ma, 0);

	return (texture1.rgb * b1 + texture2.rgb * b2) / (b1 + b2);
}

float remap(float value, float from1, float to1, float from2, float to2)
{
    return (value - from1) / (to1 - from1) * (to2 - from2) + from2;
}

float map(float value, float min1, float max1, float min2, float max2)
{
	// Convert the current value to a percentage
	// 0% - min1, 100% - max1
	float perc = (value - min1) / (max1 - min1);

	// Do the same operation backwards with min2 and max2
	value = perc * (max2 - min2) + min2;

	return clamp(value, min(min2, max2), max(min2, max2));
}

//Normal Encoding Function
half3 EncodeNormal(half3 n)
{
	return 0.5f * (normalize(n) + 1.0f);
}

//Normal Decoding Function
half3 DecodeNormal(half3 enc)
{
	return enc * 2 - 1;
}

// get normalized device coords
float2 ConvertScreenToProjection(float4 vScreen)
{
	return float2(-vScreen.x / vScreen.w / 2.0f + 0.5f, -vScreen.y / vScreen.w / 2.0f + 0.5f);
}

// get UV coords
half2 ConvertScreenToUV(float4 vScreen)
{
	half2 vScreenCoord = vScreen.xy / vScreen.w;
	vScreenCoord = 0.5h * (half2(vScreenCoord.x, -vScreenCoord.y) + 1);

	return vScreenCoord;
}

// get viewspace position from linear depth
float3 PositionVSFromDepth(float3 vView, float fDepth)
{
	return fDepth * vView.xyz / vView.z;
}

float4 SampleLinear2D(Texture2D tex, SamplerState samp, float2 uv)
{
	return pow(tex.Sample(samp, uv), 2.2f);
}

float4 SampleLinearCube(TextureCube tex, SamplerState samp, float3 uv)
{
	return pow(tex.Sample(samp, uv), 2.2f);
}

float4 SampleLinearCubeLevel(TextureCube tex, SamplerState samp, float3 uv, float level)
{
	return pow(tex.SampleLevel(samp, uv, level), 2.2f);
}

float3 SignedOctEncode(float3 n)
{
    float3 OutN;

    n /= (abs(n.x) + abs(n.y) + abs(n.z));

    OutN.y = n.y * 0.5f + 0.5f;
    OutN.x = n.x * 0.5f + OutN.y;
    OutN.y = n.x * -0.5f + OutN.y;

    OutN.z = saturate(n.z * 10000000000.0f);
    return OutN;
}

float3 SignedOctDecode(float3 n)
{
    float3 OutN;

    OutN.x = (n.x - n.y);
    OutN.y = (n.x + n.y) - 1.0f;
    OutN.z = n.z * 2.0f - 1.0f;
    OutN.z = OutN.z * (1.0f - abs(OutN.x) - abs(OutN.y));
 
    OutN = normalize(OutN);
    return OutN;
}

// A mesh's shape normal (its interpolated vertex normal, without the normal map) in the G-buffer normal's w, which plain
// meshes left at 1 and only vegetation uses (its translucency): octahedral about +y, 8 bits a side as 255 × 255 steps of
// the 16-bit SNorm, so 1 (no shape normal: every other effect) lies outside them. Deep snow follows it on meshes
// (decals_cs.hlsl); the depth's normal it took before is flat per triangle and drew a low-poly rock's facets in the snow
float PackShapeNormal(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 o = n.xz;
    if (n.y < 0.0)
        o = (1.0 - abs(n.zx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.z >= 0.0 ? 1.0 : -1.0);
    uint2 q = (uint2)round(saturate(o * 0.5 + 0.5) * 254.0);
    return ((float)(q.x * 255u + q.y) - 32512.0) / 32767.0;
}

// The shape normal PackShapeNormal wrote; false where none was (w 1). Only for meshes: vegetation keeps its translucency
// there, which would read as a normal
bool UnpackShapeNormal(float w, out float3 n)
{
    int k = (int)round(w * 32767.0) + 32512;
    n = 0;
    if (k < 0 || k > 65024)
        return false;
    float2 o = float2((uint)k / 255u, (uint)k % 255u) / 254.0 * 2.0 - 1.0;
    n = float3(o.x, 1.0 - abs(o.x) - abs(o.y), o.y);
    if (n.y < 0.0)
        n.xz = (1.0 - abs(n.zx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.z >= 0.0 ? 1.0 : -1.0);
    n = normalize(n);
    return true;
}

#endif // COMMON_FX
