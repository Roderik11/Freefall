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
    float LegacyLightScale;            // light in the engine's older units (point lights, emissive, fire, VFX) × this = pre-exposed; never above 1 (Atmosphere.LegacyLightScale)
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
    float CloudShadowStrength;         // 0 = no cloud shadows, 1 = dense cloud blocks the sun completely
    uint CloudNoiseLUTIdx;             // bindless index of the 3D cloud noise LUT (0 = none)
    float CloudShadowScale;            // shadow pattern is this many times smaller than the visible clouds (1 = matching, km-sized)

    // ── Pick hole (set per view by DeferredRenderer from RequestPick's seeThrough) ──
    float2 PickPixel;                  // centre of the pixel the pick reads this frame
    float2 _pickPad;
    uint4 PickThroughBones;            // bone buffers (InstanceDescriptor.BoneBufferIdx) of the one character that leaves that pixel open; 0 = unused

    // ── Wind (Graphics.Wind: the environment preset's, else a light breeze) — sways grass, flowers, leaves ──
    float3 WindDirection;              // horizontal unit vector the wind blows towards; zero in a calm
    float WindStrength;                // m/s
    float WindPhase;                   // sway and flutter: integral of their speed, so a change of strength never jumps
    float2 WindOffset;                 // how far the air has moved (m), wrapped every 200 m: scrolls the gust pattern
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

// ── Wind (grass cards, blades, flowers, trees) ──
// Every oscillation runs on WindPhase / WindOffset. Never Time x a wind-dependent frequency (it jumps by tens of
// radians while the wind changes and scrubs the animation), never a wave over dot(position, WindDirection) (far from
// the origin it races through hundreds of radians whenever a transition turns the wind).

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
float GustPattern(float2 p)
{
    float2 q = p - WindOffset;
    float fronts = PeriodicNoise(q / 20.0, 10.0);
    float eddies = PeriodicNoise(q / 8.0 + float2(WindPhase * 0.11, -WindPhase * 0.07), 25.0);
    return smoothstep(0.2, 0.8, fronts * 0.65 + eddies * 0.35);
}

// How hard the wind pushes at a place, gusts and a little springiness: where a front is arriving the push runs ahead
// of it, where one has passed it swings back below the base. The air that passed here 0.6 s ago is at p + wind x 0.6 s.
float WindPush(float2 p)
{
    float gust = GustPattern(p);
    float before = GustPattern(p + WindDirection.xz * WindStrength * 0.6);
    return clamp(gust + 0.7 * (gust - before), -0.2, 1.4);
}

// How a grass blade, card or flower stem sways at height fraction t of a part 'height' tall rooted at 'root': a lean
// downwind with the push (WindPush) and a flutter across it on two out-of-step rhythms, each part at its own rate,
// harder in the gusts. 'phase' (0..2π) keeps neighbours apart. Same in the shadow passes.
float3 GrassWind(float3 root, float t, float height, float phase)
{
    float s = saturate(WindStrength / 12.0);                                    // calm .. gale
    float2 dir = WindDirection.xz;
    float2 across = float2(-dir.y, dir.x);
    float push = WindPush(root.xz);
    float lean = s * (0.25 + 0.75 * push) * 0.6;
    float rate = 0.75 + 0.5 * frac(phase * 7.13);
    float wave = sin(WindPhase * 4.0 * rate + phase) * 0.65 + sin(WindPhase * 6.7 * rate + phase * 2.3) * 0.35;
    float flutter = wave * (0.02 + 0.05 * s) * (0.3 + 0.7 * saturate(push));
    float calm = saturate(1.0 - WindStrength / 0.5);                           // no direction to flutter across
    float2 offset = dir * lean + across * flutter * (1.0 - calm) + float2(flutter, flutter * 0.5) * calm;
    return float3(offset.x, -0.3 * lean * lean, offset.y) * (t * t * height);   // bending over costs a little height
}

// A whole tree bending about its base 'origin' (the world matrix' translation) — trunks, far crossmesh LODs and
// foliage alike, so a crown rides with its trunk. The bend grows with the square of the height above the base
// (15 m up: 1, capped at 4); downwind with the push at the tree, plus a slow swing on two rhythms with a per-tree
// phase. A breeze moves a 15 m point ~0.15 m, a gale ~0.9 m. Bending keeps the trunk's length (it drops a little).
float3 TreeSway(float3 worldPos, float3 origin)
{
    float h = max(0.0, worldPos.y - origin.y);
    float bend = min((h / 15.0) * (h / 15.0), 4.0);
    float s = saturate(WindStrength / 12.0);
    float treePhase = MacroHash(origin.xz) * 6.2831853;
    float push = WindPush(origin.xz);
    float swing = sin(WindPhase * 0.9 + treePhase) * 0.6 + sin(WindPhase * 1.53 + treePhase * 1.7) * 0.4;
    float calm = saturate(1.0 - WindStrength / 0.5);
    float2 dir = WindDirection.xz;
    float2 across = float2(-dir.y, dir.x);
    float2 d = (dir * (s * (0.3 + 0.5 * push)) + (dir * 0.6 + across * 0.4) * swing * (0.05 + 0.25 * s)) * (1.0 - calm)
             + float2(swing, swing * 0.5) * 0.03 * calm;
    float3 offset = float3(d.x, 0, d.y) * bend;
    offset.y = -0.5 * dot(offset.xz, offset.xz) / max(h, 1.0);
    return offset;
}
// Material data for bindless texture lookup via Material ID indirection
// 80 bytes — matches C# MaterialData layout exactly
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
// and saved under the .mat's Textures. Read it with: Texture2D noise = ResourceDescriptorHeap[S.Noise];
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
        view = entry >> 30;
        instance = entry & 0x3FFFFFFFu;
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
    uint Reserved3;
    float4 LocalBounds;        // center.xyz + radius.w
    uint TanBufferIdx;
    uint Reserved5;
    uint Reserved6;
    uint Reserved7;
    uint Reserved8;
    uint Reserved9;
};

// Interleaved gradient noise — deterministic per-pixel, no banding
float InterleavedGradientNoise(float2 screenPos)
{
    float3 magic = float3(0.06711056f, 0.00583715f, 52.9829189f);
    return frac(magic.z * frac(dot(screenPos, magic.xy)));
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

#endif // COMMON_FX
