struct PushConstantsData
{
    uint4 indices[8]; // 32 uints tightly packed as 8 vectors of 4
};

#define GET_INDEX(i) PushConstants.indices[i/4][i%4]

ConstantBuffer<PushConstantsData> PushConstants : register(b3);

cbuffer SceneConstants : register(b0)
{
    float Time;
    row_major float4x4 View;
    row_major float4x4 Projection;
    row_major float4x4 ViewProjection;
    row_major float4x4 ViewInverse;
    row_major float4x4 CameraInverse; // ViewProjection inverse for depth reconstruction
}

// Material data for bindless texture lookup via Material ID indirection
struct MaterialData
{
    uint AlbedoIdx;
    uint NormalIdx;
    uint RoughnessIdx;
    uint MetallicIdx;
    uint EmissiveIdx;
    uint AOIdx;
    uint2 Padding;
};

// Materials buffer index - slot 14 in push constants (bindless)
// Slots 2-13 are used by mesh rendering, so Materials uses 14 to avoid collision
#define MaterialsIdx GET_INDEX(14)

// Global transform buffer for GPU-driven rendering - slot 15
// All entity transforms are stored here, indexed by TransformSlot
#define GlobalTransformBufferIdx GET_INDEX(15)

// Helper to get material from instance's MaterialID
// Uses bindless access via ResourceDescriptorHeap
inline MaterialData GetMaterial(uint id)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[MaterialsIdx];
    return materials[id];
}
#define GET_MATERIAL(id) GetMaterial(id)

float3 FOG(float3 color, float depth)
{
    float3 fogColor = float3(0.5f, 0.6f, 0.7f);
    //float fog = 1 - saturate((4000 - depth) / (4000 - 300)); // linear
    float density = 0.001f;
	//fog = 1 - 1 / pow(2, linDepth * density); // exponential
    float fog = 1 - 1 / pow(2, pow(depth * density, 2)); // exponential squared
    fog = depth > 10000 ? 0 : fog;
	
    return lerp(color, fogColor, fog);
}

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
    float width, height, slices;
    tex.GetDimensions(width, height, slices);
    float texelSize = 1.0f / width;
	
    // Slope-scaled bias: surfaces at grazing angles need more bias
    float cosTheta = saturate(dot(normal, -lightDir));
    float slopeBias = 0.05f * sqrt(1.0f - cosTheta * cosTheta) / max(cosTheta, 0.05f);
	
    // World-space bias scaled by zScale (projection._33) to get NDC bias
    float bias = (0.05f + slopeBias) * zScale;
    bias = min(bias, 0.01f);
    
    float biasedDepth = depth - bias;
    
    // Per-pixel rotation from interleaved gradient noise
    float rotation = InterleavedGradientNoise(screenPos) * 6.2831853f; // 0..2π
    float radius = SHADOW_DISK_RADIUS * texelSize;
    
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
	float4 a = n1 >= 0 ? -1 : 1;
	float4 b = n1 >= 0 ? 1 : 0;
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
