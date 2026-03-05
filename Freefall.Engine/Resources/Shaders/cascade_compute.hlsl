// GPU-driven cascade matrix computation
// Reads SDSM split distances from GPU buffer and computes all cascade VP matrices.
// Eliminates the CPU readback path for adaptive shadow splits.
//
// Dispatch (1, 1, 1) — single thread group, 4 threads (one per cascade)

// Note: this shader is compiled standalone by Shader.Compile() which has no
// include handler. All types must be self-contained (no #include).

cbuffer PushConstants : register(b3)
{
    uint4 Indices[8];
};

// Push constant slots
#define SplitsBufferIdx    Indices[0].x   // SRV: StructuredBuffer<float> [4 splits]
#define CascadeOutUAVIdx   Indices[0].y   // UAV: RWStructuredBuffer<CascadeData> [4]
#define LightingOutUAVIdx  Indices[0].z   // UAV: RWStructuredBuffer<LightingCascadeData> [4]
#define ShadowMapRes       Indices[0].w   // Shadow map resolution (e.g. 2048)

#define NearPlane          asfloat(Indices[1].x)  // Camera near plane
#define CascadeCount       Indices[1].y            // Number of active cascades (always 4)
#define PrevVPBufferIdx    Indices[1].z            // SRV: previous frame VP matrices [4]
#define SmoothedSplitsIdx  Indices[1].w            // UAV: RWStructuredBuffer<float> [4] smoothed splits

// Camera parameters — passed as matrices in a cbuffer (register b1, slot 2)
cbuffer CascadeParams : register(b1)
{
    row_major float4x4 CameraView;          // Camera view matrix (zero-translation, rotation only)
    row_major float4x4 CameraProjection;    // Camera projection matrix (full, not per-cascade)
    float3 CameraPosition;                  // Absolute world-space camera position
    float _pad0;
    float3 CameraForward;                   // Camera forward direction
    float _pad1;
    float3 CameraUp;                        // Camera up direction
    float _pad2;
    float3 LightForward;                    // Directional light forward direction
    float _pad3;
    float4 CascadeSplits;                   // X,Y,Z,W = split distances for cascades 0-3
};

// Output for culling pass (frustum planes, VP matrices, split distances)
struct CascadeData
{
    float4 Planes[6];           // frustum planes
    row_major float4x4 VP;      // current frame VP
    row_major float4x4 PrevVP;  // previous frame VP (for Hi-Z)
    float4 SplitDistances;      // X=near, Y=far
};

// Output for lighting pass (camera-relative VP + split distances)
struct LightingCascadeData
{
    row_major float4x4 VP;   // Camera-relative light VP
    float4 Cascade;          // X=near, Y=far
};

// ============================================================
// Helper: Build a left-handed look-at matrix
// ============================================================
float4x4 MakeLookAtLH(float3 eye, float3 target, float3 up)
{
    float3 zAxis = normalize(target - eye);
    float3 xAxis = normalize(cross(up, zAxis));
    float3 yAxis = cross(zAxis, xAxis);
    
    float4x4 m;
    m[0] = float4(xAxis.x, yAxis.x, zAxis.x, 0);
    m[1] = float4(xAxis.y, yAxis.y, zAxis.y, 0);
    m[2] = float4(xAxis.z, yAxis.z, zAxis.z, 0);
    m[3] = float4(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1);
    return m;
}

// ============================================================
// Helper: Build a left-handed orthographic projection matrix
// ============================================================
float4x4 MakeOrthoOffCenterLH(float left, float right, float bottom, float top, float zNear, float zFar)
{
    float rml = right - left;
    float tmb = top - bottom;
    float fmn = zFar - zNear;
    
    float4x4 m = (float4x4)0;
    m[0][0] = 2.0 / rml;
    m[1][1] = 2.0 / tmb;
    m[2][2] = 1.0 / fmn;
    m[3][0] = -(right + left) / rml;
    m[3][1] = -(top + bottom) / tmb;
    m[3][2] = -zNear / fmn;
    m[3][3] = 1.0;
    return m;
}

// ============================================================
// Helper: Build perspective projection for a cascade split
// Matches Camera.GetProjectionMatrix(near, far)
// ============================================================
float4x4 MakePerspectiveLH(float near, float far)
{
    // Extract FOV params from the full camera projection
    // CameraProjection._11 = 1/(aspect*tan(fov/2)), _22 = 1/tan(fov/2)
    float sx = CameraProjection._11;
    float sy = CameraProjection._22;
    
    float4x4 m = (float4x4)0;
    m[0][0] = sx;
    m[1][1] = sy;
    
    // Reverse-Z perspective: maps [near,far] -> [1,0]
    // _33 = near / (near - far), _43 = (far * near) / (near - far)
    // but we check what System.Numerics uses for LH:
    // CreatePerspectiveFieldOfViewLeftHanded with reverseZ would be custom
    // The engine uses standard LH: _33 = far/(far-near), _43 = -(near*far)/(far-near)
    m[2][2] = far / (far - near);
    m[2][3] = 1.0;
    m[3][2] = -(near * far) / (far - near);
    return m;
}

// ============================================================
// Helper: Invert a 4x4 matrix
// We need this for computing frustum corners from cascade VP
// ============================================================
// Use a cofactor/adjugate approach for a general 4x4 inverse
float4x4 InvertMatrix(float4x4 m)
{
    float4x4 inv;
    
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];
    
    float b00 = a00*a11 - a01*a10;
    float b01 = a00*a12 - a02*a10;
    float b02 = a00*a13 - a03*a10;
    float b03 = a01*a12 - a02*a11;
    float b04 = a01*a13 - a03*a11;
    float b05 = a02*a13 - a03*a12;
    float b06 = a20*a31 - a21*a30;
    float b07 = a20*a32 - a22*a30;
    float b08 = a20*a33 - a23*a30;
    float b09 = a21*a32 - a22*a31;
    float b10 = a21*a33 - a23*a31;
    float b11 = a22*a33 - a23*a32;
    
    float det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
    float invDet = 1.0 / det;
    
    inv[0][0] = ( a11*b11 - a12*b10 + a13*b09) * invDet;
    inv[0][1] = (-a01*b11 + a02*b10 - a03*b09) * invDet;
    inv[0][2] = ( a31*b05 - a32*b04 + a33*b03) * invDet;
    inv[0][3] = (-a21*b05 + a22*b04 - a23*b03) * invDet;
    inv[1][0] = (-a10*b11 + a12*b08 - a13*b07) * invDet;
    inv[1][1] = ( a00*b11 - a02*b08 + a03*b07) * invDet;
    inv[1][2] = (-a30*b05 + a32*b02 - a33*b01) * invDet;
    inv[1][3] = ( a20*b05 - a22*b02 + a23*b01) * invDet;
    inv[2][0] = ( a10*b10 - a11*b08 + a13*b06) * invDet;
    inv[2][1] = (-a00*b10 + a01*b08 - a03*b06) * invDet;
    inv[2][2] = ( a30*b04 - a31*b02 + a33*b00) * invDet;
    inv[2][3] = (-a20*b04 + a21*b02 - a23*b00) * invDet;
    inv[3][0] = (-a10*b09 + a11*b07 - a12*b06) * invDet;
    inv[3][1] = ( a00*b09 - a01*b07 + a02*b06) * invDet;
    inv[3][2] = (-a30*b03 + a31*b01 - a32*b00) * invDet;
    inv[3][3] = ( a20*b03 - a21*b01 + a22*b00) * invDet;
    
    return inv;
}

// ============================================================
// Helper: Extract 6 frustum planes from a VP matrix
// Returns planes as float4 (normal.xyz, distance), pointing inward
// ============================================================
void ExtractFrustumPlanes(float4x4 vp, out float4 planes[6])
{
    // Left:   row3 + row0
    planes[0] = float4(vp[0][3]+vp[0][0], vp[1][3]+vp[1][0], vp[2][3]+vp[2][0], vp[3][3]+vp[3][0]);
    // Right:  row3 - row0
    planes[1] = float4(vp[0][3]-vp[0][0], vp[1][3]-vp[1][0], vp[2][3]-vp[2][0], vp[3][3]-vp[3][0]);
    // Bottom: row3 + row1
    planes[2] = float4(vp[0][3]+vp[0][1], vp[1][3]+vp[1][1], vp[2][3]+vp[2][1], vp[3][3]+vp[3][1]);
    // Top:    row3 - row1
    planes[3] = float4(vp[0][3]-vp[0][1], vp[1][3]-vp[1][1], vp[2][3]-vp[2][1], vp[3][3]-vp[3][1]);
    // Near:   row2
    planes[4] = float4(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    // Far:    row3 - row2
    planes[5] = float4(vp[0][3]-vp[0][2], vp[1][3]-vp[1][2], vp[2][3]-vp[2][2], vp[3][3]-vp[3][2]);
    
    // Normalize each plane
    [unroll]
    for (uint i = 0; i < 6; i++)
    {
        float len = length(planes[i].xyz);
        planes[i] /= len;
    }
}

// ============================================================
// CSComputeCascadeMatrices — one thread per cascade
// Port of DirectionalLight.SetupCascade()
// ============================================================
[numthreads(4, 1, 1)]
void CSComputeCascadeMatrices(uint3 id : SV_DispatchThreadID)
{
    uint cascadeIdx = id.x;
    if (cascadeIdx >= CascadeCount)
        return;
    
    // Read raw split distances from SDSM buffer
    StructuredBuffer<float> splitsBuffer = ResourceDescriptorHeap[SplitsBufferIdx];
    float rawSplit = splitsBuffer[cascadeIdx];
    
    // Temporal smoothing: blend raw SDSM splits with previous smoothed values
    // Read previous smoothed value, blend, write back
    RWStructuredBuffer<float> smoothedSplits = ResourceDescriptorHeap[SmoothedSplitsIdx];
    float prev = smoothedSplits[cascadeIdx];
    
    // On first frame (prev=0), use raw value directly; otherwise blend
    // 0.05 = ~20 frames to 63% convergence = visible but responsive smoothing
    float smoothed = (prev > 0.001) ? lerp(prev, rawSplit, 0.05) : rawSplit;
    smoothedSplits[cascadeIdx] = smoothed;
    
    // For splitNear, read the raw SDSM split for (cascadeIdx-1) and apply same
    // smoothing ratio as the previous frame would have. This avoids cross-thread
    // dependency — each cascade independently reads its own near boundary.
    float splitFar = smoothed;
    float splitNear;
    if (cascadeIdx == 0)
    {
        splitNear = NearPlane;
    }
    else
    {
        // Use the smoothed value we wrote for the previous cascade
        // Since all 4 threads are in the same wave, the write above for cascadeIdx-1
        // is NOT yet visible. Use the raw value for near instead.
        float rawNear = splitsBuffer[cascadeIdx - 1];
        float prevNear = smoothedSplits[cascadeIdx - 1];
        splitNear = (prevNear > 0.001) ? lerp(prevNear, rawNear, 0.05) : rawNear;
    }
    
    // Build cascade projection matrix (same fov/aspect as camera, different near/far)
    float4x4 cascadeProjection = MakePerspectiveLH(splitNear, splitFar);
    
    // Build camera-relative cascade VP
    float4x4 cascadeView = MakeLookAtLH(float3(0,0,0), CameraForward, CameraUp);
    float4x4 cascadeVP = mul(cascadeView, cascadeProjection);
    
    // Invert to get camera-relative world corners from NDC
    float4x4 inverseVP = InvertMatrix(cascadeVP);
    
    // 8 NDC corners: near z=0, far z=1 (left-handed D3D)
    float3 ndcCorners[8] = {
        float3(-1, -1, 0), float3( 1, -1, 0),
        float3( 1,  1, 0), float3(-1,  1, 0),
        float3(-1, -1, 1), float3( 1, -1, 1),
        float3( 1,  1, 1), float3(-1,  1, 1)
    };
    
    // Transform NDC corners to camera-relative world space
    float3 frustumCorners[8];
    [unroll]
    for (uint i = 0; i < 8; i++)
    {
        float4 corner4 = mul(float4(ndcCorners[i], 1.0), inverseVP);
        frustumCorners[i] = corner4.xyz / corner4.w;
    }
    
    // Compute frustum center (camera-relative)
    float3 frustumCenter = float3(0,0,0);
    [unroll]
    for (uint c = 0; c < 8; c++)
        frustumCenter += frustumCorners[c];
    frustumCenter /= 8.0;
    
    // Bounding sphere radius: rotation-invariant cascade size
    float radius = 0;
    [unroll]
    for (uint r = 0; r < 8; r++)
        radius = max(radius, length(frustumCorners[r] - frustumCenter));
    radius = ceil(radius * 16.0) / 16.0;
    
    float texelSize = (radius * 2.0) / (float)ShadowMapRes;
    
    // Texel snapping: snap frustum center to shadow texel grid
    // Light rotation matrix (zero-origin look at light forward along Y-up)
    float4x4 lightRotation = MakeLookAtLH(float3(0,0,0), LightForward, float3(0,1,0));
    float4x4 lightRotationInv = InvertMatrix(lightRotation);
    
    float3 frustumCenterLS = mul(float4(frustumCenter, 1.0), lightRotation).xyz;
    float3 cameraPosLS = mul(float4(CameraPosition, 1.0), lightRotation).xyz;
    
    // Sub-texel remainder of camera position
    float camRemX = fmod(cameraPosLS.x, texelSize);
    float camRemY = fmod(cameraPosLS.y, texelSize);
    if (camRemX < 0) camRemX += texelSize;
    if (camRemY < 0) camRemY += texelSize;
    
    // Offset by camera remainder, snap, then remove the offset
    float cx = frustumCenterLS.x + camRemX;
    float cy = frustumCenterLS.y + camRemY;
    float remX = fmod(cx, texelSize);
    float remY = fmod(cy, texelSize);
    if (remX < 0) remX += texelSize;
    if (remY < 0) remY += texelSize;
    
    float3 snappedLS = float3(cx - remX - camRemX, cy - remY - camRemY, frustumCenterLS.z);
    float3 snappedCenter = mul(float4(snappedLS, 1.0), lightRotationInv).xyz;
    
    // Build camera-relative light view matrix from snapped center
    float4x4 lightView = MakeLookAtLH(snappedCenter, snappedCenter + LightForward, float3(0,1,0));
    
    // Z bounds from corners
    float minZ = 1e30;
    float maxZ = -1e30;
    [unroll]
    for (uint z = 0; z < 8; z++)
    {
        float ls = mul(float4(frustumCorners[z], 1.0), lightView).z;
        minZ = min(minZ, ls);
        maxZ = max(maxZ, ls);
    }
    // Snap Z bounds to texel grid
    minZ = floor(minZ / texelSize) * texelSize;
    maxZ = ceil(maxZ / texelSize) * texelSize;
    
    // Inflate Z to catch shadow casters behind the camera
    float zRange = maxZ - minZ;
    float backExtension = zRange * pow(3.5, 3.0 - (float)cascadeIdx);
    minZ -= backExtension;
    
    float4x4 lightProj = MakeOrthoOffCenterLH(-radius, radius, -radius, radius, minZ, maxZ);
    
    // Camera-relative lightVP for light pass sampling
    float4x4 lightVP = mul(lightView, lightProj);
    
    // Apex pattern: prepend -cameraPosition translation for absolute world transforms
    float4x4 camTranslation = (float4x4)0;
    camTranslation[0][0] = 1; camTranslation[1][1] = 1; camTranslation[2][2] = 1; camTranslation[3][3] = 1;
    camTranslation[3][0] = -CameraPosition.x;
    camTranslation[3][1] = -CameraPosition.y;
    camTranslation[3][2] = -CameraPosition.z;
    
    float4x4 shadowView = mul(camTranslation, lightView);
    float4x4 shadowVP = mul(shadowView, lightProj);
    
    // Extract frustum planes from shadowVP (absolute-world space)
    float4 frustumPlanes[6];
    ExtractFrustumPlanes(shadowVP, frustumPlanes);
    
    // Read previous frame VP
    StructuredBuffer<row_major float4x4> prevVPs = ResourceDescriptorHeap[PrevVPBufferIdx];
    float4x4 prevVP = prevVPs[cascadeIdx];
    
    // Write CascadeData output
    RWStructuredBuffer<CascadeData> cascadeOut = ResourceDescriptorHeap[CascadeOutUAVIdx];
    
    CascadeData data;
    data.Planes[0] = frustumPlanes[0];
    data.Planes[1] = frustumPlanes[1];
    data.Planes[2] = frustumPlanes[2];
    data.Planes[3] = frustumPlanes[3];
    data.Planes[4] = frustumPlanes[4];
    data.Planes[5] = frustumPlanes[5];
    data.VP = shadowVP;
    data.PrevVP = prevVP;
    data.SplitDistances = float4(splitNear, splitFar, 0, 0);
    
    cascadeOut[cascadeIdx] = data;
    
    // Write LightingCascadeData output (camera-relative VP for lighting pass)
    RWStructuredBuffer<LightingCascadeData> lightingOut = ResourceDescriptorHeap[LightingOutUAVIdx];
    
    LightingCascadeData lightingData;
    lightingData.VP = lightVP;
    lightingData.Cascade = float4(splitNear, splitFar, 0, 0);
    
    lightingOut[cascadeIdx] = lightingData;
}
