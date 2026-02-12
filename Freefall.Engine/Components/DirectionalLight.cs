using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.Mathematics;
using Freefall.Graphics;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Components
{
    public class DirectionalLight : Component, IDraw
    {
        public Color3 Color = new Color3(1, 1, 1);
        public float Intensity = 1;
        public bool CastShadows = true;
        
        /// <summary>
        /// Direct cascade split distances. Each value is the far boundary of that cascade.
        /// Cascade 0: near..Splits[0], Cascade 1: Splits[0]..Splits[1], etc.
        /// </summary>
        public float[] CascadeSplits = { 16f, 64f, 128, 512 };
        


        private Material Material;
        private MaterialBlock Params;

        // Shadow Cascades
        private readonly Vector4[] cascades = new Vector4[4];
        private readonly Matrix4x4[] cascadeProjectionMatrices = new Matrix4x4[4];
        private readonly Matrix4x4[] lightSpace = new Matrix4x4[4];
        private readonly Vector3[] frustumCorners = new Vector3[8];
        
        // GPU-driven shadow rendering - frustum planes per cascade (6 planes each)
        private readonly Vector4[][] cascadeFrustumPlanes = new Vector4[4][];
        
        // Shadow cascade constant buffer (uploaded once for all 4 cascades)
        private ID3D12Resource[]? _shadowCascadeBuffers;
        private const int FrameCount = 3;

        // Dedicated shadow SceneConstants buffer (prevents overwriting opaque pass CBV)
        private ID3D12Resource[]? _shadowSceneConstantsBuffers;
        private IntPtr[]? _shadowSceneConstantsPtrs;

        public DirectionalLight()
        {
            Material = new Material(new Effect("light_directional"));
            Params = new MaterialBlock();
            
            for (int i = 0; i < 4; i++)
                cascadeFrustumPlanes[i] = new Vector4[6];
        }

        // Called by DeferredRenderer via CommandBuffer
        public void Draw()
        {
            // Enqueue shadow rendering first (runs before light pass)
            if (CastShadows && DeferredRenderer.Current?.ShadowTextureArray != null)
                CommandBuffer.Enqueue(RenderPass.ShadowMap, DrawShadows);
            
            CommandBuffer.Enqueue(RenderPass.Light, DrawLight);
        }

        private static int _diagFrameCount = 0;
        private void DrawLight(ID3D12GraphicsCommandList commandList)
        {
            _diagFrameCount++;
            var camera = Camera.Main;
            if (camera == null) return;

            // Set Shader Params
            // Zero-translation CameraInverse: even though GBuffer depth was written with full View,
            // NDC = (worldPos - camPos) × R × P, so inverse(R × P) correctly gives camera-relative pos
            var cvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
            Matrix4x4.Invert(cvp, out var cameraInverse);
            
            Params.Clear();
            Params.SetParameter("CameraInverse", cameraInverse);
            Params.SetParameter("LightColor", Color.ToVector3());
            Params.SetParameter("LightDirection", Transform.Forward);
            Params.SetParameter("LightIntensity", Intensity);
            
            // Shadow parameters - use array variant
            Params.SetParameterArray("LightSpaces", lightSpace);
            Params.SetParameterArray("Cascades", cascades);
            Params.SetParameter("DebugVisualizationMode", Engine.Settings.DebugVisualizationMode);
            
            if (DeferredRenderer.Current != null)
            {
                Params.SetTexture("ShadowMap", DeferredRenderer.Current.ShadowTextureArray);
                Params.SetTexture("NormalTex", DeferredRenderer.Current.Normals);
                Params.SetTexture("DepthTex", DeferredRenderer.Current.Depth);
                Params.SetTexture("DepthGBuf", DeferredRenderer.Current.DepthGBuffer);
                Params.SetTexture("AlbedoTex", DeferredRenderer.Current.Albedo);
                Params.SetTexture("DataTex", DeferredRenderer.Current.Data);
            }
                
            // Render fullscreen quad procedurally (shader uses SV_VertexID)
            Material.Apply(commandList, Engine.Device, Params);
            
            // One-shot diagnostic: verify matrix values reaching the GPU
            if (_diagFrameCount == 15)
            {
                var ls0 = lightSpace[0];
                Debug.Log($"[ShadowDiag] LightSpaces[0] Row0: ({ls0.M11:F4}, {ls0.M12:F4}, {ls0.M13:F4}, {ls0.M14:F4})");
                Debug.Log($"[ShadowDiag] LightSpaces[0] Row1: ({ls0.M21:F4}, {ls0.M22:F4}, {ls0.M23:F4}, {ls0.M24:F4})");
                Debug.Log($"[ShadowDiag] LightSpaces[0] Row2: ({ls0.M31:F4}, {ls0.M32:F4}, {ls0.M33:F4}, {ls0.M34:F4})");
                Debug.Log($"[ShadowDiag] LightSpaces[0] Row3: ({ls0.M41:F4}, {ls0.M42:F4}, {ls0.M43:F4}, {ls0.M44:F4})");
                Debug.Log($"[ShadowDiag] CameraInverse Row3: ({cameraInverse.M41:F4}, {cameraInverse.M42:F4}, {cameraInverse.M43:F4}, {cameraInverse.M44:F4})");
                Debug.Log($"[ShadowDiag] CamPos: {camera.Transform?.Position}");
                Debug.Log($"[ShadowDiag] LightDir: {Transform.Forward}");
            }
            
            commandList.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleStrip);
            commandList.DrawInstanced(4, 1, 0, 0);
        }

        private void DrawShadows(ID3D12GraphicsCommandList commandList)
        {
            var camera = Camera.Main;
            if (camera == null) return;
            
            var cameraPosition = camera.Position;
            var shadowTex = DeferredRenderer.Current!.ShadowTextureArray!;

            // Set up cascade split distances
            float near = camera.NearPlane;
            const int numCascades = 4;
            
            // Try adaptive splits from SDSM depth analysis (previous frame)
            float[] activeSplits = CascadeSplits;
            bool usingAdaptive = false;
            if (Engine.Settings.UseAdaptiveSplits)
            {
                var adaptiveSplits = CommandBuffer.Culler?.ReadAdaptiveSplits();
                if (adaptiveSplits != null)
                {
                    activeSplits = adaptiveSplits;
                    usingAdaptive = true;
                }
            }
            
            // Use active splits (adaptive or fixed)
            cascades[0] = new Vector4(near, activeSplits[0], 0, 0);
            for (int i = 1; i < numCascades; i++)
                cascades[i] = new Vector4(activeSplits[i - 1], activeSplits[i], 0, 0);

            // Get cascade projection matrices
            for (int i = 0; i < 4; i++)
                cascadeProjectionMatrices[i] = camera.GetProjectionMatrix(cascades[i].X, cascades[i].Y);

            // Set viewport for shadow map size
            commandList.RSSetViewport(new Viewport(0, 0, shadowTex.Width, shadowTex.Height));
            commandList.RSSetScissorRect(new RectI(0, 0, shadowTex.Width, shadowTex.Height));
            
            // Clear all cascades first
            for (int i = 0; i < 4; i++)
                commandList.ClearDepthStencilView(shadowTex.SliceDsvHandles[i], ClearFlags.Depth, 1.0f, 0);
            
            if (!CastShadows) return;
            
            if (Engine.FrameIndex % 60 == 0)
            {
                string mode = usingAdaptive ? "SDSM" : "Fixed";
                Debug.Log($"[Shadow] DrawShadows: shadowTex={shadowTex.Width}x{shadowTex.Height}, Mode={mode}");
                Debug.Log($"[Shadow] Splits: {activeSplits[0]:F1}, {activeSplits[1]:F1}, {activeSplits[2]:F1}, {activeSplits[3]:F1}");
            }
            
            // Draw each cascade using GPU-driven rendering
            // Phase 1: Compute all 4 cascade matrices and frustum planes
            for (int i = 0; i < 4; i++)
            {
                SetupCascade(i, camera, cameraPosition, shadowTex);
            }
            
            // Phase 2: GPU-driven cull all 4 cascades per batch (unified visibility pass)
            var culler = CommandBuffer.Culler;
            var allBatches = CommandBuffer.GetAllBatches(RenderPass.Opaque);
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            if (culler != null && allBatches != null)
            {
                EnsureShadowCascadeBuffer();
                UploadShadowCascadePlanes(frameIndex);
                ulong cascadeBufferAddress = _shadowCascadeBuffers![frameIndex].GPUVirtualAddress;
                
                // Cull all 4 cascades for each batch using unified visibility pass
                foreach (var batch in allBatches)
                {
                    batch.CullShadowAll(commandList, cascadeBufferAddress, culler);
                }
                
                // Phase 3: Draw each cascade
                for (int i = 0; i < 4; i++)
                {
                    // Set render target to depth-only
                    commandList.OMSetRenderTargets(0, (CpuDescriptorHandle[]?)null, false, shadowTex.SliceDsvHandles[i]);
                    
                    ulong shadowSceneCBVAddress = _shadowSceneConstantsBuffers![frameIndex].GPUVirtualAddress + (ulong)(i * 512);
                    foreach (var batch in allBatches)
                    {
                        var pass = batch.Material.GetPass(Material.Pass);
                        batch.Material.SetPass(RenderPass.Shadow);
                        batch.DrawShadow(commandList, i, shadowSceneCBVAddress);
                        batch.Material.SetPass(pass);
                    }
                }
            }
            else
            {
                // Fallback: CPU geometry submission
                CommandBuffer.Execute(RenderPass.Shadow, commandList, Engine.Device);
                CommandBuffer.Clear();
            }
        }
        
        /// <summary>
        /// Compute cascade light matrices, upload SceneConstants, and extract frustum planes.
        /// Does NOT issue any GPU commands — just CPU-side matrix setup.
        /// </summary>
        private void SetupCascade(int index, Camera camera, Vector3 cameraPosition, DepthTextureArray2D shadowTex)
        {
            var cascadeProjection = cascadeProjectionMatrices[index];
            
            // Build camera-relative cascade VP (view at origin looking along camera forward)
            var cascadeView = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up);
            var cascadeVP = cascadeView * cascadeProjection;
            
            // Invert the cascade VP to get camera-relative world corners from NDC
            Matrix4x4.Invert(cascadeVP, out var inverseVP);
            
            // 8 NDC corners: near z=0, far z=1 (left-handed D3D)
            Span<Vector3> ndcCorners = stackalloc Vector3[8]
            {
                new Vector3(-1, -1, 0), new Vector3( 1, -1, 0),
                new Vector3( 1,  1, 0), new Vector3(-1,  1, 0),
                new Vector3(-1, -1, 1), new Vector3( 1, -1, 1),
                new Vector3( 1,  1, 1), new Vector3(-1,  1, 1),
            };
            
            // Transform NDC corners to camera-relative world space
            for (int i = 0; i < 8; i++)
            {
                var corner4 = Vector4.Transform(new Vector4(ndcCorners[i], 1.0f), inverseVP);
                frustumCorners[i] = new Vector3(corner4.X, corner4.Y, corner4.Z) / corner4.W;
            }
            
            // Compute frustum center (camera-relative)
            var frustumCenter = Vector3.Zero;
            for (int i = 0; i < 8; i++)
                frustumCenter += frustumCorners[i];
            frustumCenter /= 8;
            
            // Bounding sphere radius: rotation-invariant cascade size
            // Using a sphere means the ortho projection dimensions are constant
            // regardless of camera rotation, eliminating size-change flicker
            float radius = 0;
            for (int i = 0; i < 8; i++)
                radius = MathF.Max(radius, (frustumCorners[i] - frustumCenter).Length());
            radius = MathF.Ceiling(radius * 16.0f) / 16.0f;
            
            float texelSize = (radius * 2.0f) / shadowTex.Width;
            
            // Texel snapping: snap the frustum center to the shadow map's texel grid
            // so it stays fixed in the world as the camera moves, eliminating shimmer.
            // To avoid precision loss from adding large cameraPosition to small frustumCenter,
            // we only use the sub-texel remainder of the camera position. All arithmetic
            // stays in the small-number domain — no catastrophic cancellation.
            var lightRotation = Matrix4x4.CreateLookAtLeftHanded(
                Vector3.Zero, Transform.Forward, Vector3.UnitY);
            Matrix4x4.Invert(lightRotation, out var lightRotationInv);
            
            var frustumCenterLS = Vector3.Transform(frustumCenter, lightRotation);
            var cameraPosLS = Vector3.Transform(cameraPosition, lightRotation);
            
            // Sub-texel remainder of camera position (small value, 0..texelSize)
            float camRemX = cameraPosLS.X % texelSize;
            float camRemY = cameraPosLS.Y % texelSize;
            if (camRemX < 0) camRemX += texelSize;
            if (camRemY < 0) camRemY += texelSize;
            
            // Offset by camera remainder, snap, then remove the offset
            // Equivalent to snap(F+C)-C but without ever computing F+C
            float cx = frustumCenterLS.X + camRemX;
            float cy = frustumCenterLS.Y + camRemY;
            float remX = cx % texelSize;
            float remY = cy % texelSize;
            if (remX < 0) remX += texelSize;
            if (remY < 0) remY += texelSize;
            
            var snappedLS = new Vector3(cx - remX - camRemX, cy - remY - camRemY, frustumCenterLS.Z);
            var snappedCenter = Vector3.Transform(snappedLS, lightRotationInv);
            
            // Build camera-relative light view matrix from snapped center
            var lightView = Matrix4x4.CreateLookAtLeftHanded(
                snappedCenter, snappedCenter + Transform.Forward, Vector3.UnitY);
            
            // Still need Z bounds from corners for depth range (only Z uses AABB)
            float minZ = float.MaxValue, maxZ = float.MinValue;
            for (int i = 0; i < 8; i++)
            {
                var ls = Vector3.Transform(frustumCorners[i], lightView);
                minZ = MathF.Min(minZ, ls.Z);
                maxZ = MathF.Max(maxZ, ls.Z);
            }
            // Snap Z bounds to texel grid to prevent depth range instability
            // (vertical camera movement maps to light-space Z, causing shimmer without this)
            minZ = MathF.Floor(minZ / texelSize) * texelSize;
            maxZ = MathF.Ceiling(maxZ / texelSize) * texelSize;
            
            // Inflate Z to catch shadow casters behind the camera
            float zRange = maxZ - minZ;
            float backExtension = zRange * MathF.Pow(3.5f, 3 - index);
            minZ -= backExtension;
            
            var lightProj = Matrix4x4.CreateOrthographicOffCenterLeftHanded(
                -radius, radius, -radius, radius, minZ, maxZ);
            
            // Camera-relative lightVP for light pass sampling and frustum culling
            var lightVP = lightView * lightProj;
            
            // Apex pattern: prepend -cameraPosition translation so absolute World transforms
            // get shifted to camera-relative before the camera-relative lightView
            var shadowView = Matrix4x4.CreateTranslation(-cameraPosition) * lightView;
            var shadowVP = shadowView * lightProj;
            
            // Store camera-relative lightVP for light pass sampling.
            // posFromDepth with zero-translation CameraInverse returns camera-relative positions,
            // and lightView is built from camera-relative frustumCenter, so lightVP matches.
            lightSpace[index] = lightVP;
            
            // Extract frustum planes for GPU culling (absolute-world space)
            var shadowFrustum = new Frustum(shadowVP);
            var planes = shadowFrustum.GetPlanesAsVector4();
            for (int i = 0; i < 6; i++)
                cascadeFrustumPlanes[index][i] = planes[i];
            
            // Upload shadow SceneConstants with offset shadowView for VS_Shadow
            EnsureShadowSceneConstantsBuffer();
            int frameIndex = Engine.FrameIndex % FrameCount;
            UploadShadowSceneConstants(frameIndex, index, shadowView, lightProj);
        }
               
        private void EnsureShadowCascadeBuffer()
        {
            if (_shadowCascadeBuffers != null) return;
            
            _shadowCascadeBuffers = new ID3D12Resource[FrameCount];
            int bufferSize = (24 * 16 + 255) & ~255; // 24 float4s, 256-byte aligned
            
            for (int i = 0; i < FrameCount; i++)
                _shadowCascadeBuffers[i] = Engine.Device.CreateUploadBuffer(bufferSize);
        }
        
        private unsafe void UploadShadowCascadePlanes(int frameIndex)
        {
            var constants = new GPUCuller.ShadowCascadeConstants();
            for (int c = 0; c < 4; c++)
                constants.SetCascadePlanes(c, cascadeFrustumPlanes[c]);
            
            void* pData;
            _shadowCascadeBuffers![frameIndex].Map(0, null, &pData);
            *(GPUCuller.ShadowCascadeConstants*)pData = constants;
            _shadowCascadeBuffers[frameIndex].Unmap(0);
        }

        private void EnsureShadowSceneConstantsBuffer()
        {
            if (_shadowSceneConstantsBuffers != null) return;
            
            // SceneConstants: Time(4) + View(64) + Projection(64) + ViewProjection(64) + ViewInverse(64) + CameraInverse(64) = 324 bytes
            // Aligned to 256 bytes = 512
            // SceneConstants: Time(4) + View(64) + Projection(64) + ViewProjection(64) + ViewInverse(64) + CameraInverse(64) = 324 bytes
            // Aligned to 256 bytes = 512.
            // We need 4 slots, one for each cascade, so 512 * 4 = 2048 bytes.
            int bufferSize = 512 * 4;
            _shadowSceneConstantsBuffers = new ID3D12Resource[FrameCount];
            _shadowSceneConstantsPtrs = new IntPtr[FrameCount];
            
            for (int i = 0; i < FrameCount; i++)
            {
                _shadowSceneConstantsBuffers[i] = Engine.Device.CreateUploadBuffer(bufferSize);
                unsafe
                {
                    void* pData;
                    _shadowSceneConstantsBuffers[i].Map(0, null, &pData);
                    _shadowSceneConstantsPtrs[i] = (IntPtr)pData;
                }
            }
        }

        private unsafe void UploadShadowSceneConstants(int frameIndex, int cascadeIndex, Matrix4x4 lightView, Matrix4x4 lightProj)
        {
            // Layout matches SceneConstants cbuffer in common.fx (register b0):
            // float Time;                          offset 0,  size 4
            // row_major float4x4 View;             offset 16, size 64  (16-byte aligned)
            // row_major float4x4 Projection;       offset 80, size 64
            // row_major float4x4 ViewProjection;   offset 144, size 64
            // row_major float4x4 ViewInverse;      offset 208, size 64
            // row_major float4x4 CameraInverse;    offset 272, size 64
            byte* ptr = (byte*)_shadowSceneConstantsPtrs![frameIndex];
            ptr += (cascadeIndex * 512);
            
            float time = Base.Time.TotalTime;
            var viewProj = lightView * lightProj;
            Matrix4x4.Invert(lightView, out var viewInv);
            Matrix4x4.Invert(viewProj, out var vpInv);
            
            *(float*)(ptr + 0) = time;
            *(Matrix4x4*)(ptr + 16) = lightView;
            *(Matrix4x4*)(ptr + 80) = lightProj;
            *(Matrix4x4*)(ptr + 144) = viewProj;
            *(Matrix4x4*)(ptr + 208) = viewInv;
            *(Matrix4x4*)(ptr + 272) = vpInv;
        }
    }
}
