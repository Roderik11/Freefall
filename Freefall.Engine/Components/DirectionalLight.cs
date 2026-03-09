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

        internal const int MaxCascades = 8; // Support up to 8 cascades
        
        // Shadow Cascades — arrays sized to max capacity, active count from Engine.Settings
        private readonly Vector4[] cascades = new Vector4[MaxCascades];
        private readonly Matrix4x4[] cascadeProjectionMatrices = new Matrix4x4[MaxCascades];
        private readonly Matrix4x4[] lightSpace = new Matrix4x4[MaxCascades];
        private readonly Vector3[] frustumCorners = new Vector3[8];
        
        // GPU-driven shadow rendering - frustum planes per cascade (6 planes each)
        private readonly Vector4[][] cascadeFrustumPlanes = new Vector4[MaxCascades][];
        
        // Unified cascade structured buffer: StructuredBuffer<CascadeData>
        // Holds planes + VP + PrevVP per cascade. All consumers (cull, terrain, grass) read from this.
        private GraphicsBuffer[]? _cascadeBuffers;
        private uint[]? _cascadeBufferSrvIndices;
        // TEMP DEBUG: separate VP-only buffer to test if CascadeData struct offset issue
        private GraphicsBuffer[]? _vpOnlyBuffers;
        private uint[]? _vpOnlySrvIndices;
        private const int FrameCount = 3;

        // Shadow cascade planes cbuffer (register b1) — used by compute culler
        private GraphicsBuffer[]? _shadowCascadeCBs;

        // Dedicated shadow SceneConstants buffer (prevents overwriting opaque pass CBV)
        private ID3D12Resource[]? _shadowSceneConstantsBuffers;
        private IntPtr[]? _shadowSceneConstantsPtrs;

        // VP matrices for current and previous frames
        private readonly Matrix4x4[] _shadowVPMatrices = new Matrix4x4[MaxCascades];
        private readonly Matrix4x4[] _prevShadowVPMatrices = new Matrix4x4[MaxCascades];
        
        // GPU-default cascade buffers for compute path (UAV + SRV)
        private GraphicsBuffer[]? _gpuCascadeBuffers;
        
        // Track whether GPU cascade compute was used this frame
        private bool _usedGpuCascadeCompute;
        private uint _activeLightingCascadeSrv;  // SRV for lighting pass (0 = CPU path)


        /// <summary>
        /// Current shadow scene CBV address (set per cascade during shadow rendering).
        /// Custom shadow actions can use this to rebind the light VP after Material.Apply.
        /// </summary>
        public static ulong CurrentShadowSceneCBV { get; set; }

        /// <summary>
        /// Current cascade index being rendered (for custom actions like grass).
        /// </summary>
        public static int CurrentShadowCascadeIndex { get; set; }

        /// <summary>
        /// Current shadow VP buffer SRV index (for custom actions).
        /// </summary>
        public static uint CurrentCascadeSrvIndex { get; set; }
        
        // TEMP DEBUG: VP-only buffer SRV for A/B testing
        public static uint CurrentVPOnlySrvIndex { get; set; }

        /// <summary>
        /// Get frustum planes for the outermost shadow cascade (largest coverage).
        /// Returns 6 Vector4 planes in absolute-world space. Valid during shadow pass.
        /// </summary>
        public static Vector4[]? GetOutermostCascadeFrustumPlanes()
        {
            if (_instance == null) return null;
            int last = CascadeCount - 1;
            return _instance.cascadeFrustumPlanes[last];
        }

        /// <summary>
        /// Get frustum planes for all cascades. Returns cascadeFrustumPlanes[0..N-1],
        /// each containing 6 Vector4 planes in absolute-world space.
        /// </summary>
        public static Vector4[][]? GetAllCascadeFrustumPlanes()
        {
            if (_instance == null) return null;
            return _instance.cascadeFrustumPlanes;
        }

        /// <summary>
        /// Number of active shadow cascades (from Engine.Settings).
        /// </summary>
        public static int CascadeCount => Engine.Settings.ShadowCascadeCount;

        private static DirectionalLight? _instance;

        public DirectionalLight()
        {
            Material = new Material(new Effect("light_directional"));
            Params = new MaterialBlock();
            
            for (int i = 0; i < 4; i++)
                cascadeFrustumPlanes[i] = new Vector4[6];

            _instance = this;
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
            var renderer = DeferredRenderer.Current;
            if (renderer == null) return;

            // Populate cbuffer params (same as before — Material.Apply commits them)
            var cvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
            Matrix4x4.Invert(cvp, out var cameraInverse);
            
            Params.Clear();
            Params.SetParameter("CameraInverse", cameraInverse);
            Params.SetParameter("CameraVP", cvp);
            Params.SetParameter("LightColor", Color.ToVector3());
            Params.SetParameter("LightDirection", Transform.Forward);
            Params.SetParameter("LightIntensity", Intensity);
            
            int cc = CascadeCount;
            Params.SetParameterArray("LightSpaces", lightSpace[..cc]);
            Params.SetParameterArray("Cascades", cascades[..cc]);
            Params.SetParameter("CascadeCount", cc);
            Params.SetParameter("DebugVisualizationMode", Engine.Settings.DebugVisualizationMode);
            
            // Apply params to Material cbuffers (commits data but we won't use its graphics PSO)
            Material.SetTextureIndex("LightingCascadeSRV", _activeLightingCascadeSrv);
            if (renderer.ShadowTextureArray != null)
                Params.SetTexture("ShadowMap", renderer.ShadowTextureArray);
            Params.SetTexture("NormalTex", renderer.Normals);
            Params.SetTexture("DepthTex", renderer.Depth);
            Params.SetTexture("DepthGBuf", renderer.DepthGBuffer);
            Params.SetTexture("AlbedoTex", renderer.Albedo);
            Params.SetTexture("DataTex", renderer.Data);
            
            // Commit cbuffer data via Material.Apply (populates ObjectConstants + SceneConstants)
            Material.Apply(commandList, Engine.Device, Params);
            
            // Switch to compute pipeline via ComputeShader
            var lightCS = renderer.DirectionalLightCS;
            commandList.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            commandList.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            
            // Push constants via ComputeShader reflection (10 slots)
            var desc = renderer.LightBuffer.Native.Description;
            lightCS.SetPushConstant("NormalTex", renderer.Normals.BindlessIndex);
            lightCS.SetPushConstant("DepthTex", renderer.Depth.BindlessIndex);
            lightCS.SetPushConstant("ShadowMap", renderer.ShadowTextureArray?.BindlessIndex ?? 0u);
            lightCS.SetPushConstant("DepthGBuf", renderer.DepthGBuffer.BindlessIndex);
            lightCS.SetPushConstant("AlbedoTex", renderer.Albedo.BindlessIndex);
            lightCS.SetPushConstant("DataTex", renderer.Data.BindlessIndex);
            lightCS.SetPushConstant("LightingCascadeSRV", _activeLightingCascadeSrv);
            lightCS.SetPushConstant("OutputUAV", renderer.LightBuffer.UavIndex);
            lightCS.SetPushConstant("ScreenWidth", (uint)desc.Width);
            lightCS.SetPushConstant("ScreenHeight", (uint)desc.Height);
            
            // Bind cbuffers on compute root (Material.Apply committed them on graphics root)
            foreach (var cb in Material.ConstantBuffers)
            {
                if (cb.Slot >= 0)
                {
                    cb.Commit();
                    commandList.SetComputeRootConstantBufferView((uint)cb.Slot, cb.GpuAddress);
                }
            }
            
            uint groupsX = ((uint)desc.Width + 7) / 8;
            uint groupsY = ((uint)desc.Height + 7) / 8;
            lightCS.Dispatch(0, commandList, groupsX, groupsY);
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
            
            // Check if GPU cascade compute is available and SDSM is active
            var culler = CommandBuffer.Culler;
            bool useGpuCascades = Engine.Settings.UseAdaptiveSplits 
                && culler?.CascadeComputeReady == true;
            
            if (useGpuCascades)
            {
                // GPU path: cascade compute shader reads SDSM splits directly from GPU buffer.
                // No CPU readback — all cascade data (matrices, planes, splits) computed on GPU.
                EnsureGPUCascadeBuffer();
                EnsureCascadeBuffer(); // cbuffer still needed for root sig bind (slot 2)
                int frameIndex = Engine.FrameIndex % FrameCount;
                
                uint cascadeUAV = _gpuCascadeBuffers![frameIndex].UavIndex;
                uint cascadeSRV = _gpuCascadeBuffers![frameIndex].SrvIndex;
                
                // Transition GPU cascade buffer to UAV for compute write
                commandList.ResourceBarrierTransition(_gpuCascadeBuffers![frameIndex].Native,
                    ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess);
                
                // Dispatch GPU cascade matrix computation
                culler!.ComputeCascadeMatrices(
                    commandList,
                    camera,
                    cameraPosition,
                    Transform.Forward,
                    shadowTex.Width,
                    cascadeUAV,
                    cascadeSRV);
                
                // Transition GPU cascade buffer back to SRV for culler/rendering
                commandList.ResourceBarrierTransition(_gpuCascadeBuffers[frameIndex].Native,
                    ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
                
                // Set viewport for shadow map
                commandList.RSSetViewport(new Viewport(0, 0, shadowTex.Width, shadowTex.Height));
                commandList.RSSetScissorRect(new RectI(0, 0, shadowTex.Width, shadowTex.Height));
                
                // Clear all cascades
                for (int i = 0; i < 4; i++)
                    commandList.ClearDepthStencilView(shadowTex.SliceDsvHandles[i], ClearFlags.Depth, 1.0f, 0);
                
                if (!CastShadows) return;
                
                // GPU-driven cull using GPU-computed cascade buffer
                // Culler reads planes from CascadeData structured buffer (not cbuffer)
                var allBatches = CommandBuffer.GetAllBatches(RenderPass.Opaque);
                if (allBatches != null)
                {
                    uint shadowHiZSrv = 0;
                    var shadowPyramid = DeferredRenderer.Current?.ShadowHiZPyramid;
                    if (shadowPyramid?.Ready == true) shadowHiZSrv = shadowPyramid.FullSRV;
                    
                    // cbuffer addr still bound for root sig requirement (culler ignores it now)
                    ulong cascadeCBAddr = _shadowCascadeCBs![frameIndex].Native.GPUVirtualAddress;
                    CurrentCascadeSrvIndex = cascadeSRV;
                    CurrentVPOnlySrvIndex = _vpOnlySrvIndices != null ? _vpOnlySrvIndices[frameIndex] : 0;
                    
                    foreach (var batch in allBatches)
                        batch.CullShadowAll(commandList, cascadeSRV, cascadeCBAddr, shadowHiZSrv, culler);
                    
                    commandList.OMSetRenderTargets(0, (CpuDescriptorHandle[]?)null, false, shadowTex.FullArrayDsvHandle);
                    
                    foreach (var batch in allBatches)
                    {
                        if (!batch.Material.HasPass(RenderPass.Shadow)) continue;
                        batch.Material.SetPass(RenderPass.Shadow);
                        batch.Material.Apply(commandList, Engine.Device, null);
                        batch.DrawShadowSinglePass(commandList, cascadeSRV);
                    }
                    
                    CommandBuffer.ExecuteCustomActions(RenderPass.Shadow, commandList);
                    CommandBuffer.ClearCustomActions(RenderPass.Shadow);
                }
                
                _usedGpuCascadeCompute = true;
                _activeLightingCascadeSrv = culler.LightingCascadeSRV;
            }
            else
            {
                // CPU path: existing implementation
                _usedGpuCascadeCompute = false;
                _activeLightingCascadeSrv = 0;
                
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
                
                // Draw each cascade using GPU-driven rendering
                // Phase 1: Compute cascade matrices and frustum planes
                for (int i = 0; i < CascadeCount; i++)
                    SetupCascade(i, camera, cameraPosition, shadowTex);
                
                // Phase 2: GPU-driven cull all cascades per batch (unified visibility pass)
                var allBatches = CommandBuffer.GetAllBatches(RenderPass.Opaque);
                int frameIndex = Engine.FrameIndex % FrameCount;
                
                if (culler != null && allBatches != null)
                {
                    EnsureCascadeBuffer();
                    UploadCascadeBuffer(frameIndex);
                    uint cascadeSrv = _cascadeBufferSrvIndices![frameIndex];
                    uint vpOnlySrv = _vpOnlySrvIndices![frameIndex]; // TEMP
                    CurrentCascadeSrvIndex = cascadeSrv;
                    CurrentVPOnlySrvIndex = vpOnlySrv; // TEMP
                    
                    // Shadow Hi-Z SRV for occlusion culling (0 = disabled on first frame before pyramid is ready)
                    var shadowPyramid = DeferredRenderer.Current?.ShadowHiZPyramid;
                    uint shadowHiZSrv = (shadowPyramid?.Ready == true) ? shadowPyramid.FullSRV : 0;
                    
                    // Cull all cascades for each batch using unified visibility pass
                    ulong cascadeCBAddr = _shadowCascadeCBs![frameIndex].Native.GPUVirtualAddress;
                    foreach (var batch in allBatches)
                    {
                        batch.CullShadowAll(commandList, cascadeSrv, cascadeCBAddr, shadowHiZSrv, culler);
                    }
                    
                    // Phase 3: True single-pass draw — 1 ExecuteIndirect per batch for all cascades
                    // Bind full-array DSV (all slices at once)
                    commandList.OMSetRenderTargets(0, (CpuDescriptorHandle[]?)null, false, shadowTex.FullArrayDsvHandle);
                    
                    foreach (var batch in allBatches)
                    {
                        if (!batch.Material.HasPass(RenderPass.Shadow)) continue;
                        batch.Material.SetPass(RenderPass.Shadow);
                        batch.Material.Apply(commandList, Engine.Device, null);
                        
                        batch.DrawShadowSinglePass(commandList, cascadeSrv);
                    }

                    // Custom shadow actions (grass) — single-pass via AS/MS cascade expansion
                    // FullArrayDsvHandle is already bound from batch shadows above
                    CommandBuffer.ExecuteCustomActions(RenderPass.Shadow, commandList);
                    CommandBuffer.ClearCustomActions(RenderPass.Shadow);
                }
                else
                {
                    // Fallback: CPU geometry submission
                    CommandBuffer.Execute(RenderPass.Shadow, commandList, Engine.Device);
                    CommandBuffer.Clear();
                }
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
            
            // Store combined shadowVP for the single-pass structured buffer
            _shadowVPMatrices[index] = shadowVP;
        }
               
        private void EnsureCascadeBuffer()
        {
            if (_cascadeBuffers != null) return;
            
            _cascadeBuffers = new GraphicsBuffer[FrameCount];
            _cascadeBufferSrvIndices = new uint[FrameCount];
            
            _vpOnlyBuffers = new GraphicsBuffer[FrameCount];
            _vpOnlySrvIndices = new uint[FrameCount];
            
            _shadowCascadeCBs = new GraphicsBuffer[FrameCount];
            
            for (int i = 0; i < FrameCount; i++)
            {
                _cascadeBuffers[i] = GraphicsBuffer.CreateUpload<GPUCuller.CascadeData>(MaxCascades, mapped: true);
                _cascadeBufferSrvIndices[i] = _cascadeBuffers[i].SrvIndex;
                
                _vpOnlyBuffers[i] = GraphicsBuffer.CreateUpload<Matrix4x4>(MaxCascades, mapped: true);
                _vpOnlySrvIndices[i] = _vpOnlyBuffers[i].SrvIndex;
                
                _shadowCascadeCBs[i] = GraphicsBuffer.CreateConstantBuffer<GPUCuller.ShadowCascadeConstants>();
            }
        }
        
        private void EnsureGPUCascadeBuffer()
        {
            if (_gpuCascadeBuffers != null) return;
            
            _gpuCascadeBuffers = new GraphicsBuffer[FrameCount];
            
            for (int i = 0; i < FrameCount; i++)
            {
                _gpuCascadeBuffers[i] = GraphicsBuffer.CreateStructured<GPUCuller.CascadeData>(
                    MaxCascades, srv: true, uav: true,
                    initialState: ResourceStates.NonPixelShaderResource);
            }
        }
        
        private unsafe void UploadCascadeBuffer(int frameIndex)
        {
            var dst = _cascadeBuffers![frameIndex].WritePtr<GPUCuller.CascadeData>();
            var vpDst = _vpOnlyBuffers![frameIndex].WritePtr<Matrix4x4>();
            var cbDst = _shadowCascadeCBs![frameIndex].WritePtr<Vector4>();
            for (int c = 0; c < CascadeCount; c++)
            {
                var data = new GPUCuller.CascadeData();
                data.SetPlanes(cascadeFrustumPlanes[c]);
                data.VP = _shadowVPMatrices[c];
                data.PrevVP = _prevShadowVPMatrices[c];
                data.SplitDistances = cascades[c]; // X=near, Y=far
                dst[c] = data;
                vpDst[c] = _shadowVPMatrices[c];
                
                // Write planes to cbuffer (6 Vector4s per cascade)
                for (int p = 0; p < 6; p++)
                    cbDst[c * 6 + p] = cascadeFrustumPlanes[c][p];
            }
            
            // Swap current → previous for next frame
            Array.Copy(_shadowVPMatrices, _prevShadowVPMatrices, CascadeCount);
        }

        private void EnsureShadowSceneConstantsBuffer()
        {
            if (_shadowSceneConstantsBuffers != null) return;
            
            // SceneConstants: Time(4) + View(64) + Projection(64) + ViewProjection(64) + ViewInverse(64) + CameraInverse(64) = 324 bytes
            // Aligned to 256 bytes = 512
            // SceneConstants: Time(4) + View(64) + Projection(64) + ViewProjection(64) + ViewInverse(64) + CameraInverse(64) = 324 bytes
            // Aligned to 256 bytes = 512.
            // We need CascadeCount slots, one for each cascade.
            int bufferSize = 512 * MaxCascades;
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
