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
    [Icon("icon_dirlight.png")]
    public class DirectionalLight : Component, IDraw
    {
        public Color3 Color = new Color3(1, 1, 1);
        public float Intensity = 1;
        public bool CastShadows = true;

        /// <summary>
        /// Degrees the light turns before its shadows follow (0 = every frame). A turning light puts the casters on a
        /// slightly turned shadow-map grid every frame, which the texel snapping cannot hold (it holds the grid against
        /// the camera's movement only), and every shadow edge crawls. Held, the shadows stand still and jump to the
        /// light once it has turned this far; the shading (N·L, the sky, the sun's disc) follows it every frame. The
        /// sun turns 0.25° in a game minute.
        /// </summary>
        [ValueRange(0f, 2f, 0.05f)]
        public float ShadowStep = 0.25f;

        /// <summary>
        /// Fixed cascade split distances (EngineSettings.UseAdaptiveSplits off). Each value is the far boundary of that
        /// cascade: cascade 0 near..Splits[0], cascade 1 Splits[0]..Splits[1], etc. With more cascades than values the
        /// curve they draw is resampled (FixedSplits), so four values serve six cascades.
        /// </summary>
        public float[] CascadeSplits = { 16f, 64f, 128, 512 };

        private Material Material;
        private MaterialBlock Params;

        /// <summary>Slices of the shadow array: the most cascades a frame can draw (shadow_cascades.hlsli).</summary>
        public const int MaxCascades = 6;

        /// <summary>Metres every cascade's depth range reaches toward the sun past its slice of the view, so casters
        /// above a small room (ceiling, roof) are drawn however thin the adaptive splits make the cascade.</summary>
        private const float MinCasterReach = 300f;
        
        // Shadow Cascades — arrays sized to max capacity, the count in use chosen per frame (CascadeCount)
        private readonly Vector4[] cascades = new Vector4[MaxCascades];
        private readonly Matrix4x4[] cascadeProjectionMatrices = new Matrix4x4[MaxCascades];
        private readonly Matrix4x4[] lightSpace = new Matrix4x4[MaxCascades];
        private readonly Vector3[] frustumCorners = new Vector3[8];
        private readonly float[] _cascadeRadius = new float[MaxCascades];
        private readonly float[] _splits = new float[MaxCascades];

        // Cascades drawn this frame and last frame: a cascade that sat out has no Hi-Z of last frame's casters
        private int _cascadeCount = MaxCascades;
        private int _drawnLastFrame;
        private int _shadowResolution = 2048;

        // A previous-frame VP that puts everything off the map: the shadow Hi-Z test then assumes visible
        private static readonly Matrix4x4 OffMapVP = new Matrix4x4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 1);
        
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

        /// <summary>
        /// This frame's sun shadow receivers (ShadowReceivers.Build) for the cascade culling's push constant slot 39;
        /// 0 = none: the culling tests nothing.
        /// </summary>
        public static uint CurrentShadowReceiversUav { get; private set; }
        
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
        /// This frame's view-projection per cascade (world space), as the cascade buffer holds them; valid during the
        /// shadow pass. The terrain tests its patches against the shadow receivers with them.
        /// </summary>
        public static Matrix4x4[]? CascadeViewProjections => _instance?._shadowVPMatrices;

        /// <summary>
        /// The cascades drawn this frame (EngineSettings.ShadowCascadeCount, at most MaxCascades): decided when the
        /// shadow pass starts; everything that reads the cascades later in the frame (terrain, grass, lighting, water,
        /// fog) takes this many.
        /// </summary>
        public static int CascadeCount => _instance?._cascadeCount ?? Math.Clamp(Engine.Settings.ShadowCascadeCount, 1, MaxCascades);

        /// <summary>
        /// This frame's cascades for engine_stats: view depth range, the bounding radius of its square and its texel
        /// size; null before the first shadow pass.
        /// </summary>
        public static object? DescribeCascades()
        {
            var light = _instance;
            if (light == null || light._drawnLastFrame == 0) return null;
            var result = new object[light._drawnLastFrame];
            for (int c = 0; c < result.Length; c++)
                result[c] = new
                {
                    near = MathF.Round(light.cascades[c].X, 2),
                    far = MathF.Round(light.cascades[c].Y, 1),
                    radius = MathF.Round(light._cascadeRadius[c], 1),
                    texelCm = MathF.Round(light._cascadeRadius[c] * 200f / light._shadowResolution, 1),
                };
            return new { adaptive = Engine.Settings.UseAdaptiveSplits, cascades = result };
        }

        private static DirectionalLight? _instance;

        /// <summary>
        /// Get the current directional light's forward direction.
        /// </summary>
        public static Vector3 GetLightDirection() => _instance?.Transform.Forward ?? Vector3.UnitY;

        /// <summary>The current directional light's <see cref="ShadowDirection"/>.</summary>
        public static Vector3 GetShadowDirection() => _instance?.ShadowDirection ?? Vector3.UnitY;

        /// <summary>
        /// The direction toward a light source (the sun or a moon) its shadows are cast from: the held
        /// <see cref="ShadowDirection"/> while the directional light is that source, else the source's own.
        /// </summary>
        public static Vector3 ShadowTowards(Vector3 towardSource)
        {
            var transform = _instance?.Transform;   // none while the scene that holds it is being built
            if (transform == null || Vector3.Dot(transform.Forward, towardSource) > -0.999999f)
                return towardSource;
            return -_instance!.ShadowDirection;
        }

        // The direction the shadows were last cast in (ShadowDirection)
        private Vector3 _shadowDirection;
        private bool _hasShadowDirection;

        /// <summary>
        /// The direction the shadows are cast in: the light's, held until the light has turned <see cref="ShadowStep"/>
        /// from it. Everything that draws or traces this light's shadow reads it (the cascades, SDSM, screen-space
        /// shadows, the near fog, the terrain's shadow map), so they all jump together; the shading reads Transform.Forward.
        /// </summary>
        public Vector3 ShadowDirection
        {
            get
            {
                if (Transform == null)
                    return _hasShadowDirection ? _shadowDirection : Vector3.UnitY;
                var forward = Transform.Forward;
                if (!_hasShadowDirection || Vector3.Dot(forward, _shadowDirection) < MathF.Cos(MathF.Max(ShadowStep, 0f) * (MathF.PI / 180f)))
                {
                    _shadowDirection = forward;
                    _hasShadowDirection = true;
                }
                return _shadowDirection;
            }
        }

        public DirectionalLight()
        {
            Material = new Material(new Effect("light_directional"));
            Params = new MaterialBlock();
            
            for (int i = 0; i < MaxCascades; i++)
                cascadeFrustumPlanes[i] = new Vector4[6];

            _instance = this;
        }

        public override void Destroy()
        {
            // The shadow passes of the frames in flight still read these: the cascade buffers through their
            // bindless SRVs, the planes cbuffer by address. Releasing them here risks removing the device
            // (see Animator.Destroy), so they go to the device to release once those frames are done.
            var device = Engine.Device;

            if (_cascadeBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    device.DeferDispose(_cascadeBuffers[i]);

            if (_vpOnlyBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    device.DeferDispose(_vpOnlyBuffers[i]);

            if (_shadowCascadeCBs != null)
                for (int i = 0; i < FrameCount; i++)
                    device.DeferDispose(_shadowCascadeCBs[i]);

            if (_gpuCascadeBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    device.DeferDispose(_gpuCascadeBuffers[i]);

            if (_shadowSceneConstantsBuffers != null)
                for (int i = 0; i < FrameCount; i++)
                    device.DeferDispose(_shadowSceneConstantsBuffers[i]);

            if (_instance == this)
                _instance = null;
        }

        // Called by DeferredRenderer via CommandBuffer
        public void Draw()
        {
            // Enqueue shadow rendering first (runs before light pass). Nothing queued = the renderer skips
            // its whole shadow pass (cascades, shadow Hi-Z, SDSM) and readers get no shadow map.
            if (CastShadows && Engine.Settings.EnableShadows && DeferredRenderer.Current?.ShadowTextureArray != null)
                CommandBuffer.Enqueue(RenderPass.ShadowMap, _drawShadowsAction ??= DrawShadows);

            CommandBuffer.Enqueue(RenderPass.Light, _drawLightAction ??= DrawLight);
        }

        /// <summary>
        /// Broadcast light parameters to all MasterEffects so forward-rendered shaders
        /// (transparent, glass) can read them from ObjectConstants without coupling.
        /// Called once per frame during DrawLight, after cascade setup is complete.
        /// </summary>
        private void BroadcastLightParams()
        {
            int cc = CascadeCount;
            foreach (var pair in Effect.MasterEffects)
            {
                pair.Value.SetParameter("LightColor", Color.ToVector3());
                pair.Value.SetParameter("LightDirection", Transform.Forward);
                pair.Value.SetParameter("LightIntensity", Intensity);
                pair.Value.SetParameterArray("LightSpaces", _lightSpaceParam);
                pair.Value.SetParameterArray("Cascades", _cascadesParam);
                pair.Value.SetParameter("CascadeCount", cc);
            }
        }

        // The first CascadeCount entries of lightSpace and cascades, as the arrays handed to the parameter
        // blocks (which keep the array, not a copy): refilled in place each frame. Slicing (lightSpace[..cc])
        // made two new arrays per effect per frame.
        private Matrix4x4[] _lightSpaceParam = [];
        private Vector4[] _cascadesParam = [];

        private void RefreshCascadeParams(int cc)
        {
            if (_lightSpaceParam.Length != cc)
            {
                _lightSpaceParam = new Matrix4x4[cc];
                _cascadesParam = new Vector4[cc];
            }
            Array.Copy(lightSpace, _lightSpaceParam, cc);
            Array.Copy(cascades, _cascadesParam, cc);
        }

        private Action<ID3D12GraphicsCommandList>? _drawShadowsAction, _drawLightAction;
        private static ID3D12DescriptorHeap[]? s_srvHeaps;

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
            RefreshCascadeParams(cc);
            Params.SetParameterArray("LightSpaces", _lightSpaceParam);
            Params.SetParameterArray("Cascades", _cascadesParam);
            Params.SetParameter("CascadeCount", cc);
            Params.SetParameter("DebugVisualizationMode", Engine.Settings.DebugVisualizationMode);
            
            // Broadcast light params to all effects so forward transparent shaders
            // can access them via ObjectConstants without explicit coupling
            BroadcastLightParams();
            
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
            commandList.SetDescriptorHeaps(1, s_srvHeaps ??= [Engine.Device.SrvHeap]);
            commandList.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            
            // Push constants via ComputeShader reflection (10 slots)
            var desc = renderer.LightBuffer.Native.Description;
            lightCS.SetPushConstant("NormalTex", renderer.Normals.BindlessIndex);
            lightCS.SetPushConstant("DepthTex", renderer.Depth.BindlessIndex);
            lightCS.SetPushConstant("ShadowMap", renderer.ShadowMapIndex);
            lightCS.SetPushConstant("DepthGBuf", renderer.DepthGBuffer.BindlessIndex);
            lightCS.SetPushConstant("AlbedoTex", renderer.Albedo.BindlessIndex);
            lightCS.SetPushConstant("DataTex", renderer.Data.BindlessIndex);
            lightCS.SetPushConstant("LightingCascadeSRV", renderer.ShadowMapIndex != 0 ? _activeLightingCascadeSrv : 0u);
            lightCS.SetPushConstant("OutputUAV", renderer.LightBuffer.UavIndex);
            lightCS.SetPushConstant("ScreenWidth", (uint)desc.Width);
            lightCS.SetPushConstant("ScreenHeight", (uint)desc.Height);
            lightCS.SetPushConstant("SSSTex", Engine.Settings.EnableScreenSpaceShadows ? (renderer.ScreenSpaceShadows?.OutputSrvIndex ?? 0u) : 0u);
            lightCS.SetPushConstant("SSDMTex", Engine.Settings.EnableSSDM ? (renderer.ScreenSpaceDisplacement?.OutputSrvIndex ?? 0u) : 0u);
            lightCS.SetPushConstant("GITex", 0u); // GI composed in separate CSComposeGI pass
            lightCS.SetPushConstant("SkyVisTex", renderer.SkyVisibilitySrv);  // 0: looked up in the shader
            
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
            CurrentShadowReceiversUav = 0;   // built below on the CPU-cascade path only
            var camera = Camera.Main;
            if (camera == null) return;
            
            var cameraPosition = camera.Position;
            var shadowTex = DeferredRenderer.Current!.ShadowTextureArray!;

            // Cascades 1+: casters spanning fewer texels than this are left out (clip space spans 2 units)
            float minCasterClip = MathF.Max(0f, Engine.Settings.ShadowMinCasterTexels) * 2f / shadowTex.Width;

            // The cascades this frame draws: everything later in the frame reads CascadeCount
            float near = camera.NearPlane;
            int numCascades = Math.Clamp(Engine.Settings.ShadowCascadeCount, 1, Math.Min(MaxCascades, shadowTex.Slices));
            _cascadeCount = numCascades;
            _shadowResolution = shadowTex.Width;

            // Check if GPU cascade compute is available and SDSM is active
            var culler = CommandBuffer.Culler;
            // TEMP: force CPU path to isolate GPU cascade compute bug
            // SDSM splits still made from ReadDepthDistribution() in the CPU path below
            bool useGpuCascades = false;
            // bool useGpuCascades = Engine.Settings.UseAdaptiveSplits 
            //     && culler?.CascadeComputeReady == true;
            
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
                    ShadowDirection,
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
                    
                    InstanceBatch.CullShadowAll(commandList, allBatches, cascadeSRV, cascadeCBAddr, shadowHiZSrv, culler, 4, minCasterClip);

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
                
                // Adaptive splits from the SDSM depth analysis (last frame's depth), else the fixed ones
                var distribution = Engine.Settings.UseAdaptiveSplits ? CommandBuffer.Culler?.ReadDepthDistribution() : null;
                if (distribution == null || !AdaptiveSplits(distribution, near, numCascades, _splits))
                    FixedSplits(CascadeSplits, near, numCascades, _splits);

                cascades[0] = new Vector4(near, _splits[0], 0, 0);
                for (int i = 1; i < numCascades; i++)
                    cascades[i] = new Vector4(_splits[i - 1], _splits[i], 0, 0);

                // Get cascade projection matrices
                for (int i = 0; i < numCascades; i++)
                    cascadeProjectionMatrices[i] = camera.GetProjectionMatrix(cascades[i].X, cascades[i].Y);

                // Set viewport for shadow map size
                commandList.RSSetViewport(new Viewport(0, 0, shadowTex.Width, shadowTex.Height));
                commandList.RSSetScissorRect(new RectI(0, 0, shadowTex.Width, shadowTex.Height));

                // Clear the cascades drawn this frame (a slice that sits out keeps last content, matching its Hi-Z)
                for (int i = 0; i < numCascades; i++)
                    commandList.ClearDepthStencilView(shadowTex.SliceDsvHandles[i], ClearFlags.Depth, 1.0f, 0);

                if (!CastShadows) return;

                // Draw each cascade using GPU-driven rendering
                // Phase 1: Compute cascade matrices and frustum planes
                for (int i = 0; i < numCascades; i++)
                    SetupCascade(i, camera, cameraPosition, shadowTex);
                
                // Phase 2: GPU-driven cull all cascades per batch (unified visibility pass)
                var allBatches = CommandBuffer.GetAllBatches(RenderPass.Opaque);
                int frameIndex = Engine.FrameIndex % FrameCount;
                
                if (culler != null && allBatches != null)
                {
                    EnsureCascadeBuffer();
                    UploadCascadeBuffer(frameIndex, camera);
                    _drawnLastFrame = numCascades;
                    uint cascadeSrv = _cascadeBufferSrvIndices![frameIndex];
                    uint vpOnlySrv = _vpOnlySrvIndices![frameIndex]; // TEMP
                    CurrentCascadeSrvIndex = cascadeSrv;
                    CurrentVPOnlySrvIndex = vpOnlySrv; // TEMP
                    
                    // Shadow Hi-Z SRV for occlusion culling (0 = disabled on first frame before pyramid is ready)
                    var shadowPyramid = DeferredRenderer.Current?.ShadowHiZPyramid;
                    uint shadowHiZSrv = (shadowPyramid?.Ready == true) ? shadowPyramid.FullSRV : 0;
                    
                    // Where this frame's visible surfaces sample each cascade (the opaque pass has written the linear
                    // depth): the culling tests every cascade entry against it (EngineSettings.ShadowReceiverCulling)
                    var renderer = DeferredRenderer.Current!;
                    CurrentShadowReceiversUav = renderer.ShadowReceivers?.Build(commandList, camera, renderer,
                        cascadeSrv, numCascades, shadowTex.Width) ?? 0;

                    // Cull all cascades for each batch using unified visibility pass
                    ulong cascadeCBAddr = _shadowCascadeCBs![frameIndex].Native.GPUVirtualAddress;
                    PixMarker.Begin(commandList, "Shadow culling");
                    InstanceBatch.CullShadowAll(commandList, allBatches, cascadeSrv, cascadeCBAddr, shadowHiZSrv, culler,
                        numCascades, minCasterClip,
                        casterStats: Engine.Settings.ShadowCasterStats ? renderer.ShadowReceivers : null);
                    PixMarker.End(commandList);

                    // Phase 3: True single-pass draw — 1 ExecuteIndirect per batch for all cascades
                    // Bind full-array DSV (all slices at once)
                    commandList.OMSetRenderTargets(0, (CpuDescriptorHandle[]?)null, false, shadowTex.FullArrayDsvHandle);

                    PixMarker.Begin(commandList, "Shadow draws");
                    foreach (var batch in allBatches)
                    {
                        if (!batch.Material.HasPass(RenderPass.Shadow)) continue;
                        batch.Material.SetPass(RenderPass.Shadow);
                        batch.Material.Apply(commandList, Engine.Device, null);

                        PixMarker.Begin(commandList, batch.ProfileName);
                        batch.DrawShadowSinglePass(commandList, cascadeSrv);
                        PixMarker.End(commandList);
                    }
                    PixMarker.End(commandList);

                    // Custom shadow actions (grass) — single-pass via AS/MS cascade expansion
                    // FullArrayDsvHandle is already bound from batch shadows above
                    PixMarker.Begin(commandList, "Terrain + decorators");
                    CommandBuffer.ExecuteCustomActions(RenderPass.Shadow, commandList);
                    PixMarker.End(commandList);
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
            radius = HoldRadius(index, radius);
            _cascadeRadius[index] = radius;

            float texelSize = (radius * 2.0f) / shadowTex.Width;
            
            //if (Engine.FrameIndex % 60 == 0)
            //    Debug.Log($"[SDSM] Cascade {index}: radius={radius:F1}, texelSize={texelSize:F4}, range={cascades[index].X:F1}-{cascades[index].Y:F1}");
            
            // Texel snapping: snap the frustum center to the shadow map's texel grid
            // so it stays fixed in the world as the camera moves, eliminating shimmer.
            // To avoid precision loss from adding large cameraPosition to small frustumCenter,
            // we only use the sub-texel remainder of the camera position. All arithmetic
            // stays in the small-number domain — no catastrophic cancellation.
            // The grid holds against the light's turning too while ShadowDirection is held (ShadowStep).
            var shadowDirection = ShadowDirection;
            var lightRotation = Matrix4x4.CreateLookAtLeftHanded(
                Vector3.Zero, shadowDirection, Vector3.UnitY);
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
                snappedCenter, snappedCenter + shadowDirection, Vector3.UnitY);
            
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
            
            // Inflate Z to catch shadow casters behind the camera, and at least MinCasterReach toward the sun: with adaptive
            // splits inside a small room all four cascades crowd into the few metres in view, and the last one reached
            // only about its own depth toward the sun. The ceiling and roof above lay beyond it, were clipped, and the
            // far corner of the room stood in full sun, coming and going as the camera turned (user, 2026-10-18).
            float zRange = maxZ - minZ;
            float backExtension = MathF.Max(zRange * MathF.Pow(3.5f, 3 - index), MinCasterReach);
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
        
        // The planes cbuffer (_shadowCascadeCBs, ShadowCascadePlanes in cull_instances.hlsl) is only bound because the
        // root signature wants slot 2: the kernels read the planes from this structured buffer.
        private unsafe void UploadCascadeBuffer(int frameIndex, Camera camera)
        {
            var dst = _cascadeBuffers![frameIndex].WritePtr<GPUCuller.CascadeData>();
            var vpDst = _vpOnlyBuffers![frameIndex].WritePtr<Matrix4x4>();
            int count = _cascadeCount;

            // Shadow LOD per cascade (EngineSettings.ShadowLodPerCascade, cull_instances.hlsl CascadeLodSteps): the distance
            // at which one of the camera's pixels spans one of the cascade's texels, per metre of texel (0 = off)
            float distancePerTexel = 0f;
            uint height = DeferredRenderer.Current?.DepthGBuffer?.Native.Description.Height ?? 0;
            if (Engine.Settings.ShadowLodPerCascade && camera != null && height > 0)
                distancePerTexel = height * MathF.Sqrt(camera.FoVFactor) * 0.5f;

            for (int c = 0; c < count; c++)
            {
                var data = new GPUCuller.CascadeData();
                data.SetPlanes(cascadeFrustumPlanes[c]);
                data.VP = _shadowVPMatrices[c];
                // The shadow Hi-Z holds last frame's casters: none for a cascade that was not drawn then
                data.PrevVP = c < _drawnLastFrame ? _prevShadowVPMatrices[c] : OffMapVP;
                // X = near, Y = far; Z = that LOD distance squared
                float lodDistance = _cascadeRadius[c] * 2f / _shadowResolution * distancePerTexel;
                data.SplitDistances = new Vector4(cascades[c].X, cascades[c].Y, lodDistance * lodDistance, cascades[c].W);
                dst[c] = data;
                vpDst[c] = _shadowVPMatrices[c];
            }

            // Swap current → previous for next frame
            Array.Copy(_shadowVPMatrices, _prevShadowVPMatrices, count);
        }

        // ── Cascade size ────────────────────────────────────────────────────────────

        // Steps of a cascade's square (radius), per octave: 2^(1/8), 9 % apart
        private const float RadiusStepsPerOctave = 8f;
        // A held square shrinks only when its slice fits one this many steps smaller
        private const float RadiusShrinkSteps = 2f;
        private readonly float[] _heldRadius = new float[MaxCascades];

        /// <summary>
        /// The square radius cascade <paramref name="index"/> uses for a slice that needs <paramref name="needed"/>: a
        /// step of a fixed ladder (2^(k/8) m), kept while the slice fits it and is not two steps smaller; grown at once
        /// when the slice needs more. SDSM moves the splits every frame, with a still camera too (swaying trees move
        /// the depth distribution), and a square that followed them changed its texel size every frame: the texel
        /// snapping then put the grid somewhere new each frame and the shadows crawled in cascades 1+ (2026-10-18).
        /// </summary>
        private float HoldRadius(int index, float needed)
        {
            float held = _heldRadius[index];
            float shrinkBelow = held * MathF.Pow(2f, -RadiusShrinkSteps / RadiusStepsPerOctave);
            if (needed <= held && needed > shrinkBelow)
                return held;
            float step = MathF.Ceiling(MathF.Log2(MathF.Max(needed, 0.01f)) * RadiusStepsPerOctave);
            held = MathF.Pow(2f, step / RadiusStepsPerOctave);
            _heldRadius[index] = held;
            return held;
        }

        // ── Splits ──────────────────────────────────────────────────────────────────

        /// <summary>
        /// SDSM splits for <paramref name="count"/> cascades from last frame's depth distribution: logarithmic between
        /// the depth of the nearest 1/32 of the pixels (a grass blade at the lens does not shrink cascade 0) and the
        /// farthest pixel the analysis looked at, so every cascade spans the same far/near ratio and its texels grow with
        /// distance as the pixels do. Until 2026-10-18 each inner split was the 40/70/90 % percentile of the pixels,
        /// capped at this: on a forest floor the last 10 % of the pixels reached from 125 m to 1.2 km and put every tree
        /// past 125 m in one cascade with 1.5 m texels, where their shadows vanished. False when there is no depth.
        /// </summary>
        private static bool AdaptiveSplits(GPUCuller.DepthDistribution d, float near, int count, float[] result)
        {
            if (d.Farthest <= 0f || d.Farthest - d.Nearest <= 0.0001f)
                return false;

            float floor = near + 0.1f;
            float logFrom = MathF.Max(d.Percentiles[0], floor);
            float logRatio = MathF.Max(d.Farthest, logFrom) / logFrom;
            for (int k = 1; k < count; k++)
                result[k - 1] = logFrom * MathF.Pow(logRatio, (float)k / count);
            result[count - 1] = MathF.Max(d.Farthest, floor);
            MakeIncreasing(result, count);
            return true;
        }

        /// <summary>
        /// The fixed splits (CascadeSplits) for <paramref name="count"/> cascades: as given when there are enough, else
        /// resampled along the curve they draw (log depth over k / N from the near plane), so the last split stays.
        /// </summary>
        private static void FixedSplits(float[] configured, float near, int count, float[] result)
        {
            int n = configured.Length;
            if (n == 0)
            {
                for (int k = 0; k < count; k++)
                    result[k] = near * MathF.Pow(4f, k + 1);
            }
            else if (n >= count)
            {
                Array.Copy(configured, result, count);
            }
            else
            {
                for (int k = 1; k <= count; k++)
                {
                    float x = (float)k / count * n;             // position on the configured curve, 0 = near plane
                    int i = Math.Min((int)x, n - 1);
                    float a = i == 0 ? MathF.Max(near, 0.01f) : configured[i - 1];
                    float b = configured[i];
                    result[k - 1] = a * MathF.Pow(b / a, x - i);
                }
            }
            MakeIncreasing(result, count);
        }

        private static void MakeIncreasing(float[] splits, int count)
        {
            for (int q = 1; q < count; q++)
                if (splits[q] <= splits[q - 1])
                    splits[q] = splits[q - 1] + 1f;
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
