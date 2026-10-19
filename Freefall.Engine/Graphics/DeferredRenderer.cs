using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    public class DeferredRenderer : RenderPipeline
    {
        private const int FrameCount = 3;
        
        public static DeferredRenderer Current { get; private set; } = null!;

        public RenderTexture2D Albedo;
        public RenderTexture2D Normals;
        public RenderTexture2D Data;
        public RenderTexture2D DepthGBuffer;  // Linear view-space depth (R32_Float), cleared to 0
        public RenderTexture2D EntityIdBuffer; // Per-pixel entity ID (R32_UInt) for mouse picking
        public RenderTexture2D HeightBuffer;   // Per-pixel displacement vectors for SSDM (RG16_Float)
        public DepthTexture2D Depth;
        
        public RenderTexture2D LightBuffer = null!;
        /// <summary>How much of the sky each opaque pixel sees past the terrain and the meshes (sky_visibility_cs.hlsl),
        /// R8; read by the directional light.</summary>
        public RenderTexture2D SkyVisibilityTex = null!;
        /// <summary>Its bindless SRV this frame, 0 when not drawn (the directional light then looks it up itself).</summary>
        public uint SkyVisibilitySrv { get; private set; }
        private ResourceStates _skyVisState = ResourceStates.Common;
        private ComputeShader _skyVisCS = null!;
        private int _kSkyVis;
        public RenderTexture2D Composite = null!;     // HDR intermediate (R16G16B16A16_Float)
        public RenderTexture2D LdrOutput = null!;     // Final tonemapped output (R8G8B8A8_UNorm)
        public RenderTexture2D? CompositeSnapshot;
        public DepthTextureArray2D ShadowTextureArray = null!;

        /// <summary>
        /// Bindless index of the shadow cascades rendered this frame, or 0 when none were (shadows switched off
        /// in the settings, or no light queued its cascades). Shaders skip shadowing for 0, so pass this rather
        /// than ShadowTextureArray.BindlessIndex, which would show the last cascades ever rendered.
        /// </summary>
        public uint ShadowMapIndex => _shadowsRendered ? ShadowTextureArray.BindlessIndex : 0u;
        private bool _shadowsRendered;

        // The shadow array is created once in DepthWrite and survives resizes, so its state is tracked here and
        // not through _isFirstFrame (which a resize resets)
        private ResourceStates _shadowMapState = ResourceStates.DepthWrite;

        /// <summary>Hi-Z depth pyramid, matched to this pipeline's depth buffer dimensions.</summary>
        public HiZPyramid? HiZPyramid { get; private set; }

        /// <summary>Shadow Hi-Z pyramid for occlusion culling shadow casters.</summary>
        public ShadowHiZPyramid? ShadowHiZPyramid { get; private set; }

        /// <summary>Screen-space shadow pass (Bend Studio technique).</summary>
        public ScreenSpaceShadows? ScreenSpaceShadows { get; private set; }

        /// <summary>The near fog volume (god rays): the air near the camera lit with the sun's shadows.</summary>
        public VolumetricFog? VolumetricFog { get; private set; }
        private uint _fogVolumeIdx;             // this frame's SceneConstants.FogVolumeIdx (0 = off)
        // This view's aerial perspective volume (the far haze), sized to its target: SceneConstants.AerialPerspectiveIdx
        private readonly AerialPerspectiveVolume _aerialPerspective = new();
        // This view's volumetric clouds (half size): SceneConstants.CloudTargetIdx
        private readonly CloudView _cloudView = new();
        private uint _cloudTargetIdx;
        private Matrix4x4 _relativeVP;          // this frame's view-projection without translation

        // Screen-space shadows of the highest ranked point lights (TracePointLightScreenShadows): a mask each, and
        // the registry slot it was traced for this frame (NoScreenShadow = none), for the point light pass
        private const int MaxScreenShadowedPointLights = 4;
        private const uint NoScreenShadow = uint.MaxValue;
        private readonly ScreenSpaceShadows?[] _pointScreenShadows = new ScreenSpaceShadows?[MaxScreenShadowedPointLights];
        private readonly uint[] _pointScreenShadowSlots = { NoScreenShadow, NoScreenShadow, NoScreenShadow, NoScreenShadow };
        private readonly uint[] _topPickSlots = new uint[MaxScreenShadowedPointLights];

        /// <summary>Ground truth ambient occlusion (spatial-only GTAO).</summary>
        public GTAO? Gtao { get; private set; }

        /// <summary>Screen-space displacement mapping (wavefront cooperative march).</summary>
        public ScreenSpaceDisplacement? ScreenSpaceDisplacement { get; private set; }

        /// <summary>SMAA 1x anti-aliasing (Jimenez et al. 2012).</summary>
        public SMAA? Smaa { get; private set; }

        /// <summary>Sparse 3D Radiance Cascades (world-space GI).</summary>
        public RadianceCascades? RadianceCascades { get; private set; }

        /// <summary>HDR Bloom (COD-style dual-filter pyramid).</summary>
        public BloomPyramid? BloomPyramid { get; private set; }

        private Material matClear = null!;
        private Material matDirectionalLight = null!;
        /// <summary>Directional light material — used by RC to borrow SceneConstants binding.</summary>
        internal Material DirectionalLightMaterial => matDirectionalLight;
        private ComputeShader _compositionCS = null!;
        private int _kCompose;
        public ComputeShader DirectionalLightCS { get; private set; } = null!;
        private int _kDirectionalLight;
        
        // Point lights: the GPU culls the light registry (PointLightBuffer) into this renderer's visible list after
        // the opaque pass; the shadow selection and the tiled lighting (light_point_cs.hlsl) read it
        private ComputeShader _pointLightCS = null!;
        private int _kPointLight;
        private ComputeShader _lightCullCS = null!;   // point_light_cull.hlsl
        private int _kCountLights, _kCullLights, _kOccludeLights;
        private GraphicsBuffer? _visibleLights;       // UAV: RWStructuredBuffer<PointLightData>[MaxVisible]
        private GraphicsBuffer? _visibleLightCount;   // UAV: [0] lights in the list, [1] lights in view (RWByteAddressBuffer)
        private GraphicsBuffer? _lightImportance;     // UAV: lights in view per importance band (which make the list)
        private GraphicsBuffer[]? _lightFrameParams;  // per-frame upload: PointLightFrame (camera, budget, fade)
        private uint _lightFrame;                     // this renderer's frames, for the shadow selection's state
        private bool _pointLightsCulled;

        /// <summary>Point light shadows: the local shadow atlas and the lights picked for it each frame.</summary>
        public LocalShadows? LocalShadows { get; private set; }

        /// <summary>Sun shadow receivers: this frame's visible surfaces per cascade tile, tested by the cascade culling.</summary>
        public ShadowReceivers? ShadowReceivers { get; private set; }

        /// <summary>
        /// This frame's visible point lights (RWStructuredBuffer&lt;PointLightData&gt; UAV) and their count
        /// (RWByteAddressBuffer UAV, [0]) once the light cull has run, else 0 and 0: the shadow receivers mark the sun
        /// samples the shadow selection takes around each.
        /// </summary>
        internal (uint lights, uint count) VisiblePointLights =>
            _pointLightsCulled ? (_visibleLights!.UavIndex, _visibleLightCount!.UavIndex) : (0u, 0u);

        private bool _isFirstFrame = true;
        private bool _compositeSnapshotFirstFrame = true;
        
        // Bloom compute shader and kernel indices
        private ComputeShader _bloomCS = null!;
        private int _kBloomThreshold;
        private int _kBloomDownsample;
        private int _kBloomUpsample;
        
        // Finalize compute shader (HDR → LDR: bloom + tonemap + gamma)
        private ComputeShader _finalizeCS = null!;
        private int _kFinalize;
        
        // Readback staging for picking (1x1 pixel each)
        private GraphicsBuffer? _entityIdReadback;
        private GraphicsBuffer? _depthReadback;
        private GraphicsBuffer? _normalReadback;
        private int _pickRequestX = -1;
        private int _pickRequestY = -1;
        private Animator? _pickSeeThrough;
        private bool _pickResultReady;
        private bool _pickCopyPending;
        
        // Saved pick coordinates for world position reconstruction
        private int _readbackPixelX;
        private int _readbackPixelY;
        
        // Cached array to avoid per-frame allocation
        private CpuDescriptorHandle[]? _cachedGBufferRtvHandles;
        
        // Frustum constant buffers (per-frame, uploaded once before batch loop)
        private GraphicsBuffer[]? _frustumConstantsBuffers;
        private bool _frustumBuffersInitialized;
        
        // Previous frame's ViewProjection — Hi-Z depth is 1 frame behind,
        // so occlusion must project using the VP that matches the depth data
        private System.Numerics.Matrix4x4 _previousFrameViewProjection;

        /// <summary>
        /// GPU address of this view's frustum + Hi-Z constants (GPUCuller.FrustumConstants), valid from the
        /// Opaque pass on. For compute passes that cull against the same frustum as the mesh culler.
        /// </summary>
        public ulong FrustumConstantsAddress { get; private set; }

        public DeferredRenderer()
        {
        }

        public override void Initialize(int width, int height)
        {
            // First renderer sets Current; Render() handles save/restore per viewport
            Current ??= this;

            // Create render textures first
            CreateRenderTextures(width, height);

            // Create Materials
            matClear = new Material(new Effect("cleargbuffer"));
            matDirectionalLight = new Material(new Effect("light_directional"));

            // Compile composition compute shader
            _compositionCS = new ComputeShader("composition_cs.hlsl");
            _kCompose = _compositionCS.FindKernel("CSCompose");

            // Compile directional light compute shader
            DirectionalLightCS = new ComputeShader("light_directional_cs.hlsl");
            _kDirectionalLight = DirectionalLightCS.FindKernel("CSDirectionalLight");

            // Sky visibility of the opaque surfaces (terrain + sky captures), for the directional light and composition
            _skyVisCS = new ComputeShader("sky_visibility_cs.hlsl");
            _kSkyVis = _skyVisCS.FindKernel("CSSkyVisibility");

            // Point lights: GPU cull of the registry, then tiled lighting on the visible list
            _pointLightCS = new ComputeShader("light_point_cs.hlsl");
            _kPointLight = _pointLightCS.FindKernel("CSPointLight");
            _lightCullCS = new ComputeShader("point_light_cull.hlsl");
            _kCountLights = _lightCullCS.FindKernel("CSCountLights");
            _kCullLights = _lightCullCS.FindKernel("CSCullLights");
            _kOccludeLights = _lightCullCS.FindKernel("CSOccludeLights");
            _visibleLights = GraphicsBuffer.CreateStructured<PointLightBuffer.VisibleLight>(PointLightBuffer.MaxVisible, uav: true);
            _visibleLightCount = GraphicsBuffer.CreateRaw(2, uav: true, clearable: true);
            _lightImportance = GraphicsBuffer.CreateStructured<uint>(64, uav: true);
            _lightFrameParams = new GraphicsBuffer[FrameCount];
            for (int i = 0; i < FrameCount; i++)
                _lightFrameParams[i] = GraphicsBuffer.CreateUpload<PointLightBuffer.FrameParams>(1, mapped: true);

            // Shadow Map Array (Cascades)
            ShadowTextureArray = new DepthTextureArray2D(2048, 2048, DirectionalLight.MaxCascades);

            // Point light shadows (own atlas, independent of the cascades)
            LocalShadows = new LocalShadows();

            // Where visible surfaces sample the cascades, for the cascade culling (DirectionalLight.DrawShadows)
            ShadowReceivers = new ShadowReceivers();

            // Hi-Z pyramid (matched to depth buffer size)
            HiZPyramid = new HiZPyramid();
            HiZPyramid.Create(Engine.Device, width, height);
            
            // Shadow Hi-Z pyramid (matched to shadow map size)
            ShadowHiZPyramid = new ShadowHiZPyramid();
            ShadowHiZPyramid.Create(Engine.Device, 2048, 2048, DirectionalLight.MaxCascades);
            
            // Screen-space shadows
            ScreenSpaceShadows = new ScreenSpaceShadows();

            // Near fog volume
            VolumetricFog = new VolumetricFog();

            // Ambient occlusion
            Gtao = new GTAO();

            // Screen-space displacement
            ScreenSpaceDisplacement = new ScreenSpaceDisplacement();

            // Radiance Cascades GI
            RadianceCascades = new RadianceCascades();
            RadianceCascades.Initialize(width, height);

            // SMAA anti-aliasing
            Smaa = new SMAA();

            // Bloom pyramid
            BloomPyramid = new BloomPyramid();
            BloomPyramid.Create(Engine.Device, width, height);
            _bloomCS = new ComputeShader("bloom_cs.hlsl");
            _kBloomThreshold = _bloomCS.FindKernel("CSBloomThreshold");
            _kBloomDownsample = _bloomCS.FindKernel("CSBloomDownsample");
            _kBloomUpsample = _bloomCS.FindKernel("CSBloomUpsample");

            // Finalize pass (HDR → LDR)
            _finalizeCS = new ComputeShader("finalize_cs.hlsl");
            _kFinalize = _finalizeCS.FindKernel("CSFinalize");
        }

        private void CreateRenderTextures(int width, int height)
        {
            Albedo = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float);
            Normals = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_SNorm);
            Data = new RenderTexture2D(Engine.Device, width, height, Format.R8G8B8A8_UNorm);
            DepthGBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R32_Float);
            EntityIdBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R32_UInt);
            HeightBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R16G16_Float);
            
            // 1x1 readback buffers for picking
            _entityIdReadback?.Dispose();
            _entityIdReadback = GraphicsBuffer.CreateReadback<uint>(1);
            _depthReadback?.Dispose();
            _depthReadback = GraphicsBuffer.CreateReadback<float>(1);
            _normalReadback?.Dispose();
            _normalReadback = GraphicsBuffer.CreateReadback<ulong>(1); // 4 × int16 = 8 bytes
            _cachedGBufferRtvHandles = null; // force re-creation
            
            // Depth Texture
            Depth = new DepthTexture2D(width, height, Format.D32_Float, true);

            LightBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float, randomWrite: true);
            SkyVisibilityTex = new RenderTexture2D(Engine.Device, width, height, Format.R8_UNorm, randomWrite: true);
            _skyVisState = ResourceStates.Common;
            Composite = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float, randomWrite: true);
            LdrOutput = new RenderTexture2D(Engine.Device, width, height, Format.R8G8B8A8_UNorm, randomWrite: true);
        }

        public override void Resize(int width, int height)
        {
            Albedo?.Dispose();
            Normals?.Dispose();
            Data?.Dispose();
            DepthGBuffer?.Dispose();
            EntityIdBuffer?.Dispose();
            HeightBuffer?.Dispose();
            Depth?.Dispose();
            LightBuffer?.Dispose();
            SkyVisibilityTex?.Dispose();
            Composite?.Dispose();
            LdrOutput?.Dispose();

            CreateRenderTextures(width, height);
            _isFirstFrame = true; // Resource states reset to Common/Texture
            _compositeSnapshotFirstFrame = true;
            CompositeSnapshot?.Dispose();
            CompositeSnapshot = null;

            
            // Recreate Hi-Z pyramid to match new depth buffer dimensions
            HiZPyramid?.Dispose();
            HiZPyramid = new HiZPyramid();
            HiZPyramid.Create(Engine.Device, width, height);

            // Resize RC GI buffer
            RadianceCascades?.Resize(width, height);

            // Resize bloom pyramid
            BloomPyramid?.Dispose();
            BloomPyramid = new BloomPyramid();
            BloomPyramid.Create(Engine.Device, width, height);
        }

        public override void Clear(Camera camera)
        {
            // Clear Render Targets logic? 
            // In DX12, we Clear via CommandList
            // We'll rely on the beginning of Render frame to clear
        }

        private void EnsureFrustumBuffers(GraphicsDevice device)
        {
            if (_frustumBuffersInitialized) return;
            
            _frustumConstantsBuffers = new GraphicsBuffer[FrameCount];
            for (int i = 0; i < FrameCount; i++)
                _frustumConstantsBuffers[i] = GraphicsBuffer.CreateConstantBuffer<GPUCuller.FrustumConstants>();
            
            _frustumBuffersInitialized = true;
        }
        
        /// <summary>
        /// Upload frustum planes + Hi-Z parameters to a GPU constant buffer.
        /// Returns the GPU virtual address for the constant buffer.
        /// </summary>
        private ulong UploadFrustumConstants(Camera camera, GraphicsDevice device)
        {
            EnsureFrustumBuffers(device);
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            // Build frustum from camera (or frozen VP)
            var vpMatrix = Engine.Settings.FreezeFrustum 
                ? Engine.Settings.FrozenViewProjection 
                : camera.ViewProjection;
            var frustum = new Frustum(vpMatrix);
            var frustumPlanes = frustum.GetPlanesAsVector4();
            
            var constants = new GPUCuller.FrustumConstants
            {
                Plane0 = frustumPlanes[0],
                Plane1 = frustumPlanes[1],
                Plane2 = frustumPlanes[2],
                Plane3 = frustumPlanes[3],
                Plane4 = frustumPlanes[4],
                Plane5 = frustumPlanes[5],
            };
            
            // Add Hi-Z occlusion data if available and pyramid has been generated
            var culler = CommandBuffer.Culler;
            if (culler != null && HiZPyramid != null && HiZPyramid.FullSRV != 0 
                && HiZPyramid.Ready && !Engine.Settings.DisableHiZ)
            {
                // Hi-Z depth is from the previous frame, so project spheres
                // using the previous frame's VP to match the depth data
                constants.OcclusionProjection = Engine.Settings.FreezeFrustum
                    ? Engine.Settings.FrozenViewProjection
                    : _previousFrameViewProjection;
                constants.HiZSrvIdx = HiZPyramid.FullSRV;
                constants.HiZWidth = HiZPyramid.Width;
                constants.HiZHeight = HiZPyramid.Height;
                constants.HiZMipCount = (uint)HiZPyramid.MipCount;
                constants.NearPlane = camera.NearPlane;
                constants.ProjScale = camera.Projection.M22; // cot(fovY/2) — correct for sphere→screen
            }
            
            // Debug visualization mode
            constants.DebugMode = (uint)Engine.Settings.DebugVisualizationMode;
            
            // Cull stats UAV (always set if culler is initialized)
            if (culler != null)
                constants.CullStatsUAVIdx = culler.CullStatsUAV;
            
            // Camera position + sort direction for GPU depth sorting (opaque = front-to-back)
            constants.CameraPosition = camera.Position;
            constants.SortDirection = 0; // Front-to-back for opaque
            
            // Store current VP for next frame's occlusion projection
            _previousFrameViewProjection = camera.ViewProjection;

            unsafe
            {
                *_frustumConstantsBuffers![frameIndex].WritePtr<GPUCuller.FrustumConstants>() = constants;
            }

            FrustumConstantsAddress = _frustumConstantsBuffers![frameIndex].Native.GPUVirtualAddress;
            return FrustumConstantsAddress;
        }

        // CPU time of recording each pass (what the GPU spends on it is not measured here)
        private static readonly ProfilerMarker _gbufferMarker = ProfilerMarker.Get("G-buffer");
        private static readonly ProfilerMarker _drawSourcesMarker = ProfilerMarker.Get("Draw sources");
        private static readonly ProfilerMarker _transformUploadMarker = ProfilerMarker.Get("Transform upload");
        private static readonly ProfilerMarker _opaqueMarker = ProfilerMarker.Get("Opaque");
        private static readonly ProfilerMarker _shadowsMarker = ProfilerMarker.Get("Shadows");
        private static readonly ProfilerMarker _skyMarker = ProfilerMarker.Get("Sky");
        private static readonly ProfilerMarker _transparentMarker = ProfilerMarker.Get("Transparent");
        private static readonly ProfilerMarker _lightingMarker = ProfilerMarker.Get("Lighting");
        private static readonly ProfilerMarker _composeMarker = ProfilerMarker.Get("Compose");
        private static readonly ProfilerMarker _forwardMarker = ProfilerMarker.Get("Forward");
        private static readonly ProfilerMarker _postMarker = ProfilerMarker.Get("Post-processing");
        private static readonly ProfilerMarker _blitMarker = ProfilerMarker.Get("Blit");

        public override void Render(Camera camera, ID3D12GraphicsCommandList list)
        {
            // Set Current for duration of this render so DirectionalLight/other components
            // can access G-buffer textures (essential for non-primary editor viewports)
            var previousCurrent = Current;
            Current = this;

            PixMarker.Begin(list, "Deferred renderer");

            // Ensure Materials buffer is bound (creates SRV for bindless access)
            Material.BindMaterialsBuffer(list, Engine.Device);

            // Per-object effect data (struct Instance) that changed since the last frame
            InstanceData.Upload(Engine.Device);

            // This frame's exposure (once per tick: the metering read back from an earlier frame), before any pass
            // pre-exposes its light with it
            AutoExposure.Update();

            // 1. G-Buffer Pass (includes shadow rendering internally)
            using (_gbufferMarker.Auto())
                FillGBuffer(camera, list);

            // 2. Light Pass
            using (_lightingMarker.Auto())
                FillLightBuffer(camera, list);

            // 3. Composition (HDR — no tonemapping)
            using (_composeMarker.Auto())
                Compose(camera, list);

            // 4. Forward pass — renders directly to HDR Composite with depth read.
            //    Depth is already in PixelShaderResource after FillGBuffer.
            //    Composite is still in RenderTarget from Compose.
            using (_forwardMarker.Auto())
                RenderForward(camera, list);

            // Auto exposure metering of the finished HDR image (deferred + forward + sky), before bloom
            var compositeDesc = Composite.Native.Description;
            AutoExposure.Measure(list, Composite.BindlessIndex, (int)compositeDesc.Width, (int)compositeDesc.Height, Atmosphere.Exposure);

            using (_postMarker.Auto())
            {
                // 5. Bloom (reads HDR Composite — now includes both deferred + forward)
                bool bloom = BloomPyramid != null && Engine.Settings.EnableBloom;
                if (bloom)
                {
                    // Depth to NonPixelShaderResource: the bloom prefilter masks the sky with it, and so does
                    // the finalize pass when the bloom spreads light
                    Transition(list, Depth.Native, ResourceStates.PixelShaderResource, ResourceStates.NonPixelShaderResource);

                    PixMarker.Begin(list, "Bloom");
                    DispatchBloom(list);
                    PixMarker.End(list);
                }

                // 6. Finalize (HDR → LDR: bloom + tonemap + gamma + dither)
                PixMarker.Begin(list, "Finalize");
                Finalize(list);
                PixMarker.End(list);

                // Restore depth state for next frame
                if (bloom)
                    Transition(list, Depth.Native, ResourceStates.NonPixelShaderResource, ResourceStates.PixelShaderResource);

                // 7. SMAA Anti-Aliasing (optional, on LDR output)
                if (Smaa != null && Engine.Settings.EnableSMAA)
                {
                    PixMarker.Begin(list, "SMAA");
                    var desc = LdrOutput.Native.Description;
                    Smaa.Execute(list, LdrOutput.Native, LdrOutput.BindlessIndex, (int)desc.Width, (int)desc.Height);
                    PixMarker.End(list);
                }
            }

            // 8. Blit to Backbuffer
            using (_blitMarker.Auto())
                BlitToBackBuffer(camera, list);

            PixMarker.End(list); // Deferred renderer

            _isFirstFrame = false;

            // Restore previous Current (stack-safe for nested viewports)
            Current = previousCurrent;
        }
        
        private void Transition(ID3D12GraphicsCommandList list, ID3D12Resource res, ResourceStates before, ResourceStates after)
        {
            if (before == after) return;
            list.ResourceBarrierTransition(res, before, after);
        }

        private void FillGBuffer(Camera camera, ID3D12GraphicsCommandList list)
        {
             // A pick asked for this frame that sees through a character: its skin leaves that pixel open.
             // Read here, not in RequestPick: the bone buffers' indices are those of the frame being drawn.
             bool pickHole = _pickRequestX >= 0 && _pickSeeThrough != null;
             camera.PickHolePixel = pickHole ? new Vector2(_pickRequestX + 0.5f, _pickRequestY + 0.5f) : Vector2.Zero;
             camera.PickHoleBones = pickHole ? _pickSeeThrough!.BoneBufferIndices() : default;

             // Set shader parameters globally on all effects (Apex pattern)
             camera.SetShaderParams();

             // Zero-translation CameraInverse: even though GBuffer depth was written with full View,
             // NDC = (worldPos - camPos) × R × P, so inverse(R × P) correctly gives camera-relative pos.
             // Set before anything is drawn: the atmosphere's aerial perspective volume needs it too.
             var relativeVP = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
             Matrix4x4.Invert(relativeVP, out var cameraInverse);

             // The near fog volume this view's passes read (filled after the shadows, 2e below, before anything reads it):
             // its index stays the same from frame to frame, so it is set here with the camera, not mid-frame
             var compositeSize = Composite.Native.Description;
             _fogVolumeIdx = VolumetricFog != null && Engine.Settings.Fog && Engine.Settings.VolumetricFog
                 ? VolumetricFog.Prepare((int)compositeSize.Width, (int)compositeSize.Height) : 0u;
             _relativeVP = relativeVP;
             // ... and this view's aerial perspective volume, sized to it (filled by Atmosphere.Render just below)
             uint aerialPerspectiveIdx = _aerialPerspective.Prepare((int)compositeSize.Width, (int)compositeSize.Height);
             // ... and its volumetric clouds (marched in 2g below, before the sky reads them; 0 = the 2D layer)
             _cloudTargetIdx = _cloudView.Prepare((int)compositeSize.Width, (int)compositeSize.Height);

             foreach (var pair in Effect.MasterEffects)
             {
                 pair.Value.SetParameter("CameraInverse", cameraInverse);
                 pair.Value.SetParameter("CameraRelativeVP", relativeVP);
                 pair.Value.SetParameter("FogVolumeIdx", _fogVolumeIdx);
                 pair.Value.SetParameter("FogVolumeRange", VolumetricFog?.Range ?? 0f);
                 pair.Value.SetParameter("AerialPerspectiveIdx", aerialPerspectiveIdx);
                 pair.Value.SetParameter("CloudTargetIdx", _cloudTargetIdx);
             }

             // The atmosphere's LUTs for this view (sky-view, aerial perspective, ambient): the sky dome, the
             // lighting, the composition and the forward passes all read them
             Atmosphere.Render(list, BindSceneConstantsForCompute, _aerialPerspective);

             PixMarker.Begin(list, "GBuffer");
             // First frame: textures start in Common state; subsequent frames: PixelShaderResource
             var fromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
             var depthFromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
             
             Transition(list, Albedo.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, Normals.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, Data.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, DepthGBuffer.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, EntityIdBuffer.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, HeightBuffer.Native, fromState, ResourceStates.RenderTarget);
             Transition(list, Depth.Native, depthFromState, ResourceStates.DepthWrite);

             // Run the RenderGroup: IDraw components (ScriptDrawSystem) and draw systems enqueue their commands.
             // This enqueues DrawShadows callback into ShadowMap pass AND opaque batches
             Systems.RunDraw();
             // Components with GPU-resident draws (no Draw()) update their registrations here
             using (_drawSourcesMarker.Auto())
                 CommandBuffer.RefreshDrawSources();

             // Upload transforms AFTER Draw() so per-frame SetTransform() calls
             // (e.g. terrain patch pool-slot reassignment) are captured before GPU reads
             using (_transformUploadMarker.Auto())
                 TransformBuffer.Instance?.Upload();

             // === GBuffer Draw ===
             // Cache GBuffer RTV handles to avoid per-frame allocation
             _cachedGBufferRtvHandles ??= new CpuDescriptorHandle[6];
             _cachedGBufferRtvHandles[0] = Albedo.RtvHandle;
             _cachedGBufferRtvHandles[1] = Normals.RtvHandle;
             _cachedGBufferRtvHandles[2] = Data.RtvHandle;
             _cachedGBufferRtvHandles[3] = DepthGBuffer.RtvHandle;
             _cachedGBufferRtvHandles[4] = EntityIdBuffer.RtvHandle;
             _cachedGBufferRtvHandles[5] = HeightBuffer.RtvHandle;
             list.OMSetRenderTargets(_cachedGBufferRtvHandles, Depth.DsvHandle);
             
             list.ClearRenderTargetView(Albedo.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(Normals.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(Data.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(DepthGBuffer.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(EntityIdBuffer.RtvHandle, new Color4(0,0,0,0));
             list.ClearRenderTargetView(HeightBuffer.RtvHandle, new Color4(0,0,0,0));
             list.ClearDepthStencilView(Depth.DsvHandle, ClearFlags.Depth, 0.0f, 0); // Reverse depth: far=0

             // Viewport
             var vp = new Viewport(0, 0, Albedo.Native.Description.Width, Albedo.Native.Description.Height);
             var scissor = new RectI(0, 0, (int)Albedo.Native.Description.Width, (int)Albedo.Native.Description.Height);
             
             list.RSSetViewport(vp);
             list.RSSetScissorRect(scissor);

             // 1. Opaque pass FIRST — builds InstanceBatch GPU buffers
             using (_opaqueMarker.Auto())
             {
                 PixMarker.Begin(list, "Opaque");
                 ulong frustumGpuAddr = UploadFrustumConstants(camera, Engine.Device);
                 CommandBuffer.Execute(RenderPass.Opaque, list, Engine.Device, frustumGpuAddr);
                 PixMarker.End(list); // Opaque
             }

             // Point lights: the GPU culls the light registry into this frame's visible list. Before the shadow pass,
             // whose receiver culling keeps the sun samples the shadow selection (2b) takes around each visible light.
             _pointLightsCulled = CullPointLights(camera, list);

             // 2. Shadow pass — uses opaque batches (now built) for CullShadow/DrawShadow
             //    Must run AFTER Opaque so InstanceBatch GPU state is valid.
             //    Only when shadows are on and a light queued its cascades (DirectionalLight.Draw, above);
             //    otherwise readers get ShadowMapIndex 0 and the grass/terrain shadow actions are dropped
             //    with the rest of the frame's queues (CommandBuffer.Clear).
             _shadowsRendered = false;
             if (ShadowTextureArray != null && Engine.Settings.EnableShadows
                 && CommandBuffer.HasPendingCommands(RenderPass.ShadowMap))
             {
                 using var _ = _shadowsMarker.Auto();
                 PixMarker.Begin(list, "Shadows");
                 // Transition shadow texture to DepthWrite for rendering
                 if (_shadowMapState != ResourceStates.DepthWrite)
                 {
                     list.ResourceBarrierTransition(ShadowTextureArray.Native,
                         _shadowMapState, ResourceStates.DepthWrite);
                 }

                 CommandBuffer.Execute(RenderPass.ShadowMap, list, Engine.Device);
                 
                 // Generate shadow Hi-Z pyramid from this frame's shadow depth
                 if (ShadowHiZPyramid != null && CommandBuffer.Culler != null)
                 {
                     PixMarker.Begin(list, "Shadow Hi-Z");
                     // Merge barriers: skip intermediate PixelShaderResource state
                     list.ResourceBarrierTransition(ShadowTextureArray.Native,
                         ResourceStates.DepthWrite, ResourceStates.NonPixelShaderResource);
                     
                     CommandBuffer.Culler.GenerateShadowHiZPyramid(
                         list, ShadowTextureArray.BindlessIndex, ShadowHiZPyramid, DirectionalLight.CascadeCount);
                     
                     // Transition to PixelShaderResource for light pass sampling
                     list.ResourceBarrierTransition(ShadowTextureArray.Native,
                         ResourceStates.NonPixelShaderResource, ResourceStates.PixelShaderResource);
                     PixMarker.End(list); // Shadow Hi-Z
                 }
                 else
                 {
                     // No Hi-Z — transition directly to PixelShaderResource for light pass
                     list.ResourceBarrierTransition(ShadowTextureArray.Native,
                         ResourceStates.DepthWrite, ResourceStates.PixelShaderResource);
                 }
                 _shadowMapState = ResourceStates.PixelShaderResource;
                 _shadowsRendered = true;

                 // Shadow pass changes render targets and viewport for depth rendering.
                 // Restore GBuffer state before Sky pass.
                 // Note: camera.SetShaderParams() is NOT needed here because the shadow pass
                 // now uses a dedicated SceneConstants CBV instead of modifying MasterEffects.
                 list.OMSetRenderTargets(_cachedGBufferRtvHandles, Depth.DsvHandle);
                 list.RSSetViewport(vp);
                 list.RSSetScissorRect(scissor);
                 PixMarker.End(list); // Shadows
             }

             // 2b. Point lights: picks the shadowed ones out of the visible list (culled above) and renders their faces
             //     into their own atlas (independent of the sun's cascades). No point light shadows in daylight.
             if (LocalShadows != null)
             {
                 if (_pointLightsCulled && Engine.Settings.EnableShadows && Engine.Settings.MaxShadowedPointLights > 0)
                 {
                     var frameParams = _lightFrameParams![Engine.FrameIndex % FrameCount];

                     // The selection reads the linear depth (light centres hidden behind walls) and the daylight at each
                     // light judged last frame (2d: lights the daylight drowns get no shadow)
                     Transition(list, DepthGBuffer.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
                     LocalShadows.Select(list, camera, frameParams.SrvIndex, _visibleLights!.UavIndex,
                         _visibleLightCount!, DepthGBuffer.BindlessIndex, PointLightBuffer.Instance!.LampStatesUav(list));
                     Transition(list, DepthGBuffer.Native, ResourceStates.NonPixelShaderResource, ResourceStates.RenderTarget);

                     if (LocalShadows.Render(list, frameParams.Native.GPUVirtualAddress))
                     {
                         list.OMSetRenderTargets(_cachedGBufferRtvHandles, Depth.DsvHandle);
                         list.RSSetViewport(vp);
                         list.RSSetScissorRect(scissor);
                     }
                 }
                 else
                 {
                     LocalShadows.Skip();
                 }
             }

             // 2c. Sky captures: the meshes round the camera seen from directions of the sky, for the sky visibility
             //     (sky_capture.hlsli). Four directions per frame while any is due; after the shadows, whose per-batch
             //     culling buffers it reuses.
             if (SkyCapture.Render(list, camera))
             {
                 list.OMSetRenderTargets(_cachedGBufferRtvHandles, Depth.DsvHandle);
                 list.RSSetViewport(vp);
                 list.RSSetScissorRect(scissor);
             }

             // 2d. Lamps: the daylight where each light hangs past the terrain and the meshes, and whether each light
             //     marked OnlyWhenDark is lit by it (point_light_lamps.hlsl); after the captures it reads, for next frame's
             //     cull and shadow selection. Once per frame for all renderers.
             if (_pointLightsCulled)
                 PointLightBuffer.Instance?.UpdateLamps(list, _lightFrameParams![Engine.FrameIndex % FrameCount].SrvIndex);

             // 2e. Near fog volume: the air near the camera lit with this frame's cascades and sky captures, before the
             //     transparent pass, the composition and the forward passes read it (sky_common.fx AtmosphereFog)
             if (_fogVolumeIdx != 0)
                 VolumetricFog!.Render(list, camera.Position, _relativeVP, ShadowMapIndex, BindSceneConstantsForCompute);

             // 2f. Exposure meter: the light at the camera past the terrain, the sky captures and the clouds, and the
             //     lamps near it, for the auto exposure (read back a few frames later)
             AutoExposure.MeterIncident(list, camera.Position, BindSceneConstantsForCompute);

             // 2g. Volumetric clouds: this view's march through the field baked in Atmosphere.Render, before the sky pass
             //     draws them (mesh_skybox.fx, SceneConstants.CloudTargetIdx)
             if (_cloudTargetIdx != 0)
                 _cloudView.Render(list, camera.Position, _relativeVP, BindSceneConstantsForCompute);

             // 3. Sky pass
             using (_skyMarker.Auto())
             {
                 PixMarker.Begin(list, "Sky");
                 CommandBuffer.Execute(RenderPass.Sky, list, Engine.Device);
                 PixMarker.End(list); // Sky
             }

             // 4. Transparent pass — renders after opaque+sky, can depth-test against both
             using (_transparentMarker.Auto())
             {
                 PixMarker.Begin(list, "Transparent");
                 CommandBuffer.Execute(RenderPass.Transparent, list, Engine.Device);
                 PixMarker.End(list); // Transparent
             }

             // End GBuffer
             Transition(list, Albedo.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
             Transition(list, Normals.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
             Transition(list, Data.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);

             // Entity ID readback: promote pending copy from previous frame (GPU has finished it by now)
             if (_pickCopyPending)
             {
                 _pickResultReady = true;
                 _pickCopyPending = false;
             }

             // EntityIdBuffer stays in RenderTarget — Forward pass will also write to it.
             // Transition happens in RenderForward after all draws + readback.
             
             // DepthGBuffer: transition to NonPixelShaderResource for Hi-Z compute pass
             Transition(list, DepthGBuffer.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
             
             // HeightBuffer: transition to SRV for displacement compute pass
             Transition(list, HeightBuffer.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
             
             // Generate Hi-Z depth pyramid from LINEAR GBuffer depth (not hardware depth)
             // Linear depth cleared to 0 = no occluder, max() naturally keeps farthest geometry
             // Skip when frustum is frozen — frozen pyramid must match frozen VP for correct occlusion
             if (!Engine.Settings.FreezeFrustum)
             {
                 PixMarker.Begin(list, "Hi-Z Pyramid");
                 CommandBuffer.Culler?.GenerateHiZPyramid(list, DepthGBuffer.BindlessIndex, HiZPyramid);
                 PixMarker.End(list); // Hi-Z Pyramid
             }
             
             // SDSM: Analyze depth buffer for adaptive cascade splits
             // DepthGBuffer is still in NonPixelShaderResource — perfect for compute SRV reads
             // Only feeds adaptive cascade splits: skipped while no shadows are rendered or the splits are fixed
             // (DirectionalLight reads it only with UseAdaptiveSplits; 0.24 ms at 2176×1249 spent for nothing before)
             if (_shadowsRendered && Engine.Settings.UseAdaptiveSplits && CommandBuffer.Culler?.SdsmReady == true)
             {
                 PixMarker.Begin(list, "SDSM Depth Analysis");
                 int depthWidth = (int)DepthGBuffer.Native.Description.Width;
                 int depthHeight = (int)DepthGBuffer.Native.Description.Height;
                 CommandBuffer.Culler.AnalyzeDepth(list, DepthGBuffer.BindlessIndex,
                     depthWidth, depthHeight, camera, -DirectionalLight.GetLightDirection(), Engine.Settings.MaxShadowDistance);
                 PixMarker.End(list); // SDSM Depth Analysis
             }
             
             // Screen-space displacement mapping: uses height buffer + depth buffer.
             // Produces displaced depth (GBuffer depth remapped through B buffer).
             // Must run BEFORE SSS so shadows see the displaced surface.
             if (ScreenSpaceDisplacement != null && Engine.Settings.EnableSSDM)
             {
                 PixMarker.Begin(list, "Screen-Space Displacement");
                 int sdw = (int)HeightBuffer.Native.Description.Width;
                 int sdh = (int)HeightBuffer.Native.Description.Height;
                 var scvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
                 ScreenSpaceDisplacement.Execute(list, HeightBuffer.BindlessIndex, 
                     DepthGBuffer.BindlessIndex, sdw, sdh, scvp);
                 PixMarker.End(list);
             }

             // Screen-space shadows: ray-march against GBuffer linear depth (includes heightmap bias)
             // Must execute before DepthGBuffer transitions to PixelShaderResource (SSS is compute)
             if (ScreenSpaceShadows != null && Engine.Settings.EnableScreenSpaceShadows)
             {
                 PixMarker.Begin(list, "Screen-Space Shadows");
                 var cvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
                 var lightDir = -Components.DirectionalLight.GetLightDirection();
                 int dw = (int)DepthGBuffer.Native.Description.Width;
                 int dh = (int)DepthGBuffer.Native.Description.Height;
                 ScreenSpaceShadows.Execute(list, DepthGBuffer.BindlessIndex, dw, dh, lightDir, cvp, camera.NearPlane);
                 PixMarker.End(list);

                 TracePointLightScreenShadows(camera, list, cvp, dw, dh);
             }
             else
                 Array.Fill(_pointScreenShadowSlots, NoScreenShadow);

             // Ambient occlusion: horizon search against GBuffer linear depth + world normals.
             // Consumed by composition (ambient term only).
             if (Gtao != null && Engine.Settings.EnableGTAO)
             {
                 PixMarker.Begin(list, "GTAO");
                 var viewRotation = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up);
                 int aw = (int)DepthGBuffer.Native.Description.Width;
                 int ah = (int)DepthGBuffer.Native.Description.Height;
                 Gtao.Execute(list, DepthGBuffer.BindlessIndex, Normals.BindlessIndex, aw, ah, viewRotation, camera.Projection);
                 PixMarker.End(list);
             }

             // Transition GBuffer depth to PixelShaderResource for light pass sampling
             Transition(list, DepthGBuffer.Native, ResourceStates.NonPixelShaderResource, ResourceStates.PixelShaderResource);

             // Hardware depth to PixelShaderResource for light pass
             Transition(list, Depth.Native, ResourceStates.DepthWrite, ResourceStates.PixelShaderResource);
             
             PixMarker.End(list); // GBuffer
        }

        private void FillLightBuffer(Camera camera, ID3D12GraphicsCommandList list)
        {
             PixMarker.Begin(list, "Lighting");

             // CameraInverse / CameraRelativeVP: set in FillGBuffer, before the atmosphere pass

             // Radiance Cascades GI — runs while LightBuffer is still in SRV state
             // so it can read last frame's lighting for bounce radiance estimation.
             if (RadianceCascades != null && Engine.Settings.EnableRadianceCascades)
             {
                 RadianceCascades.Execute(list, camera, this);
             }

             // NOW transition LightBuffer to UAV for directional light (after RC has read it as SRV)
             var fromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
             Transition(list, LightBuffer.Native, fromState, ResourceStates.UnorderedAccess);

             // How much sky each opaque pixel sees past the terrain and the meshes, for the directional light below
             SkyVisibilitySrv = 0;
             if (Engine.Settings.SkyVisibility)
             {
                 PixMarker.Begin(list, "Sky visibility");
                 if (_skyVisState != ResourceStates.UnorderedAccess)
                     Transition(list, SkyVisibilityTex.Native, _skyVisState, ResourceStates.UnorderedAccess);
                 var size = SkyVisibilityTex.Native.Description;
                 _skyVisCS.SetPushConstant("NormalTex", Normals.BindlessIndex);
                 _skyVisCS.SetPushConstant("DepthTex", Depth.BindlessIndex);
                 _skyVisCS.SetPushConstant("OutputUAV", SkyVisibilityTex.UavIndex);
                 _skyVisCS.SetPushConstant("ScreenWidth", (uint)size.Width);
                 _skyVisCS.SetPushConstant("ScreenHeight", (uint)size.Height);
                 list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
                 list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
                 BindSceneConstantsForCompute(list);
                 _skyVisCS.Dispatch(_kSkyVis, list, ((uint)size.Width + 7) / 8, ((uint)size.Height + 7) / 8);
                 Transition(list, SkyVisibilityTex.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
                 _skyVisState = ResourceStates.NonPixelShaderResource;
                 SkyVisibilitySrv = SkyVisibilityTex.BindlessIndex;
                 PixMarker.End(list);
             }

             // Execute custom Light actions (directional light compute dispatch writes to UAV)
             PixMarker.Begin(list, "Directional Light");
             CommandBuffer.ExecuteCustomActions(RenderPass.Light, list);
             CommandBuffer.ClearCustomActions(RenderPass.Light);
             PixMarker.End(list); // Directional Light
             
             // UAV barrier between directional and point light dispatches
             list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(LightBuffer.Native)));
             
             // Point light compute dispatch (tiled culling, writes additively to same UAV)
             PixMarker.Begin(list, "Point Lights");
             DispatchPointLights(camera, list);
             PixMarker.End(list); // Point Lights
             
             // UAV barrier, then compose GI into LightBuffer
             list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(LightBuffer.Native)));
             
             // GI composition — adds indirect diffuse to LightBuffer AFTER all direct lighting
             if (RadianceCascades != null && Engine.Settings.EnableRadianceCascades)
             {
                 RadianceCascades.ComposeGI(list, this);
                 list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(LightBuffer.Native)));
             }
             
             Transition(list, LightBuffer.Native, ResourceStates.UnorderedAccess, ResourceStates.PixelShaderResource);
             PixMarker.End(list); // Lighting
        }

        private const float PointShadowFadeSeconds = 0.3f;

        // The GPU culls the light registry into this frame's visible list (point_light_cull.hlsl): position from the
        // light's transform, flicker, camera frustum. False when there are no lights.
        private bool CullPointLights(Camera camera, ID3D12GraphicsCommandList list)
        {
             var registry = PointLightBuffer.Instance;
             if (registry == null || registry.SlotCount == 0)
                 return false;
             registry.Upload();

             int frameIdx = Engine.FrameIndex % FrameCount;
             _lightFrame++;
             WriteLightFrameParams(camera, frameIdx);

             PixMarker.Begin(list, "Point light cull");
             _lightCullCS.SetPushConstant("Records", registry.SrvIndex);
             _lightCullCS.SetPushConstant("RecordCount", (uint)registry.SlotCount);
             _lightCullCS.SetPushConstant("Transforms", TransformBuffer.Instance.SrvIndex);
             _lightCullCS.SetPushConstant("FrameParams", _lightFrameParams![frameIdx].SrvIndex);
             _lightCullCS.SetPushConstant("VisibleUAV", _visibleLights!.UavIndex);
             _lightCullCS.SetPushConstant("VisibleCountUAV", _visibleLightCount!.UavIndex);
             _lightCullCS.SetPushConstant("HistogramUAV", _lightImportance!.UavIndex);
             _lightCullCS.SetPushConstant("HiZTex", 0u);

             list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
             list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
             // Lamps (OnlyWhenDark) as lit as the daylight where each hangs let them, judged last frame (2d)
             _lightCullCS.SetPushConstant("LampsUAV", registry.LampStatesUav(list));
             _visibleLightCount.ClearUAV(list, new Vortice.Mathematics.Int4(0, 0, 0, 0));
             _lightImportance.ClearUAV(list, new Vortice.Mathematics.Int4(0, 0, 0, 0));
             list.Barriers(
                 new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibleLightCount.Native)),
                 new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_lightImportance.Native)));

             // Count by importance first: past PointLightBuffer.MaxVisible the least important lights are left out
             uint groups = (uint)((registry.SlotCount + 63) / 64);
             _lightCullCS.Dispatch(_kCountLights, list, groups);
             list.Barriers(
                 new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibleLightCount.Native)),
                 new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_lightImportance.Native)));
             _lightCullCS.Dispatch(_kCullLights, list, groups);

             list.Barriers(
                 new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibleLights.Native)),
                 new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibleLightCount.Native)));
             PixMarker.End(list);
             return true;
        }

        // Camera and frame for the light cull and the shadow selection, per renderer (PointLightFrame)
        private unsafe void WriteLightFrameParams(Camera camera, int frameIdx)
        {
             // As FillLightBuffer builds it: the view without translation
             var cvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, camera.Forward, camera.Up) * camera.Projection;
             var c0 = new Vector4(cvp.M11, cvp.M21, cvp.M31, cvp.M41);
             var c1 = new Vector4(cvp.M12, cvp.M22, cvp.M32, cvp.M42);
             var c3 = new Vector4(cvp.M14, cvp.M24, cvp.M34, cvp.M44);
             var terrain = Atmosphere.TerrainCover;

             var p = _lightFrameParams![frameIdx].WritePtr<PointLightBuffer.FrameParams>();
             *p = new PointLightBuffer.FrameParams
             {
                 CameraRelativeVP = cvp,
                 Plane0 = NormalisePlane(c3 + c0),     // left
                 Plane1 = NormalisePlane(c3 - c0),     // right
                 Plane2 = NormalisePlane(c3 + c1),     // bottom
                 Plane3 = NormalisePlane(c3 - c1),     // top
                 Plane4 = new Vector4(Vector3.Normalize(camera.Forward), -camera.NearPlane),
                 CameraPosition = camera.Position,
                 Time = Time.TotalTime,
                 PixelsPerMetre = 0.5f * Albedo.Native.Description.Height * camera.Projection.M22,
                 FadeStep = Math.Clamp(Time.Delta, 0f, 0.1f) / PointShadowFadeSeconds,
                 SunLight = Daylight.SunIlluminance(),
                 Ambient = Daylight.SkyIlluminance(),
                 ShadowBudget = (uint)Math.Clamp(Engine.Settings.MaxShadowedPointLights, 0, LocalShadows.MaxLights),
                 Frame = _lightFrame,
                 Lamps = Daylight.Lamps,
                 TerrainHeightmap = terrain.Heightmap,
                 TerrainMaxHeight = terrain.MaxHeight,
                 TerrainOrigin = terrain.Origin,
                 TerrainSize = terrain.Size,
                 SunHeight = Daylight.SunHeight(),
                 LampFadeStep = Math.Clamp(Time.Delta, 0f, 0.1f) / Daylight.LampFadeSeconds,
             };
        }

        private static Vector4 NormalisePlane(Vector4 plane) =>
             plane / new Vector3(plane.X, plane.Y, plane.Z).Length();

        // Screen-space shadows toward the point lights the shadow selection ranked highest 2 frames ago (its
        // readback): they carry what the atlas cannot resolve, grass blades and flowers, at screen resolution. Each
        // light is traced within its sphere's rectangle on screen, from where its transform is now (a carried torch
        // moves). The point light pass fades a mask with that light's atlas shadow, so it comes and goes with it.
        private void TracePointLightScreenShadows(Camera camera, ID3D12GraphicsCommandList list, Matrix4x4 cameraRelativeVP,
            int width, int height)
        {
             Array.Fill(_pointScreenShadowSlots, NoScreenShadow);
             var registry = PointLightBuffer.Instance;
             int wanted = Math.Clamp(Engine.Settings.ScreenShadowedPointLights, 0, MaxScreenShadowedPointLights);
             if (LocalShadows == null || registry == null || wanted == 0 || !Engine.Settings.EnableShadows
                 || Engine.Settings.MaxShadowedPointLights <= 0)
                 return;

             int count = LocalShadows.TopPicks(_topPickSlots.AsSpan(0, wanted));
             if (count == 0)
                 return;

             PixMarker.Begin(list, "Point light screen-space shadows");
             for (int i = 0; i < count; i++)
             {
                 var light = registry.LightAt((int)_topPickSlots[i]);
                 var transform = light?.Entity?.Transform;
                 if (light == null || transform == null || !light.Enabled)
                     continue;
                 var tracer = _pointScreenShadows[i] ??= new ScreenSpaceShadows();
                 if (tracer.ExecutePoint(list, DepthGBuffer.BindlessIndex, width, height,
                         transform.WorldPosition - camera.Position, light.Range, cameraRelativeVP, camera.NearPlane))
                     _pointScreenShadowSlots[i] = _topPickSlots[i];
             }
             PixMarker.End(list);
        }

        private unsafe void DispatchPointLights(Camera camera, ID3D12GraphicsCommandList list)
        {
             if (!_pointLightsCulled)
             {
                 LocalShadows?.EndFrame(list);
                 return;
             }

             var desc = LightBuffer.Native.Description;
             _pointLightCS.SetPushConstant("NormalTex", Normals.BindlessIndex);
             _pointLightCS.SetPushConstant("DepthTex", Depth.BindlessIndex);
             _pointLightCS.SetPushConstant("AlbedoTex", Albedo.BindlessIndex);
             _pointLightCS.SetPushConstant("DataTex", Data.BindlessIndex);
             _pointLightCS.SetPushConstant("OutputUAV", LightBuffer.UavIndex);
             _pointLightCS.SetPushConstant("ScreenWidth", (uint)desc.Width);
             _pointLightCS.SetPushConstant("ScreenHeight", (uint)desc.Height);
             _pointLightCS.SetPushConstant("VisibleLightUAV", _visibleLights!.UavIndex);
             _pointLightCS.SetPushConstant("VisibleLightCountUAV", _visibleLightCount!.UavIndex);

             // Point light shadows (atlas 0 = none rendered this frame: the shader skips them)
             _pointLightCS.SetPushConstant("LocalShadowAtlas", LocalShadows?.AtlasIndex ?? 0u);
             _pointLightCS.SetPushConstant("LocalShadowViews", LocalShadows?.ViewBufferSrv ?? 0u);
             _pointLightCS.SetPushConstant("LocalShadowLights", LocalShadows?.LightBufferSrv ?? 0u);

             // Screen-space shadows of the top lights: a light whose registry slot matches reads that mask
             for (int i = 0; i < MaxScreenShadowedPointLights; i++)
             {
                 bool traced = _pointScreenShadowSlots[i] != NoScreenShadow;
                 _pointLightCS.SetPushConstant($"ScreenShadowSlot{i}", _pointScreenShadowSlots[i]);
                 _pointLightCS.SetPushConstant($"ScreenShadowTex{i}", traced ? _pointScreenShadows[i]!.OutputSrvIndex : 0u);
             }

             // Bind root signature, descriptor heap, and SceneConstants cbuffer (b0)
             list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
             list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

             // Bind SceneConstants (b0) for CameraInverse + CameraRelativeVP
             foreach (var cb in matDirectionalLight.ConstantBuffers)
             {
                 if (cb.Slot >= 0)
                 {
                     matDirectionalLight.Effect!.GetMaterialBlock().Apply(cb);
                     cb.Commit();
                     list.SetComputeRootConstantBufferView((uint)cb.Slot, cb.GpuAddress);
                 }
             }

             // Lights wholly behind this frame's Hi-Z (built after the cull) are skipped by the tiles
             uint hiZSrv = (HiZPyramid != null && HiZPyramid.Ready && !Engine.Settings.DisableHiZ)
                 ? HiZPyramid.FullSRV : 0u;
             if (hiZSrv != 0)
             {
                 _lightCullCS.SetPushConstant("HiZTex", hiZSrv);
                 _lightCullCS.Dispatch(_kOccludeLights, list, (uint)(PointLightBuffer.MaxVisible / 64));
                 list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(_visibleLights.Native)));
             }

             // Tiled culling + PBR lighting
             uint groupsX = ((uint)desc.Width + 7) / 8;
             uint groupsY = ((uint)desc.Height + 7) / 8;
             _pointLightCS.Dispatch(_kPointLight, list, groupsX, groupsY);

             LocalShadows?.EndFrame(list);
        }

        /// <summary>
        /// Bind SceneConstants (b0) for a compute pass: the directional light material's copy, which every
        /// SetShaderParams writes like any other effect's. One buffer per frame in flight, so every pass of the
        /// frame reads what was last committed before the list ran.
        /// </summary>
        private void BindSceneConstantsForCompute(ID3D12GraphicsCommandList list)
        {
            foreach (var cb in matDirectionalLight.ConstantBuffers)
            {
                if (cb.Slot == 1) // SceneConstants only
                {
                    matDirectionalLight.Effect!.GetMaterialBlock().Apply(cb);
                    cb.Commit();
                    list.SetComputeRootConstantBufferView((uint)cb.Slot, cb.GpuAddress);
                }
            }
        }

        private void Compose(Camera camera, ID3D12GraphicsCommandList list)
        {
            PixMarker.Begin(list, "Composition");
            // Compute shader path — write directly to Composite UAV
            var fromState = _isFirstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            Transition(list, Composite.Native, fromState, ResourceStates.UnorderedAccess);
            
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            
            // Push constants via ComputeShader reflection
            _compositionCS.SetPushConstant("AlbedoTex", Albedo.BindlessIndex);
            _compositionCS.SetPushConstant("LightTex", LightBuffer.BindlessIndex);
            _compositionCS.SetPushConstant("DataTex", Data.BindlessIndex);
            _compositionCS.SetPushConstant("NormalTex", Normals.BindlessIndex);
            _compositionCS.SetPushConstant("OutputUAV", Composite.UavIndex);
            
            var desc = Composite.Native.Description;
            _compositionCS.SetPushConstant("ScreenWidth", (uint)desc.Width);
            _compositionCS.SetPushConstant("ScreenHeight", (uint)desc.Height);
            _compositionCS.SetPushConstant("DepthGBuf", DepthGBuffer.BindlessIndex);
            _compositionCS.SetPushConstant("AOTex", Engine.Settings.EnableGTAO ? (Gtao?.OutputSrvIndex ?? 0u) : 0u);
            
            // Pass SSDM displacement texture to composition for ambient parallax
            _compositionCS.SetParam("AODirectStrength", Engine.Settings.GTAODirectStrength);
            _compositionCS.SetParam("SSDMTexIdx",Engine.Settings.EnableSSDM ? (ScreenSpaceDisplacement?.OutputSrvIndex ?? 0u) : 0u);

            BindSceneConstantsForCompute(list);

            uint groupsX = ((uint)desc.Width + 7) / 8;
            uint groupsY = ((uint)desc.Height + 7) / 8;
            _compositionCS.Dispatch(_kCompose, list, groupsX, groupsY);
            
            // UAV barrier, then transition to RenderTarget for forward pass
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(Composite.Native)));
            Transition(list, Composite.Native, ResourceStates.UnorderedAccess, ResourceStates.RenderTarget);
            
            // NOTE: Composite stays in RenderTarget for the Forward pass
            PixMarker.End(list); // Composition
        }

        /// <summary>
        /// Execute bloom: threshold → downsample chain → upsample chain.
        /// Composite (HDR) is in NonPixelShaderResource when this is called.
        /// BloomPyramid texture starts in Common (first frame) or NonPixelShaderResource.
        /// </summary>
        private void DispatchBloom(ID3D12GraphicsCommandList list)
        {
            var bloom = BloomPyramid!;
            var settings = Engine.Settings;

            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            // Transition entire bloom texture to UAV for writing.
            // All mips are in the same state: Common (first frame) or NonPixelShaderResource (subsequent).
            var bloomFrom = _isFirstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            Transition(list, bloom.Texture!, bloomFrom, ResourceStates.UnorderedAccess);

            // --- Pass 1: Threshold (full-res HDR Composite → half-res mip 0) ---
            var compDesc = Composite.Native.Description;
            _bloomCS.SetPushConstant(_kBloomThreshold, "InputSrv", Composite.BindlessIndex);
            _bloomCS.SetPushConstant(_kBloomThreshold, "OutputUAV", bloom.MipUAVs[0]);
            _bloomCS.SetPushConstant(_kBloomThreshold, "OutputWidth", (uint)compDesc.Width);
            _bloomCS.SetPushConstant(_kBloomThreshold, "OutputHeight", (uint)compDesc.Height);
            _bloomCS.SetPushConstant(_kBloomThreshold, "Param0Bits", BitConverter.SingleToUInt32Bits(settings.BloomThreshold));
            _bloomCS.SetPushConstant(_kBloomThreshold, "Param1Bits", BitConverter.SingleToUInt32Bits(settings.BloomSoftKnee));
            _bloomCS.SetPushConstant(_kBloomThreshold, "Param2Bits", BitConverter.SingleToUInt32Bits(MathF.Max(settings.BloomMaxBrightness, 0f)));
            _bloomCS.SetPushConstant(_kBloomThreshold, "DepthSrv", Depth.BindlessIndex);

            uint threshX = ((uint)bloom.Width + 7) / 8;
            uint threshY = ((uint)bloom.Height + 7) / 8;
            _bloomCS.Dispatch(_kBloomThreshold, list, threshX, threshY);

            // UAV barrier after threshold write
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(bloom.Texture!)));

            // --- Pass 2: Downsample chain (mip 0 → mip 1 → ... → mip N-1) ---
            // After threshold: all mips in UAV. Source mip transitions to SRV per-iteration.
            for (int i = 1; i < bloom.MipCount; i++)
            {
                int srcW = Math.Max(1, bloom.Width >> (i - 1));
                int srcH = Math.Max(1, bloom.Height >> (i - 1));
                int dstW = Math.Max(1, bloom.Width >> i);
                int dstH = Math.Max(1, bloom.Height >> i);

                // Per-subresource: source mip → SRV, dest mip stays UAV
                list.ResourceBarrier(new ResourceBarrier(
                    new ResourceTransitionBarrier(bloom.Texture!, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource, (uint)(i - 1))));

                _bloomCS.SetPushConstant(_kBloomDownsample, "InputSrv", bloom.MipSRVs[i - 1]);
                _bloomCS.SetPushConstant(_kBloomDownsample, "OutputUAV", bloom.MipUAVs[i]);
                _bloomCS.SetPushConstant(_kBloomDownsample, "OutputWidth", (uint)dstW);
                _bloomCS.SetPushConstant(_kBloomDownsample, "OutputHeight", (uint)dstH);
                _bloomCS.SetPushConstant(_kBloomDownsample, "Param0Bits", BitConverter.SingleToUInt32Bits(1.0f / srcW));
                _bloomCS.SetPushConstant(_kBloomDownsample, "Param1Bits", BitConverter.SingleToUInt32Bits(1.0f / srcH));

                uint downX = ((uint)dstW + 7) / 8;
                uint downY = ((uint)dstH + 7) / 8;
                _bloomCS.Dispatch(_kBloomDownsample, list, downX, downY);

                list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(bloom.Texture!)));
            }
            // State after downsample: mips 0..N-2 = SRV, mip N-1 = UAV

            // --- Pass 3: Upsample chain (mip N-1 → mip N-2 → ... → mip 0) ---
            for (int i = bloom.MipCount - 2; i >= 0; i--)
            {
                int srcW = Math.Max(1, bloom.Width >> (i + 1));
                int srcH = Math.Max(1, bloom.Height >> (i + 1));
                int dstW = Math.Max(1, bloom.Width >> i);
                int dstH = Math.Max(1, bloom.Height >> i);

                // Per-subresource: source mip[i+1] → SRV, dest mip[i] → UAV
                list.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(bloom.Texture!, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource, (uint)(i + 1))),
                    new ResourceBarrier(new ResourceTransitionBarrier(bloom.Texture!, ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess, (uint)i)));

                _bloomCS.SetPushConstant(_kBloomUpsample, "InputSrv", bloom.MipSRVs[i + 1]);
                _bloomCS.SetPushConstant(_kBloomUpsample, "OutputUAV", bloom.MipUAVs[i]);
                _bloomCS.SetPushConstant(_kBloomUpsample, "OutputWidth", (uint)dstW);
                _bloomCS.SetPushConstant(_kBloomUpsample, "OutputHeight", (uint)dstH);
                _bloomCS.SetPushConstant(_kBloomUpsample, "Param0Bits", BitConverter.SingleToUInt32Bits(1.0f / srcW));
                _bloomCS.SetPushConstant(_kBloomUpsample, "Param1Bits", BitConverter.SingleToUInt32Bits(1.0f / srcH));
                _bloomCS.SetPushConstant(_kBloomUpsample, "Param2Bits", BitConverter.SingleToUInt32Bits(settings.BloomRadius));

                uint upX = ((uint)dstW + 7) / 8;
                uint upY = ((uint)dstH + 7) / 8;
                _bloomCS.Dispatch(_kBloomUpsample, list, upX, upY);

                list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(bloom.Texture!)));
            }
            // State after upsample: mip 0 = UAV, mips 1..N-1 = SRV

            // Transition mip 0 to SRV for finalize to sample
            list.ResourceBarrier(new ResourceBarrier(
                new ResourceTransitionBarrier(bloom.Texture!, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource, 0)));
        }

        /// <summary>
        /// Finalize pass: reads HDR Composite + bloom, applies tonemapping, writes LDR output.
        /// Composite is in NonPixelShaderResource when this is called (after forward/post-process).
        /// </summary>
        private void Finalize(ID3D12GraphicsCommandList list)
        {
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            // Transition LdrOutput to UAV for writing
            var ldrFrom = _isFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource;
            Transition(list, LdrOutput.Native, ldrFrom, ResourceStates.UnorderedAccess);

            var desc = Composite.Native.Description;
            _finalizeCS.SetPushConstant(_kFinalize, "InputSrv", Composite.BindlessIndex);
            _finalizeCS.SetPushConstant(_kFinalize, "OutputUAV", LdrOutput.UavIndex);
            _finalizeCS.SetPushConstant(_kFinalize, "ScreenWidth", (uint)desc.Width);
            _finalizeCS.SetPushConstant(_kFinalize, "ScreenHeight", (uint)desc.Height);

            // Bloom params (push constants — packed as uint)
            bool bloomActive = BloomPyramid != null && Engine.Settings.EnableBloom && BloomPyramid.MipCount > 0;
            _finalizeCS.SetPushConstant(_kFinalize, "BloomTex", bloomActive ? BloomPyramid!.MipSRVs[0] : 0u);
            _finalizeCS.SetPushConstant(_kFinalize, "BloomIntBits", BitConverter.SingleToUInt32Bits(bloomActive ? Engine.Settings.BloomIntensity : 0f));
            _finalizeCS.SetPushConstant(_kFinalize, "TimeBits", BitConverter.SingleToUInt32Bits(Time.TotalTime));
            _finalizeCS.SetPushConstant(_kFinalize, "DepthSrv", Depth.BindlessIndex);
            _finalizeCS.SetPushConstant(_kFinalize, "BloomThresholdBits", BitConverter.SingleToUInt32Bits(Engine.Settings.BloomThreshold));
            _finalizeCS.SetPushConstant(_kFinalize, "BloomKneeBits", BitConverter.SingleToUInt32Bits(Engine.Settings.BloomSoftKnee));
            _finalizeCS.SetPushConstant(_kFinalize, "BloomMipCount", bloomActive ? (uint)BloomPyramid!.MipCount : 1u);
            _finalizeCS.SetPushConstant(_kFinalize, "BloomSpreads", bloomActive && Engine.Settings.BloomSpreadsLight ? 1u : 0u);
            _finalizeCS.SetPushConstant(_kFinalize, "BloomCapBits", BitConverter.SingleToUInt32Bits(MathF.Max(Engine.Settings.BloomMaxBrightness, 0f)));

            uint groupsX = ((uint)desc.Width + 7) / 8;
            uint groupsY = ((uint)desc.Height + 7) / 8;
            _finalizeCS.Dispatch(_kFinalize, list, groupsX, groupsY);

            // UAV barrier, then transition to PixelShaderResource for SMAA/Blit
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(LdrOutput.Native)));
            Transition(list, LdrOutput.Native, ResourceStates.UnorderedAccess, ResourceStates.PixelShaderResource);
        }

        private void RenderForward(Camera camera, ID3D12GraphicsCommandList list)
        {
             PixMarker.Begin(list, "Forward");
             // Snapshot the Composite buffer for forward-pass shaders to read
             // (avoids read/write hazard — ocean writes to Composite while reading the snapshot)
             if (CompositeSnapshot == null) 
             {
                 CompositeSnapshot = new RenderTexture2D(Engine.Device,
                     (int)Composite.Native.Description.Width,
                     (int)Composite.Native.Description.Height,
                     Composite.Native.Description.Format);
             }
             Transition(list, Composite.Native, ResourceStates.RenderTarget, ResourceStates.CopySource);
             Transition(list, CompositeSnapshot.Native, 
                 _compositeSnapshotFirstFrame ? ResourceStates.Common : ResourceStates.PixelShaderResource, 
                 ResourceStates.CopyDest);
             list.CopyResource(CompositeSnapshot.Native, Composite.Native);
             Transition(list, CompositeSnapshot.Native, ResourceStates.CopyDest, ResourceStates.PixelShaderResource);
             Transition(list, Composite.Native, ResourceStates.CopySource, ResourceStates.RenderTarget);
             _compositeSnapshotFirstFrame = false;


             // Depth is in PixelShaderResource — forward objects can sample it via bindless SRV.
             // Transition depth back to DepthWrite so forward objects can depth-test.
             Transition(list, Depth.Native, ResourceStates.PixelShaderResource, ResourceStates.DepthWrite);

             // Bind render targets for forward pass
             list.OMSetRenderTargets(new[] { Composite.RtvHandle, EntityIdBuffer.RtvHandle }, Depth.DsvHandle);
             list.RSSetViewport(new Viewport(0, 0, Composite.Native.Description.Width, Composite.Native.Description.Height));
             list.RSSetScissorRect(new RectI(0, 0, (int)Composite.Native.Description.Width, (int)Composite.Native.Description.Height));

             // Execute forward pass draws (ocean, gizmos, transparent, etc.)
             PixMarker.Begin(list, "Forward Draws");
             CommandBuffer.Execute(RenderPass.Forward, list, Engine.Device);
             PixMarker.End(list); // Forward Draws

             // Pick readback: entity ID + depth + normal
              if (_pickRequestX >= 0 && _entityIdReadback != null)
              {
                  var pickBox = new Vortice.Mathematics.Box(_pickRequestX, _pickRequestY, 0, _pickRequestX + 1, _pickRequestY + 1, 1);

                  // Entity ID (RenderTarget -> CopySource -> PixelShaderResource)
                  Transition(list, EntityIdBuffer.Native, ResourceStates.RenderTarget, ResourceStates.CopySource);
                  list.CopyTextureRegion(
                      new TextureCopyLocation(_entityIdReadback.Native, new PlacedSubresourceFootPrint { Offset = 0, Footprint = new SubresourceFootPrint(Format.R32_UInt, 1, 1, 1, 256) }),
                      0, 0, 0,
                      new TextureCopyLocation(EntityIdBuffer.Native, 0), pickBox);
                  Transition(list, EntityIdBuffer.Native, ResourceStates.CopySource, ResourceStates.PixelShaderResource);

                  // Depth (PixelShaderResource -> CopySource -> PixelShaderResource)
                  Transition(list, DepthGBuffer.Native, ResourceStates.PixelShaderResource, ResourceStates.CopySource);
                  list.CopyTextureRegion(
                      new TextureCopyLocation(_depthReadback!.Native, new PlacedSubresourceFootPrint { Offset = 0, Footprint = new SubresourceFootPrint(Format.R32_Float, 1, 1, 1, 256) }),
                      0, 0, 0,
                      new TextureCopyLocation(DepthGBuffer.Native, 0), pickBox);
                  Transition(list, DepthGBuffer.Native, ResourceStates.CopySource, ResourceStates.PixelShaderResource);

                  // Normal (PixelShaderResource -> CopySource -> PixelShaderResource)
                  Transition(list, Normals.Native, ResourceStates.PixelShaderResource, ResourceStates.CopySource);
                  list.CopyTextureRegion(
                      new TextureCopyLocation(_normalReadback!.Native, new PlacedSubresourceFootPrint { Offset = 0, Footprint = new SubresourceFootPrint(Format.R16G16B16A16_SNorm, 1, 1, 1, 256) }),
                      0, 0, 0,
                      new TextureCopyLocation(Normals.Native, 0), pickBox);
                  Transition(list, Normals.Native, ResourceStates.CopySource, ResourceStates.PixelShaderResource);

                  _readbackPixelX = _pickRequestX;
                  _readbackPixelY = _pickRequestY;
                  _pickCopyPending = true;
                  _pickRequestX = -1;
                  _pickSeeThrough = null;
              }
              else
              {
                  Transition(list, EntityIdBuffer.Native, ResourceStates.RenderTarget, ResourceStates.PixelShaderResource);
              }


             // Transition Composite to NonPixelShaderResource for compute reads (bloom + finalize)
             Transition(list, Composite.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
             // Depth for PostProcess reads
             Transition(list, Depth.Native, ResourceStates.DepthWrite, ResourceStates.PixelShaderResource);

             // PostProcess pass — only do expensive snapshot copy + transitions if commands are queued
             if (CommandBuffer.HasPendingCommands(RenderPass.PostProcess))
             {
                 // Snapshot composite for PostProcess manual blending
                 Transition(list, Composite.Native, ResourceStates.NonPixelShaderResource, ResourceStates.CopySource);
                 Transition(list, CompositeSnapshot.Native, ResourceStates.PixelShaderResource, ResourceStates.CopyDest);
                 list.CopyResource(CompositeSnapshot.Native, Composite.Native);
                 Transition(list, CompositeSnapshot.Native, ResourceStates.CopyDest, ResourceStates.PixelShaderResource);
                 Transition(list, Composite.Native, ResourceStates.CopySource, ResourceStates.RenderTarget);

                 // PostProcess reads CompositeSnapshot + Depth, writes to Composite
                 list.OMSetRenderTargets(Composite.RtvHandle);
                 list.RSSetViewport(new Viewport(0, 0, Composite.Native.Description.Width, Composite.Native.Description.Height));
                 list.RSSetScissorRect(new RectI(0, 0, (int)Composite.Native.Description.Width, (int)Composite.Native.Description.Height));
                 PixMarker.Begin(list, "PostProcess");
                 CommandBuffer.Execute(RenderPass.PostProcess, list, Engine.Device);
                 PixMarker.End(list); // PostProcess

                 // Transition Composite back to NonPixelShaderResource for compute reads
                 Transition(list, Composite.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
             }
             PixMarker.End(list); // Forward
        }
        

        private void BlitToBackBuffer(Camera camera, ID3D12GraphicsCommandList list)
        {
             PixMarker.Begin(list, "Blit");
             var backBuffer = camera.Target.CurrentBackBuffer;

             // If SMAA is active, blit from its output; otherwise from LdrOutput
             bool useSmaa = Smaa != null && Engine.Settings.EnableSMAA && Smaa.BlitSource != null;
             var blitSource = useSmaa ? Smaa!.BlitSource! : LdrOutput.Native;

             Transition(list, blitSource, ResourceStates.PixelShaderResource, ResourceStates.CopySource);
             Transition(list, backBuffer, ResourceStates.RenderTarget, ResourceStates.CopyDest);
             
             list.CopyResource(backBuffer, blitSource);
             
             Transition(list, blitSource, ResourceStates.CopySource, ResourceStates.PixelShaderResource);
             Transition(list, backBuffer, ResourceStates.CopyDest, ResourceStates.RenderTarget);
             PixMarker.End(list); // Blit
        }

        /// <summary>
        /// Request entity ID readback at the specified pixel coordinate.
        /// Result is available next frame via GetPickedEntityId.
        /// </summary>
        /// <param name="seeThrough">
        /// A character whose skin leaves the picked pixel open in the frame it is read from
        /// (gbuffer_skinned's pick hole), so the pick gets what is behind it: the player's own, for
        /// a crosshair that its back or head is in the way of. That one pixel, that one character;
        /// what it holds in its hands (not skinned) still counts.
        /// </param>
        public void RequestPick(int x, int y, Animator? seeThrough = null)
        {
            _pickRequestX = x;
            _pickRequestY = y;
            _pickSeeThrough = seeThrough;
            _pickResultReady = false;
        }

        /// <summary>
        /// Promote a pending pick copy to a ready result.
        /// Call from OnUpdate before checking HasPickResult to eliminate the
        /// 1-frame delay caused by promote normally running at render start.
        /// Safe to call even if no copy is pending.
        /// </summary>
        public void TryPromotePickResult()
        {
            if (_pickCopyPending)
            {
                _pickResultReady = true;
                _pickCopyPending = false;
            }
        }

        /// <summary>
        /// Whether a pick result is ready to read.
        /// </summary>
        public bool HasPickResult => _pickResultReady;

        /// <summary>
        /// Read the picked entity's TransformSlot from the readback buffer.
        /// Returns 0 if no entity (sky/grass), the TransformSlot otherwise.
        /// Call after HasPickResult is true. Also populates PickedMeshPartId.
        /// </summary>
        public unsafe uint GetPickedEntityId()
        {
            if (!_pickResultReady || _entityIdReadback == null) return 0;
            _pickResultReady = false;

            uint packed = *_entityIdReadback.ReadPtr<uint>();
            PickedMeshPartId = packed & 0xFF;        // lower 8 bits = meshpart

            // Read depth (linear view-space)
            PickedDepth = _depthReadback != null ? *_depthReadback.ReadPtr<float>() : 0f;

            // Read normal (SNorm16 × 4 → float)
            if (_normalReadback != null)
            {
                short* nPtr = (short*)_normalReadback.ReadPtr<ulong>();
                PickedNormal = new Vector3(
                    nPtr[0] / 32767f,
                    nPtr[1] / 32767f,
                    nPtr[2] / 32767f);
            }
            else
            {
                PickedNormal = Vector3.UnitY;
            }

            PickedPixelX = _readbackPixelX;
            PickedPixelY = _readbackPixelY;

            PickedEntitySlot = packed >> 8;           // upper 24 bits = entity ID
            PickSerial++;
            return PickedEntitySlot;
        }

        /// <summary>
        /// The TransformSlot read by the last <see cref="GetPickedEntityId"/> (0 = none), kept with
        /// the pixel, depth and normal it came with. A result can only be taken once, so when two
        /// ask (the editor's viewport and a game script in play mode) the one that comes second
        /// reads it here, and tells by <see cref="PickSerial"/> that it is a new one.
        /// </summary>
        public uint PickedEntitySlot { get; private set; }

        /// <summary>Counts the pick results taken so far.</summary>
        public int PickSerial { get; private set; }

        /// <summary>MeshPart index from the last pick result.</summary>
        public uint PickedMeshPartId { get; private set; }

        /// <summary>Linear view-space depth at the picked pixel. 0 = sky/empty.</summary>
        public float PickedDepth { get; private set; }

        /// <summary>World-space surface normal at the picked pixel.</summary>
        public Vector3 PickedNormal { get; private set; }

        /// <summary>Pixel coordinates of the last pick (viewport-local).</summary>
        public int PickedPixelX { get; private set; }
        public int PickedPixelY { get; private set; }

        public override void Dispose()
        {
            Albedo?.Dispose();
            Normals?.Dispose();
            Data?.Dispose();
            DepthGBuffer?.Dispose();
            EntityIdBuffer?.Dispose();
            HeightBuffer?.Dispose();
            Depth?.Dispose();
            LightBuffer?.Dispose();
            SkyVisibilityTex?.Dispose();
            Composite?.Dispose();
            CompositeSnapshot?.Dispose();

            ShadowTextureArray?.Dispose();
            LocalShadows?.Dispose();
            ShadowReceivers?.Dispose();
            ScreenSpaceShadows?.Dispose();
            VolumetricFog?.Dispose();
            _aerialPerspective.Dispose();
            _cloudView.Dispose();
            foreach (var tracer in _pointScreenShadows) tracer?.Dispose();
            Gtao?.Dispose();
            ScreenSpaceDisplacement?.Dispose();
            RadianceCascades?.Dispose();
            Smaa?.Dispose();
            BloomPyramid?.Dispose();
            _bloomCS?.Dispose();
            LdrOutput?.Dispose();
            _finalizeCS?.Dispose();
            _entityIdReadback?.Dispose();
            _depthReadback?.Dispose();
            _normalReadback?.Dispose();
            _pointLightCS?.Dispose();
            _lightCullCS?.Dispose();
            _visibleLights?.Dispose();
            _visibleLightCount?.Dispose();
            _lightImportance?.Dispose();
            if (_lightFrameParams != null)
                foreach (var b in _lightFrameParams) b?.Dispose();
        }
    }
}
