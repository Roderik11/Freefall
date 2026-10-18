using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Assets;
using Freefall.Graphics;
using Freefall.Base;
using Freefall.Components;
using Vortice.XAudio2;
using Vortice.Multimedia;

namespace Freefall
{
    public static class Engine
    {
        private static bool _isRunning;

        public static AssetManager Assets { get; private set; }
        public static GraphicsDevice Device { get; private set; }
        public static EngineSettings Settings { get; private set; }
        public static string RootDirectory { get; private set; }
        public static FreefallProject Project { get; private set; }

        public static IXAudio2 AudioDevice { get; private set; }
        public static X3DAudio Audio3D { get; private set; }
        private static IXAudio2MasteringVoice _masteringVoice;
        
        public static int TickCount { get; private set; }

        // Snapshot list to avoid modification during iteration (Apex pattern)
        private static readonly List<RenderView> _renderViews = new List<RenderView>();
        
        /// <summary>
        /// The current frame buffer index (0, 1, or 2). Uses swapchain's index for proper GPU synchronization.
        /// Falls back to TickCount % 3 before RenderView is initialized.
        /// </summary>
        public static int FrameIndex => RenderView.Primary?.FrameIndex ?? (TickCount % 3);
        
        public static bool Running => _isRunning;

        /// <summary>
        /// True when running inside the editor, false for standalone game.
        /// Controls whether terrain loads editable layer data or just the pre-baked result.
        /// </summary>
        public static bool IsEditor { get; private set; }

        public static void SetPlaymode(bool playmode)
        {
            IsEditor = !playmode;
        }

        // Main-thread work queue for marshaling from background threads
        private sealed class WorkItem
        {
            public Action Action;
            public TaskCompletionSource<bool> Tcs;
        }

        private static readonly ConcurrentQueue<WorkItem> _mainThreadQueue = new ConcurrentQueue<WorkItem>();

        /// <summary>
        /// Run an action on the main thread at the end of a tick (after Present; see Tick). The caller's continuation
        /// runs on the thread pool, never inline inside the tick.
        /// </summary>
        public static Task RunOnMainThreadAsync(Action action)
        {
            if (action == null) return Task.CompletedTask;
            var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            _mainThreadQueue.Enqueue(new WorkItem { Action = action, Tcs = tcs });
            return tcs.Task;
        }

        static Engine()
        {
            Settings = new EngineSettings();
        }

        /// <summary>
        /// Initialize core engine systems (Device, Assets, Audio, Physics).
        /// The caller is responsible for creating the Window, RenderView, and Input.
        /// </summary>
        public static void Initialize()
        {
            // Set high-resolution timer for smooth high-FPS gameplay
            Kernel32.timeBeginPeriod(1);
            
            RootDirectory = AppContext.BaseDirectory;

            LoadPixGpuCapturer();
            Device = new GraphicsDevice();
            
            // Initialize persistent transform buffer for GPU-driven rendering
            TransformBuffer.Initialize(Device);
            PointLightBuffer.Initialize(Device); // GPU light registry: slots follow ComponentCache<PointLight>
            
            // Initialize streaming manager before assets (AssetManager creates default textures that need it)
            var streaming = new StreamingManager(Device);
            
            // Centralized Asset Manager Init
            Assets = new AssetManager(Device);

            // Engine-internal default textures & materials
            Freefall.Assets.InternalAssets.Initialize(Device);
            Freefall.Assets.InternalAssets.Register(Assets);
            
            // Initialize Audio
            AudioDevice = XAudio2.XAudio2Create(ProcessorSpecifier.DefaultProcessor, true);
            _masteringVoice = AudioDevice.CreateMasteringVoice();
            Audio3D = new X3DAudio(Speakers.FrontLeft | Speakers.FrontRight);
            Debug.Log("[Engine] XAudio2 + X3DAudio initialized");
            
            // Initialize Physics
            PhysicsWorld.Initialize();
        }

        /// <summary>
        /// Initialize the engine with an external window handle (for editor mode).
        /// The caller owns the window and is responsible for resize events.
        /// </summary>
        public static void Initialize(IntPtr handle, int width, int height)
        {
            Kernel32.timeBeginPeriod(1);

            RootDirectory = AppContext.BaseDirectory;

            LoadPixGpuCapturer();
            Device = new GraphicsDevice();
            TransformBuffer.Initialize(Device);
            PointLightBuffer.Initialize(Device); // GPU light registry: slots follow ComponentCache<PointLight>

            var streaming = new StreamingManager(Device);
            Assets = new AssetManager(Device);

            Freefall.Assets.InternalAssets.Initialize(Device);
            Freefall.Assets.InternalAssets.Register(Assets);

            // Initialize Rendering Foundation — view auto-registers in RenderView.All
            var view = new RenderView(handle, width, height, Device);
            view.Pipeline = new DeferredRenderer();
            view.Pipeline.Initialize(width, height);

            Input.Init(handle);

            AudioDevice = XAudio2.XAudio2Create(ProcessorSpecifier.DefaultProcessor, true);
            _masteringVoice = AudioDevice.CreateMasteringVoice();
            Audio3D = new X3DAudio(Speakers.FrontLeft | Speakers.FrontRight);
            Debug.Log("[Engine] Initialized (editor mode)");
            IsEditor = true;

            PhysicsWorld.Initialize();
        }

        /// <summary>
        /// Open a project at runtime. Initializes AssetDatabase and sets asset base directory.
        /// Can be called after Initialize() — enables the landing-page-first workflow. Any thread: it reads the
        /// project's files and fills the asset database, nothing that needs the main thread (the editor runs it on a
        /// worker while the landing page shows progress). Progress: what it is doing, fraction 0..1.
        /// </summary>
        public static void OpenProject(string path, Action<string, float> progress = null)
        {
            Project = FreefallProject.Open(path);
            AssetDatabase.Initialize(Project, progress);
            Assets.BaseDirectory = Project.AssetsDirectory;
            Debug.Log($"[Engine] Project opened: {Project.Name}");
        }

        /// <summary>
        /// Mark the engine as running and initialize timing.
        /// The caller owns the message loop (Apex pattern).
        /// </summary>
        public static void Run()
        {
            _isRunning = true;
            Time.Initialize();
        }

        private static readonly ProfilerMarker _disposalsMarker = ProfilerMarker.Get("Deferred disposals");
        private static readonly ProfilerMarker _assetReimportMarker = ProfilerMarker.Get("Asset reimport");
        private static readonly ProfilerMarker _assetReloadMarker = ProfilerMarker.Get("Asset reload");
        private static readonly ProfilerMarker _shaderReloadMarker = ProfilerMarker.Get("Shader reload");
        private static readonly ProfilerMarker _physicsMarker = ProfilerMarker.Get("Physics");
        private static readonly ProfilerMarker _navigationMarker = ProfilerMarker.Get("Navigation");
        private static readonly ProfilerMarker _updateMarker = ProfilerMarker.Get("Update");
        private static readonly ProfilerMarker _resizeMarker = ProfilerMarker.Get("Resize");
        private static readonly ProfilerMarker _renderMarker = ProfilerMarker.Get("Render");
        private static readonly ProfilerMarker _cameraMarker = ProfilerMarker.Get("Camera");
        private static readonly ProfilerMarker _queueMarker = ProfilerMarker.Get("Main thread queue");

        public static void Tick()
        {
            TickCount++;

            Time.Update();
            Graphics.Wind.Advance(Time.Delta);
            Input.Update();

            // Release GPU buffers from previous batch resizes (deferred N frames for safety)
            using (_disposalsMarker.Auto())
            {
                Graphics.InstanceBatch.FlushDeferredDisposals();
                Device?.FlushFrameDeferredDisposals();
            }

            // A forced reimport (folder Reimport, MCP asset_reimport) imports a few files per frame
            using (_assetReimportMarker.Auto())
                AssetDatabase.TickReimports();

            // Hot-reload loaded assets whose source was reimported (auto-import, asset_refresh, inspector
            // reimport). Before Update/Draw so the whole frame sees the new data.
            using (_assetReloadMarker.Auto())
                Assets?.ReloadReimported();

            // Recompile shaders whose .fx/.hlsl file changed on disk (same point: nothing is recording yet)
            using (_shaderReloadMarker.Auto())
                Graphics.ShaderHotReload.Update();

            // Physics (fixed-timestep)
            using (_physicsMarker.Auto())
                PhysicsWorld.Update((float)Time.Delta);
            using (_navigationMarker.Auto())
                NavMeshWorld.Update((float)Time.Delta);

            // Update Logic
            using (_updateMarker.Auto())
            {
                Graphics.Overlay.Clear();   // what game code draws over the picture lasts one frame
                try
                {
                    Update();
                }
                catch when (!Device.IsDeviceLost && Device.NativeDevice.DeviceRemovedReason.Failure)
                {
                    // A component creating GPU resources in Update is often the first to notice a removed
                    // device. Write the reason (+ DRED) before the exception takes the process down.
                    Device.LogDeviceRemoved("Engine.Update");
                    throw;
                }
            }

            // --- Process all deferred resizes before rendering ---
            // Must sync GPU first — previous frames may still be using resources
            bool anyResizePending = false;
            foreach (var v in RenderView.All)
            {
                if (v.IsResizePending) { anyResizePending = true; break; }
            }

            if (anyResizePending)
            {
                using var _ = _resizeMarker.Auto();
                RenderView.Primary?.WaitForGpu();
                foreach (var v in RenderView.All)
                    v.ProcessPendingResize();
            }

            // A removed device makes every later Map fail (null pointer): report the reason (+ DRED) once, up front,
            // instead of dying with an NRE somewhere in the renderer.
            if (!Device.IsDeviceLost && Device.NativeDevice.DeviceRemovedReason.Failure)
                Device.LogDeviceRemoved("Engine.Tick");

            // --- Apex multi-viewport render loop ---
            Device.ResetTemporaryDescriptors();

            _renderViews.Clear();
            _renderViews.AddRange(RenderView.All);

            using (_renderMarker.Auto())
            {
                foreach (var view in _renderViews)
                {
                    if (!view.Enabled) continue;
                    if (!view.HasSwapChain) continue; // Headless views rendered inside primary view's OnRender

                    if (view.OnRender != null)
                    {
                        view.OnRender(view);
                    }
                    else
                    {
                        view.Prepare();

                        using (_cameraMarker.Auto())
                        {
                            var cam = Camera.Main;
                            if (cam != null && cam.Target == view)
                                cam.Render(view.CommandList.Native);

                            view.OnAfterRender?.Invoke(view);
                        }

                        view.Present();
                    }
                }
            }

            CommandBuffer.Clear();

            // Drain main-thread queue AFTER Present (GPU fence ensures no commands in flight,
            // so creating GPU resources like upload buffers and SRV descriptors is safe here).
            // Time-budgeted: process items until 4ms spent to avoid frame hitches.
            if (!_mainThreadQueue.IsEmpty)
            {
                using var _ = _queueMarker.Auto();
                var budgetStart = System.Diagnostics.Stopwatch.GetTimestamp();
                long budgetTicks = System.Diagnostics.Stopwatch.Frequency * 4 / 1000; // 4ms
                while (_mainThreadQueue.TryDequeue(out var work))
                {
                    try
                    {
                        work.Action?.Invoke();
                        work.Tcs?.SetResult(true);
                    }
                    catch (Exception ex)
                    {
                        work.Tcs?.SetException(ex);
                    }
                    if (System.Diagnostics.Stopwatch.GetTimestamp() - budgetStart >= budgetTicks)
                        break;
                }
            }

            Input.ClearFrameCallbacks();

            Profiler.EndFrame();
        }

        private static void Update()
        {
            // F5: Toggle adaptive shadow cascade splits (SDSM)
            if (Input.IsKeyPressed(Keys.F5))
            {
                Settings.UseAdaptiveSplits = !Settings.UseAdaptiveSplits;
                Debug.Log($"[Engine] Adaptive Splits {(Settings.UseAdaptiveSplits ? "Enabled" : "Disabled")}");
            }
            
            // F6: Cycle debug visualization mode
            if (Input.IsKeyPressed(Keys.F6))
            {
                Settings.DebugVisualizationMode = (DebugVizMode)(((int)Settings.DebugVisualizationMode + 1) % 10);
                Debug.Log($"[Engine] Debug Viz: {Settings.DebugVisualizationMode}");
            }
            
            // F7: Toggle Hi-Z occlusion culling
            if (Input.IsKeyPressed(Keys.F7))
            {
                Settings.DisableHiZ = !Settings.DisableHiZ;
                Debug.Log($"[Engine] Hi-Z Occlusion {(Settings.DisableHiZ ? "DISABLED" : "ENABLED")}");
            }

            // F8: Toggle wireframe mode
            if (Input.IsKeyPressed(Keys.F8))
            {
                Settings.Wireframe = !Settings.Wireframe;
                Debug.Log($"[Engine] Wireframe {(Settings.Wireframe ? "Enabled" : "Disabled")}");
            }

            // F9: Toggle VSync
            if (Input.IsKeyPressed(Keys.F9))
            {
                Settings.VSync = !Settings.VSync;
                Debug.Log($"[Engine] VSync {(Settings.VSync ? "Enabled" : "Disabled")}");
            }
            
            // F10: Toggle Frustum Freeze - freezes culling frustum for debug visualization
            if (Input.IsKeyPressed(Keys.F10))
            {
                Settings.FreezeFrustum = !Settings.FreezeFrustum;
                if (Settings.FreezeFrustum && Camera.Main != null)
                {
                    Settings.FrozenViewProjection = Camera.Main.ViewProjection;
                    Debug.Log("[Engine] Frustum FROZEN - walk around to see culled objects");
                }
                else
                {
                    Debug.Log("[Engine] Frustum UNFROZEN - culling follows camera");
                }
            }
            
            // Entity logic (includes component Updates like Camera, CharacterController, etc.)
            EntityManager.Update();

            // Regenerate PCG components invalidated by edits, once the edit has finished
            Freefall.PCG.PCGScheduler.Update();
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr LoadLibraryW([MarshalAs(UnmanagedType.LPWStr)] string lpFileName);

        /// <summary>
        /// Load WinPixGpuCapturer.dll so PIX can attach to the running process.
        /// Must be called before any D3D12 API calls.
        /// </summary>
        private static void LoadPixGpuCapturer()
        {
            return;

#if DEBUG
            // Find latest PIX install
            var pixDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "Microsoft PIX");
            
            if (!Directory.Exists(pixDir)) return;
            
            // PIX installs into versioned subdirectories (e.g. "2401.08.002")
            var latest = Directory.GetDirectories(pixDir)
                .OrderByDescending(d => d)
                .FirstOrDefault();
            
            if (latest == null) return;
            
            var dllPath = Path.Combine(latest, "WinPixGpuCapturer.dll");
            if (!File.Exists(dllPath)) return;
            
            var handle = LoadLibraryW(dllPath);
            if (handle != IntPtr.Zero)
                Debug.Log("[Engine]", $"PIX GPU Capturer loaded from {dllPath}");
            else
                Debug.LogWarning("[Engine]", $"Failed to load PIX GPU Capturer: {Marshal.GetLastWin32Error()}");
#endif
        }

        public static void Shutdown()
        {
            _isRunning = false;
            
            PhysicsWorld.Shutdown();
            NavMeshWorld.Shutdown();
            
            // Restore timer resolution
            Kernel32.timeEndPeriod(1);
            
            TransformBuffer.Instance?.Dispose();

            StreamingManager.Instance?.Dispose();

            // Dispose all registered render views
            for (int i = RenderView.All.Count - 1; i >= 0; i--)
                RenderView.All[i].Dispose();

            Assets?.Dispose();
            _masteringVoice?.Dispose();
            AudioDevice?.Dispose();
            Device?.Dispose();
        }
    }

    public enum DebugVizMode
    {
        Off = 0,
        CascadeColors = 1,
        ShadowFactor = 2,
        ContactShadow = 3,
        LinearDepth = 4,
        DecoControlMap = 5,
        HeightMap = 6,
        RCDebug = 7,
        SkyVisibility = 8,          // light_directional_cs: the sky's share a surface sees past terrain and meshes
        MeshSkyVisibility = 9       // the meshes' part of it alone (SkyCapture)
    }

    /// <summary>EngineSettings.ShadowReceiverCulling</summary>
    public enum ShadowReceiverCulling
    {
        Off = 0,
        Count = 1,      // receiver pass + counts only
        On = 2,         // entries that shade no receiver are left out of the cascades
    }

    public class EngineSettings
    {
         [Category("Display")]
         public bool VSync { get; set; } = true;                            // F1 - VSync
         public bool Wireframe { get; set; } = false;                       // F2 - Global wireframe
         public bool EnableSMAA { get; set; } = false;                      // Anti-aliasing
         public bool ShowGrid { get; set; } = true;                      // Show editor grid in scene view

        [Category("Culling")]
         public bool FreezeFrustum { get; set; } = false;                   // F3 - Freeze culling frustum
         public bool DisableHiZ { get; set; } = false;                      // F6 - Disable Hi-Z occlusion
         public bool DisableDepthSort { get; set; } = true;                 // GPU doesn't order instances within a draw call — sort has no visual effect

         /// <summary>
         /// Global LOD distance scale. Default 1.0. Linear: 2.0 doubles every LOD and cull distance.
         /// Higher = LODs stay high-detail longer (quality). Lower = transition sooner (performance).
         /// Multiplied with per-mesh Mesh.LODBias.
         /// </summary>
         [ValueRange(0.1f, 20.0f)]
         public float LODScale { get; set; } = 1.0f;

         [Category("Shadows")]
         public bool EnableShadows { get; set; } = true;

         /// <summary>
         /// Cascades after the first leave out casters whose bounding sphere spans fewer of that cascade's
         /// shadow-map texels than this (diameter): far out a small thing darkens a texel or two, which
         /// screen-space shadows already give it. 0 = every caster in every cascade.
         /// </summary>
         [ValueRange(0f, 16f)]
         public float ShadowMinCasterTexels { get; set; } = 4f;

         /// <summary>
         /// Receiver-mask caster culling (ShadowReceivers): a cascade entry whose caster lies deeper than every
         /// visible receiver under its footprint darkens nothing on screen. On = such entries are left out of the
         /// cascades; Count = the receiver pass only counts them (engine_stats: sunShadowReceivers).
         /// </summary>
         public ShadowReceiverCulling ShadowReceiverCulling { get; set; } = ShadowReceiverCulling.On;
         public bool UseAdaptiveSplits { get; set; } = true;                // F4 - SDSM adaptive cascade splits

         [ValueRange(3, 8)]
         public int ShadowCascadeCount { get; set; } = 4;                   // Number of shadow cascades

         [ValueRange(50f, 1000f)]
         public float MaxShadowDistance { get; set; } = 300f;               // SDSM max depth for split analysis

         /// <summary>
         /// Point lights that get a shadow each frame, at night: every point light in view whose sphere is at least
         /// 32 px in radius on screen competes, ranked on the GPU by brightness times the screen share of its sphere,
         /// lower when its centre is hidden (LocalShadows). Fewer when the atlas is full. 0 = none.
         /// </summary>
         [Category("Point Light Shadows")]
         [ValueRange(0, 32)]
         public int MaxShadowedPointLights { get; set; } = 12;

         [Category("Screen-Space Shadows")]                                 // Bend Studio technique
         public bool EnableScreenSpaceShadows { get; set; } = true;

         [ValueRange(0.001f, 0.2f)]
         public float SSSSurfaceThickness { get; set; } = 0.005f;

         [ValueRange(0.005f, 0.1f)]
         public float SSSBilinearThreshold { get; set; } = 0.02f;

         [ValueRange(1f, 8f)]
         public float SSSShadowContrast { get; set; } = 2.0f;

         /// <summary>
         /// Point lights that also get screen-space shadows: the highest ranked of the shadowed ones, each traced
         /// within its sphere's rectangle on screen. They carry what the atlas cannot resolve, grass blades and
         /// flowers. 0 = only the sun has them.
         /// </summary>
         [ValueRange(0, 4)]
         public int ScreenShadowedPointLights { get; set; } = 4;

         [Category("Ambient Occlusion")]                                    // GTAO (spatial-only, no TAA)
         public bool EnableGTAO { get; set; } = true;

         [ValueRange(0.1f, 4f)]
         public float GTAORadius { get; set; } = 1.0f;                      // World-space radius (meters)

         [ValueRange(0f, 1f)]
         public float GTAOIntensity { get; set; } = 1.0f;

         [ValueRange(0f, 1f)]
         public float GTAODirectStrength { get; set; } = 0.5f;              // AO applied to direct light (0 = ambient only, physically correct)

         [ValueRange(0.5f, 4f)]
         public float GTAOPower{ get; set; } = 1.5f;                       // Visibility exponent (higher = darker)

         [ValueRange(1, 8)]
         public int GTAOSlices { get; set; } = 8;                           // Horizon slices per pixel (quality vs cost)

         /// <summary>
         /// Trace and denoise AO at half resolution, then upsample it depth-aware to the screen. At 3 slices it shimmered
         /// in motion; at 8 slices the user found it looks great and stays cheap, so on by default (2026-10-16).
         /// </summary>
         public bool GTAOHalfResolution { get; set; } = true;

         [Category("Displacement")]                                      // Screen-Space Displacement Mapping (SSDM)
         public bool EnableSSDM { get; set; } = true;
         public bool UseSSDMPyramid { get; set; } = true;

         [ValueRange(0f, 30f)]
         public float SSDMHeightScale { get; set; } = 1f;

         [ValueRange(1f, 128f)]
         public float PixelErrorThreshold { get; set; } = 128.0f;

         [Category("Fog")]
         public bool Fog { get; set; } = true;                              // Aerial perspective: the atmosphere and its ground fog between camera and scene (Graphics.Atmosphere)

         // Camera exposure as EV100: light in physical units (lux, cd/m²) times 1 / (1.2 · 2^EV) is what the
         // tonemapper sees; higher = darker image. Auto: metered from the scene each frame (Graphics.AutoExposure).
         // Off: ExposureEV, which the captures lock (13.36 = Atmosphere.ReferenceEV shows light sources in the
         // engine's older units exactly as before).
         // The sky's ambient light reaches only as much of a surface as sees the sky past the terrain (a cave under a
         // hill gets none: sky_common.fx SkyVisibility). Off: every surface gets the whole sky (before stage 4).
         [Category("Ambient")]
         public bool SkyVisibility { get; set; } = true;
         // With SkyVisibility: the meshes round the camera shade the sky too (walls, roofs, tree crowns; a room is lit
         // only through its door and windows), from depth maps drawn from 16 directions of the sky (SkyCapture).
         // Off: only the terrain shades it.
         public bool SkyCaptures { get; set; } = true;
         // With SkyCaptures: the least share of the sky's ambient light a place the meshes cover keeps, a stand-in for
         // daylight bouncing in through doors and windows (a room without a lamp was black at noon). Caves under the
         // terrain do not get it.
         public float SkyCaptureBounce { get; set; } = 0.04f;

         [Category("Exposure")]
         public bool AutoExposure { get; set; } = true;

         [ValueRange(-6f, 20f)]
         public float ExposureEV { get; set; } = Graphics.Atmosphere.ReferenceEV;

         // EV added to the metering, + = brighter. The metering puts the scene's average at the standard 12.5 % grey
         // (K = 12.5), darker than this engine's look was tuned at: the default brings a sunny noon back to ~EV 13.4.
         [ValueRange(-4f, 6f)]
         public float ExposureCompensation { get; set; } = 1.35f;

         // Dim light: below ExposureDimLightEV (the metered EV100 before compensation; a clear noon meters ~14.8, an
         // overcast one ~14) the exposure follows the scene only partly. Of each EV the scene loses there, this share
         // still darkens the image, so dusk reads darker than day and a moonlit night darker than dusk; 0 = the eye
         // adapts fully (dusk as bright as noon until ExposureMinEV, then a sudden fall to black).
         [ValueRange(0f, 1f)]
         public float ExposureDimLightDarkening { get; set; } = 0.3f;

         [ValueRange(0f, 16f)]
         public float ExposureDimLightEV { get; set; } = 12f;

         // The darkest and brightest the auto exposure goes: how bright a moonless night may get (MinEV) is a choice
         [ValueRange(-8f, 20f)]
         public float ExposureMinEV { get; set; } = -1f;

         [ValueRange(-8f, 20f)]
         public float ExposureMaxEV { get; set; } = 16f;

         // Adaptation rates (1/s, of the remaining EV difference): up = toward a darker image (into the light, fast),
         // down = toward a brighter one (into the dark, slow, as the eye)
         [ValueRange(0.05f, 20f)]
         public float ExposureSpeedUp { get; set; } = 3f;

         [ValueRange(0.05f, 20f)]
         public float ExposureSpeedDown { get; set; } = 1f;

         // How far the eye adapts to light sources (emissive materials, flames, embers, VFX glow) in the dark: below
         // EV 13.36 (Atmosphere.ReferenceEV, a sunny noon) their pre-exposure follows the camera's only by 1 - this.
         // 0 = physically (a 40000-nit flame blows out white at night), 1 = they look at night as at noon. 0.8 is the
         // user's pick (2026-10-18): flames and lantern glass a little hotter than with 1. Graphics.Atmosphere.EmissiveExposure.
         [ValueRange(0f, 1f)]
         public float EmissiveAdaptation { get; set; } = 0.8f;

         [Category("Bloom")]
         public bool EnableBloom { get; set; } = true;

         [ValueRange(0f, 5f)]
         public float BloomThreshold { get; set; } = 0f;

         [ValueRange(0f, 1f)]
         public float BloomSoftKnee { get; set; } = 0.3f;

         [ValueRange(0f, 2f)]
         public float BloomIntensity { get; set; } = 0.3f;

         // true: BloomIntensity (0-1) is the share of each pixel's light above the threshold that is spread
         // out; the image keeps its total brightness. false: the bloom is added on top (mip 0 holds the sum of
         // all mips, so a flat area gains about MipCount × BloomIntensity of its own light).
         public bool BloomSpreadsLight { get; set; } = true;

         [ValueRange(0.5f, 3f)]
         public float BloomRadius { get; set; } = 1.0f;

         [Category("Radiance Cascades")]                                    // Sparse 3D GI
         public bool EnableRadianceCascades { get; set; } = false;

         [ValueRange(0f, 5f)]
         public float RCIntensity { get; set; } = 1.0f;

         /// <summary>
         /// Blades per clump where a terrain decorator variant's distance band draws Blades (grass_blades.fx):
         /// one clump stands in for one card.
         /// </summary>
         [Category("Grass")]
         [ValueRange(1, 28)]
         public int GrassBladesPerClump { get; set; } = 24;

         /// <summary>
         /// Ignore the environment's wind and sway vegetation in OverrideWindStrength / OverrideWindDirection
         /// instead (Graphics.Wind), for tuning the wind animation without switching presets.
         /// </summary>
         [Category("Wind")]
         public bool OverrideWind { get; set; } = false;

         /// <summary>Wind speed (m/s) while OverrideWind is on.</summary>
         [ValueRange(0f, 25f)]
         public float OverrideWindStrength { get; set; } = 6f;

         /// <summary>Direction the wind blows towards (degrees, as EnvironmentPreset) while OverrideWind is on.</summary>
         [ValueRange(0f, 360f)]
         public float OverrideWindDirection { get; set; } = 45f;

         [Category("Debug")]
         public DebugVizMode DebugVisualizationMode { get; set; } = DebugVizMode.Off; // F5 - Debug viz
         public int RCDebugSubMode { get; set; } = 1;                       // 0=off, 1=tile lookup, 2=per-level radiance, 3=constant bypass, 4=raw pool readback

         public System.Numerics.Matrix4x4 FrozenViewProjection { get; set; } // VP matrix when frustum frozen
    }
}
