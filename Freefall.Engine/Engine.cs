using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System.Numerics;
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

        // Main-thread work queue for marshaling from background threads
        private sealed class WorkItem
        {
            public Action Action;
            public TaskCompletionSource<bool> Tcs;
        }

        private static readonly ConcurrentQueue<WorkItem> _mainThreadQueue = new ConcurrentQueue<WorkItem>();

        public static Task RunOnMainThreadAsync(Action action)
        {
            if (action == null) return Task.CompletedTask;
            var tcs = new TaskCompletionSource<bool>();
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
            
            RootDirectory = Directory.GetCurrentDirectory(); 

            Device = new GraphicsDevice();
            
            // Initialize persistent transform buffer for GPU-driven rendering
            TransformBuffer.Initialize(Device);
            
            // Initialize streaming manager before assets (AssetManager creates default textures that need it)
            var streaming = new StreamingManager(Device);
            
            // Centralized Asset Manager Init
            Assets = new AssetManager(Device);

            // Engine-internal default textures & materials
            Freefall.Assets.InternalAssets.Initialize(Device);
            
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

            Device = new GraphicsDevice();
            TransformBuffer.Initialize(Device);

            var streaming = new StreamingManager(Device);
            Assets = new AssetManager(Device);
            Freefall.Assets.InternalAssets.Initialize(Device);

            // Initialize Rendering Foundation — view auto-registers in RenderView.All
            var view = new RenderView(handle, width, height, Device);
            view.Pipeline = new DeferredRenderer();
            view.Pipeline.Initialize(width, height);
            
            Input.Init(handle);
            
            AudioDevice = XAudio2.XAudio2Create(ProcessorSpecifier.DefaultProcessor, true);
            _masteringVoice = AudioDevice.CreateMasteringVoice();
            Audio3D = new X3DAudio(Speakers.FrontLeft | Speakers.FrontRight);
            Debug.Log("[Engine] Initialized (editor mode)");
            
            PhysicsWorld.Initialize();
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

        public static void Tick()
        {
            TickCount++;

            Time.Update();
            Input.Update();

            // Release GPU buffers from previous batch resizes (deferred N frames for safety)
            Graphics.InstanceBatch.FlushDeferredDisposals();


            // Flush pending entity additions before Update/Render
            Base.EntityManager.FlushPending();

            // Physics (fixed-timestep)
            PhysicsWorld.Update((float)Time.Delta);
            
            // Update Logic
            Update();

            // --- Process all deferred resizes before rendering (Apex pattern) ---
            // Must sync GPU first — previous frames may still be using resources
            bool anyResizePending = false;
            foreach (var v in RenderView.All)
            {
                if (v.IsResizePending) { anyResizePending = true; break; }
            }

            if (anyResizePending)
            {
                RenderView.Primary?.WaitForGpu();
                foreach (var v in RenderView.All)
                    v.ProcessPendingResize();
            }

            // --- Apex multi-viewport render loop ---
            Device.ResetTemporaryDescriptors();

            _renderViews.Clear();
            _renderViews.AddRange(RenderView.All);

            foreach (var view in _renderViews)
            {
                if (!view.Enabled) continue;
                if (!view.HasSwapChain) continue; // Headless views rendered manually (e.g. RenderGui)

                if (view.OnRender != null)
                {
                    view.OnRender(view);
                }
                else
                {
                    view.Prepare();

                    var cam = Camera.Main;
                    if (cam != null && cam.Target == view)
                        cam.Render(view.CommandList.Native);

                    view.Present();
                }
            }

            CommandBuffer.Clear();
            
            // Drain main-thread queue AFTER Present (GPU fence ensures no commands in flight,
            // so creating GPU resources like upload buffers and SRV descriptors is safe here).
            // Time-budgeted: process items until 4ms spent to avoid frame hitches.
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

            Input.ClearFrameCallbacks();
        }

        private static void Update()
        {
            // F1: Toggle VSync
            if (Input.IsKeyPressed(Keys.F1))
            {
                Settings.VSync = !Settings.VSync;
                Debug.Log($"[Engine] VSync {(Settings.VSync ? "Enabled" : "Disabled")}");
            }
            
            // F2: Toggle wireframe mode
            if (Input.IsKeyPressed(Keys.F2))
            {
                Settings.TerrainWireframe = !Settings.TerrainWireframe;
                Debug.Log($"[Engine] Wireframe {(Settings.TerrainWireframe ? "Enabled" : "Disabled")}");
            }
            
            // F3: Toggle Frustum Freeze - freezes culling frustum for debug visualization
            if (Input.IsKeyPressed(Keys.F3))
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
            
            // F4: Toggle adaptive shadow cascade splits (SDSM)
            if (Input.IsKeyPressed(Keys.F4))
            {
                Settings.UseAdaptiveSplits = !Settings.UseAdaptiveSplits;
                Debug.Log($"[Engine] Adaptive Splits {(Settings.UseAdaptiveSplits ? "Enabled" : "Disabled")}");
            }
            
            // F5: Cycle debug visualization mode (0=off, 1=cascade colors, 2=shadow factor, 3=depth, 4=x-ray occlusion)
            if (Input.IsKeyPressed(Keys.F5))
            {
                Settings.DebugVisualizationMode = (Settings.DebugVisualizationMode + 1) % 5;
                string[] modeNames = { "Off", "Cascade Colors", "Shadow Factor", "GBuffer Depth", "X-Ray Occlusion" };
                Debug.Log($"[Engine] Debug Viz: {modeNames[Settings.DebugVisualizationMode]}");
            }
            
            // F6: Toggle Hi-Z occlusion culling
            if (Input.IsKeyPressed(Keys.F6))
            {
                Settings.DisableHiZ = !Settings.DisableHiZ;
                Debug.Log($"[Engine] Hi-Z Occlusion {(Settings.DisableHiZ ? "DISABLED" : "ENABLED")}");
            }
            
            // Entity logic (includes component Updates like Camera, CharacterController, etc.)
            EntityManager.Update();
        }

        public static void Shutdown()
        {
            _isRunning = false;
            
            PhysicsWorld.Shutdown();
            
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

    public class EngineSettings
    {
         public bool VSync { get; set; } = true;                            // F1 - VSync
         public bool TerrainWireframe { get; set; } = false;                // F2 - Global wireframe
         public bool FreezeFrustum { get; set; } = false;                   // F3 - Freeze culling frustum
         public bool UseAdaptiveSplits { get; set; } = false;               // F4 - SDSM adaptive cascade splits
         public int  DebugVisualizationMode { get; set; } = 0;              // F5 - Debug viz (0-4)
         public bool DisableHiZ { get; set; } = false;                      // F6 - Disable Hi-Z occlusion
         public System.Numerics.Matrix4x4 FrozenViewProjection { get; set; } // VP matrix when frustum frozen
    }
}
