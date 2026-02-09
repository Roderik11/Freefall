using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Assets;
using Freefall.Graphics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall
{
    public static class Engine
    {
        private static Window _window;
        public static RenderView RenderView { get; private set; } // Changed to public property
        private static bool _isRunning;

        public static AssetManager Assets { get; private set; }
        public static GraphicsDevice Device { get; private set; }
        public static EngineSettings Settings { get; private set; }
        public static string RootDirectory { get; private set; }
        
        public static int TickCount { get; private set; }
        
        /// <summary>
        /// The current frame buffer index (0, 1, or 2). Uses swapchain's index for proper GPU synchronization.
        /// Falls back to TickCount % 3 before RenderView is initialized.
        /// </summary>
        public static int FrameIndex => RenderView?.FrameIndex ?? (TickCount % 3);
        
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

        public static void Initialize(string title, int width, int height)
        {
            // Set high-resolution timer for smooth high-FPS gameplay
            Kernel32.timeBeginPeriod(1);
            
            RootDirectory = Directory.GetCurrentDirectory(); 

            _window = new Window(title, width, height);
            Device = new GraphicsDevice();
            
            // Initialize persistent transform buffer for GPU-driven rendering
            TransformBuffer.Initialize(Device);
            
            // Initialize streaming manager before assets (AssetManager creates default textures that need it)
            var streaming = new StreamingManager(Device);
            
            // Centralized Asset Manager Init
            Assets = new AssetManager(Device);

            // Engine-internal default textures & materials
            Freefall.Assets.InternalAssets.Initialize(Device);

            // Initialize Rendering Foundation
            RenderView = new RenderView(_window, Device);
            RenderView.Pipeline = new DeferredRenderer(); // Use Deferred
            RenderView.Pipeline.Initialize(_window.Width, _window.Height);
            
            Input.Init(_window.Handle);
            
            // Initialize Subsystems
            // Audio.Initialize(Device); // Stub
            // Physics.Initialize(); // Stub
        }

        public static void Run()
        {
            _isRunning = true;
            Time.Initialize();
            
            float titleUpdateTimer = 0;
            string baseTitle = _window.Title;
            while (_isRunning && _window.ProcessEvents())
            {
                Time.Update();
                Input.Update(_window.Handle);
                
                Tick();

                // Update performance metrics in title bar every 0.1s for responsiveness
                titleUpdateTimer += (float)Time.Delta;
                if (titleUpdateTimer >= 0.1f)
                {
                    string vsyncStatus = Settings.VSync ? "ON" : "OFF";
                    string hizStatus = Settings.DisableHiZ ? "OFF" : "ON";
                    int occluded = CommandBuffer.Culler?.LastHiZOccludedCount ?? 0;
                    _window.SetTitle($"{baseTitle} | FPS: {Time.FPS:0} | Draw: {CommandBuffer.LastDrawCallCount} | Batches: {CommandBuffer.LastBatchCount} | HiZ: {hizStatus} Occl: {occluded}");
                    titleUpdateTimer = 0;
                }
                
                Input.ClearFrameCallbacks();
            }
            
            Shutdown();
        }

        public static void Tick()
        {
            TickCount++;

            // Release GPU buffers from previous batch resizes (deferred N frames for safety)
            Graphics.InstanceBatch.FlushDeferredDisposals();

            // Flush pending entity additions before Update/Render
            Base.EntityManager.FlushPending();

            // Update Logic
            Update();

            // Render Logic
            Render();
            
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
            
            // F9: Toggle Hi-Z occlusion culling
            if (Input.IsKeyPressed(Keys.F6))
            {
                Settings.DisableHiZ = !Settings.DisableHiZ;
                Debug.Log($"[Engine] Hi-Z Occlusion {(Settings.DisableHiZ ? "DISABLED" : "ENABLED")}");
            }
            // Entity logic (includes component Updates like Camera, CharacterController, etc.)
            EntityManager.Update();
        }

        private static void Render()
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            
            // Reset per-frame descriptor allocations
            Device.ResetTemporaryDescriptors();
            
            var prepareTime = System.Diagnostics.Stopwatch.StartNew();
            RenderView.Prepare();
            prepareTime.Stop();
            
            // Use the main camera if available
            var cam = Camera.Main;
            var renderTime = System.Diagnostics.Stopwatch.StartNew();
            if (cam != null)
            {
                cam.Render(RenderView.CommandList.Native);
            }
            renderTime.Stop();
            
            var presentTime = System.Diagnostics.Stopwatch.StartNew();
            RenderView.Present();
            presentTime.Stop();

            // Clear commands for next frame
            CommandBuffer.Clear();
            
            sw.Stop();
            
            // Log every 60 frames
            if (FrameIndex % 60 == 0)
            {
                Debug.Log($"[Render] Total: {sw.Elapsed.TotalMilliseconds:F2}ms | Prepare: {prepareTime.Elapsed.TotalMilliseconds:F2}ms | Camera.Render: {renderTime.Elapsed.TotalMilliseconds:F2}ms | Present: {presentTime.Elapsed.TotalMilliseconds:F2}ms");
            }
        }

        public static void Shutdown()
        {
            _isRunning = false;
            
            // Restore timer resolution
            Kernel32.timeEndPeriod(1);
            
            TransformBuffer.Instance?.Dispose();
            StreamingManager.Instance?.Dispose();
            RenderView?.Dispose();
            Assets?.Dispose();
            Device?.Dispose();
            _window?.Dispose();
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
