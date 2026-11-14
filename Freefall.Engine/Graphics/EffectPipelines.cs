using System;
using System.Threading;
using System.Threading.Tasks;

namespace Freefall.Graphics
{
    /// <summary>
    /// The pipeline states of an effect's passes. Every material of an effect draws with the same ones: they are made
    /// from the effect's shaders and render state alone (what a material adds is its constant buffers and textures),
    /// so they belong to the effect, are made once per compile, on a worker, from the moment the effect is compiled
    /// (an engine effect at start, a project effect when it loads, either on a hot reload), and are shared by every
    /// instance of its key like its techniques.
    /// <para>
    /// Making a pipeline state is where the driver compiles the shaders for the GPU, and it keeps the result between
    /// runs only in its own cache (NVIDIA writes it when the process exits). Until 2026-11-14 every material made its
    /// own on the main thread when it was loaded: the first spell look on vfx_volume froze the game for 757 ms
    /// (three passes and their wireframes) and the first on vfx_layers for 378 ms, whenever the driver had none saved
    /// (after a shader edit, a driver update, a player's first run); a second material of the effect then took 2 ms.
    /// </para>
    /// </summary>
    internal sealed class EffectPipelines
    {
        private static readonly ProfilerMarker WaitMarker = ProfilerMarker.Get("Effect pipelines wait");
        private const double SlowBuildMs = 250;   // as AssetManager's slow loads

        private Task<Set>? _build;
        private Set? _ready;

        /// <summary>
        /// Starts making the pipeline states of <paramref name="effect"/>'s compile on a worker. What an earlier compile
        /// made is released once its build is done (deferred: frames in flight may still draw with it). Main thread,
        /// after the effect's techniques and render state are in place; nothing is made for an effect without a pass.
        /// </summary>
        internal void Start(Effect effect)
        {
            var device = Freefall.Engine.Device;
            Retire(device);
            if (device == null || effect.Techniques.Count == 0 || effect.Techniques[0].Passes.Count == 0)
                return;

            // What the build reads, taken now: a hot reload replaces the effect's lists and render state in place
            var technique = effect.Techniques[0];
            var renderState = effect.RenderState;
            var blend = effect.BlendState;
            var name = effect.Name;
            _build = Task.Run(() => Build(technique, renderState, blend, device, name));
        }

        private static Set Build(EffectTechnique technique, ShaderRenderState renderState, BlendState blend,
            GraphicsDevice device, string? name)
        {
            try
            {
                long start = System.Diagnostics.Stopwatch.GetTimestamp();
                var set = Material.BuildPipelines(technique, renderState, blend, device, name);
                double ms = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                if (ms >= SlowBuildMs)   // the driver compiled them (none saved from an earlier run): off the main thread now
                    Debug.LogAlways($"[Effect] {name}: {set.Solid.Length} pipeline state(s) made on a worker in {ms:0} ms");
                return set;
            }
            catch (Exception ex)
            {
                Debug.LogError("Effect", $"{name}: its pipeline states were not made; its materials draw nothing: {ex.Message}");
                return new Set(0);
            }
        }

        /// <summary>
        /// The pipeline states draws use. A draw that comes before the build is done waits for it (the time it still
        /// needs, once per compile: the profiler's "Effect pipelines wait"). Null for an effect with no compiled pass.
        /// </summary>
        internal Set? Ready
        {
            get
            {
                var ready = Volatile.Read(ref _ready);
                if (ready != null) return ready;
                var build = _build;
                if (build == null) return null;
                if (!build.IsCompleted)
                    using (WaitMarker.Auto()) build.Wait();
                ready = build.Result;
                Volatile.Write(ref _ready, ready);
                return ready;
            }
        }

        private void Retire(GraphicsDevice? device)
        {
            var old = _build?.Result ?? _ready;   // a build still running is waited for: its pipelines are released too
            _build = null;
            _ready = null;
            old?.Release(device);
        }

        /// <summary>One compile's pipeline states, by pass index.</summary>
        internal sealed class Set
        {
            /// <summary>What each pass draws with; null where it could not be made (the pass draws nothing).</summary>
            public readonly PipelineState?[] Solid;

            /// <summary>
            /// A Shadow pass drawn into the sky captures (SkyCapture.cs): the same shaders with a small, clamped slope
            /// bias. Null for other passes.
            /// </summary>
            public readonly PipelineState?[] Capture;

            // The editor's wireframe view: made on a worker the first time a draw asks, the solid one drawn meanwhile
            internal readonly Func<PipelineState?>?[] MakeWireframe;
            private PipelineState?[]? _wireframes;
            private Task? _wireframeBuild;

            public Set(int passes)
            {
                Solid = new PipelineState?[passes];
                Capture = new PipelineState?[passes];
                MakeWireframe = new Func<PipelineState?>?[passes];
            }

            /// <summary>The wireframe variant of a pass, or null until it is made (the first ask starts making them all).</summary>
            public PipelineState? WireframeFor(int pass)
            {
                var made = Volatile.Read(ref _wireframes);
                if (made == null)
                {
                    var fresh = new PipelineState?[Solid.Length];
                    if (Interlocked.CompareExchange(ref _wireframes, fresh, null) == null)
                        _wireframeBuild = Task.Run(() =>
                        {
                            for (int i = 0; i < fresh.Length; i++)
                            {
                                try { Volatile.Write(ref fresh[i], MakeWireframe[i]?.Invoke()); }
                                catch (Exception ex) { Debug.LogWarning("Effect", $"A wireframe pipeline state was not made: {ex.Message}"); }
                            }
                        });
                    return null;
                }
                return Volatile.Read(ref made[pass]);
            }

            internal void Release(GraphicsDevice? device)
            {
                try { _wireframeBuild?.Wait(); } catch (AggregateException) { }
                foreach (var pso in Solid) Dispose(pso);
                foreach (var pso in Capture) Dispose(pso);
                if (_wireframes != null)
                    foreach (var pso in _wireframes) Dispose(pso);

                // Only the native object: PipelineState.Dispose() also releases its root signature, the device's
                // shared GlobalRootSignature (disposing it crashes the next draw)
                void Dispose(PipelineState? pso)
                {
                    if (pso?.Native == null) return;
                    if (device != null) device.DeferDispose(pso.Native);
                    else pso.Native.Dispose();
                }
            }
        }
    }
}
