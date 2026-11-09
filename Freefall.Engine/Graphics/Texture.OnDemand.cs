using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Graphics
{
    // ── Textures loaded on demand (TextureImporter.LoadOnDemand) ──
    // The terrain's brush heightmaps: read only by a height bake, which runs when a stamp over them changes or the saved
    // heights do not match the stamps. The asset loads without its GPU copy; whoever reads it on the GPU acquires it first
    // and releases it after, and the copy is there only while somebody holds it (and, in the editor, a while after, within
    // OnDemandTextures.IdleBudget). compute_prepasses.md "Brush heightmaps".
    public partial class Texture
    {
        private string? _demandCache;               // the cache file the GPU copy is read from
        private int _readers;
        private Task<CpuTextureData?>? _demandLoad;  // the CPU copy being read on a worker
        private bool _demandFailed;
        internal long DemandBytes;                  // what the GPU copy takes while it is resident
        internal LinkedListNode<Texture>? IdleNode; // in OnDemandTextures' idle list (resident, nobody holding it)

        /// <summary>
        /// Imported with LoadOnDemand: loaded without its GPU copy. That copy is made while somebody holds the texture
        /// (<see cref="Acquire"/>) and goes when nobody does, and its <see cref="BindlessIndex"/> is 0 meanwhile. Code that
        /// reads such a texture on the GPU acquires it, waits for <see cref="GpuReady"/>, reads BindlessIndex then, and
        /// releases it once its list is recorded (a copy that goes is disposed after the frames in flight).
        /// A material sampling it sees the null descriptor.
        /// </summary>
        public bool OnDemand => _demandCache != null;

        /// <summary>How many hold the texture (<see cref="Acquire"/> without <see cref="Release"/> yet).</summary>
        public int Readers => _readers;

        /// <summary>The cache file of an on-demand texture could not be read: it stays without a GPU copy.</summary>
        public bool DemandFailed => _demandFailed;

        /// <summary>An on-demand texture over this cache file, without its GPU copy (TextureLoader).</summary>
        internal static Texture OnDemandFrom(string cachePath, string name)
            => new() { _demandCache = cachePath, Name = name, AssetPath = name };

        /// <summary>
        /// Count a reader of the GPU copy and start loading it if it is not resident (read on a worker, uploaded once
        /// <see cref="GpuReady"/> sees it read). Pair every call with <see cref="Release"/>. No-op for a texture that is not
        /// on demand: it is always resident. Main thread.
        /// </summary>
        public void Acquire()
        {
            if (!OnDemand) return;
            if (_readers++ == 0) OnDemandTextures.LeaveIdle(this);
            StartLoad();
        }

        /// <summary>
        /// One reader fewer. When none is left the copy goes into the idle list: kept while it fits
        /// <see cref="OnDemandTextures.IdleBudget"/>, the least recently released ones going first, and dropped after the
        /// frames in flight. A load still on its way is dropped at once. Main thread.
        /// </summary>
        public void Release()
        {
            if (!OnDemand) return;
            if (_readers <= 0)
            {
                Debug.LogWarning("Texture", $"'{Name}' released more often than acquired: ignored");
                return;
            }
            if (--_readers == 0) OnDemandTextures.Idle(this);
        }

        /// <summary>
        /// A list recorded now may read <see cref="BindlessIndex"/>: the GPU copy is resident and its upload has run. Also
        /// true when the copy cannot be made (<see cref="DemandFailed"/>: BindlessIndex stays 0, which the readers take as
        /// "none"), and always for a texture that is not on demand. Takes the worker's read in and uploads it when it is
        /// done; asked while held. Main thread.
        /// </summary>
        public bool GpuReady
        {
            get
            {
                if (!OnDemand || _demandFailed) return true;
                if (_demandLoad != null)
                {
                    if (!_demandLoad.IsCompleted) return false;
                    FinishLoad();
                    if (_demandFailed) return true;
                }
                if (_resource == null)
                {
                    // Dropped while held (evicted, reimported): load it again
                    if (_readers > 0) StartLoad();
                    return false;
                }
                return IsReady(StreamingManager.Instance?.GetCompletedFence() ?? long.MaxValue);
            }
        }

        /// <summary>
        /// Block until <see cref="GpuReady"/>: the read waited for, the upload flushed. For code that records its own list
        /// and submits it at once (the readback's scratch bakes, checks, thumbnails). Acquire first. Main thread.
        /// </summary>
        public void WaitGpuReady() => OnDemandTextures.WaitGpuReady(new[] { this });

        // The worker's read waited for and uploaded (the upload still on its way); nothing when there is none
        internal void FinishReadNow()
        {
            if (!OnDemand || _demandFailed) return;
            if (_resource == null && _readers > 0) StartLoad();
            if (_demandLoad == null) return;
            try { _demandLoad.Wait(); } catch { /* FinishLoad reports it */ }
            FinishLoad();
        }

        private void StartLoad()
        {
            if (_resource != null || _demandLoad != null || _demandFailed) return;
            string path = _demandCache!, name = Name;
            _demandLoad = Task.Run(() => Assets.Loaders.TextureLoader.ReadCpuData(path, name));
        }

        // The CPU copy read on a worker goes up: the resource and its slot now, the pixels on the copy queue
        private void FinishLoad()
        {
            var load = _demandLoad!;
            _demandLoad = null;
            CpuTextureData? cpu = null;
            try { cpu = load.Result; }
            catch (Exception ex)
            {
                Debug.LogWarning("Texture", $"'{Name}' (on demand) could not be read: {ex.GetBaseException().Message}");
            }
            if (cpu == null)
            {
                _demandFailed = true;
                return;
            }

            var device = Engine.Device;
            CreateGpuCopy(device, cpu, trackUpload: true);
            DemandBytes = (long)device.NativeDevice.GetResourceAllocationInfo(0, _resource.Description).SizeInBytes;
            OnDemandTextures.Loaded(this);
        }

        /// <summary>
        /// Drop the GPU copy: its resource and its bindless slot go once the frames in flight are done with them, and a
        /// load on its way is dropped with it. Main thread.
        /// </summary>
        internal void DropGpuCopy()
        {
            _demandLoad = null;
            if (_resource == null) return;
            var device = Engine.Device;
            device.DeferReleaseBindlessIndex(BindlessIndex);
            device.DeferDispose(_resource);
            _resource = null;
            BindlessIndex = 0;
            SrvHandle = default;
            SrvCpuHandle = default;
            OnDemandTextures.Dropped(this);
            DemandBytes = 0;
        }

        // Evicted from the asset cache: the copy goes after the frames in flight; a reader still holding it loads it again
        private void DisposeOnDemand()
        {
            OnDemandTextures.LeaveIdle(this);
            DropGpuCopy();
        }

        /// <summary>
        /// Hot reload (TextureLoader.Reload): read from this cache file from now on. The copy there is dropped after the
        /// frames in flight, and loaded again from the new file when somebody holds the texture. Main thread.
        /// </summary>
        internal void ReloadOnDemand(string cachePath)
        {
            _demandCache = cachePath;
            _demandFailed = false;
            OnDemandTextures.LeaveIdle(this);
            DropGpuCopy();
            if (_readers > 0) StartLoad();
        }
    }

    /// <summary>
    /// The GPU copies of the on-demand textures (<see cref="Texture.OnDemand"/>): how many and how large, and the idle
    /// ones nobody holds, kept within <see cref="IdleBudget"/> so a stamp edited again soon finds its brush still there.
    /// Every scene load drops the idle ones. Main thread.
    /// </summary>
    public static class OnDemandTextures
    {
        private static readonly LinkedList<Texture> _idle = new();
        private static long _idleBytes;
        private static bool _hooked;

        /// <summary>
        /// Bytes of GPU copies kept after their last reader let go, the least recently released going first beyond it.
        /// 0 (the player) drops a copy as soon as nobody holds it; the editor keeps a few brushes (EditorApp).
        /// </summary>
        public static long IdleBudget { get; set; }

        /// <summary>On-demand textures with a GPU copy now, and the bytes those copies take.</summary>
        public static int ResidentCount { get; private set; }
        public static long ResidentBytes { get; private set; }

        /// <summary>GPU copies made since the start (loads, not reuses from the idle list).</summary>
        public static int Loads { get; private set; }

        internal static void Loaded(Texture texture)
        {
            ResidentCount++;
            ResidentBytes += texture.DemandBytes;
            Loads++;
            if (!_hooked)
            {
                _hooked = true;
                SceneManager.LoadStarted += _ => DropIdle();
            }
        }

        internal static void Dropped(Texture texture)
        {
            ResidentCount--;
            ResidentBytes -= texture.DemandBytes;
        }

        // Nobody holds it any more: a resident copy waits in the idle list, a load still on its way goes
        internal static void Idle(Texture texture)
        {
            if (texture.Native == null)
            {
                texture.DropGpuCopy();
                return;
            }
            texture.IdleNode = _idle.AddLast(texture);
            _idleBytes += texture.DemandBytes;
            Trim(IdleBudget);
        }

        internal static void LeaveIdle(Texture texture)
        {
            if (texture.IdleNode == null) return;
            _idle.Remove(texture.IdleNode);
            texture.IdleNode = null;
            _idleBytes -= texture.DemandBytes;
        }

        /// <summary>Drop idle copies, the least recently released first, until they take at most <paramref name="budget"/> bytes.</summary>
        public static void Trim(long budget)
        {
            while (_idleBytes > budget && _idle.First is { } oldest)
            {
                var texture = oldest.Value;
                LeaveIdle(texture);
                texture.DropGpuCopy();
            }
        }

        /// <summary>
        /// Block until every one of these is <see cref="Texture.GpuReady"/>: their reads (started together by Acquire, on
        /// workers) waited for and uploaded, then one flush of the copy queue. For code that records its own list and
        /// submits it at once. Acquire them first. Main thread.
        /// </summary>
        public static void WaitGpuReady(IReadOnlyList<Texture> textures)
        {
            bool ready = true;
            foreach (var texture in textures)
            {
                texture.FinishReadNow();
                ready &= texture.GpuReady;
            }
            if (!ready)
                StreamingManager.Instance?.Flush();
        }

        /// <summary>Drop every idle copy (a scene load: what the last scene's edits kept is not this one's).</summary>
        public static void DropIdle() => Trim(0);

        /// <summary>For engine_stats.</summary>
        public static object Describe() => new
        {
            resident = ResidentCount,
            residentMiB = Math.Round(ResidentBytes / 1048576.0, 1),
            idle = _idle.Count,
            idleMiB = Math.Round(_idleBytes / 1048576.0, 1),
            idleBudgetMiB = Math.Round(IdleBudget / 1048576.0, 1),
            loads = Loads,
        };
    }
}
