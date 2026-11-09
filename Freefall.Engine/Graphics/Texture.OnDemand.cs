using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Graphics
{
    // ── Textures loaded on demand ──
    // Two kinds load without their GPU copy and have it only while somebody holds them (and a while after, within
    // OnDemandTextures.IdleBudget):
    // - imported with TextureImporter.LoadOnDemand: the terrain's brush heightmaps, read only by a height bake
    //   (compute_prepasses.md "Brush heightmaps");
    // - material textures (cold): a texture only a material's loader asked for, of at least OnDemandTextures.ColdMinBytes.
    //   Materials hold their textures while something draws with them (Material.Acquire, Touch) and show a fallback
    //   until the copy has arrived. Anything else that loads such a texture (the UI, a terrain layer, a script) makes it
    //   resident for good at once (Pin). rendering_engine/performance/vram.md "Textures only while drawn".
    // All of it under OnDemandTextures.Sync: holds come from parallel draws and loads from workers.
    public partial class Texture
    {
        private string? _demandCache;               // the cache file the GPU copy is read from
        private int _readers;
        private Task<CpuTextureData?>? _demandLoad;  // the CPU copy being read on a worker
        private bool _demandFailed;
        private bool _materialCold;                 // cold because only a material asked for it: Pin makes it resident
        internal long DemandBytes;                  // what the GPU copy takes while it is resident
        internal LinkedListNode<Texture>? IdleNode; // in OnDemandTextures' idle list (resident, nobody holding it)
        internal bool InLoading;                    // in OnDemandTextures' list of copies on their way (Update)
        internal int TrackedTick;                   // Engine.TickCount when it went on its way (for the arrival times)
        internal bool PinPending;                   // pinned, its holders not told yet (Update)

        // Materials holding this texture (Material.Acquire): told when its copy arrives or goes. One entry per hold.
        internal readonly List<Material> DemandHolders = new();

        /// <summary>
        /// Loaded without its GPU copy: imported with LoadOnDemand, or a material texture loaded cold. That copy is made
        /// while somebody holds the texture (<see cref="Acquire"/>) and goes when nobody does, and its
        /// <see cref="BindlessIndex"/> is 0 meanwhile. Code that reads such a texture on the GPU acquires it, waits for
        /// <see cref="GpuReady"/>, reads BindlessIndex then, and releases it once its list is recorded (a copy that goes is
        /// disposed after the frames in flight). Materials do this themselves (Material.Acquire).
        /// </summary>
        public bool OnDemand => _demandCache != null;

        /// <summary>Loaded cold for a material (not imported with LoadOnDemand): made resident for good by any other load.</summary>
        public bool MaterialCold => _materialCold;

        /// <summary>How many hold the texture (<see cref="Acquire"/> without <see cref="Release"/> yet).</summary>
        public int Readers => _readers;

        /// <summary>The cache file of an on-demand texture could not be read: it stays without a GPU copy.</summary>
        public bool DemandFailed => _demandFailed;

        /// <summary>Its cache file is being read on a worker.</summary>
        internal bool ReadInFlight => _demandLoad != null;

        /// <summary>
        /// The GPU copy is resident and its upload has run, as <see cref="OnDemandTextures.Update"/> last saw (or a
        /// <see cref="OnDemandTextures.WaitGpuReady"/>). What materials bind instead of their fallback.
        /// </summary>
        public bool Arrived { get; internal set; }

        /// <summary>An on-demand texture over this cache file, without its GPU copy (TextureLoader, LoadOnDemand).</summary>
        internal static Texture OnDemandFrom(string cachePath, string name)
            => new() { _demandCache = cachePath, Name = name, AssetPath = name };

        /// <summary>A material texture loaded cold (TextureLoader): on demand until something other than a material loads it.</summary>
        internal static Texture ColdFrom(string cachePath, string name)
        {
            System.Threading.Interlocked.Increment(ref OnDemandTextures.LoadedColdCount);   // loaders run on workers too
            return new() { _demandCache = cachePath, _materialCold = true, Name = name, AssetPath = name };
        }

        /// <summary>
        /// Count a reader of the GPU copy and start loading it if it is not resident (read on a worker, uploaded once
        /// <see cref="GpuReady"/> sees it read). Pair every call with <see cref="Release"/>. No-op for a texture that is not
        /// on demand: it is always resident. Any thread.
        /// </summary>
        public void Acquire()
        {
            if (!OnDemand) return;
            lock (OnDemandTextures.Sync)
                if (OnDemand) AcquireLocked();
        }

        /// <summary>
        /// One reader fewer. When none is left the copy goes into the idle list: kept while it fits
        /// <see cref="OnDemandTextures.IdleBudget"/> (trimmed once a frame), the least recently released ones going first,
        /// and dropped after the frames in flight. A load still on its way is dropped at once. Any thread.
        /// </summary>
        public void Release()
        {
            if (!OnDemand) return;
            lock (OnDemandTextures.Sync)
                if (OnDemand) ReleaseLocked();
        }

        // A material's hold: counted like any reader, and the material is told when the copy arrives or goes
        internal void AcquireFor(Material material)
        {
            if (!OnDemand) return;
            DemandHolders.Add(material);
            AcquireLocked();
        }

        internal void ReleaseFor(Material material)
        {
            DemandHolders.Remove(material);
            if (OnDemand) ReleaseLocked();   // pinned meanwhile: resident for good, nothing to count
        }

        private void AcquireLocked()
        {
            if (_readers++ == 0) OnDemandTextures.LeaveIdle(this);
            StartLoad();
            if (!Arrived) OnDemandTextures.Track(this);
        }

        private void ReleaseLocked()
        {
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
                lock (OnDemandTextures.Sync)
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
        }

        /// <summary>
        /// Block until <see cref="GpuReady"/>: the read waited for, the upload flushed. For code that records its own list
        /// and submits it at once (the readback's scratch bakes, checks, thumbnails). Acquire first. Main thread.
        /// </summary>
        public void WaitGpuReady() => OnDemandTextures.WaitGpuReady(new[] { this });

        // The worker's read waited for and uploaded (the upload still on its way); nothing when there is none. Under the lock
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
        /// load on its way is dropped with it. The materials holding it bind their fallback at once. Main thread, under
        /// <see cref="OnDemandTextures.Sync"/>.
        /// </summary>
        internal void DropGpuCopy()
        {
            _demandLoad = null;
            bool wasArrived = Arrived;
            Arrived = false;
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

            // Still held (evicted, reimported): its holders show the fallback until the copy is back
            if (wasArrived) NotifyHolders();
            if (_readers > 0) OnDemandTextures.Track(this);
        }

        // The copy arrived or went (or the texture became resident for good): every material holding it writes its slots
        internal void NotifyHolders()
        {
            for (int i = 0; i < DemandHolders.Count; i++)
            {
                var material = DemandHolders[i];
                if (DemandHolders.IndexOf(material) == i)   // once per material, however many slots it holds this in
                    material.RefreshTextureIndices();
            }
        }

        /// <summary>
        /// Something other than a material loaded this cold material texture (the UI, a terrain layer, a script): it
        /// becomes resident for good, as if it had never been cold, read now if it is not there yet (its users read
        /// BindlessIndex at once). Explicit on-demand textures (LoadOnDemand) are never pinned. Any thread.
        /// </summary>
        public void Pin()
        {
            if (!_materialCold) return;
            lock (OnDemandTextures.Sync)
            {
                if (!_materialCold) return;
                _materialCold = false;
                OnDemandTextures.LeaveIdle(this);
                if (_resource != null)
                {
                    // Resident already (held, or idle): kept, no longer counted among the on-demand copies
                    OnDemandTextures.Dropped(this);
                    _demandLoad = null;
                    _demandCache = null;
                }
                else
                {
                    // Read now (the read on its way if there is one) and made as any texture is: ready at once
                    CpuTextureData? cpu = null;
                    try
                    {
                        StartLoad();
                        cpu = _demandLoad?.Result;
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("Texture", $"'{Name}' could not be read: {ex.GetBaseException().Message}");
                    }
                    _demandLoad = null;
                    _demandCache = null;                                     // resident for good, like any texture
                    if (cpu != null) CreateGpuCopy(Engine.Device, cpu, trackUpload: false);
                }
                Arrived = false;
                OnDemandTextures.Pinned(this);
            }
        }

        // Evicted from the asset cache: the copy goes after the frames in flight; a reader still holding it loads it again
        private void DisposeOnDemand()
        {
            lock (OnDemandTextures.Sync)
            {
                OnDemandTextures.LeaveIdle(this);
                DropGpuCopy();
            }
        }

        /// <summary>
        /// Hot reload (TextureLoader.Reload): read from this cache file from now on. The copy there is dropped after the
        /// frames in flight, and loaded again from the new file when somebody holds the texture. Main thread.
        /// </summary>
        internal void ReloadOnDemand(string cachePath)
        {
            lock (OnDemandTextures.Sync)
            {
                _demandCache = cachePath;
                _demandFailed = false;
                OnDemandTextures.LeaveIdle(this);
                DropGpuCopy();
                if (_readers > 0)
                {
                    StartLoad();
                    OnDemandTextures.Track(this);
                }
            }
        }
    }

    /// <summary>
    /// The GPU copies of the on-demand textures (<see cref="Texture.OnDemand"/>): how many and how large, the ones on their
    /// way (taken in and told to their materials once a frame, <see cref="Update"/>), and the idle ones nobody holds, kept
    /// within <see cref="IdleBudget"/> so a stamp edited again soon or a person walking back into view finds them still
    /// there. Every scene load drops the idle ones.
    /// </summary>
    public static class OnDemandTextures
    {
        /// <summary>Guards every texture's and material's on-demand state.</summary>
        public static readonly object Sync = new();

        private static readonly LinkedList<Texture> _idle = new();
        private static readonly List<Texture> _loading = new();     // acquired, their copy not arrived yet
        private static readonly List<Texture> _pinned = new();      // made resident for good, their holders not told yet
        private static readonly List<Material> _leased = new();     // held by Material.Touch
        private static long _idleBytes;
        private static bool _hooked;

        /// <summary>
        /// Bytes of GPU copies kept after their last reader let go, the least recently released going first beyond it
        /// (trimmed once a frame). The editor and the player both keep 128 MiB: a person who walks out of view and back,
        /// or a spell cast again, finds its textures still there.
        /// </summary>
        public static long IdleBudget { get; set; } = 128L << 20;

        /// <summary>
        /// Material textures load cold (on demand) at all. Off, a material's textures load resident as every texture did
        /// before 2026-11-09. Takes effect for textures loaded from then on.
        /// </summary>
        public static bool MaterialTexturesOnDemand { get; set; } = true;

        /// <summary>
        /// A material texture loads cold only when its cache file holds at least this many bytes (512 KiB: 1024² compressed
        /// and up). The small ones (effects' noise, masks, 512² and under) stay resident: little memory, and an effect
        /// that flashes on for a moment never waits for them.
        /// </summary>
        public static long ColdMinBytes { get; set; } = 512L << 10;

        /// <summary>
        /// Milliseconds a frame's <see cref="Update"/> spends taking finished reads in (each makes its resource and queues
        /// its upload); the rest wait for the next frame.
        /// </summary>
        public static double UpdateBudgetMs { get; set; } = 4;

        /// <summary>Frames a material used by per-frame draws (Material.Touch) stays held after its last use.</summary>
        public static int LeaseTicks { get; set; } = 120;

        /// <summary>On-demand textures with a GPU copy now, and the bytes those copies take.</summary>
        public static int ResidentCount { get; private set; }
        public static long ResidentBytes { get; private set; }

        /// <summary>GPU copies made since the start (loads, not reuses from the idle list).</summary>
        public static int Loads { get; private set; }

        /// <summary>Material textures loaded cold since the start, and how many of them something else then pinned.</summary>
        public static int LoadedCold => LoadedColdCount;
        internal static int LoadedColdCount;
        public static int PinnedCount { get; private set; }

        /// <summary>Acquired textures whose copy has not arrived yet (a scene load's Finishing step waits for them).</summary>
        public static int Pending
        {
            get
            {
                lock (Sync)
                {
                    int n = 0;
                    foreach (var texture in _loading)
                        if (texture.Readers > 0 && !texture.DemandFailed) n++;
                    return n;
                }
            }
        }

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

        internal static void Pinned(Texture texture)
        {
            PinnedCount++;
            if (!texture.PinPending)
            {
                texture.PinPending = true;
                _pinned.Add(texture);
            }
        }

        // Acquired and not arrived: Update takes its read in and tells its materials when the upload has run
        internal static void Track(Texture texture)
        {
            if (texture.InLoading) return;
            texture.InLoading = true;
            texture.TrackedTick = Engine.TickCount;
            _loading.Add(texture);
        }

        // Frames from being held to bound, per copy: how long its materials showed their fallback
        private static readonly int[] _arrivalFrames = new int[6];   // 1, 2, 3, 4-5, 6-10, more
        private static int _arrivalFramesMax;

        private static void CountArrival(Texture texture)
        {
            int frames = Math.Max(1, Engine.TickCount - texture.TrackedTick);
            _arrivalFrames[frames <= 3 ? frames - 1 : frames <= 5 ? 3 : frames <= 10 ? 4 : 5]++;
            _arrivalFramesMax = Math.Max(_arrivalFramesMax, frames);
        }

        internal static void Lease(Material material) => _leased.Add(material);

        // Nobody holds it any more: a resident copy waits in the idle list (Update trims it). A read still on its way goes
        // on and its copy goes idle when it is done (Update): a renderer that registers its draws again a moment later
        // (a refresh lets go and takes hold in one call) does not read it twice
        internal static void Idle(Texture texture)
        {
            if (texture.Native == null || texture.IdleNode != null) return;
            texture.IdleNode = _idle.AddLast(texture);
            _idleBytes += texture.DemandBytes;
        }

        internal static void LeaveIdle(Texture texture)
        {
            if (texture.IdleNode == null) return;
            _idle.Remove(texture.IdleNode);
            texture.IdleNode = null;
            _idleBytes -= texture.DemandBytes;
        }

        /// <summary>
        /// Once a frame, before rendering (Engine.Tick): reads that are done go up, copies whose upload has run count as
        /// arrived and their materials bind them, pinned textures' materials bind them, leases not renewed end, and the
        /// idle copies are trimmed to the budget. Works only on what is on its way: nothing when nothing is. Main thread.
        /// </summary>
        public static void Update()
        {
            lock (Sync)
            {
                long start = System.Diagnostics.Stopwatch.GetTimestamp();
                for (int i = _loading.Count - 1; i >= 0; i--)
                {
                    var texture = _loading[i];

                    // Taking a read in makes its resource and queues its upload: within a budget a frame, the rest next frame
                    if (texture.Native == null && texture.ReadInFlight
                        && System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds > UpdateBudgetMs)
                        continue;

                    bool done;
                    if (!texture.OnDemand) done = true;                                    // pinned meanwhile
                    else if (texture.Readers == 0 && texture.Native == null && !texture.ReadInFlight)
                        done = true;                                                       // dropped before it came
                    else if (texture.GpuReady)
                    {
                        if (!texture.DemandFailed && texture.Native != null)
                        {
                            texture.Arrived = true;
                            CountArrival(texture);
                            texture.NotifyHolders();
                            if (texture.Readers == 0) Idle(texture);                       // let go of while it came
                        }
                        done = true;
                    }
                    else done = false;

                    if (done)
                    {
                        texture.InLoading = false;
                        _loading[i] = _loading[^1];
                        _loading.RemoveAt(_loading.Count - 1);
                    }
                }

                foreach (var texture in _pinned)
                {
                    texture.PinPending = false;
                    texture.NotifyHolders();
                    texture.DemandHolders.Clear();   // resident for good: nothing to tell them any more
                }
                _pinned.Clear();

                if (_leased.Count > 0)
                {
                    int tick = Engine.TickCount;
                    for (int i = _leased.Count - 1; i >= 0; i--)
                    {
                        if (!_leased[i].LeaseExpired(tick)) continue;
                        _leased[i].EndLease();
                        _leased[i] = _leased[^1];
                        _leased.RemoveAt(_leased.Count - 1);
                    }
                }

                Trim(IdleBudget);
            }
        }

        /// <summary>Drop idle copies, the least recently released first, until they take at most <paramref name="budget"/> bytes.</summary>
        public static void Trim(long budget)
        {
            lock (Sync)
            {
                while (_idleBytes > budget && _idle.First is { } oldest)
                {
                    var texture = oldest.Value;
                    LeaveIdle(texture);
                    texture.DropGpuCopy();
                }
            }
        }

        /// <summary>
        /// Block until every one of these is <see cref="Texture.GpuReady"/>: their reads (started together by Acquire, on
        /// workers) waited for and uploaded, then one flush of the copy queue. They count as arrived at once and their
        /// materials bind them. For code that records its own list and submits it at once. Acquire them first. Main thread.
        /// </summary>
        public static void WaitGpuReady(IReadOnlyList<Texture> textures)
        {
            lock (Sync)
            {
                bool ready = true;
                foreach (var texture in textures)
                {
                    texture.FinishReadNow();
                    ready &= texture.GpuReady;
                }
                if (!ready)
                    StreamingManager.Instance?.Flush();
                foreach (var texture in textures)
                {
                    if (!texture.OnDemand || texture.Arrived || texture.Native == null || !texture.GpuReady) continue;
                    texture.Arrived = true;
                    texture.NotifyHolders();
                }
            }
        }

        /// <summary>
        /// Hold these materials (<see cref="Material.Acquire"/>) and block until their textures are on the GPU and bound:
        /// for code that draws them once into its own list (item icons). Release each afterwards. Main thread.
        /// </summary>
        public static void HoldNow(IReadOnlyCollection<Material> materials)
        {
            var textures = new List<Texture>();
            lock (Sync)
            {
                foreach (var material in materials)
                {
                    material.Acquire();
                    foreach (var texture in material.Textures.Values)
                        if (texture is { OnDemand: true }) textures.Add(texture);
                }
            }
            if (textures.Count > 0) WaitGpuReady(textures);
        }

        /// <summary>Drop every idle copy (a scene load: what the last scene used is not this one's).</summary>
        public static void DropIdle() => Trim(0);

        /// <summary>For engine_stats.</summary>
        public static object Describe()
        {
            lock (Sync)
            {
                return new
                {
                    resident = ResidentCount,
                    residentMiB = Math.Round(ResidentBytes / 1048576.0, 1),
                    idle = _idle.Count,
                    idleMiB = Math.Round(_idleBytes / 1048576.0, 1),
                    idleBudgetMiB = Math.Round(IdleBudget / 1048576.0, 1),
                    pending = Pending,
                    leasedMaterials = _leased.Count,
                    loads = Loads,
                    loadedCold = LoadedCold,
                    pinned = PinnedCount,
                    // Frames from held to bound (the fallback showing meanwhile), by arrivals: 1, 2, 3, 4-5, 6-10, more
                    arrivalFrames = string.Join(" ", _arrivalFrames),
                    arrivalFramesMax = _arrivalFramesMax,
                    materialTexturesOnDemand = MaterialTexturesOnDemand,
                    coldMinKiB = ColdMinBytes / 1024,
                };
            }
        }
    }
}
