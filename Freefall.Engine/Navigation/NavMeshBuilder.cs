using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using DotRecast.Core;
using DotRecast.Core.Numerics;
using DotRecast.Detour;
using DotRecast.Recast;
using DotRecast.Recast.Geom;

using Debug = Freefall.Debug;

namespace Freefall.Navigation
{
    public enum NavMeshBakeStage
    {
        PreparingMeshes,
        BuildingTiles,
        Assembling,
        Done,
        Cancelled,
        Failed,
    }

    public sealed class NavMeshBakeResult
    {
        public required NavMeshTileSet Tiles;

        /// <summary>The runtime navmesh assembled from the tiles. Null if nothing is walkable.</summary>
        public DtNavMesh? NavMesh;

        /// <summary>Cells whose inputs changed since the previous bake: built anew, or emptied.</summary>
        public int RebuiltCells;

        /// <summary>Cells carried over from the previous bake because their inputs did not change.</summary>
        public int ReusedCells;

        /// <summary>
        /// World-space XZ rects (min, max) of the first MaxChangedRects cells that changed against the
        /// previous bake: where in the scene something is different. Empty for a full rebuild.
        /// </summary>
        public (Vector2 min, Vector2 max)[] ChangedRects = [];

        public const int MaxChangedRects = 256;

        public double Seconds;

        /// <summary>Where the time went, summed over the worker threads. For the log.</summary>
        public string Profile = "";
    }

    /// <summary>A running (or finished) navmesh bake. All members are safe to read from any thread.</summary>
    public sealed class NavMeshBake
    {
        private readonly CancellationTokenSource _cancel = new();
        private volatile NavMeshBakeStage _stage;

        internal int Done;
        internal int Rebuilt;
        internal int Reused;

        public NavMeshBakeStage Stage { get => _stage; internal set => _stage = value; }

        /// <summary>Grid cells of the bake. 0 until the grid is known.</summary>
        public int TotalCells { get; internal set; }

        public int DoneCells => Volatile.Read(ref Done);

        public float Progress => TotalCells > 0 ? (float)DoneCells / TotalCells : 0f;

        public bool IsCompleted => _stage >= NavMeshBakeStage.Done;

        public NavMeshBakeResult? Result { get; internal set; }

        public string? Error { get; internal set; }

        /// <summary>Completes when the bake has finished, failed or was cancelled. Never faults.</summary>
        public Task Completion { get; internal set; } = Task.CompletedTask;

        public void Cancel() => _cancel.Cancel();

        internal CancellationToken Token => _cancel.Token;
    }

    /// <summary>
    /// Bakes a navmesh from the scene's terrain and static meshes with Recast.
    ///
    /// The world is cut into a fixed grid of tiles and each tile is built on its own: it rasterizes
    /// only the terrain patch and the mesh chunks that reach into it, so memory stays at a few MB per
    /// worker thread no matter how large the scene is. Nothing is ever merged into one big triangle
    /// soup. Each tile stores a hash of its inputs; a rebake skips every tile whose hash is unchanged.
    ///
    /// Only Start() touches the scene (main thread). The build runs on its own low-priority threads.
    /// </summary>
    public static partial class NavMeshBuilder
    {
        // Bump when a change to the bake alters its output, so stored tiles are rebuilt
        private const int BakeVersion = 1;

        private const int MaxCells = 4_000_000;

        private readonly struct BakeSettings
        {
            public readonly float CellSize, CellHeight;
            public readonly float AgentHeight, AgentRadius, MaxClimb, MaxSlope;
            public readonly float TerrainStep;
            public readonly int TileSize;

            public BakeSettings(Assets.NavMesh asset)
            {
                CellSize = MathF.Max(asset.CellSize, 0.05f);
                CellHeight = MathF.Max(asset.CellHeight, 0.02f);
                AgentHeight = asset.AgentHeight;
                AgentRadius = asset.AgentRadius;
                MaxClimb = asset.MaxClimb;
                MaxSlope = Math.Clamp(asset.MaxSlope, 0f, 89f);
                TerrainStep = MathF.Max(asset.TerrainSampleStep, 0.25f);
                TileSize = Math.Clamp(asset.TileSize, 16, 512);
            }

            public ulong Hash()
            {
                ulong h = NavHash.Mix(BakeVersion, (ulong)TileSize);
                h = NavHash.Mix(NavHash.Mix(h, CellSize), CellHeight);
                h = NavHash.Mix(NavHash.Mix(h, AgentHeight), AgentRadius);
                h = NavHash.Mix(NavHash.Mix(h, MaxClimb), MaxSlope);
                return NavHash.Mix(h, TerrainStep);
            }
        }

        /// <summary>
        /// Start a bake of the current scene. Call on the main thread; returns as soon as the scene
        /// has been captured. Tiles of <paramref name="previous"/> whose inputs are unchanged are
        /// carried over; pass null to rebuild everything.
        /// </summary>
        public static NavMeshBake Start(Assets.NavMesh settings, NavMeshTileSet? previous)
        {
            var bake = new NavMeshBake();
            var bakeSettings = new BakeSettings(settings);
            var input = NavMeshBakeInput.Collect();

            bake.Completion = Task.Factory.StartNew(
                () => Run(bake, bakeSettings, input, previous),
                CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);

            return bake;
        }

        private static void Run(NavMeshBake bake, BakeSettings settings, NavMeshBakeInput input, NavMeshTileSet? previous)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                var token = bake.Token;

                // Leave cores for the editor, which keeps rendering while this runs
                int threads = Math.Clamp(Environment.ProcessorCount - 2, 1, 32);

                // ── Mesh chunks + bounds ──
                bake.Stage = NavMeshBakeStage.PreparingMeshes;
                RunWorkers(threads, input.Geometries.Count, token, () => 0, (i, _) => input.Geometries[i].Prepare());
                input.PrepareInstances();

                // ── Grid ──
                var context = BakeContext.Create(settings, input, previous);
                bake.TotalCells = context.Tiles.CellCount;

                Debug.Log($"[NavMesh] Baking {context.Tiles.TilesX}x{context.Tiles.TilesZ} tiles of {context.TileWorld:0.#} m from " +
                          $"{input.Terrains.Count} terrain(s), {input.Instances.Length} mesh instance(s) of {input.Geometries.Count} mesh(es), {threads} threads" +
                          (context.Previous != null ? " (incremental)" : ""));

                // ── Tiles ──
                bake.Stage = NavMeshBakeStage.BuildingTiles;
                var profile = new BakeProfile();
                long allocatedBefore = GC.GetTotalAllocatedBytes();
                int gen0Before = GC.CollectionCount(0);

                // Which cells changed against the previous bake (the first few), to tell where
                var changedCells = new int[NavMeshBakeResult.MaxChangedRects];
                int changedCount = 0;

                RunWorkers(threads, bake.TotalCells, token, () => new TileWorker(context), (cell, worker) =>
                {
                    switch (worker.Build(cell))
                    {
                        case CellResult.Rebuilt:
                            Interlocked.Increment(ref bake.Rebuilt);
                            if (context.Previous != null)
                            {
                                int slot = Interlocked.Increment(ref changedCount) - 1;
                                if (slot < changedCells.Length) changedCells[slot] = cell;
                            }
                            break;
                        case CellResult.Reused: Interlocked.Increment(ref bake.Reused); break;
                    }
                    Interlocked.Increment(ref bake.Done);
                }, worker => profile.Add(worker.Profile));

                var changedRects = new (Vector2 min, Vector2 max)[Math.Min(changedCount, changedCells.Length)];
                Array.Sort(changedCells, 0, changedRects.Length);
                for (int i = 0; i < changedRects.Length; i++)
                {
                    var tiles = context.Tiles;
                    var min = new Vector2(
                        tiles.Origin.X + changedCells[i] % tiles.TilesX * tiles.TileWorldSize,
                        tiles.Origin.Z + changedCells[i] / tiles.TilesX * tiles.TileWorldSize);
                    changedRects[i] = (min, min + new Vector2(tiles.TileWorldSize));
                }

                string profileText = profile.Describe(threads,
                    GC.GetTotalAllocatedBytes() - allocatedBefore, GC.CollectionCount(0) - gen0Before);

                // ── Runtime navmesh ──
                bake.Stage = NavMeshBakeStage.Assembling;
                var navMesh = context.Tiles.CreateNavMesh();
                token.ThrowIfCancellationRequested();

                bake.Result = new NavMeshBakeResult
                {
                    Tiles = context.Tiles,
                    NavMesh = navMesh,
                    RebuiltCells = bake.Rebuilt,
                    ReusedCells = bake.Reused,
                    ChangedRects = changedRects,
                    Seconds = sw.Elapsed.TotalSeconds,
                    Profile = profileText,
                };
                bake.Stage = NavMeshBakeStage.Done;
            }
            catch (OperationCanceledException)
            {
                bake.Stage = NavMeshBakeStage.Cancelled;
            }
            catch (Exception ex)
            {
                bake.Error = ex.Message;
                bake.Stage = NavMeshBakeStage.Failed;
                Debug.LogError("NavMesh", $"Bake failed: {ex}");
            }
        }

        /// <summary>
        /// Run body(0..count-1) on dedicated below-normal threads. The thread pool is left alone: the
        /// engine's own parallel loops keep using it while a bake runs for a minute.
        /// </summary>
        private static void RunWorkers<TLocal>(int threads, int count, CancellationToken token, Func<TLocal> createLocal, Action<int, TLocal> body,
            Action<TLocal>? finish = null)
        {
            int next = -1;
            Exception? error = null;

            void Loop()
            {
                try
                {
                    var local = createLocal();
                    while (Volatile.Read(ref error) == null && !token.IsCancellationRequested)
                    {
                        int i = Interlocked.Increment(ref next);
                        if (i >= count) break;
                        body(i, local);
                    }
                    finish?.Invoke(local);
                }
                catch (Exception ex)
                {
                    Interlocked.CompareExchange(ref error, ex, null);
                }
            }

            var pool = new Thread[Math.Min(threads, count)];
            for (int i = 0; i < pool.Length; i++)
            {
                pool[i] = new Thread(Loop) { IsBackground = true, Priority = ThreadPriority.BelowNormal, Name = "NavMeshBake" };
                pool[i].Start();
            }
            foreach (var thread in pool)
                thread.Join();

            if (error != null)
                ExceptionDispatchInfo.Capture(error).Throw();
            token.ThrowIfCancellationRequested();
        }

        /// <summary>Triangle list of the walkable polygons, for debug drawing. Any thread.</summary>
        public static void BuildDebugGeometry(DtNavMesh navMesh, out Vector3[] vertices, out uint[] indices)
        {
            int vertexCount = 0, indexCount = 0;
            for (int i = 0; i < navMesh.GetMaxTiles(); i++)
            {
                var data = navMesh.GetTile(i)?.data;
                if (data?.header == null) continue;

                vertexCount += data.header.vertCount;
                for (int p = 0; p < data.header.polyCount; p++)
                {
                    var poly = data.polys[p];
                    if (poly.GetPolyType() != DtPolyTypes.DT_POLYTYPE_OFFMESH_CONNECTION)
                        indexCount += Math.Max(0, poly.vertCount - 2) * 3;
                }
            }

            vertices = new Vector3[vertexCount];
            indices = new uint[indexCount];

            int v = 0, o = 0;
            for (int i = 0; i < navMesh.GetMaxTiles(); i++)
            {
                var data = navMesh.GetTile(i)?.data;
                if (data?.header == null) continue;

                uint baseVertex = (uint)v;
                for (int k = 0; k < data.header.vertCount; k++)
                    vertices[v++] = new Vector3(data.verts[k * 3], data.verts[k * 3 + 1], data.verts[k * 3 + 2]);

                for (int p = 0; p < data.header.polyCount; p++)
                {
                    var poly = data.polys[p];
                    if (poly.GetPolyType() == DtPolyTypes.DT_POLYTYPE_OFFMESH_CONNECTION) continue;

                    for (int j = 2; j < poly.vertCount; j++)
                    {
                        indices[o++] = baseVertex + (uint)poly.verts[0];
                        indices[o++] = baseVertex + (uint)poly.verts[j - 1];
                        indices[o++] = baseVertex + (uint)poly.verts[j];
                    }
                }
            }
        }

        // ── Shared, read-only state of one bake ──

        private sealed class BakeContext
        {
            public required BakeSettings Settings;
            public required NavMeshBakeInput Input;
            public required RcConfig Config;
            public required NavMeshTileSet Tiles;

            /// <summary>The previous bake, if its tiles can be carried over (same grid).</summary>
            public NavMeshTileSet? Previous;

            public ulong SettingsHash;
            public float TileWorld;

            /// <summary>World-space width of the border a tile rasterizes beyond its own rect.</summary>
            public float Pad;

            /// <summary>Heightfield edge length in cells: the tile plus its border on both sides.</summary>
            public int FieldSize;

            public float WalkableThresholdSq;
            public int WalkableArea;

            // Instances per cell: CellItems[CellStart[cell] .. CellStart[cell + 1]) index Input.Instances
            public required int[] CellStart;
            public required int[] CellItems;

            public static BakeContext Create(BakeSettings settings, NavMeshBakeInput input, NavMeshTileSet? previous)
            {
                int walkableRadius = (int)MathF.Ceiling(settings.AgentRadius / settings.CellSize);

                var config = new RcConfig(
                    useTiles: true,
                    tileSizeX: settings.TileSize,
                    tileSizeZ: settings.TileSize,
                    borderSize: walkableRadius + 3,
                    partition: RcPartition.WATERSHED,
                    cellSize: settings.CellSize,
                    cellHeight: settings.CellHeight,
                    agentMaxSlope: settings.MaxSlope,
                    agentHeight: settings.AgentHeight,
                    agentRadius: settings.AgentRadius,
                    agentMaxClimb: settings.MaxClimb,
                    minRegionArea: 8f,
                    mergeRegionArea: 20f,
                    edgeMaxLen: 12f,
                    edgeMaxError: 1.3f,
                    vertsPerPoly: NavMeshTileSet.MaxVertsPerPoly,
                    detailSampleDist: 6f,
                    detailSampleMaxError: 1f,
                    filterLowHangingObstacles: true,
                    filterLedgeSpans: true,
                    filterWalkableLowHeightSpans: true,
                    walkableAreaMod: new RcAreaModification(1),
                    buildMeshDetail: true
                );

                // ── Bounds: the terrains, or without a terrain everything there is ──
                var min = new Vector3(float.MaxValue);
                var max = new Vector3(float.MinValue);
                if (input.Terrains.Count > 0)
                {
                    foreach (var terrain in input.Terrains)
                    {
                        min = Vector3.Min(min, terrain.Position);
                        max = Vector3.Max(max, terrain.Position + new Vector3(terrain.Size.X, terrain.MaxHeight, terrain.Size.Y));
                    }
                }
                else
                {
                    foreach (ref readonly var instance in input.Instances.AsSpan())
                    {
                        min = Vector3.Min(min, instance.Min);
                        max = Vector3.Max(max, instance.Max);
                    }
                }

                if (min.X > max.X)
                    throw new InvalidOperationException("No geometry to bake: the scene has no terrain and no static meshes.");

                // ── Grid: anchored at the world origin, so it only changes when the bounds do ──
                float tileWorld = settings.TileSize * settings.CellSize;
                var origin = new Vector3(MathF.Floor(min.X / tileWorld) * tileWorld, 0, MathF.Floor(min.Z / tileWorld) * tileWorld);
                long tilesX = Math.Max(1, (long)MathF.Ceiling((max.X - origin.X) / tileWorld - 1e-3f));
                long tilesZ = Math.Max(1, (long)MathF.Ceiling((max.Z - origin.Z) / tileWorld - 1e-3f));

                if (tilesX * tilesZ > MaxCells)
                    throw new InvalidOperationException(
                        $"NavMesh bounds {max.X - min.X:0} x {max.Z - min.Z:0} m need {tilesX}x{tilesZ} tiles. Use a larger CellSize or TileSize" +
                        (input.Terrains.Count == 0 ? ", or check for a mesh far away from the rest of the scene." : "."));

                var tiles = new NavMeshTileSet(origin, tileWorld, (int)tilesX, (int)tilesZ);
                float pad = config.BorderSize * settings.CellSize;

                BinInstances(input.Instances, tiles, pad, out var cellStart, out var cellItems);

                float walkableThreshold = MathF.Cos(settings.MaxSlope / 180f * MathF.PI);

                return new BakeContext
                {
                    Settings = settings,
                    Input = input,
                    Config = config,
                    Tiles = tiles,
                    Previous = previous != null && previous.SameGrid(origin, tileWorld, (int)tilesX, (int)tilesZ) ? previous : null,
                    SettingsHash = settings.Hash(),
                    TileWorld = tileWorld,
                    Pad = pad,
                    FieldSize = settings.TileSize + config.BorderSize * 2,
                    WalkableThresholdSq = walkableThreshold * walkableThreshold,
                    WalkableArea = config.WalkableAreaMod.Apply(0),
                    CellStart = cellStart,
                    CellItems = cellItems,
                };
            }

            /// <summary>Counting sort of the instances into the cells their bounds (plus border) touch.</summary>
            private static void BinInstances(MeshInstance[] instances, NavMeshTileSet tiles, float pad, out int[] cellStart, out int[] cellItems)
            {
                bool CellRange(in MeshInstance instance, out int x0, out int x1, out int z0, out int z1)
                {
                    float inv = 1f / tiles.TileWorldSize;
                    x0 = Math.Max(0, (int)MathF.Floor((instance.Min.X - pad - tiles.Origin.X) * inv));
                    x1 = Math.Min(tiles.TilesX - 1, (int)MathF.Floor((instance.Max.X + pad - tiles.Origin.X) * inv));
                    z0 = Math.Max(0, (int)MathF.Floor((instance.Min.Z - pad - tiles.Origin.Z) * inv));
                    z1 = Math.Min(tiles.TilesZ - 1, (int)MathF.Floor((instance.Max.Z + pad - tiles.Origin.Z) * inv));
                    return x0 <= x1 && z0 <= z1;
                }

                cellStart = new int[tiles.CellCount + 1];

                for (int i = 0; i < instances.Length; i++)
                {
                    if (!CellRange(instances[i], out int x0, out int x1, out int z0, out int z1)) continue;
                    for (int z = z0; z <= z1; z++)
                        for (int x = x0; x <= x1; x++)
                            cellStart[z * tiles.TilesX + x + 1]++;
                }

                for (int c = 0; c < tiles.CellCount; c++)
                    cellStart[c + 1] += cellStart[c];

                cellItems = new int[cellStart[tiles.CellCount]];
                var cursor = new int[tiles.CellCount];
                Array.Copy(cellStart, cursor, cursor.Length);

                for (int i = 0; i < instances.Length; i++)
                {
                    if (!CellRange(instances[i], out int x0, out int x1, out int z0, out int z1)) continue;
                    for (int z = z0; z <= z1; z++)
                        for (int x = x0; x <= x1; x++)
                            cellItems[cursor[z * tiles.TilesX + x]++] = i;
                }
            }
        }

        /// <summary>Time per phase of the tile build (Stopwatch ticks) and the triangles rasterized.</summary>
        private sealed class BakeProfile
        {
            public long Inputs, Rasterize, Recast, Detour, Triangles;

            public void Add(BakeProfile other)
            {
                lock (this)
                {
                    Inputs += other.Inputs;
                    Rasterize += other.Rasterize;
                    Recast += other.Recast;
                    Detour += other.Detour;
                    Triangles += other.Triangles;
                }
            }

            public string Describe(int threads, long allocatedBytes, int gen0Collections)
            {
                double s = 1.0 / Stopwatch.Frequency;
                return $"{threads} threads, cpu: inputs {Inputs * s:0.0} s, rasterize {Rasterize * s:0.0} s ({Triangles / 1e6:0.0} M triangles), " +
                       $"recast {Recast * s:0.0} s, detour {Detour * s:0.0} s; allocated {allocatedBytes / (1024.0 * 1024 * 1024):0.0} GB, {gen0Collections} gen0 GCs";
            }
        }

        private enum CellResult
        {
            /// <summary>Nothing reaches into the cell.</summary>
            Empty,
            Reused,
            Rebuilt,
        }

        /// <summary>
        /// Recast asks its geometry provider for convex volumes after rasterization. The triangles
        /// are rasterized by TileWorker, so there is nothing else to provide.
        /// </summary>
        private sealed class NoGeometry : IRcInputGeomProvider
        {
            public static readonly NoGeometry Instance = new();

            private readonly List<RcConvexVolume> _volumes = [];
            private readonly List<RcOffMeshConnection> _connections = [];

            public RcTriMesh GetMesh() => null!;
            public RcVec3f GetMeshBoundsMin() => default;
            public RcVec3f GetMeshBoundsMax() => default;
            public IEnumerable<RcTriMesh> Meshes() => [];
            public void AddConvexVolume(RcConvexVolume convexVolume) { }
            public IList<RcConvexVolume> ConvexVolumes() => _volumes;
            public List<RcOffMeshConnection> GetOffMeshConnections() => _connections;
            public void AddOffMeshConnection(RcVec3f start, RcVec3f end, float radius, bool bidir, int area, int flags) { }
            public void RemoveOffMeshConnections(Predicate<RcOffMeshConnection> filter) { }
        }
    }
}
