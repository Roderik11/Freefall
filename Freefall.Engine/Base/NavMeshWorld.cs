using System;
using System.Collections.Generic;
using System.Numerics;
using DotRecast.Core.Numerics;
using DotRecast.Detour;
using DotRecast.Detour.Crowd;
using Freefall.Components;

namespace Freefall.Base
{
    /// <summary>
    /// Global navigation state. Owns the DtNavMesh and DtCrowd instances.
    /// Initialized per-scene when a NavMesh asset is available.
    /// Mirrors the PhysicsWorld pattern.
    /// </summary>
    public static class NavMeshWorld
    {
        private static DtNavMesh? _navMesh;
        private static DtCrowd? _crowd;

        // Half-extents for navmesh point queries (search volume)
        private static readonly RcVec3f QueryExtents = new(2f, 4f, 2f);

        // The searches, in the profiler under whoever asks (main thread)
        private static readonly ProfilerMarker FindPathMarker = ProfilerMarker.Get("NavMesh find path");
        private static readonly ProfilerMarker RandomPointMarker = ProfilerMarker.Get("NavMesh random point");
        private static readonly ProfilerMarker WayOutMarker = ProfilerMarker.Get("NavMesh way outdoors");
        private static readonly ProfilerMarker WaysMarker = ProfilerMarker.Get("NavMesh agents' ways");

        /// <summary>Recast area ids a bake gives: walkable ground, and a building's ground floor (NavMeshArea Indoors).</summary>
        public const int WalkArea = 1, IndoorsArea = 2;

        /// <summary>Polygon flags a bake gives, by area: what the query filters include and exclude.</summary>
        public const int WalkFlag = 1, IndoorsFlag = 2;

        /// <summary>
        /// What a metre indoors costs a way against a metre outdoors. Indoors is walkable, and a way that ends there
        /// goes in at the nearest door; one that only passes goes round by the street unless that is this much longer.
        /// 20 was too little: round one of Eldermoor's blocks is over 120 m, and walkers went in at a house's front door
        /// and out at its back (2026-10-25).
        /// </summary>
        public const float IndoorsCost = 1000f;

        // Ways go anywhere, indoors at its cost; points a body is put at keep outdoors (a pilgrim the crowd put on "the
        // nearest navmesh" stood in somebody's front room, 2026-10-25)
        private static readonly IDtQueryFilter DefaultFilter = Costly();
        private static readonly IDtQueryFilter OutdoorsFilter = Outdoors();

        // Every area at 1 a metre but indoors (Detour's DT_MAX_AREAS of them)
        private static DtQueryDefaultFilter Costly()
        {
            var costs = new float[64];
            Array.Fill(costs, 1f);
            costs[IndoorsArea] = IndoorsCost;
            return new DtQueryDefaultFilter(0xffff, 0, costs);
        }

        private static DtQueryDefaultFilter Outdoors()
        {
            var filter = new DtQueryDefaultFilter();
            filter.SetExcludeFlags(IndoorsFlag);
            return filter;
        }

        /// <summary>
        /// Most polygons one <see cref="FindPath"/> looks at. DotRecast's node pool grows without bound (C++ Detour's
        /// stops at its maxNodes, commonly 2048), so a target the start cannot reach (another island of the navmesh,
        /// across water) had one search walk every polygon connected to the start: 12 s and 30 MB in one frame on
        /// Eldermoor, an animal fleeing the player (2026-10-24). Stopped here, the way leads as near the target as the
        /// search got.
        /// </summary>
        public const int MaxSearchNodes = 2048;

        private static bool _cappedLogged;

        /// <summary>
        /// Polygons the crowd's queue of ways looks at in a frame, for every agent waiting for one together (DotRecast's
        /// 100 by default), and in the quick search a new destination gets at once (its 20). A way across a town of
        /// Eldermoor's size looks at well over a thousand: with seventy townsfolk asking, one waited seconds, walking
        /// meanwhile the quick search's way, which ends where it got nearest the target, often just inside a house's
        /// door; there it stood until its walk was given up and asked for again (2026-10-25).
        /// </summary>
        public const int PathStepsPerFrame = 500, QuickPathSteps = 48;

        /// <summary>How many crowd targets were out of reach and given as the nearest point a way leads to (the path service).</summary>
        public static int OutOfReach { get; private set; }

        /// <summary>True if a navmesh is loaded and ready for queries.</summary>
        public static bool IsReady => _navMesh != null;

        // ── Lifecycle ──

        /// <summary>Load a baked NavMesh into the runtime.</summary>
        public static void Initialize(DtNavMesh navMesh)
        {
            _navMesh = navMesh;
            _wayQuery = new DtNavMeshQuery(navMesh);
            ClearAsks();
            _waySearching = null;
            _cappedLogged = false;

            // Create crowd with sane defaults
            var crowdCfg = new DtCrowdConfig(0.6f) // max agent radius
            {
                maxFindPathIterations = PathStepsPerFrame,
                maxTargetFindPathIterations = QuickPathSteps,
            };
            _crowd = new DtCrowd(crowdCfg, navMesh, _ => Costly());   // its agents' ways as FindPath's

            // The phases that work on one agent at a time run on workers (our DotRecast: a query per thread), and every
            // phase shows in the profiler under Navigation, with its threads where it has them
            _crowd.ParallelFor = ParallelWork.For;
            _crowd.Telemetry().Observer = CrowdPhases.Instance;

            // Log navmesh bounds for diagnostics
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            int totalPolys = 0;
            for (int i = 0; i < navMesh.GetMaxTiles(); i++)
            {
                var tile = navMesh.GetTile(i);
                if (tile?.data == null) continue;
                totalPolys += tile.data.header.polyCount;
                var h = tile.data.header;
                minX = MathF.Min(minX, h.bmin.X); minY = MathF.Min(minY, h.bmin.Y); minZ = MathF.Min(minZ, h.bmin.Z);
                maxX = MathF.Max(maxX, h.bmax.X); maxY = MathF.Max(maxY, h.bmax.Y); maxZ = MathF.Max(maxZ, h.bmax.Z);
            }
            Debug.Log($"[NavMeshWorld] Initialized: {totalPolys} polys, bounds=({minX:F1},{minY:F1},{minZ:F1})-({maxX:F1},{maxY:F1},{maxZ:F1})");
        }

        /// <summary>Initialize from a NavMesh asset (assembles the runtime navmesh from its baked tiles).</summary>
        public static void Initialize(Assets.NavMesh asset)
        {
            var navMesh = asset.Tiles?.CreateNavMesh();
            if (navMesh == null)
            {
                Debug.LogWarning("NavMeshWorld", "NavMesh asset has no baked data.");
                return;
            }

            Initialize(navMesh);
        }

        /// <summary>Tear down — called on scene unload.</summary>
        public static void Shutdown()
        {
            _crowd = null;
            _wayQuery = null;
            ClearAsks();
            _waySearching = null;
            _navMesh = null;
            Debug.Log("[NavMeshWorld] Shutdown");
        }

        /// <summary>Tick crowd simulation. Called from Engine.Update().</summary>
        public static void Update(float deltaTime)
        {
            _crowd?.Update(deltaTime, null);
        }

        // ── Path Queries ──

        /// <summary>
        /// Find a path between two world positions.
        /// Returns smoothed waypoint list, or empty if no path found. The search stops after
        /// <see cref="MaxSearchNodes"/> polygons; a target out of reach gets the way to the nearest the search came.
        /// Indoors (NavMeshArea) costs <see cref="IndoorsCost"/> times as much a metre.
        /// </summary>
        public static List<Vector3> FindPath(Vector3 from, Vector3 to)
        {
            var result = new List<Vector3>();
            FindPath(from, to, result);
            return result;
        }

        /// <summary>
        /// <see cref="FindPath(Vector3, Vector3)"/> into the caller's list, emptied first, without allocating. False when
        /// no way was found. Any thread (a query of its own, <see cref="Local"/>).
        /// </summary>
        public static bool FindPath(Vector3 from, Vector3 to, List<Vector3> into)
        {
            into.Clear();
            var local = Local();
            if (local == null) return false;
            var query = local.Query;
            using var timed = FindPathMarker.Auto();

            var startPos = ToRc(from);
            var endPos = ToRc(to);
            var filter = DefaultFilter;

            query.FindNearestPoly(startPos, QueryExtents, filter,
                out long startRef, out var startPt, out _);
            query.FindNearestPoly(endPos, QueryExtents, filter,
                out long endRef, out var endPt, out _);

            if (startRef == 0 || endRef == 0) return false;

            // Find polygon path, by a sliced search so that it stops after MaxSearchNodes (see there)
            var pathPolys = local.PathPolys;
            query.InitSlicedFindPath(startRef, endRef, startPt, endPt, filter, 0);
            var searched = query.UpdateSlicedFindPath(MaxSearchNodes, out _);
            if (searched.InProgress() && !_cappedLogged)
            {
                _cappedLogged = true;
                Debug.Log($"[NavMeshWorld] FindPath stopped after {MaxSearchNodes} polygons, the target out of reach or far: " +
                          $"({from.X:F0}, {from.Z:F0}) to ({to.X:F0}, {to.Z:F0}); the way goes as near as it got (logged once per navmesh)");
            }
            query.FinalizeSlicedFindPath(pathPolys, out int pathCount, pathPolys.Length);

            if (pathCount == 0) return false;

            // Build straight path (smoothed waypoints)
            var straightPath = local.PathCorners;
            query.FindStraightPath(startPt, endPt, pathPolys, pathCount,
                straightPath, out int straightCount, straightPath.Length, 0);

            for (int i = 0; i < straightCount; i++)
            {
                into.Add(FromRc(straightPath[i].pos));
            }

            return into.Count > 0;
        }

        /// <summary>
        /// A thread's query and the scratch of the queries here (a query's node pools, open list and queues are its own):
        /// any thread may ask FindPath, SamplePosition, RandomReachablePoint, WayOutdoors and IsOnNavMesh at once, while
        /// the navmesh is not changed (its tiles are put in on the main thread, between the parallel loops). Made on a
        /// thread's first ask, and again for a new navmesh.
        /// </summary>
        private sealed class QueryScratch
        {
            public DtNavMesh? Mesh;
            public DtNavMeshQuery Query = null!;
            public readonly long[] PathPolys = new long[256];                      // FindPath's: 256 polygons of a way, as it always had
            public readonly DtStraightPath[] PathCorners = new DtStraightPath[256];
            public readonly long[] WayPolys = new long[MaxSearchNodes];            // the way into RandomReachablePoint's circle
            public readonly long[] AroundRefs = new long[256], AroundParents = new long[256];   // WayOutdoors' polygons round a point
            public readonly float[] AroundCosts = new float[256];
            public readonly NavRandom Random = new();
        }

        [ThreadStatic] private static QueryScratch? t_query;

        // This thread's query for the navmesh there is now, or null when there is none
        private static QueryScratch? Local()
        {
            var mesh = _navMesh;
            if (mesh == null) return null;
            var local = t_query ??= new QueryScratch();
            if (local.Mesh != mesh)
            {
                local.Mesh = mesh;
                local.Query = new DtNavMeshQuery(mesh);
            }
            return local;
        }

        /// <summary>
        /// Find the nearest valid navmesh point to a world position. With <paramref name="outdoors"/>, never on a
        /// building's ground floor (NavMeshArea): for putting a body somewhere.
        /// </summary>
        public static bool SamplePosition(Vector3 position, float radius, out Vector3 result, bool outdoors = false)
        {
            result = position;
            var local = Local();
            if (local == null) return false;

            var extents = new RcVec3f(radius, radius, radius);
            var status = local.Query.FindNearestPoly(ToRc(position), extents, outdoors ? OutdoorsFilter : DefaultFilter,
                out long polyRef, out var nearestPt, out _);

            if (status.Failed() || polyRef == 0) return false;

            result = FromRc(nearestPt);
            return true;
        }

        /// <summary>
        /// A random point on the navmesh that a way leads to from <paramref name="from"/>, within
        /// <paramref name="radius"/> of <paramref name="center"/>, any walkable spot as likely as
        /// another: for wandering. A position picked blind and put on the navmesh can lie on a
        /// piece of it nothing leads to (the floor of a fountain's basin, a walled yard), and an
        /// agent sent there goes as near as a way leads and never arrives. False where there is
        /// no navmesh under <paramref name="from"/>.
        /// <para>
        /// From outside the circle the spread begins where a way from <paramref name="from"/> towards
        /// the centre ends (a search bounded as <see cref="FindPath"/>'s is): Detour spreads only over
        /// polygons inside the circle, and a start outside it gave a point on the start polygon
        /// itself, at <paramref name="from"/>'s feet. Guards sent on their rounds to a stall 100 m
        /// off stood still at their posts (2026-10-24). Where no way gets into the circle, the
        /// point is as near as one leads.
        /// </para>
        /// Outdoors only: the spread does not go through a building's ground floor (NavMeshArea), and the point is
        /// never on one. A walk about the town ends in the street, and a body put down stands there.
        /// </summary>
        public static bool RandomReachablePoint(Vector3 from, Vector3 center, float radius, out Vector3 point)
        {
            point = from;
            var local = Local();
            if (local == null) return false;
            var query = local.Query;
            using var timed = RandomPointMarker.Auto();

            var filter = OutdoorsFilter;
            query.FindNearestPoly(ToRc(from), QueryExtents, filter,
                out long startRef, out var startPt, out _);
            if (startRef == 0) return false;

            var away = from - center;
            if (away.X * away.X + away.Z * away.Z > radius * radius)
            {
                query.FindNearestPoly(ToRc(center), QueryExtents, filter,
                    out long centerRef, out var centerPt, out _);
                if (centerRef == 0) return false;

                var wayPolys = local.WayPolys;
                query.InitSlicedFindPath(startRef, centerRef, startPt, centerPt, filter, 0);
                query.UpdateSlicedFindPath(MaxSearchNodes, out _);
                query.FinalizeSlicedFindPath(wayPolys, out int count, wayPolys.Length);
                if (count == 0) return false;
                startRef = wayPolys[count - 1];
            }

            // Spreads from that polygon over its neighbours, as far as the circle reaches
            var status = query.FindRandomPointAroundCircle(startRef, ToRc(center), radius,
                filter, local.Random, out long randomRef, out var randomPt);
            if (status.Failed() || randomRef == 0) return false;

            point = FromRc(randomPt);
            return true;
        }

        /// <summary>
        /// Where the nearest way out of a building's ground floor (NavMeshArea) comes out, within <paramref name="radius"/>
        /// of <paramref name="from"/>: a point outdoors about <paramref name="beyond"/> metres past where the way leaves the
        /// building, which a way from <paramref name="from"/> reaches. The nearest outdoors navmesh to a point indoors is
        /// often behind a wall, in a yard or a gap between houses no way leads to, and a townsman sent there went as near
        /// as a way led, into a corner of the room, for good (2026-10-25). From outdoors, the navmesh under
        /// <paramref name="from"/>. False where there is no navmesh there, or no way out within the radius.
        /// </summary>
        public static bool WayOutdoors(Vector3 from, float radius, out Vector3 point, float beyond = 1.5f)
        {
            point = from;
            var local = Local();
            if (local == null) return false;
            var query = local.Query;
            var mesh = local.Mesh!;
            using var timed = WayOutMarker.Auto();

            query.FindNearestPoly(ToRc(from), QueryExtents, DefaultFilter, out long startRef, out var startPt, out _);
            if (startRef == 0) return false;
            if (!Indoors(mesh, startRef))
            {
                point = FromRc(startPt);
                return true;
            }

            // Spreads over the polygons round it cheapest first, a metre indoors dear: the cheapest outdoors one is where
            // the nearest door lets out
            var aroundRefs = local.AroundRefs;
            var aroundCosts = local.AroundCosts;
            var status = query.FindPolysAroundCircle(startRef, startPt, radius, DefaultFilter,
                aroundRefs, local.AroundParents, aroundCosts, out int count, aroundRefs.Length);
            if (status.Failed()) return false;
            int best = -1;
            for (int i = 0; i < count; i++)
                if (!Indoors(mesh, aroundRefs[i]) && (best < 0 || aroundCosts[i] < aroundCosts[best])) best = i;
            if (best < 0) return false;

            // On it nearest the room, then a little further out over the ground outdoors (sliding along the walls)
            long outRef = aroundRefs[best];
            query.ClosestPointOnPoly(outRef, startPt, out var door, out _);
            var away = new RcVec3f(door.X - startPt.X, 0f, door.Z - startPt.Z);
            float length = MathF.Sqrt(away.X * away.X + away.Z * away.Z);
            if (length > 1e-3f)
            {
                var to = new RcVec3f(door.X + away.X / length * beyond, door.Y, door.Z + away.Z / length * beyond);
                if (query.MoveAlongSurface(outRef, door, to, OutdoorsFilter, out var moved, aroundRefs, out _, aroundRefs.Length).Succeeded())
                    door = new RcVec3f(moved.X, door.Y, moved.Z);
            }
            point = FromRc(door);
            return true;
        }

        private static bool Indoors(DtNavMesh mesh, long polyRef) =>
            mesh.GetPolyFlags(polyRef, out int flags).Succeeded() && (flags & IndoorsFlag) != 0;

        private sealed class NavRandom : DotRecast.Core.IRcRand
        {
            private readonly Random _random = new();

            public float Next() => _random.NextSingle();
            public double NextDouble() => _random.NextDouble();
            public int NextInt32() => _random.Next();
        }

        /// <summary>Is this world position on the navmesh?</summary>
        public static bool IsOnNavMesh(Vector3 position, float tolerance = 0.5f)
        {
            return SamplePosition(position, tolerance, out _);
        }

        // ── Crowd Management (internal, used by NavMeshAgent) ──

        /// <summary>
        /// Register an agent with the crowd. Not logged: an NPC that sleeps far from every viewer leaves the crowd and
        /// joins again each time it wakes (NavMeshAgent.OnEnabledChanged).
        /// </summary>
        internal static DtCrowdAgent? AddCrowdAgent(Vector3 position, DtCrowdAgentParams agentParams)
        {
            if (_crowd == null) return null;
            return _crowd.AddAgent(ToRc(position), agentParams);
        }

        /// <summary>Remove an agent from the crowd.</summary>
        internal static void RemoveCrowdAgent(DtCrowdAgent agent)
        {
            _crowd?.RemoveAgent(agent);
        }

        /// <summary>
        /// An agent the crowd has given up on (its state invalid: where it was added, or once stood, the crowd found no
        /// polygon within its own reach of a little over its radius) is put on the nearest polygon within
        /// <see cref="QueryExtents"/> of <paramref name="position"/>, and walks again. The crowd never does this itself:
        /// an invalid agent is left out of every update from then on. False where no polygon is that near.
        /// </summary>
        internal static bool PlaceAgent(DtCrowdAgent agent, Vector3 position)
        {
            if (_crowd == null) return false;
            _crowd.GetNavMeshQuery().FindNearestPoly(ToRc(position), QueryExtents, _crowd.GetFilter(0),
                out long polyRef, out var nearest, out _);
            if (polyRef == 0) return false;

            agent.corridor.Reset(polyRef, nearest);
            agent.boundary.Reset();
            agent.partial = false;
            agent.npos = nearest;
            agent.state = DtCrowdAgentState.DT_CROWDAGENT_STATE_WALKING;
            return true;
        }

        /// <summary>
        /// A destination put on the navmesh for the crowd, with the crowd's own query, filter and extents (what it checks
        /// its ways against): the polygon and the point on it. False where no navmesh is near enough. The way there is
        /// the path service's (<see cref="AskWay"/>).
        /// </summary>
        internal static bool TargetOnNavMesh(Vector3 target, out long polyRef, out Vector3 onNavMesh)
        {
            polyRef = 0;
            onNavMesh = target;
            if (_crowd == null) return false;

            var status = _crowd.GetNavMeshQuery().FindNearestPoly(ToRc(target), _crowd.GetQueryExtents(), _crowd.GetFilter(0),
                out polyRef, out var point, out _);
            if (polyRef == 0)
            {
                Debug.Log($"[NavMeshWorld] TargetOnNavMesh: FindNearestPoly failed for {target} (status={status})");
                return false;
            }
            onNavMesh = FromRc(point);
            return true;
        }

        // ── The agents' ways: the path service ──
        //
        // An agent's destination is put on the navmesh at once (TargetOnNavMesh). The search for the way there runs here,
        // served by NavMeshAgentSystem: one agent at a time in the order they asked, on a query of its own, for up to
        // WayMilliseconds of a frame for all of them together, a search carried over frames where it takes more. Its end
        // is the crowd's target and its corners the agent's Way. Searched on the spot, as SetDestination did, a target out
        // of reach (all of MaxSearchNodes) took 7 ms of a frame with 400 characters about (2026-11-03). A budget of 256
        // polygons a frame was too little where hundreds ask at once (a scene's people, a crowd spawned): some waited
        // most of a minute for their ways.
        //
        // Never a target out of reach: the nearest point a way leads to instead, found by a search bounded as
        // FindPath's. The crowd's own queue searches without a bound (DotRecast's node pool grows), and re-plans a way
        // that ends short of its target every second once near its end: a few townsfolk sent where no way led searched
        // the whole navmesh over and over, and the rest waited behind them for their ways, up to 122 s (2026-10-25).

        /// <summary>Milliseconds of a frame the path service searches for, for every agent waiting together.</summary>
        public const double WayMilliseconds = 1.0;

        // Polygons a search goes on by between two looks at the clock
        private const int WaySlice = 64;

        private static DtNavMeshQuery? _wayQuery;
        private static readonly Queue<NavMeshAgent> _wayAsked = new();
        private static NavMeshAgent? _waySearching;    // whose search is under way, carried over frames
        private static long _wayTarget;                 // what it searches for: the target's polygon and point
        private static RcVec3f _wayFrom, _wayTargetPt;
        private static int _waySearched;                // polygons looked at so far
        private static readonly long[] _wayPolys = new long[MaxSearchNodes];
        private static readonly DtStraightPath[] _wayCorners = new DtStraightPath[256];

        /// <summary>Agents waiting for their way, the one being searched for included (asks since the last gathering not counted).</summary>
        public static int WaysWaiting => _wayAsked.Count + (_waySearching != null ? 1 : 0);

        // Some agent has asked since the service last gathered the asks
        private static volatile bool _asksWaiting;

        /// <summary>
        /// The agent's way is looked for by the path service, once however often it asks before its turn. Any thread (a
        /// brain's think on a worker): the ask only marks the agent, and the service gathers the marked ones in the agents'
        /// order as it begins its frame (<see cref="GatherAsks"/>), so the queue's order is the same every run.
        /// </summary>
        internal static void AskWay(NavMeshAgent agent)
        {
            if (agent.WayAsked) return;
            agent.WayAsked = true;
            _asksWaiting = true;
        }

        // The queue emptied (a navmesh replaced or gone): those in it are no longer asking, and may ask again
        private static void ClearAsks()
        {
            while (_wayAsked.TryDequeue(out var agent))
                agent.WayQueued = agent.WayAsked = agent.WayOnly = false;
            if (_waySearching != null) _waySearching.WayAsked = _waySearching.WayOnly = false;
            _asksWaiting = false;
        }

        // The agents that have asked and are not in the queue yet, in the agents' order (their component cache's)
        private static void GatherAsks()
        {
            if (!_asksWaiting) return;
            _asksWaiting = false;
            var agents = ComponentCache<NavMeshAgent>.All;
            for (int i = 0; i < agents.Count; i++)
            {
                var agent = agents[i];
                if (agent == null || !agent.WayAsked || agent.WayQueued || agent == _waySearching) continue;
                agent.WayQueued = true;
                _wayAsked.Enqueue(agent);
            }
        }

        /// <summary>
        /// The path service's frame (NavMeshAgentSystem, ahead of the agents): searches for WayMilliseconds at most. In the
        /// profiler as "NavMesh agents' ways", its items the agents waiting as it began.
        /// </summary>
        internal static void ServeWays()
        {
            if (_crowd == null || _wayQuery == null) return;
            GatherAsks();
            if (WaysWaiting == 0) return;
            using var timed = WaysMarker.Auto(WaysWaiting);

            long until = System.Diagnostics.Stopwatch.GetTimestamp() + (long)(WayMilliseconds * System.Diagnostics.Stopwatch.Frequency / 1000.0);
            while (System.Diagnostics.Stopwatch.GetTimestamp() < until)
            {
                var agent = _waySearching;
                if (agent == null)
                {
                    if (!BeginNextWay()) return;
                    continue;
                }

                // Gone, stopped, or sent somewhere else since its search began: dropped, or begun again for there
                if (!agent.WayWanted(out long target, out var targetPt))
                {
                    agent.WayAsked = false;
                    _waySearching = null;
                    continue;
                }
                if (target != _wayTarget || ToRc(targetPt) != _wayTargetPt)
                {
                    BeginWay(agent);
                    continue;
                }

                var status = _wayQuery.UpdateSlicedFindPath(Math.Min(WaySlice, MaxSearchNodes - _waySearched), out int done);
                _waySearched += Math.Max(done, 1);
                if (status.InProgress() && _waySearched < MaxSearchNodes) continue;   // on next frame once the time is spent
                FinishWay(agent, searched: true);
            }
        }

        private static bool BeginNextWay()
        {
            while (_wayAsked.TryDequeue(out var agent))
            {
                agent.WayQueued = false;
                if (!agent.WayWanted(out _, out _))
                {
                    agent.WayAsked = false;
                    continue;
                }
                BeginWay(agent);
                return true;
            }
            return false;
        }

        private static void BeginWay(NavMeshAgent agent)
        {
            var crowdAgent = agent.CrowdAgent!;
            agent.WayWanted(out _wayTarget, out var targetPt);
            _wayTargetPt = ToRc(targetPt);
            _wayFrom = crowdAgent.npos;
            _waySearched = 0;
            _waySearching = agent;

            long start = crowdAgent.corridor.GetFirstPoly();
            if (start == 0 || start == _wayTarget)
            {
                FinishWay(agent, searched: false);   // on the target's polygon already, or not on the navmesh: straight there
                return;
            }
            _wayQuery!.InitSlicedFindPath(start, _wayTarget, _wayFrom, _wayTargetPt, DefaultFilter, 0);
        }

        private static void FinishWay(NavMeshAgent agent, bool searched)
        {
            _waySearching = null;
            long targetRef = _wayTarget;
            var targetPt = _wayTargetPt;

            int count = 0;
            if (searched)
            {
                _wayQuery!.FinalizeSlicedFindPath(_wayPolys, out count, _wayPolys.Length);
                if (count > 0 && _wayPolys[count - 1] != targetRef)
                {
                    targetRef = _wayPolys[count - 1];
                    _wayQuery.ClosestPointOnPoly(targetRef, targetPt, out targetPt, out _);
                    OutOfReach++;
                }
            }

            // Its corners, from where it stood when the search began
            int corners = 0;
            if (count > 0)
                _wayQuery!.FindStraightPath(_wayFrom, targetPt, _wayPolys, count, _wayCorners, out corners, _wayCorners.Length, 0);
            if (corners == 0)
            {
                _wayCorners[0] = new DtStraightPath(_wayFrom, DtStraightPathFlags.DT_STRAIGHTPATH_START, 0);
                _wayCorners[1] = new DtStraightPath(targetPt, DtStraightPathFlags.DT_STRAIGHTPATH_END, targetRef);
                corners = 2;
            }

            bool ok = agent.WayOnly || _crowd!.RequestMoveTarget(agent.CrowdAgent!, targetRef, targetPt);
            if (!ok)
                Debug.Log($"[NavMeshWorld] RequestMoveTarget failed (agent={agent.CrowdAgent!.idx}, targetRef={targetRef})");
            agent.WayFound(ok, FromRc(targetPt), _wayCorners.AsSpan(0, corners));
        }

        /// <summary>Stop an agent.</summary>
        internal static void ResetAgentTarget(DtCrowdAgent agent)
        {
            _crowd?.ResetMoveTarget(agent);
        }

        // ── Coordinate Conversion ──
        // DotRecast uses RcVec3f (its own vec3), Freefall uses System.Numerics.Vector3.
        // Both are Y-up, same coordinate system — just type conversion.

        internal static RcVec3f ToRc(Vector3 v) => new(v.X, v.Y, v.Z);
        internal static Vector3 FromRc(RcVec3f v) => new(v.X, v.Y, v.Z);
    }

    /// <summary>
    /// The crowd's phases as profiler sections under Navigation ("Crowd plan velocity", ...), from DotRecast's telemetry
    /// as each begins and ends (our DotRecast: <see cref="IDtCrowdPhaseObserver"/>). The path queue's update runs inside
    /// the move requests', so the open sections are kept as a stack.
    /// </summary>
    internal sealed class CrowdPhases : IDtCrowdPhaseObserver
    {
        public static readonly CrowdPhases Instance = new();

        private readonly Dictionary<DtCrowdTimerLabel, ProfilerMarker> _markers = new();
        private readonly ProfilerScope[] _open = new ProfilerScope[8];
        private int _depth;

        public void Begin(DtCrowdTimerLabel phase)
        {
            if (!_markers.TryGetValue(phase, out var marker))
                _markers[phase] = marker = ProfilerMarker.Get("Crowd " + Words(phase.Label));
            if (_depth < _open.Length) _open[_depth] = marker.Auto();
            _depth++;
        }

        public void End(DtCrowdTimerLabel phase)
        {
            if (_depth == 0) return;
            if (--_depth < _open.Length) _open[_depth].Dispose();
        }

        // "PlanVelocity" → "plan velocity"
        private static string Words(string label)
        {
            var text = new System.Text.StringBuilder(label.Length + 4);
            for (int i = 0; i < label.Length; i++)
            {
                char c = label[i];
                if (char.IsUpper(c) && i > 0) text.Append(' ');
                text.Append(char.ToLowerInvariant(c));
            }
            return text.ToString();
        }
    }
}
