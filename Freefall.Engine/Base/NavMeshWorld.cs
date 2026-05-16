using System;
using System.Collections.Generic;
using System.Numerics;
using DotRecast.Core;
using DotRecast.Core.Numerics;
using DotRecast.Detour;
using DotRecast.Detour.Crowd;
using DotRecast.Detour.Io;

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
        private static DtNavMeshQuery? _navQuery;
        private static DtCrowd? _crowd;

        // Half-extents for navmesh point queries (search volume)
        private static readonly RcVec3f QueryExtents = new(2f, 4f, 2f);

        private static readonly IDtQueryFilter DefaultFilter = new DtQueryDefaultFilter();

        /// <summary>True if a navmesh is loaded and ready for queries.</summary>
        public static bool IsReady => _navMesh != null;

        // ── Lifecycle ──

        /// <summary>Load a baked NavMesh into the runtime.</summary>
        public static void Initialize(DtNavMesh navMesh)
        {
            _navMesh = navMesh;
            _navQuery = new DtNavMeshQuery(navMesh);

            // Create crowd with sane defaults
            var crowdCfg = new DtCrowdConfig(0.6f); // max agent radius
            _crowd = new DtCrowd(crowdCfg, navMesh);

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

        /// <summary>Initialize from a NavMesh asset (loads MeshData bytes).</summary>
        public static void Initialize(Assets.NavMesh asset)
        {
            if (asset.MeshData == null || asset.MeshData.Length == 0)
            {
                Debug.LogWarning("NavMeshWorld", "NavMesh asset has no baked data.");
                return;
            }

            // Deserialize DtNavMesh from bytes
            var navMesh = NavMeshSerializer.Deserialize(asset.MeshData);
            if (navMesh == null)
            {
                Debug.LogError("NavMeshWorld", "Failed to deserialize navmesh data.");
                return;
            }

            Initialize(navMesh);
        }

        /// <summary>Tear down — called on scene unload.</summary>
        public static void Shutdown()
        {
            _crowd = null;
            _navQuery = null;
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
        /// Returns smoothed waypoint list, or empty if no path found.
        /// </summary>
        public static List<Vector3> FindPath(Vector3 from, Vector3 to)
        {
            var result = new List<Vector3>();
            if (_navQuery == null) return result;

            var startPos = ToRc(from);
            var endPos = ToRc(to);

            _navQuery.FindNearestPoly(startPos, QueryExtents, DefaultFilter,
                out long startRef, out var startPt, out _);
            _navQuery.FindNearestPoly(endPos, QueryExtents, DefaultFilter,
                out long endRef, out var endPt, out _);

            if (startRef == 0 || endRef == 0) return result;

            // Find polygon path
            var pathPolys = new long[256];
            _navQuery.FindPath(startRef, endRef, startPt, endPt, DefaultFilter,
                pathPolys, out int pathCount, 256);

            if (pathCount == 0) return result;

            // Build straight path (smoothed waypoints)
            var straightPath = new DtStraightPath[256];
            _navQuery.FindStraightPath(startPt, endPt, pathPolys, pathCount,
                straightPath, out int straightCount, 256, 0);

            for (int i = 0; i < straightCount; i++)
            {
                result.Add(FromRc(straightPath[i].pos));
            }

            return result;
        }

        /// <summary>Find the nearest valid navmesh point to a world position.</summary>
        public static bool SamplePosition(Vector3 position, float radius, out Vector3 result)
        {
            result = position;
            if (_navQuery == null) return false;

            var extents = new RcVec3f(radius, radius, radius);
            var status = _navQuery.FindNearestPoly(ToRc(position), extents, DefaultFilter,
                out long polyRef, out var nearestPt, out _);

            if (status.Failed() || polyRef == 0) return false;

            result = FromRc(nearestPt);
            return true;
        }

        /// <summary>Is this world position on the navmesh?</summary>
        public static bool IsOnNavMesh(Vector3 position, float tolerance = 0.5f)
        {
            return SamplePosition(position, tolerance, out _);
        }

        // ── Crowd Management (internal, used by NavMeshAgent) ──

        /// <summary>Register an agent with the crowd.</summary>
        internal static DtCrowdAgent? AddCrowdAgent(Vector3 position, DtCrowdAgentParams agentParams)
        {
            if (_crowd == null) return null;

            var agent = _crowd.AddAgent(ToRc(position), agentParams);
            Debug.Log($"[NavMeshWorld] AddCrowdAgent: idx={agent?.idx}, state={agent?.state}, pos={position}");
            return agent;
        }

        /// <summary>Remove an agent from the crowd.</summary>
        internal static void RemoveCrowdAgent(DtCrowdAgent agent)
        {
            _crowd?.RemoveAgent(agent);
        }

        /// <summary>Set an agent's move target.</summary>
        internal static bool SetAgentTarget(DtCrowdAgent agent, Vector3 target)
        {
            if (_crowd == null) return false;

            // Use the crowd's own query, filter, and extents — must match what
            // the crowd uses internally for path validation (see DotRecast demo).
            var navQuery = _crowd.GetNavMeshQuery();
            var filter = _crowd.GetFilter(0);
            var halfExtents = _crowd.GetQueryExtents();

            var pos = ToRc(target);
            var status = navQuery.FindNearestPoly(pos, halfExtents, filter,
                out long targetRef, out var targetPt, out _);

            if (targetRef == 0)
            {
                Debug.Log($"[NavMeshWorld] SetAgentTarget: FindNearestPoly failed for {target} (status={status}, extents={halfExtents})");
                return false;
            }

            bool ok = _crowd.RequestMoveTarget(agent, targetRef, targetPt);
            if (!ok)
                Debug.Log($"[NavMeshWorld] SetAgentTarget: RequestMoveTarget failed (agent={agent.idx}, targetRef={targetRef})");

            return ok;
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
    /// Handles serialization/deserialization of DtNavMesh to/from byte arrays.
    /// </summary>
    internal static class NavMeshSerializer
    {
        /// <summary>Serialize a DtNavMesh to a byte array for storage.</summary>
        public static byte[] Serialize(DtNavMesh navMesh)
        {
            var writer = new DtMeshSetWriter();
            using var ms = new System.IO.MemoryStream();
            using var bw = new System.IO.BinaryWriter(ms);
            writer.Write(bw, navMesh, RcByteOrder.LITTLE_ENDIAN, false);
            return ms.ToArray();
        }

        /// <summary>Deserialize a DtNavMesh from a byte array.</summary>
        public static DtNavMesh? Deserialize(byte[] data)
        {
            try
            {
                var reader = new DtMeshSetReader();
                using var ms = new System.IO.MemoryStream(data);
                using var br = new System.IO.BinaryReader(ms);
                return reader.Read(br, 6); // maxVertsPerPoly = 6
            }
            catch (Exception ex)
            {
                Debug.Log($"[NavMeshWorld] Deserialize failed: {ex.Message}");
                return null;
            }
        }
    }
}
