using System;
using System.Collections.Generic;
using System.Numerics;
using DotRecast.Core.Numerics;
using DotRecast.Recast;
using DotRecast.Recast.Geom;
using DotRecast.Detour;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Navigation
{
    /// <summary>
    /// Collects scene geometry (terrain + static meshes) and runs the
    /// Recast/Detour pipeline to produce a baked DtNavMesh.
    /// </summary>
    public static class NavMeshBuilder
    {
        /// <summary>
        /// Bake a navmesh from scene geometry using the given parameters.
        /// </summary>
        public static DtNavMesh? Build(Assets.NavMesh settings, out int polyCount, out int vertCount)
        {
            polyCount = 0;
            vertCount = 0;

            // ── Step 1: Collect geometry ──
            CollectGeometry(settings.TerrainSampleStep,
                out var verts, out var faces);

            if (verts.Count == 0 || faces.Count == 0)
            {
                Debug.Log("[NavMesh] No geometry collected — nothing to bake.");
                return null;
            }

            Debug.Log($"[NavMesh] Collected {verts.Count / 3} vertices, {faces.Count / 3} triangles");

            // ── Step 2: Create input geometry provider ──
            var geom = new SimpleInputGeomProvider(verts, faces);

            // ── Step 3: Configure Recast ──
            int walkableHeight = (int)MathF.Ceiling(settings.AgentHeight / settings.CellHeight);
            int walkableClimb = (int)MathF.Floor(settings.MaxClimb / settings.CellHeight);
            int walkableRadius = (int)MathF.Ceiling(settings.AgentRadius / settings.CellSize);
            float maxSlopeRad = settings.MaxSlope;

            var rcCfg = new RcConfig(
                useTiles: true,
                tileSizeX: 64,
                tileSizeZ: 64,
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
                vertsPerPoly: 6,
                detailSampleDist: 6f,
                detailSampleMaxError: 1f,
                filterLowHangingObstacles: true,
                filterLedgeSpans: true,
                filterWalkableLowHeightSpans: true,
                walkableAreaMod: new RcAreaModification(1),
                buildMeshDetail: true
            );

            // ── Step 4: Build tiles ──
            var builder = new RcBuilder();
            var results = builder.BuildTiles(geom, rcCfg, false, true);

            if (results == null || results.Count == 0)
            {
                Debug.Log("[NavMesh] Recast build produced no tiles.");
                return null;
            }

            // ── Step 5: Create DtNavMesh from tiles ──
            var meshParams = new DtNavMeshParams();
            var bmin = geom.GetMeshBoundsMin();
            meshParams.orig = bmin;
            meshParams.tileWidth = 64 * settings.CellSize;
            meshParams.tileHeight = 64 * settings.CellSize;

            // Determine grid size
            var bmax = geom.GetMeshBoundsMax();
            RcRecast.CalcTileCount(bmin, bmax, settings.CellSize, 64, 64, out var tw, out var th);
            meshParams.maxTiles = tw * th;
            meshParams.maxPolys = 32768;

            var navMesh = new DtNavMesh();
            navMesh.Init(meshParams, 6);

            int totalPolys = 0;
            int totalVerts = 0;

            foreach (var result in results)
            {
                var pmesh = result.Mesh;
                if (pmesh == null || pmesh.npolys == 0)
                    continue;

                // Recast leaves poly flags at 0 — set walkable flag so queries find them
                for (int i = 0; i < pmesh.npolys; i++)
                    pmesh.flags[i] = 1;

                // Build Detour navmesh data for this tile
                var option = GetNavMeshCreateParams(result, settings, pmesh);
                var meshData = DtNavMeshBuilder.CreateNavMeshData(option);
                if (meshData == null)
                    continue;

                navMesh.AddTile(meshData, 0, 0, out _);
                totalPolys += pmesh.npolys;
                totalVerts += pmesh.nverts;
            }

            polyCount = totalPolys;
            vertCount = totalVerts;

            Debug.Log($"[NavMesh] Built navmesh: {totalPolys} polys, {totalVerts} verts, {tw}x{th} tiles");
            return navMesh;
        }

        /// <summary>
        /// Collect all terrain and static mesh geometry into flat vertex/face lists.
        /// </summary>
        private static void CollectGeometry(float terrainStep,
            out List<float> verts, out List<int> faces)
        {
            verts = new List<float>();
            faces = new List<int>();

            // ── Terrain ──
            var terrains = ComponentCache<TerrainRenderer>.All;
            foreach (var terrain in terrains)
            {
                CollectTerrainGeometry(terrain, terrainStep, verts, faces);
            }

            // ── Static meshes with MeshRenderers ──
            var renderers = ComponentCache<MeshRenderer>.All;
            foreach (var mr in renderers)
            {
                CollectMeshGeometry(mr, verts, faces);
            }
        }

        private static void CollectTerrainGeometry(TerrainRenderer terrainRenderer,
            float step, List<float> verts, List<int> faces)
        {
            var terrain = terrainRenderer.Terrain;
            if (terrain?.HeightField == null) return;

            var terrainPos = terrainRenderer.Transform?.Position ?? Vector3.Zero;
            var size = terrain.TerrainSize;
            int baseVertex = verts.Count / 3;

            int stepsX = (int)MathF.Ceiling(size.X / step);
            int stepsZ = (int)MathF.Ceiling(size.Y / step);

            // Sample height grid
            for (int z = 0; z <= stepsZ; z++)
            {
                for (int x = 0; x <= stepsX; x++)
                {
                    float wx = x * step;
                    float wz = z * step;
                    wx = MathF.Min(wx, size.X);
                    wz = MathF.Min(wz, size.Y);

                    float wy = terrainRenderer.GetHeight(
                        new Vector3(wx, 0, wz) + terrainPos);

                    verts.Add(wx + terrainPos.X);
                    verts.Add(wy);
                    verts.Add(wz + terrainPos.Z);
                }
            }

            // Generate triangle pairs for each grid cell
            int rowWidth = stepsX + 1;
            for (int z = 0; z < stepsZ; z++)
            {
                for (int x = 0; x < stepsX; x++)
                {
                    int i00 = baseVertex + z * rowWidth + x;
                    int i10 = i00 + 1;
                    int i01 = i00 + rowWidth;
                    int i11 = i01 + 1;

                    // Triangle 1
                    faces.Add(i00);
                    faces.Add(i01);
                    faces.Add(i10);

                    // Triangle 2
                    faces.Add(i10);
                    faces.Add(i01);
                    faces.Add(i11);
                }
            }
        }

        private static void CollectMeshGeometry(MeshRenderer mr,
            List<float> verts, List<int> faces)
        {
            var mesh = mr.Mesh;
            if (mesh?.Positions == null || mesh.CpuIndices == null) return;

            // Only include static geometry (entities without RigidBody or with static RigidBody)
            var entity = mr.Entity;
            if (entity == null) return;

            var rb = entity.GetComponentInParents<RigidBody>();
            if (rb != null && !rb.IsStatic) return;

            var worldMatrix = mr.Transform?.Matrix ?? Matrix4x4.Identity;
            int baseVertex = verts.Count / 3;

            // Transform and add all vertices
            foreach (var pos in mesh.Positions)
            {
                var wp = Vector3.Transform(pos, worldMatrix);
                verts.Add(wp.X);
                verts.Add(wp.Y);
                verts.Add(wp.Z);
            }

            // Add all indices (offset by base)
            foreach (var idx in mesh.CpuIndices)
            {
                faces.Add(baseVertex + (int)idx);
            }
        }

        private static DtNavMeshCreateParams GetNavMeshCreateParams(
            RcBuilderResult result, Assets.NavMesh settings, RcPolyMesh pmesh)
        {
            var dmesh = result.MeshDetail;
            var option = new DtNavMeshCreateParams();

            option.verts = pmesh.verts;
            option.vertCount = pmesh.nverts;
            option.polys = pmesh.polys;
            option.polyAreas = pmesh.areas;
            option.polyFlags = pmesh.flags;
            option.polyCount = pmesh.npolys;
            option.nvp = pmesh.nvp;

            if (dmesh != null)
            {
                option.detailMeshes = dmesh.meshes;
                option.detailVerts = dmesh.verts;
                option.detailVertsCount = dmesh.nverts;
                option.detailTris = dmesh.tris;
                option.detailTriCount = dmesh.ntris;
            }

            option.walkableHeight = settings.AgentHeight;
            option.walkableRadius = settings.AgentRadius;
            option.walkableClimb = settings.MaxClimb;
            option.cs = settings.CellSize;
            option.ch = settings.CellHeight;
            option.buildBvTree = true;

            option.bmin = pmesh.bmin;
            option.bmax = pmesh.bmax;

            option.tileX = result.TileX;
            option.tileZ = result.TileZ;

            return option;
        }
    }

    /// <summary>
    /// Minimal implementation of IRcInputGeomProvider for DotRecast.
    /// Wraps flat vertex/face arrays.
    /// </summary>
    internal class SimpleInputGeomProvider : IRcInputGeomProvider
    {
        private readonly float[] _verts;
        private readonly int[] _faces;
        private readonly RcVec3f _bmin;
        private readonly RcVec3f _bmax;
        private readonly RcTriMesh _mesh;
        private readonly List<RcConvexVolume> _convexVolumes = new();
        private readonly List<RcOffMeshConnection> _offMeshConnections = new();

        public SimpleInputGeomProvider(List<float> verts, List<int> faces)
        {
            _verts = verts.ToArray();
            _faces = faces.ToArray();

            _bmin = new RcVec3f(float.MaxValue, float.MaxValue, float.MaxValue);
            _bmax = new RcVec3f(float.MinValue, float.MinValue, float.MinValue);

            for (int i = 0; i < _verts.Length; i += 3)
            {
                _bmin.X = MathF.Min(_bmin.X, _verts[i]);
                _bmin.Y = MathF.Min(_bmin.Y, _verts[i + 1]);
                _bmin.Z = MathF.Min(_bmin.Z, _verts[i + 2]);
                _bmax.X = MathF.Max(_bmax.X, _verts[i]);
                _bmax.Y = MathF.Max(_bmax.Y, _verts[i + 1]);
                _bmax.Z = MathF.Max(_bmax.Z, _verts[i + 2]);
            }

            _mesh = new RcTriMesh(_verts, _faces);
        }

        public RcTriMesh GetMesh() => _mesh;
        public RcVec3f GetMeshBoundsMin() => _bmin;
        public RcVec3f GetMeshBoundsMax() => _bmax;
        public IEnumerable<RcTriMesh> Meshes() => new[] { _mesh };
        public void AddConvexVolume(RcConvexVolume vol) => _convexVolumes.Add(vol);
        public IList<RcConvexVolume> ConvexVolumes() => _convexVolumes;
        public List<RcOffMeshConnection> GetOffMeshConnections() => _offMeshConnections;
        public void AddOffMeshConnection(RcVec3f start, RcVec3f end, float radius, bool bidir, int area, int flags)
            => _offMeshConnections.Add(new RcOffMeshConnection(start, end, radius, bidir, area, flags));
        public void RemoveOffMeshConnections(Predicate<RcOffMeshConnection> filter)
            => _offMeshConnections.RemoveAll(filter);
    }
}
