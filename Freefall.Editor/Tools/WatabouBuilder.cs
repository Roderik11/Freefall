using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Freefall.Procedural;
using Vortice.Mathematics;

namespace Freefall.Editor.Tools
{
    [CreateAsset("Watabou Theme")]
    public class WatabouTheme : Asset
    {
        public Material WallMaterial = InternalAssets.DefaultMaterial;

        public Material RoadMaterial = InternalAssets.DefaultMaterial;
        public Material RoadCurbMaterial = InternalAssets.DefaultMaterial;

        public Material SidewalkMaterial = InternalAssets.DefaultMaterial;
        public Material SidewalkCurbMaterial = InternalAssets.DefaultMaterial;

        public Prefab TowerPrefab;
        public Prefab GatePrefab;

        public List<Prefab> Houses = [];
    }

    /// <summary>
    /// Import settings for Watabou village/city.
    /// Exposed via GenericInspector in the importer window.
    /// </summary>
    public class WatabouSettings
    {
        [FilePath("Watabout File", "*.json")]
        public string FilePath;

        /// <summary>Target road width in meters. Scale is computed as TargetRoadWidth / JSON roadWidth.</summary>
        [ValueRange(1f, 20f)]
        public float RoadWidthMeters = 8f;

        [ValueRange(0.1f, 1f)]
        public float ShrinkBuildings = 1f;

        [ValueRange(2f, 20f)]
        public float BuildingHeightMin = 6f;

        [ValueRange(2f, 30f)]
        public float BuildingHeightMax = 12f;

        [ValueRange(2f, 15f)]
        public float WallHeight = 13f;

        [ValueRange(3f, 25f)]
        public float TowerHeight = 30f;

        /// <summary>Terrain layer painted under the roads (none = roads leave the ground as it is).</summary>
        public TerrainLayer RoadLayer;

        public bool ImportBuildings = true;
        public bool ImportRoads = true;
        public bool ImportWalls = true;
        public bool ImportTowers = true;
        public bool ImportRivers = true;
        public bool AddTerrainInfluence = true;

        /// <summary>Use WFC modular building meshes instead of simple extrusion.</summary>
        public bool ModularBuildings = false;

        /// <summary>Interpret building polygons as city blocks (pavement + house filling) instead of individual houses.</summary>
        public bool BuildingsAreBlocks = false;

        [ValueRange(2.5f, 5f)]
        public float StoryHeight = 3.5f;

        [ValueRange(1.5f, 4f)]
        public float CellWidth = 2.5f;

        [ValueRange(0.6f, 2f)]
        public float WindowWidth = 1.2f;

        [ValueRange(0.8f, 2f)]
        public float WindowHeight = 1.5f;

        [ValueRange(0.05f, 0.3f)]
        public float WindowInset = 0.15f;

        [ValueRange(1f, 2.5f)]
        public float DoorWidth = 1.5f;

        [ValueRange(2f, 3.5f)]
        public float DoorHeight = 2.5f;

        public WatabouTheme Theme;
    }

    /// <summary>
    /// Builds a scene hierarchy from parsed Watabou data.
    /// Creates entities with MeshRenderer, Spline, and HeightStamp components.
    /// </summary>
    public static class WatabouBuilder
    {
        private static Random _rng = new();

        public static Entity Build(WatabouData data, WatabouSettings settings, string villageName)
        {
            // Compute scale factor from anchor: road width
            float scale = settings.RoadWidthMeters / Math.Max(data.RoadWidth, 0.1f);
            _rng = new Random(villageName.GetHashCode());

            // Root entity
            var root = new Entity(villageName);

            // Walls first — collects gate positions for tower exclusion
            var gatePositions = new List<Vector2>();
            if (settings.ImportWalls)
                gatePositions = BuildWalls(root, data, settings, scale);

            if (settings.ImportRoads)
                BuildRoads(root, data, settings, scale);

            if (settings.ImportTowers)
                BuildTowers(root, data, settings, scale, gatePositions);

            if (settings.ImportRivers)
                BuildRivers(root, data, settings, scale);

            var theme = settings.Theme;
            // Building / house placement
            if (settings.BuildingsAreBlocks && theme?.Houses.Count > 0 && settings.ImportBuildings)
                HousePlacer.PlaceInBlocks(root, data, settings, scale, _rng);
            else if (theme?.Houses.Count > 0)
                HousePlacer.PlaceAlongRoads(root, data, settings, scale, _rng);
            else if (settings.ImportBuildings)
                BuildBuildings(root, data, settings, scale, settings.ShrinkBuildings);

            MessageDispatcher.Send(Msg.RefreshExplorer);
            return root;
        }

        // ═══════════════════════════
        // ── Buildings ──
        // ═══════════════════════════

        private static void BuildBuildings(Entity root, WatabouData data, WatabouSettings settings, float scale, float shrink)
        {
            var buildingsRoot = new Entity("Buildings");
            buildingsRoot.Transform.Parent = root.Transform;

            // Precompute city center and median area for height heuristic
            var allCentroids = new List<Vector2>();
            var allAreas = new List<float>();
            foreach (var fp in data.Buildings)
            {
                if (fp.Count < 3) continue;
                var scaled = new List<Vector2>();
                foreach (var p in fp) scaled.Add(new Vector2(p.X * scale, p.Y * scale));
                allCentroids.Add(ComputeCentroid(scaled));
                allAreas.Add(Math.Abs(GetSignedArea(scaled)));
            }

            var cityCenter = ComputeCentroid(allCentroids);
            allAreas.Sort();
            float medianArea = allAreas.Count > 0 ? allAreas[allAreas.Count / 2] : 1f;

            // Compute max distance from center for normalization
            float maxDistFromCenter = 1f;
            foreach (var c in allCentroids)
                maxDistFromCenter = Math.Max(maxDistFromCenter, (c - cityCenter).Length());

            // Prefab building placer (used when ModularBuildings is enabled)
            PrefabBuildingPlacer prefabPlacer = null;
            if (settings.ModularBuildings)
                prefabPlacer = new PrefabBuildingPlacer(settings, _rng.Next());

            int areaIdx = 0;
            for (int i = 0; i < data.Buildings.Count; i++)
            {
                var footprint = data.Buildings[i];
                if (footprint.Count < 3) { continue; }

                // Scale and convert coordinates (Watabou Y → engine Z)
                var scaledFootprint = new List<Vector2>();
                foreach (var p in footprint)
                    scaledFootprint.Add(new Vector2(p.X * scale, p.Y * scale));

                // Compute centroid
                var centroid = ComputeCentroid(scaledFootprint);

                // Center the polygon around origin
                var centered = new List<Vector2>();
                foreach (var p in scaledFootprint)
                    centered.Add(p - centroid);

                for (int j = 0; j < centered.Count; j++)
                    centered[j] *= shrink;

                // ── Height heuristic ──
                float area = Math.Abs(GetSignedArea(centered));
                float areaFactor = Math.Clamp(area / Math.Max(medianArea, 0.01f), 0.4f, 2.0f);
                float distFromCenter = (centroid - cityCenter).Length();
                float centerFactor = 1.0f + 0.3f * (1.0f - distFromCenter / maxDistFromCenter);
                float noise = 0.85f + (float)_rng.NextDouble() * 0.3f; // ±15%

                float height = settings.BuildingHeightMin +
                    (settings.BuildingHeightMax - settings.BuildingHeightMin) * areaFactor * 0.5f;
                height *= centerFactor * noise;
                height = Math.Clamp(height, settings.BuildingHeightMin, settings.BuildingHeightMax * 1.5f);

                // ── Prefab-based building ──
                if (prefabPlacer != null)
                {
                    var building = prefabPlacer.Build(centered, height, i);
                    if (building != null)
                    {
                        building.Transform.Parent = buildingsRoot.Transform;
                        building.Transform.Position = new Vector3(centroid.X, 0, centroid.Y);

                        if (settings.AddTerrainInfluence)
                        {
                            var hs = building.AddComponent<HeightStamp>();
                            hs.Radius = ComputePolygonRadius(centered) + 1f;
                            hs.Falloff = 2f;

                            var ds = building.AddComponent<DecoStamp>();
                            ds.Radius = hs.Radius;
                            ds.Falloff = hs.Falloff;
                        }

                        areaIdx++;
                        continue;
                    }
                    // Fall through to mesh extrusion if prefab placement fails
                }
                
                var theme = settings.Theme;

                if (theme?.Houses.Count > 0)
                {
                    // Attempt to place a house prefab (non-modular, purely decorative)
                    var housePrefab = theme.Houses[_rng.Next(theme.Houses.Count)];
                    var house = housePrefab.Instantiate();
                    if (house != null)
                    {
                        house.Transform.Parent = buildingsRoot.Transform;
                        house.Transform.Position = new Vector3(centroid.X, 0, centroid.Y);
                        house.Transform.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, (float)_rng.NextDouble() * MathF.PI * 2f);
                        areaIdx++;
                        continue;
                    }
                }
                else
                {
                    // ── Mesh-based building (fallback) ──
                    var entity = new Entity($"Building_{i}");
                    entity.Transform.Parent = buildingsRoot.Transform;
                    entity.Transform.Position = new Vector3(centroid.X, 0, centroid.Y);

                    Mesh mesh = ExtrudePolygon(centered, height);

                    if (mesh == null) continue;

                    var renderer = entity.AddComponent<MeshRenderer>();
                    renderer.Mesh = mesh;
                    renderer.Material = InternalAssets.DefaultMaterial;

                    // Terrain influence
                    if (settings.AddTerrainInfluence)
                    {
                        var hs = entity.AddComponent<HeightStamp>();
                        hs.Radius = ComputePolygonRadius(centered) + 1f;
                        hs.Falloff = 2f;

                        var ds = entity.AddComponent<DecoStamp>();
                        ds.Radius = hs.Radius;
                        ds.Falloff = hs.Falloff;
                    }
                }

                areaIdx++;
            }
        }

        // ═════════════════════════════════════════════
        // ── WFC Modular Building ──
        // ═════════════════════════════════════════════

        /// <summary>
        /// Generate a modular building mesh using WFC per facade.
        /// Each polygon edge becomes a facade grid (columns × stories),
        /// solved with WFC for tile placement, then meshed.
        /// </summary>
        private static Mesh BuildModularBuilding(List<Vector2> polygon, float height,
            WatabouSettings settings, int buildingSeed)
        {
            // Ensure CCW winding
            if (GetSignedArea(polygon) > 0)
                polygon.Reverse();

            int n = polygon.Count;
            int stories = Math.Max(1, (int)(height / settings.StoryHeight));
            float actualStoryH = height / stories;

            var verts = new List<Vector3>();
            var norms = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();

            // ── Floor cap ──
            var floorTris = EarClipTriangulate(polygon);
            if (floorTris != null && floorTris.Count >= 3)
            {
                uint floorBase = (uint)verts.Count;
                for (int i = 0; i < n; i++)
                {
                    verts.Add(new Vector3(polygon[i].X, 0, polygon[i].Y));
                    norms.Add(-Vector3.UnitY);
                    uvs.Add(polygon[i] * 0.1f);
                }
                for (int i = 0; i < floorTris.Count / 3; i++)
                {
                    indices.Add(floorBase + (uint)floorTris[i * 3 + 0]);
                    indices.Add(floorBase + (uint)floorTris[i * 3 + 1]);
                    indices.Add(floorBase + (uint)floorTris[i * 3 + 2]);
                }
            }

            // ── Roof ──
            GenerateRoof(polygon, height, buildingSeed, verts, norms, uvs, indices);

            // ── Per-edge facade via WFC ──
            var ruleSet = BuildingTiles.CreateRuleSet();

            // Find longest edge (gets the door)
            int longestEdge = 0;
            float longestLen = 0;
            for (int i = 0; i < n; i++)
            {
                float len = (polygon[(i + 1) % n] - polygon[i]).Length();
                if (len > longestLen) { longestLen = len; longestEdge = i; }
            }

            for (int edge = 0; edge < n; edge++)
            {
                int next = (edge + 1) % n;
                var p0 = polygon[edge];
                var p1 = polygon[next];

                var edge2D = p1 - p0;
                float edgeLen = edge2D.Length();
                if (edgeLen < 0.01f) continue;

                int columns = Math.Max(1, (int)(edgeLen / settings.CellWidth));
                float actualCellW = edgeLen / columns;

                // Grid: columns wide × stories tall (wall rows only, no WFC roof row)
                var solver = new WFCSolver(columns, stories, ruleSet,
                    seed: buildingSeed * 7919 + edge * 31);

                // Doors on faces with enough room (2+ columns)
                bool hasDoor = (columns >= 2);
                BuildingTiles.ApplyBuildingConstraints(solver, stories, hasDoor);

                // Solve
                solver.Solve();

                // Convert solved tiles to mesh geometry
                var dir2D = edge2D / edgeLen;
                var right = new Vector3(dir2D.X, 0, dir2D.Y);
                var up = Vector3.UnitY;
                var outNormal = Vector3.Normalize(new Vector3(dir2D.Y, 0, -dir2D.X));

                for (int col = 0; col < columns; col++)
                {
                    for (int row = 0; row < stories; row++)
                    {
                        int tileId = solver.GetTile(col, row);
                        if (tileId < 0) tileId = BuildingTiles.WallSolid; // Fallback

                        var origin = new Vector3(p0.X, 0, p0.Y)
                                   + right * (col * actualCellW)
                                   + up * (row * actualStoryH);

                        BuildingTiles.GenerateTileMesh(
                            tileId, actualCellW, actualStoryH,
                            settings.WindowWidth, settings.WindowHeight, settings.WindowInset,
                            settings.DoorWidth, settings.DoorHeight,
                            verts, norms, uvs, indices,
                            origin, right, up, outNormal);
                    }
                }
            }

            if (verts.Count == 0) return null;

            var mesh = new Mesh(Engine.Device, verts.ToArray(), norms.ToArray(), uvs.ToArray(), indices.ToArray());
            mesh.BoundingBox = ComputeBounds(verts);
            mesh.MeshParts.Add(new MeshPart
            {
                NumIndices = indices.Count,
                BoundingBox = mesh.BoundingBox,
                BoundingSphere = mesh.LocalBoundingSphere
            });
            return mesh;
        }

        // ═════════════════════════════════════════════
        // ── Roof Generation ──
        // ═════════════════════════════════════════════

        private enum RoofType { Flat, Pyramid, Gabled }

        /// <summary>
        /// Generate a roof on top of the building walls.
        /// Type is chosen based on footprint aspect ratio + seed-based randomness.
        /// </summary>
        private static void GenerateRoof(List<Vector2> polygon, float wallHeight, int seed,
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices)
        {
            int n = polygon.Count;
            if (n < 3) return;

            // Compute footprint bounding box for aspect ratio
            var bbMin = new Vector2(float.MaxValue);
            var bbMax = new Vector2(float.MinValue);
            foreach (var p in polygon)
            {
                bbMin = Vector2.Min(bbMin, p);
                bbMax = Vector2.Max(bbMax, p);
            }
            float bbW = bbMax.X - bbMin.X;
            float bbH = bbMax.Y - bbMin.Y;
            float aspect = Math.Max(bbW, bbH) / Math.Max(Math.Min(bbW, bbH), 0.01f);

            // Choose roof type
            var rng = new Random(seed * 13 + 7);
            float roll = (float)rng.NextDouble();

            RoofType roofType;
            if (aspect > 1.8f)
                roofType = roll < 0.15f ? RoofType.Flat : RoofType.Gabled; // Rectangular → mostly gabled
            else if (aspect > 1.3f)
                roofType = roll < 0.3f ? RoofType.Pyramid : (roll < 0.5f ? RoofType.Flat : RoofType.Gabled);
            else
                roofType = roll < 0.4f ? RoofType.Flat : RoofType.Pyramid; // Square → mostly pyramid

            // Roof peak height (relative to wall top)
            float footprintMinDim = Math.Min(bbW, bbH);
            float roofRise = footprintMinDim * (0.3f + (float)rng.NextDouble() * 0.3f); // 30-60% of narrow side

            switch (roofType)
            {
                case RoofType.Flat:
                    GenerateFlatRoof(polygon, wallHeight, verts, norms, uvs, indices);
                    break;
                case RoofType.Pyramid:
                    GeneratePyramidRoof(polygon, wallHeight, roofRise, verts, norms, uvs, indices);
                    break;
                case RoofType.Gabled:
                    GenerateGabledRoof(polygon, wallHeight, roofRise, bbMin, bbMax, verts, norms, uvs, indices);
                    break;
            }
        }

        /// <summary>Flat roof — simple ear-clipped cap at wall top.</summary>
        private static void GenerateFlatRoof(List<Vector2> polygon, float y,
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices)
        {
            int n = polygon.Count;
            var tris = EarClipTriangulate(polygon);
            if (tris == null || tris.Count < 3) return;

            uint roofBase = (uint)verts.Count;
            for (int i = 0; i < n; i++)
            {
                verts.Add(new Vector3(polygon[i].X, y, polygon[i].Y));
                norms.Add(Vector3.UnitY);
                uvs.Add(polygon[i] * 0.1f);
            }
            for (int i = 0; i < tris.Count / 3; i++)
            {
                indices.Add(roofBase + (uint)tris[i * 3 + 2]);
                indices.Add(roofBase + (uint)tris[i * 3 + 1]);
                indices.Add(roofBase + (uint)tris[i * 3 + 0]);
            }
        }

        /// <summary>
        /// Pyramid/hip roof — raised centroid with triangle fan from each edge.
        /// Works for any polygon shape.
        /// </summary>
        private static void GeneratePyramidRoof(List<Vector2> polygon, float wallHeight, float rise,
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices)
        {
            int n = polygon.Count;
            var centroid = ComputeCentroid(polygon);
            var peak = new Vector3(centroid.X, wallHeight + rise, centroid.Y);

            // One triangle per edge → peak
            for (int i = 0; i < n; i++)
            {
                int next = (i + 1) % n;
                var p0 = new Vector3(polygon[i].X, wallHeight, polygon[i].Y);
                var p1 = new Vector3(polygon[next].X, wallHeight, polygon[next].Y);

                // Outward slope normal: Cross(toPeak, toNext) for CW polygon
                var toPeak = peak - p0;
                var toNext = p1 - p0;
                var normal = Vector3.Normalize(Vector3.Cross(toPeak, toNext));

                uint baseIdx = (uint)verts.Count;
                verts.Add(p0);
                verts.Add(p1);
                verts.Add(peak);
                norms.Add(normal);
                norms.Add(normal);
                norms.Add(normal);
                uvs.Add(new Vector2(0, 0));
                uvs.Add(new Vector2(toNext.Length(), 0));
                uvs.Add(new Vector2(toNext.Length() * 0.5f, toPeak.Length()));

                // Winding: p0, peak, p1 (CW from outside for CW polygon)
                indices.Add(baseIdx + 0);
                indices.Add(baseIdx + 2);
                indices.Add(baseIdx + 1);
            }
        }

        /// <summary>
        /// Gabled roof — ridge along the longest bounding box axis.
        /// Each polygon edge creates a slope from wall-top to its ridge projection.
        /// Per-triangle normals handle non-planar quads correctly.
        /// </summary>
        private static void GenerateGabledRoof(List<Vector2> polygon, float wallHeight, float rise,
            Vector2 bbMin, Vector2 bbMax,
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices)
        {
            int n = polygon.Count;
            float bbW = bbMax.X - bbMin.X;
            float bbH = bbMax.Y - bbMin.Y;

            // Ridge direction: along the longer axis
            Vector2 ridgeDir;
            Vector2 center = (bbMin + bbMax) * 0.5f;

            if (bbW >= bbH)
                ridgeDir = Vector2.UnitX;
            else
                ridgeDir = Vector2.UnitY;

            float ridgeHalfLen = Math.Max(bbW, bbH) * 0.5f;
            float peakY = wallHeight + rise;

            for (int i = 0; i < n; i++)
            {
                int next = (i + 1) % n;
                var v0 = polygon[i];
                var v1 = polygon[next];

                var r0 = ProjectOntoRidge(v0, center, ridgeDir, ridgeHalfLen);
                var r1 = ProjectOntoRidge(v1, center, ridgeDir, ridgeHalfLen);

                var p0 = new Vector3(v0.X, wallHeight, v0.Y);
                var p1 = new Vector3(v1.X, wallHeight, v1.Y);
                var r0_3d = new Vector3(r0.X, peakY, r0.Y);
                var r1_3d = new Vector3(r1.X, peakY, r1.Y);

                // Skip degenerate quads (edge lies on the ridge line)
                var toRidge = r0_3d - p0;
                if (toRidge.LengthSquared() < 0.01f) continue;

                // Triangle 1: p0, r0, p1
                var n1 = Vector3.Normalize(Vector3.Cross(r0_3d - p0, p1 - p0));

                // Triangle 2: p1, r0, r1
                var n2 = Vector3.Normalize(Vector3.Cross(r0_3d - p1, r1_3d - p1));

                uint baseIdx = (uint)verts.Count;
                // Tri 1 verts
                verts.Add(p0);
                verts.Add(p1);
                verts.Add(r0_3d);
                norms.Add(n1);
                norms.Add(n1);
                norms.Add(n1);
                uvs.Add(new Vector2(0, 0));
                uvs.Add(new Vector2((p1 - p0).Length(), 0));
                uvs.Add(new Vector2(0, toRidge.Length()));

                // Tri 2 verts
                verts.Add(p1);
                verts.Add(r0_3d);
                verts.Add(r1_3d);
                norms.Add(n2);
                norms.Add(n2);
                norms.Add(n2);
                uvs.Add(new Vector2(0, 0));
                uvs.Add(new Vector2(toRidge.Length(), 0));
                uvs.Add(new Vector2((r1_3d - r0_3d).Length(), toRidge.Length()));

                // Tri 1: p0, r0, p1 (CW from outside)
                indices.Add(baseIdx + 0);
                indices.Add(baseIdx + 2);
                indices.Add(baseIdx + 1);
                // Tri 2: p1, r0, r1 (CW from outside)
                indices.Add(baseIdx + 3);
                indices.Add(baseIdx + 4);
                indices.Add(baseIdx + 5);
            }
        }

        /// <summary>Project a 2D point onto the ridge line, clamped to ridge endpoints.</summary>
        private static Vector2 ProjectOntoRidge(Vector2 point, Vector2 ridgeCenter, Vector2 ridgeDir, float ridgeHalfLen)
        {
            float t = Vector2.Dot(point - ridgeCenter, ridgeDir);
            t = Math.Clamp(t, -ridgeHalfLen, ridgeHalfLen);
            return ridgeCenter + ridgeDir * t;
        }

        // ═══════════════════════════
        // ── Roads ──
        // ═══════════════════════════

        private static void BuildRoads(Entity root, WatabouData data, WatabouSettings settings, float scale)
        {
            var roadsRoot = new Entity("Roads");
            roadsRoot.Transform.Parent = root.Transform;

            var theme = settings.Theme;

            for (int i = 0; i < data.Roads.Count; i++)
            {
                var road = data.Roads[i];
                if (road.Points.Count < 2) continue;

                float roadWidth = road.Width * scale;

                var entity = new Entity($"Road_{i}");
                entity.Transform.Parent = roadsRoot.Transform;

                // Add spline with scaled points
                var spline = entity.AddComponent<Spline>();
                spline.Points.Clear();
                foreach (var p in road.Points)
                    spline.Points.Add(new Vector3(p.X * scale, 0, p.Y * scale));

                // RuntimeMesh generates the road surface strip
                var rtMesh = entity.AddComponent<RuntimeMesh>();
                rtMesh.Width = roadWidth;
                rtMesh.HeightMode = RuntimeMeshHeightMode.Surface;
                rtMesh.EnableCurbs = true;
                rtMesh.Smoothness = 4;

                var renderer = entity.AddComponent<MeshRenderer>();
                renderer.Material = theme?.RoadMaterial ?? InternalAssets.DefaultMaterial;

                // Terrain stamps along the road
                if (settings.AddTerrainInfluence)
                {
                    var hs = entity.AddComponent<HeightStamp>();
                    hs.Radius = roadWidth * 0.5f + 1f;
                    hs.Falloff = 1f;

                    var ss = entity.AddComponent<SplatStamp>();
                    ss.Radius = hs.Radius;
                    ss.Falloff = hs.Falloff;
                    ss.Layer = settings.RoadLayer;

                    var ds = entity.AddComponent<DecoStamp>();
                    ds.Radius = hs.Radius;
                    ds.Falloff = hs.Falloff;
                }
            }
        }

        // ═══════════════════════════
        // ── Walls ──
        // ═══════════════════════════

        /// <summary>Returns gate positions (scaled 2D) for tower exclusion.</summary>
        private static List<Vector2> BuildWalls(Entity root, WatabouData data, WatabouSettings settings, float scale)
        {
            var wallsRoot = new Entity("Walls");
            wallsRoot.Transform.Parent = root.Transform;
            var allGatePositions = new List<Vector2>();
            var theme = settings.Theme;

            for (int i = 0; i < data.Walls.Count; i++)
            {
                var wall = data.Walls[i];
                if (wall.Points.Count < 3) continue;

                // Scale wall polygon to 2D
                var scaledPoints2D = new List<Vector2>();
                foreach (var p in wall.Points)
                    scaledPoints2D.Add(new Vector2(p.X * scale, p.Y * scale));

                float wallWidth = wall.Width * scale;
                float wallHeight = settings.WallHeight;

                // Try splitting wall at road crossings
                var split = WallSplitter.Split(scaledPoints2D, data.Roads, scale);

                if (split == null)
                {
                    // No road crossings — single closed wall
                    CreateWallSegmentEntity(wallsRoot, $"Wall_{i}", scaledPoints2D,
                        closed: true, wallWidth, wallHeight, settings);
                }
                else
                {
                    // Open wall segments between gates
                    for (int s = 0; s < split.Segments.Count; s++)
                    {
                        CreateWallSegmentEntity(wallsRoot, $"Wall_{i}_Seg{s}", split.Segments[s],
                            closed: false, wallWidth, wallHeight, settings);
                    }

                    // Gate prefabs at road crossings
                    if (theme?.GatePrefab != null)
                    {   
                        for (int g = 0; g < split.Gates.Count; g++)
                        {
                            var crossing = split.Gates[g];
                            var gate = theme?.GatePrefab.Instantiate();
                            if (gate == null) continue;

                            gate.Name = $"Gate_{i}_{g}";
                            gate.Transform.Parent = wallsRoot.Transform;
                            gate.Transform.Position = new Vector3(crossing.Position.X, 0, crossing.Position.Y);

                            // Orient gate perpendicular to wall tangent (facing along the road)
                            float angle = MathF.Atan2(crossing.WallTangent.X, crossing.WallTangent.Y) + MathF.PI * 0.5f;
                            gate.Transform.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, angle);
                        }
                    }

                    // Collect gate positions for tower exclusion
                    foreach (var g in split.Gates)
                        allGatePositions.Add(g.Position);
                }
            }

            return allGatePositions;
        }

        /// <summary>
        /// Create a wall entity with Spline + RuntimeMesh + MeshRenderer.
        /// </summary>
        private static void CreateWallSegmentEntity(Entity parent, string name, List<Vector2> points2D,
            bool closed, float wallWidth, float wallHeight, WatabouSettings settings)
        {
            if (points2D.Count < 2) return;
            var theme = settings.Theme;

            var entity = new Entity(name);
            entity.Transform.Parent = parent.Transform;

            var spline = entity.AddComponent<Spline>();
            spline.Points.Clear();
            spline.Closed = closed;
            spline.Tension = 1f; // tight — keep polygon corners sharp
            foreach (var p in points2D)
                spline.Points.Add(new Vector3(p.X, 0, p.Y));

            // RuntimeMesh generates the wall geometry from the spline
            var rtMesh = entity.AddComponent<RuntimeMesh>();
            rtMesh.Width = wallWidth;
            rtMesh.Height = wallHeight;
            rtMesh.HeightMode = RuntimeMeshHeightMode.Surface;
            rtMesh.Smoothness = 2;

            var renderer = entity.AddComponent<MeshRenderer>();
            if (theme?.WallMaterial != null)
                renderer.Material = theme.WallMaterial;
            else
                renderer.Material = InternalAssets.DefaultMaterial;

            if (settings.AddTerrainInfluence)
            {
                var hs = entity.AddComponent<HeightStamp>();
                hs.Radius = wallWidth * 0.5f + 1f;
                hs.Falloff = 1f;

                var ds = entity.AddComponent<DecoStamp>();
                ds.Radius = hs.Radius;
                ds.Falloff = hs.Falloff;
            }
        }

        // ═══════════════════════════
        // ── Towers ──
        // ═══════════════════════════

        private static void BuildTowers(Entity root, WatabouData data, WatabouSettings settings, float scale, List<Vector2> gatePositions)
        {
            if (data.Walls.Count == 0) return;

            var towersRoot = new Entity("Towers");
            towersRoot.Transform.Parent = root.Transform;
            var theme = settings.Theme;

            float radius = data.TowerRadius * scale;
            float height = settings.TowerHeight;
            int towerIndex = 0;

            for (int w = 0; w < data.Walls.Count; w++)
            {
                var wall = data.Walls[w];
                if (wall.Points.Count < 3) continue;

                foreach (var vertex in wall.Points)
                {
                    var center = new Vector2(vertex.X * scale, vertex.Y * scale);

                    // Skip tower if a gate occupies this vertex
                    bool isGate = false;
                    foreach (var gp in gatePositions)
                    {
                        if (Vector2.Distance(center, gp) < 1f)
                        { isGate = true; break; }
                    }
                    if (isGate) continue;

                    var entity = new Entity($"Tower_{towerIndex++}");
                    entity.Transform.Parent = towersRoot.Transform;
                    entity.Transform.Position = new Vector3(center.X, 0, center.Y);

                    if (theme?.TowerPrefab != null)
                    {
                        var tower = theme?.TowerPrefab.Instantiate();
                        if (tower != null)
                        {
                            tower.Transform.Parent = towersRoot.Transform;
                            tower.Transform.Position = new Vector3(center.X, 0, center.Y);
                            tower.Name = entity.Name;
                            entity.Destroy();
                        }
                    }
                    else
                    {
                        // Fallback: 12-sided cylinder extrusion
                        const int sides = 12;
                        var footprint = new List<Vector2>();
                        for (int s = 0; s < sides; s++)
                        {
                            float angle = s * MathF.Tau / sides;
                            footprint.Add(new Vector2(
                                MathF.Cos(angle) * radius,
                                MathF.Sin(angle) * radius));
                        }

                        var mesh = ExtrudePolygon(footprint, height);
                        if (mesh == null) continue;

                        var renderer = entity.AddComponent<MeshRenderer>();
                        renderer.Mesh = mesh;
                        renderer.Material = InternalAssets.DefaultMaterial;
                    }

                    if (settings.AddTerrainInfluence)
                    {
                        var hs = entity.AddComponent<HeightStamp>();
                        hs.Radius = radius + 1f;
                        hs.Falloff = 2f;

                        var ds = entity.AddComponent<DecoStamp>();
                        ds.Radius = hs.Radius;
                        ds.Falloff = hs.Falloff;
                    }
                }
            }
        }

        // ═══════════════════════════
        // ── Rivers ──
        // ═══════════════════════════

        private static void BuildRivers(Entity root, WatabouData data, WatabouSettings settings, float scale)
        {
            var riversRoot = new Entity("Rivers");
            riversRoot.Transform.Parent = root.Transform;

            for (int i = 0; i < data.Rivers.Count; i++)
            {
                var river = data.Rivers[i];
                if (river.Points.Count < 2) continue;

                var entity = new Entity($"River_{i}");
                entity.Transform.Parent = riversRoot.Transform;

                var spline = entity.AddComponent<Spline>();
                spline.Points.Clear();
                foreach (var p in river.Points)
                    spline.Points.Add(new Vector3(p.X * scale, 0, p.Y * scale));

                if (settings.AddTerrainInfluence)
                {
                    var hs = entity.AddComponent<HeightStamp>();
                    hs.Radius = river.Width * scale * 0.5f;
                    hs.Falloff = river.Width * scale * 0.3f;
                }
            }
        }

        // ═══════════════════════════════════════
        // ── Mesh Generation Helpers ──
        // ═══════════════════════════════════════

        /// <summary>
        /// Extrude a 2D polygon (XZ plane) into a 3D mesh with floor, walls, and roof.
        /// Polygon points are in local space (centered around origin).
        /// </summary>
        private static Mesh ExtrudePolygon(List<Vector2> polygon, float height)
        {
            // Ensure winding is CCW for correct normals
            if (GetSignedArea(polygon) > 0)
                polygon.Reverse();

            var triangles = EarClipTriangulate(polygon);
            if (triangles == null || triangles.Count < 3) return null;

            int n = polygon.Count;
            int triCount = triangles.Count / 3;

            // Vertex layout:
            // Floor cap: n verts
            // Roof cap: n verts
            // Walls: 4 verts per edge (n edges)
            int vertCount = n * 2 + n * 4;
            var verts = new Vector3[vertCount];
            var norms = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];

            // Floor vertices (Y = 0)
            for (int i = 0; i < n; i++)
            {
                verts[i] = new Vector3(polygon[i].X, 0, polygon[i].Y);
                norms[i] = -Vector3.UnitY;
                uvs[i] = polygon[i] * 0.1f; // Simple UV
            }

            // Roof vertices (Y = height)
            for (int i = 0; i < n; i++)
            {
                verts[n + i] = new Vector3(polygon[i].X, height, polygon[i].Y);
                norms[n + i] = Vector3.UnitY;
                uvs[n + i] = polygon[i] * 0.1f;
            }

            // Wall vertices (4 per edge)
            int wallBase = n * 2;
            for (int i = 0; i < n; i++)
            {
                int next = (i + 1) % n;
                var p0 = polygon[i];
                var p1 = polygon[next];

                // Wall normal (outward in XZ)
                var edge = new Vector2(p1.X - p0.X, p1.Y - p0.Y);
                var normal = Vector3.Normalize(new Vector3(edge.Y, 0, -edge.X));

                int vi = wallBase + i * 4;
                verts[vi + 0] = new Vector3(p0.X, 0, p0.Y);       // bottom-left
                verts[vi + 1] = new Vector3(p1.X, 0, p1.Y);       // bottom-right
                verts[vi + 2] = new Vector3(p0.X, height, p0.Y);  // top-left
                verts[vi + 3] = new Vector3(p1.X, height, p1.Y);  // top-right

                norms[vi + 0] = normal;
                norms[vi + 1] = normal;
                norms[vi + 2] = normal;
                norms[vi + 3] = normal;

                float edgeLen = edge.Length() * 0.2f;
                uvs[vi + 0] = new Vector2(0, 0);
                uvs[vi + 1] = new Vector2(edgeLen, 0);
                uvs[vi + 2] = new Vector2(0, height * 0.2f);
                uvs[vi + 3] = new Vector2(edgeLen, height * 0.2f);
            }

            // Indices
            // Floor: triangulated (reversed winding for bottom face)
            // Roof: triangulated
            // Walls: 2 tris per edge
            int idxCount = triCount * 3 * 2 + n * 6;
            var indices = new uint[idxCount];
            int idx = 0;

            // Floor (facing down: CCW in 2D maps to -Y normal in XZ, so use as-is)
            for (int i = 0; i < triCount; i++)
            {
                indices[idx++] = (uint)triangles[i * 3 + 0];
                indices[idx++] = (uint)triangles[i * 3 + 1];
                indices[idx++] = (uint)triangles[i * 3 + 2];
            }

            // Roof (facing up: reverse winding to flip normal to +Y)
            for (int i = 0; i < triCount; i++)
            {
                indices[idx++] = (uint)(n + triangles[i * 3 + 2]);
                indices[idx++] = (uint)(n + triangles[i * 3 + 1]);
                indices[idx++] = (uint)(n + triangles[i * 3 + 0]);
            }

            // Walls
            for (int i = 0; i < n; i++)
            {
                uint vi = (uint)(wallBase + i * 4);
                indices[idx++] = vi + 0;
                indices[idx++] = vi + 2;
                indices[idx++] = vi + 1;
                indices[idx++] = vi + 1;
                indices[idx++] = vi + 2;
                indices[idx++] = vi + 3;
            }

            var mesh = new Mesh(Engine.Device, verts, norms, uvs, indices);
            mesh.BoundingBox = ComputeBounds(verts);
            mesh.MeshParts.Add(new MeshPart
            {
                NumIndices = indices.Length,
                BoundingBox = mesh.BoundingBox,
                BoundingSphere = mesh.LocalBoundingSphere
            });
            return mesh;
        }

        // ═══════════════════════════════════
        // ── Ear-Clipping Triangulation ──
        // ═══════════════════════════════════

        /// <summary>
        /// Simple ear-clipping triangulation for a simple polygon.
        /// Returns list of triangle indices (into the input polygon).
        /// Assumes CCW winding.
        /// </summary>
        private static List<int> EarClipTriangulate(List<Vector2> polygon)
        {
            var result = new List<int>();
            if (polygon.Count < 3) return result;

            // Build index list
            var indices = new List<int>();
            for (int i = 0; i < polygon.Count; i++)
                indices.Add(i);

            int safety = polygon.Count * 3;
            while (indices.Count > 2 && safety-- > 0)
            {
                bool earFound = false;
                for (int i = 0; i < indices.Count; i++)
                {
                    int prev = indices[(i - 1 + indices.Count) % indices.Count];
                    int curr = indices[i];
                    int next = indices[(i + 1) % indices.Count];

                    var a = polygon[prev];
                    var b = polygon[curr];
                    var c = polygon[next];

                    // Check if this is a convex vertex (CCW: convex has positive cross)
                    float cross = Cross2D(b - a, c - b);
                    if (cross <= 0) continue; // Concave or collinear, skip

                    // Check if any other vertex is inside this triangle
                    bool isEar = true;
                    for (int j = 0; j < indices.Count; j++)
                    {
                        int idx = indices[j];
                        if (idx == prev || idx == curr || idx == next) continue;
                        if (PointInTriangle(polygon[idx], a, b, c))
                        {
                            isEar = false;
                            break;
                        }
                    }

                    if (isEar)
                    {
                        result.Add(prev);
                        result.Add(curr);
                        result.Add(next);
                        indices.RemoveAt(i);
                        earFound = true;
                        break;
                    }
                }

                if (!earFound)
                    break; // Degenerate polygon
            }

            return result;
        }

        // ═══════════════════════════
        // ── Math Helpers ──
        // ═══════════════════════════

        private static float Cross2D(Vector2 a, Vector2 b) => a.X * b.Y - a.Y * b.X;

        private static float GetSignedArea(List<Vector2> poly)
        {
            float area = 0;
            for (int i = 0; i < poly.Count; i++)
            {
                var a = poly[i];
                var b = poly[(i + 1) % poly.Count];
                area += (b.X - a.X) * (b.Y + a.Y);
            }
            return area * 0.5f;
        }

        private static bool PointInTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            float d1 = Cross2D(b - a, p - a);
            float d2 = Cross2D(c - b, p - b);
            float d3 = Cross2D(a - c, p - c);

            bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
            bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

            return !(hasNeg && hasPos);
        }

        private static Vector2 ComputeCentroid(List<Vector2> polygon)
        {
            var sum = Vector2.Zero;
            foreach (var p in polygon) sum += p;
            return sum / polygon.Count;
        }

        private static float ComputePolygonRadius(List<Vector2> centeredPolygon)
        {
            float maxDist = 0;
            foreach (var p in centeredPolygon)
                maxDist = Math.Max(maxDist, p.Length());
            return maxDist;
        }

        private static BoundingBox ComputeBounds(Vector3[] verts)
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var v in verts)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            return new BoundingBox(min, max);
        }

        private static BoundingBox ComputeBounds(List<Vector3> verts)
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var v in verts)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            return new BoundingBox(min, max);
        }
    }
}
