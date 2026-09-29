using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Procedural;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Builds buildings from prefab pieces on a quantized grid.
    /// Takes an arbitrary polygon footprint, rasterizes it to a 2M grid,
    /// decomposes into maximal rectangles, and places modular wall/roof pieces.
    /// </summary>
    public class PrefabBuildingPlacer
    {
        private readonly WatabouSettings _settings;
        private readonly Random _rng;

        // Resolved piece lists (lazily populated)
        private readonly Dictionary<string, Prefab> _prefabCache = new();
        private bool _catalogResolved;

    // Resolved piece lists: store GUIDs (not names) to avoid ambiguity
        // with .staticmesh sub-assets that share the same name.
        private List<string> _stoneWalls2M;
        private List<string> _stoneWalls4M;
        private List<string> _plasterWalls2M;
        private List<string> _plasterWalls4M;
        private List<string> _stoneWindowWalls2M;
        private List<string> _plasterWindowWalls2M;
        private List<string> _stoneDoorWalls2M;
        private List<string> _plasterDoorWalls2M;
        private List<string> _windows2M;
        private List<string> _doors;
        private List<string> _stoneCorners;

        public PrefabBuildingPlacer(WatabouSettings settings, int seed)
        {
            _settings = settings;
            _rng = new Random(seed);
        }

        /// <summary>
        /// Build a prefab-based building from a polygon footprint.
        /// Returns the root entity, or null if the footprint is too small.
        /// </summary>
        public Entity Build(List<Vector2> polygon, float height, int buildingIndex)
        {
            ResolveCatalog();

            if (polygon.Count < 3) return null;

            const float cellSize = 2f;

            // 1. Find longest edge → defines building's local X axis
            float bestLen = 0;
            Vector2 bestDir = Vector2.UnitX;
            for (int i = 0; i < polygon.Count; i++)
            {
                var a = polygon[i];
                var b = polygon[(i + 1) % polygon.Count];
                float len = Vector2.Distance(a, b);
                if (len > bestLen)
                {
                    bestLen = len;
                    bestDir = Vector2.Normalize(b - a);
                }
            }

            // Building orientation angle (world Y rotation)
            float buildingAngle = MathF.Atan2(bestDir.Y, bestDir.X);

            // Local axes
            var localX = bestDir;
            var localZ = new Vector2(-localX.Y, localX.X); // perpendicular

            // 2. Transform polygon to local space
            var centroid = Vector2.Zero;
            foreach (var p in polygon) centroid += p;
            centroid /= polygon.Count;

            var localPoly = new List<Vector2>(polygon.Count);
            foreach (var p in polygon)
            {
                var d = p - centroid;
                localPoly.Add(new Vector2(
                    Vector2.Dot(d, localX),
                    Vector2.Dot(d, localZ)));
            }

            // 3. Rasterize in local space
            var grid = RasterizePolygon(localPoly, cellSize, out var gridOrigin);
            if (grid == null) return null;

            int gridW = grid.GetLength(0);
            int gridH = grid.GetLength(1);

            int occupied = 0;
            for (int x = 0; x < gridW; x++)
                for (int z = 0; z < gridH; z++)
                    if (grid[x, z]) occupied++;

            if (occupied < 2) return null;

            // 4. Derive building properties from footprint size
            const float wallHeight = 3f;

            // Larger footprint → more stories, more likely stone ground floor
            int maxStories;
            bool useStoneGround;
            if (occupied >= 8)
            {
                maxStories = 2 + (_rng.Next(2)); // 2-3
                useStoneGround = _rng.NextDouble() < 0.8;
            }
            else if (occupied >= 4)
            {
                maxStories = _rng.NextDouble() < 0.3 ? 2 : 1; // mostly 1, sometimes 2
                useStoneGround = _rng.NextDouble() < 0.4;
            }
            else
            {
                maxStories = _rng.NextDouble() < 0.1 ? 2 : 1; // almost always 1
                useStoneGround = _rng.NextDouble() < 0.15;
            }

            // Watabou height still caps the maximum
            int stories = Math.Min(maxStories, Math.Max(1, (int)(height / wallHeight)));

            // 5. Create building root at world centroid, rotated to match longest edge
            var root = new Entity($"PrefabBuilding_{buildingIndex}");
            root.Transform.Position = new Vector3(centroid.X, 0, centroid.Y);
            root.Transform.Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, -buildingAngle);

            // 6. Place walls per floor
            for (int floor = 0; floor < stories; floor++)
            {
                bool isStone = useStoneGround && (floor == 0);
                float y = floor * wallHeight;
                bool doorPlaced = false;

                for (int gx = 0; gx < gridW; gx++)
                {
                    for (int gz = 0; gz < gridH; gz++)
                    {
                        if (!grid[gx, gz]) continue;

                        const float R = MathF.PI * 0.5f;

                        // -Z face
                        if (gz == 0 || !grid[gx, gz - 1])
                        {
                            var pos = new Vector3(gridOrigin.X + (gx + 0.5f) * cellSize, y, gridOrigin.Y + gz * cellSize);
                            PlaceWallPiece(root, pos, R, cellSize, isStone, ref doorPlaced);
                        }

                        // +Z face
                        if (gz == gridH - 1 || !grid[gx, gz + 1])
                        {
                            var pos = new Vector3(gridOrigin.X + (gx + 0.5f) * cellSize, y, gridOrigin.Y + (gz + 1) * cellSize);
                            PlaceWallPiece(root, pos, R + MathF.PI, cellSize, isStone, ref doorPlaced);
                        }

                        // -X face
                        if (gx == 0 || !grid[gx - 1, gz])
                        {
                            var pos = new Vector3(gridOrigin.X + gx * cellSize, y, gridOrigin.Y + (gz + 0.5f) * cellSize);
                            PlaceWallPiece(root, pos, R + MathF.PI * 0.5f, cellSize, isStone, ref doorPlaced);
                        }

                        // +X face
                        if (gx == gridW - 1 || !grid[gx + 1, gz])
                        {
                            var pos = new Vector3(gridOrigin.X + (gx + 1) * cellSize, y, gridOrigin.Y + (gz + 0.5f) * cellSize);
                            PlaceWallPiece(root, pos, R - MathF.PI * 0.5f, cellSize, isStone, ref doorPlaced);
                        }
                    }
                }
            }

            return root;
        }

        #region Wall Placement

        // Pre-computed rotation quaternions for each wall face direction.
        // Base rotation from Tower_Single analysis: Q ≈ (0, 0.707, -0.707, 0)
        // which maps the Unity mesh into Freefall's coordinate system.
        // Additional Y rotations are composed to face each cardinal direction.
        private static readonly Quaternion BaseRotation = new Quaternion(0, 0.7071068f, -0.7071068f, 0);

        /// <summary>
        /// Position and orient an instantiated prefab piece at a wall slot.
        /// Single-entity prefabs need the full BaseRotation (coordinate conversion).
        /// Composite prefabs already have it baked into their children — only apply Y rotation.
        /// </summary>
        private void PlaceAndOrient(Entity piece, Entity root, Vector3 position, float rotationY)
        {
            var yRot = Quaternion.CreateFromAxisAngle(Vector3.UnitY, rotationY);
            bool hasChildren = piece.Transform.GetChildCount() > 0;

            piece.Transform.Parent = root.Transform;
            piece.Transform.Position = position;
            piece.Transform.Rotation = hasChildren ? yRot : yRot * BaseRotation;
        }

        private void PlaceWallPiece(Entity root, Vector3 position, float rotationY,
            float cellSize, bool isStone, ref bool doorPlaced)
        {
            string guid;
            var walls = isStone ? _stoneWalls2M : _plasterWalls2M;

            // Door on ground floor (once per building, random face)
            if (!doorPlaced && _rng.NextDouble() < 0.15)
            {
                var doorWalls = isStone ? _stoneDoorWalls2M : _plasterDoorWalls2M;
                guid = Pick(doorWalls);
                if (guid != null)
                {
                    doorPlaced = true;
                    var entity = InstantiatePiece(guid);
                    if (entity != null)
                        PlaceAndOrient(entity, root, position, rotationY);
                    return;
                }
            }

            // Wall with window opening, or solid wall
            var windowWalls = isStone ? _stoneWindowWalls2M : _plasterWindowWalls2M;
            if (_rng.NextDouble() < 0.4 && windowWalls?.Count > 0)
                guid = Pick(windowWalls);
            else
                guid = Pick(walls);

            if (guid != null)
            {
                var entity = InstantiatePiece(guid);
                if (entity != null)
                    PlaceAndOrient(entity, root, position, rotationY);
            }
        }

        #endregion

        #region Polygon Rasterization

        /// <summary>
        /// Rasterize a 2D polygon onto a grid of cellSize × cellSize cells.
        /// Polygon should already be in local space.
        /// Returns the boolean occupancy grid and the local-space origin of cell (0,0).
        /// </summary>
        private bool[,] RasterizePolygon(List<Vector2> polygon, float cellSize, out Vector2 gridOrigin)
        {
            gridOrigin = Vector2.Zero;
            if (polygon.Count < 3) return null;

            // Compute AABB in local space
            var min = new Vector2(float.MaxValue);
            var max = new Vector2(float.MinValue);
            foreach (var p in polygon)
            {
                min = Vector2.Min(min, p);
                max = Vector2.Max(max, p);
            }

            // Snap origin to grid
            float originX = MathF.Floor(min.X / cellSize) * cellSize;
            float originZ = MathF.Floor(min.Y / cellSize) * cellSize;
            gridOrigin = new Vector2(originX, originZ);

            int gridW = (int)MathF.Ceiling((max.X - originX) / cellSize);
            int gridH = (int)MathF.Ceiling((max.Y - originZ) / cellSize);

            if (gridW < 1 || gridH < 1 || gridW > 100 || gridH > 100) return null;

            var grid = new bool[gridW, gridH];

            for (int x = 0; x < gridW; x++)
            {
                for (int z = 0; z < gridH; z++)
                {
                    float cx = originX + (x + 0.5f) * cellSize;
                    float cz = originZ + (z + 0.5f) * cellSize;
                    grid[x, z] = PointInPolygon(new Vector2(cx, cz), polygon);
                }
            }

            return grid;
        }

        /// <summary>Ray-casting point-in-polygon test.</summary>
        private static bool PointInPolygon(Vector2 point, List<Vector2> polygon)
        {
            bool inside = false;
            int n = polygon.Count;
            for (int i = 0, j = n - 1; i < n; j = i++)
            {
                var pi = polygon[i];
                var pj = polygon[j];
                if ((pi.Y > point.Y) != (pj.Y > point.Y) &&
                    point.X < (pj.X - pi.X) * (point.Y - pi.Y) / (pj.Y - pi.Y) + pi.X)
                {
                    inside = !inside;
                }
            }
            return inside;
        }

        #endregion

        #region Piece Catalog

        private void ResolveCatalog()
        {
            if (_catalogResolved) return;

            // Build a lookup: name → GUID, filtered to .prefab sources only
            var prefabGuids = new Dictionary<string, List<string>>();
            foreach (var meta in AssetDatabase.GetAllMeta())
            {
                if (meta.SourcePath == null || !meta.SourcePath.EndsWith(".prefab"))
                    continue;

                var name = AssetDatabase.ResolveFriendlyName(meta.Guid);
                if (name == null) continue;

                if (!prefabGuids.ContainsKey(name))
                    prefabGuids[name] = new List<string>();
                prefabGuids[name].Add(meta.Guid);
            }

            _stoneWalls2M = FindByPrefix(prefabGuids, "SM_WallS_2X3M");
            _stoneWalls4M = FindByPrefix(prefabGuids, "SM_WallS_4X3M");
            _plasterWalls2M = FindByPrefix(prefabGuids, "SM_WallP_2X3M");
            _plasterWalls4M = FindByPrefix(prefabGuids, "SM_WallP_4X3M");

            // _Wall suffix = wall with window opening (e.g. SM_WallS_Window_2X3M_A_Wall)
            // Without _Wall = standalone window insert (NOT a wall piece)
            _stoneWindowWalls2M = FindByPrefix(prefabGuids, "SM_WallS_Window_2X3M_", "_XLarge",  requiredSuffix: "_Wall");
            _plasterWindowWalls2M = FindByPrefix(prefabGuids, "SM_WallP_Window_2X3M_", "_XLarge", requiredSuffix: "_wall");
            _stoneDoorWalls2M = FindByPrefix(prefabGuids, "SM_WallS_Door_2X3M");
            _plasterDoorWalls2M = FindByPrefix(prefabGuids, "SM_WallP_Door_2X3M");

            _windows2M = FindByPrefix(prefabGuids, "SM_Window_2X3M_");
            _doors = FindByPrefix(prefabGuids, "Door_2X2");

            _stoneCorners = FindByPrefix(prefabGuids, "SM_Stone_Corner_3M");

            int total = (_stoneWalls2M?.Count ?? 0) + (_plasterWalls2M?.Count ?? 0) +
                        (_stoneWindowWalls2M?.Count ?? 0) + (_plasterWindowWalls2M?.Count ?? 0) +
                        (_stoneDoorWalls2M?.Count ?? 0) + (_windows2M?.Count ?? 0) + (_doors?.Count ?? 0);

            Debug.Log($"[PrefabBuildingPlacer] Resolved {total} piece GUIDs " +
                $"(walls: {_stoneWalls2M?.Count ?? 0}S/{_plasterWalls2M?.Count ?? 0}P, " +
                $"windows: {_stoneWindowWalls2M?.Count ?? 0}S/{_plasterWindowWalls2M?.Count ?? 0}P, " +
                $"doors: {_stoneDoorWalls2M?.Count ?? 0}, inserts: {_windows2M?.Count ?? 0}W/{_doors?.Count ?? 0}D)");

            _catalogResolved = true;
        }

        /// <summary>
        /// Find all prefab GUIDs whose name starts with the given prefix.
        /// Returns GUIDs (not names) to avoid ambiguity with .staticmesh sub-assets.
        /// </summary>
        private static List<string> FindByPrefix(Dictionary<string, List<string>> prefabGuids, string namePrefix,
            string exclude = null, string requiredSuffix = null)
        {
            var results = new List<string>();
            foreach (var kvp in prefabGuids)
            {
                if (!kvp.Key.StartsWith(namePrefix, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (exclude != null && kvp.Key.Contains(exclude, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (requiredSuffix != null && !kvp.Key.EndsWith(requiredSuffix, StringComparison.OrdinalIgnoreCase))
                    continue;
                results.AddRange(kvp.Value);
            }
            return results;
        }

        private Prefab LoadPrefab(string guid)
        {
            if (_prefabCache.TryGetValue(guid, out var cached))
                return cached;

            var prefab = Engine.Assets.LoadByGuid<Prefab>(guid);
            if (prefab != null)
                _prefabCache[guid] = prefab;

            return prefab;
        }

        private Entity InstantiatePiece(string guid)
        {
            if (guid == null) return null;
            var prefab = LoadPrefab(guid);
            var entity = prefab?.Instantiate();
            if (entity == null) return null;

            // Scale normalization:
            // - Composite prefabs (with children): children have 100x scale for cm→m.
            //   If root ALSO has 100x, it's double-scaled → reset root to 1.
            // - Simple prefabs (no children): mesh is in cm, root scale [1,1,1] → set to 100.
            var s = entity.Transform.Scale;
            bool hasChildren = entity.Transform.GetChildCount() > 0;

            if (hasChildren && (s.X > 50f || s.Y > 50f || s.Z > 50f))
                entity.Transform.Scale = Vector3.One;
            else if (!hasChildren && s.X < 50f && s.Y < 50f && s.Z < 50f)
                entity.Transform.Scale = new Vector3(100f);

            return entity;
        }

        private string Pick(List<string> list)
        {
            if (list == null || list.Count == 0) return null;
            return list[_rng.Next(list.Count)];
        }

        #endregion
    }
}

