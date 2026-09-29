using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Builds a 3D building from Watabou Dwellings data using prefab pieces.
    /// Walks the cell grid per floor, places wall/window/door prefabs on exterior faces,
    /// and generates roofs for cells without a floor above.
    /// </summary>
    public class DwellingsBuilder
    {
        private readonly float _cellSize;
        private readonly float _storyHeight;
        private readonly Random _rng;

        // Prefab cache
        private readonly Dictionary<string, Prefab> _prefabCache = new();
        private bool _catalogResolved;

        // Wall pieces (GUIDs)
        private List<string> _stoneWalls2M;
        private List<string> _stoneWalls1M;
        private List<string> _plasterWalls2M;
        private List<string> _plasterWalls1M;
        private List<string> _stoneWindowWalls;
        private List<string> _plasterWindowWalls;
        private List<string> _stoneDoorWalls;
        private List<string> _plasterDoorWalls;
        private List<string> _stoneCorners;

        // Roof pieces
        private List<string> _roofSlopesS_1M;
        private List<string> _roofSlopesS_2M;
        private List<string> _roofSlopesS_3M;
        private List<string> _roofSlopesM_1M;
        private List<string> _roofSlopesM_2M;
        private List<string> _roofSlopesM_3M;
        private List<string> _roofSlopesLM_1M;
        private List<string> _roofSlopesLM_2M;
        private List<string> _roofSlopesLM_3M;
        private List<string> _roofSlopesLM_3M_cutoff;
        private List<string> _roofRidges1M;
        private List<string> _roofRidges2M;
        private List<string> _roofRidges3M;
        private List<string> _roofRidges6M;
        private List<string> _roofRidges10M;
        private List<string> _roofCorners;
        private List<string> _roofHipTri1M;
        private List<string> _roofHipTri2M;
        private List<string> _gableTriStone;
        private List<string> _gableTriPlaster;

        // Accessories
        private List<string> _chimneys;
        private List<string> _stairs;

        // BaseRotation from PrefabBuildingPlacer — handles cm→m mesh coordinate conversion.
        private static readonly Quaternion BaseRotation = new(0, 0.7071068f, -0.7071068f, 0);

        // Roof slope rotations extracted from existing house prefabs.
        // Ridge along X: from House02 (slopes extend along Z from ridge)
        private static readonly Quaternion RoofSlopeA_RidgeX = new(0, 0.7071068f, -0.7071068f, 0);
        private static readonly Quaternion RoofSlopeB_RidgeX = new(0.7071068f, 0, 0, 0.7071068f);
        // Ridge along Z: from House01 (slopes extend along X from ridge)
        private static readonly Quaternion RoofSlopeA_RidgeZ = new(0.5f, 0.5f, -0.5f, 0.5f);
        private static readonly Quaternion RoofSlopeB_RidgeZ = new(-0.5f, 0.5f, -0.5f, -0.5f);

        // Ridge cap rotation (from House02)
        private static readonly Quaternion RidgeCapRot_RidgeX = new(0, 0.7071068f, -0.7071068f, 0);
        private static readonly Quaternion RidgeCapRot_RidgeZ = new(0.5f, 0.5f, -0.5f, 0.5f);

        // LM roof rise: measured from House02 (wall top 6M, ridge 10.55M = 4.55M rise)
        private const float LM_RIDGE_RISE = 4.55f;
        // Slope piece width along the ridge (3M for LM_3M)
        private const float SLOPE_TILE_WIDTH = 3f;

        public DwellingsBuilder(float cellSize = 2f, float storyHeight = 3f, int seed = 42)
        {
            _cellSize = cellSize;
            _storyHeight = storyHeight;
            _rng = new Random(seed);
        }

        /// <summary>
        /// Build a complete building entity from Dwellings data.
        /// </summary>
        public Entity Build(DwellingsData data, string name)
        {
            ResolveCatalog();

            var root = new Entity(name);
            int floorCount = data.Floors.Count;
            if (floorCount == 0) return root;

            // ── 1. Build per-floor occupancy and room maps ──
            var floorCells = new List<HashSet<CellCoord>>(floorCount);
            var floorRoomMap = new List<Dictionary<CellCoord, string>>(floorCount);

            int minI = int.MaxValue, maxI = int.MinValue;
            int minJ = int.MaxValue, maxJ = int.MinValue;

            foreach (var floor in data.Floors)
            {
                var cells = new HashSet<CellCoord>();
                var roomMap = new Dictionary<CellCoord, string>();

                foreach (var room in floor.Rooms)
                {
                    foreach (var cell in room.Cells)
                    {
                        cells.Add(cell);
                        roomMap[cell] = room.Name ?? "Room";
                        minI = Math.Min(minI, cell.I);
                        maxI = Math.Max(maxI, cell.I);
                        minJ = Math.Min(minJ, cell.J);
                        maxJ = Math.Max(maxJ, cell.J);
                    }
                }

                floorCells.Add(cells);
                floorRoomMap.Add(roomMap);
            }

            // Centering offset so building origin is at its center
            float centerX = (minJ + maxJ + 1) * 0.5f * _cellSize;
            float centerZ = (minI + maxI + 1) * 0.5f * _cellSize;

            // ── 2. Process each floor ──
            for (int f = 0; f < floorCount; f++)
            {
                var floor = data.Floors[f];
                var cells = floorCells[f];
                float floorY = floor.Level * _storyHeight;
                bool isGroundFloor = floor.Level == 0;

                // Build edge lookups
                var windowEdges = new HashSet<(CellCoord cell, string dir)>();
                foreach (var w in floor.Windows)
                    windowEdges.Add((w.Cell, w.Dir));

                var doorEdges = new HashSet<(CellCoord cell, string dir)>();
                foreach (var d in floor.Doors)
                    if (d.Edge != null)
                        doorEdges.Add((d.Edge.Cell, d.Edge.Dir));

                var floorEntity = new Entity($"Floor_{floor.Level}");
                floorEntity.Transform.Parent = root.Transform;

                // ── Place walls ──
                foreach (var cell in cells)
                {
                    foreach (var dir in new[] { "n", "s", "e", "w" })
                    {
                        var neighbor = cell.Neighbor(dir);

                        if (cells.Contains(neighbor))
                        {
                            // Interior face — only place wall if there's a door between rooms
                            // To avoid double-placing, only place from the cell with lower hash
                            if (doorEdges.Contains((cell, dir)) && cell.GetHashCode() < neighbor.GetHashCode())
                            {
                                var doorGuid = Pick(isGroundFloor ? _stoneDoorWalls : _plasterDoorWalls);
                                if (doorGuid != null)
                                {
                                    var pos = GetWallPosition(cell, dir, floorY, centerX, centerZ);
                                    PlacePiece(floorEntity, pos, GetWallRotation(dir), doorGuid);
                                }
                            }
                            continue;
                        }

                        // Exterior face — determine piece type from JSON annotations
                        bool isExit = isGroundFloor &&
                                      data.Exit != null &&
                                      data.Exit.Cell.Equals(cell) &&
                                      data.Exit.Dir == dir;
                        bool isWindow = windowEdges.Contains((cell, dir));
                        bool useStone = isGroundFloor;

                        string guid;
                        if (isExit)
                            guid = Pick(useStone ? _stoneDoorWalls : _plasterDoorWalls);
                        else if (isWindow)
                            guid = Pick(useStone ? _stoneWindowWalls : _plasterWindowWalls);
                        else
                            guid = Pick(useStone ? _stoneWalls2M : _plasterWalls2M);

                        if (guid != null)
                        {
                            var pos = GetWallPosition(cell, dir, floorY, centerX, centerZ);
                            PlacePiece(floorEntity, pos, GetWallRotation(dir), guid);
                        }
                    }
                }

                // ── Place corners at convex exterior vertices ──
                PlaceCorners(floorEntity, cells, floorY, isGroundFloor, centerX, centerZ);
            }

            // ── 3. Place roofs ──
            PlaceRoofs(root, data.Floors, floorCells, centerX, centerZ);

            MessageDispatcher.Send(Msg.RefreshExplorer);
            return root;
        }

        #region Wall Positioning

        /// <summary>
        /// Get the world position for a wall piece on a given cell face.
        /// Position is at the face center, bottom of the wall.
        /// </summary>
        private Vector3 GetWallPosition(CellCoord cell, string dir, float floorY, float cx, float cz)
        {
            float cellX = cx - (cell.J + 1) * _cellSize; // Negate J for correct mirror
            float cellZ = cell.I * _cellSize - cz;

            return dir switch
            {
                "n" => new Vector3(cellX + _cellSize * 0.5f, floorY, cellZ),
                "s" => new Vector3(cellX + _cellSize * 0.5f, floorY, cellZ + _cellSize),
                "w" => new Vector3(cellX + _cellSize, floorY, cellZ + _cellSize * 0.5f),
                "e" => new Vector3(cellX, floorY, cellZ + _cellSize * 0.5f),
                _ => Vector3.Zero
            };
        }

        /// <summary>
        /// Get the Y rotation for a wall piece facing a given direction.
        /// Based on the rotation conventions from PrefabBuildingPlacer.
        /// </summary>
        private static float GetWallRotation(string dir)
        {
            const float R = MathF.PI * 0.5f;
            return dir switch
            {
                "n" => R,              // face -Z
                "s" => R + MathF.PI,   // face +Z
                "w" => 0f,             // face +X (swapped)
                "e" => R + R,          // face -X (swapped)
                _ => 0f
            };
        }

        /// <summary>
        /// Instantiate and position a wall/door/corner prefab piece.
        /// Uses the same orientation logic as PrefabBuildingPlacer.
        /// </summary>
        private void PlacePiece(Entity parent, Vector3 position, float rotationY, string guid, bool isCorner = false)
        {
            var entity = InstantiatePiece(guid);
            if (entity == null) return;

            var yRot = Quaternion.CreateFromAxisAngle(Vector3.UnitY, rotationY);
            bool hasChildren = entity.Transform.GetChildCount() > 0;

            entity.Transform.Parent = parent.Transform;
            entity.Transform.Position = position;
            entity.Transform.Rotation = hasChildren ? yRot : yRot * (isCorner ? Quaternion.Identity : BaseRotation);
        }

        /// <summary>
        /// Place a roof piece with a direct rotation quaternion.
        /// Roof pieces use their own orientation conventions, not wall BaseRotation.
        /// </summary>
        private void PlaceRoofDirect(Entity parent, Vector3 position, Quaternion rotation, string guid)
        {
            var entity = InstantiatePiece(guid);
            if (entity == null) return;

            entity.Transform.Parent = parent.Transform;
            entity.Transform.Position = position;
            entity.Transform.Rotation = rotation;
        }

        #endregion

        #region Corners

        /// <summary>
        /// Place corner pieces at convex exterior vertices of the floor plan.
        /// A convex corner exists where two adjacent exterior wall faces meet.
        /// </summary>
        private void PlaceCorners(Entity parent, HashSet<CellCoord> cells, float floorY,
            bool isGroundFloor, float cx, float cz)
        {
            if (_stoneCorners == null || _stoneCorners.Count == 0) return;
            if (!isGroundFloor) return; // Corners only on stone ground floor

            foreach (var cell in cells)
            {
                // Check all 4 corner vertices of this cell
                // NW corner: exterior on both N and W
                CheckCorner(parent, cells, cell, "n", "w", floorY, cx, cz);
                // NE corner: exterior on both N and E
                CheckCorner(parent, cells, cell, "n", "e", floorY, cx, cz);
                // SW corner: exterior on both S and W
                CheckCorner(parent, cells, cell, "s", "w", floorY, cx, cz);
                // SE corner: exterior on both S and E
                CheckCorner(parent, cells, cell, "s", "e", floorY, cx, cz);
            }
        }

        private void CheckCorner(Entity parent, HashSet<CellCoord> cells, CellCoord cell,
            string dirA, string dirB, float floorY, float cx, float cz)
        {
            var neighborA = cell.Neighbor(dirA);
            var neighborB = cell.Neighbor(dirB);

            // Both faces must be exterior (neighbor not occupied)
            if (cells.Contains(neighborA) || cells.Contains(neighborB)) return;

            // The diagonal neighbor must also be empty for a true convex corner
            var diagonal = neighborA.Neighbor(dirB);
            if (cells.Contains(diagonal)) return;

            float cellX = cx - (cell.J + 1) * _cellSize;
            float cellZ = cell.I * _cellSize - cz;

            // Corner vertex position (e/w swapped for mirrored X)
            float vx = cellX + (dirB == "w" ? _cellSize : 0);
            float vz = cellZ + (dirA == "s" ? _cellSize : 0);

            // Corner rotation based on which quadrant (e/w swapped)
            float rotY = (dirA, dirB) switch
            {
                ("n", "w") => MathF.PI,
                ("n", "e") => -MathF.PI * 0.5f,
                ("s", "w") => MathF.PI * 0.5f,
                ("s", "e") => 0f,
                _ => 0f
            };

            var guid = Pick(_stoneCorners);
            if (guid != null)
                PlacePiece(parent, new Vector3(vx, floorY, vz), rotY, guid, isCorner: true);
        }

        #endregion

        #region Roofs

        /// <summary>
        /// Generate procedural hip roof meshes using the straight skeleton algorithm.
        /// For rectilinear polygons, the straight skeleton height at each point equals
        /// the minimum perpendicular distance to any boundary edge — computed here by
        /// walking N/S/E/W from each vertex to find the nearest exterior edge.
        /// </summary>
        private void PlaceRoofs(Entity root, List<DwellingsFloor> floors,
            List<HashSet<CellCoord>> floorCells, float cx, float cz)
        {
            var roofsEntity = new Entity("Roofs");
            roofsEntity.Transform.Parent = root.Transform;

            for (int f = 0; f < floors.Count; f++)
            {
                var cells = floorCells[f];
                HashSet<CellCoord> cellsAbove = (f + 1 < floorCells.Count) ? floorCells[f + 1] : null;

                var roofCells = new HashSet<CellCoord>();
                foreach (var cell in cells)
                {
                    if (cellsAbove == null || !cellsAbove.Contains(cell))
                        roofCells.Add(cell);
                }

                if (roofCells.Count == 0) continue;

                float roofY = (floors[f].Level + 1) * _storyHeight;

                var regions = FloodFillRegions(roofCells);

                int regionIdx = 0;
                foreach (var region in regions)
                    GenerateStraightSkeletonRoof(roofsEntity, region, roofY, cx, cz, regionIdx++);
            }
        }

        /// <summary>
        /// Generate A-frame roof mesh for an arbitrary rectilinear region.
        /// 
        /// Algorithm:
        /// 1. Find all maximal rectangles in the cell grid
        /// 2. Each rectangle gets an A-frame: ridge along long axis, slopes perpendicular
        /// 3. Final height at each vertex = MIN of all overlapping rectangle contributions
        /// This produces intersecting A-frame roofs with valley lines at junctions.
        /// </summary>
        private void GenerateStraightSkeletonRoof(Entity parent, HashSet<CellCoord> region,
            float roofY, float cx, float cz, int index)
        {
            const float pitchSlope = 0.7f; // Rise per meter of horizontal run (~35° pitch)
            const float overhang = 0.5f;   // Eave overhang in meters

            // Bounding box of the region
            int rMinI = int.MaxValue, rMaxI = int.MinValue;
            int rMinJ = int.MaxValue, rMaxJ = int.MinValue;
            foreach (var c in region)
            {
                rMinI = Math.Min(rMinI, c.I);
                rMaxI = Math.Max(rMaxI, c.I);
                rMinJ = Math.Min(rMinJ, c.J);
                rMaxJ = Math.Max(rMaxJ, c.J);
            }

            // Build 2D grid for maximal rectangle search
            int gridRows = rMaxI - rMinI + 1;
            int gridCols = rMaxJ - rMinJ + 1;
            bool[,] grid = new bool[gridRows, gridCols];
            foreach (var c in region)
                grid[c.I - rMinI, c.J - rMinJ] = true;

            // Find minimum covering set of large A-frame rectangles
            var coverRects = MinimumCoveringRectangles(grid, gridRows, gridCols);

            Debug.Log($"[Roof_{index}] {coverRects.Count} cover rects");
            foreach (var r in coverRects)
                Debug.Log($"  rect: i=[{r.i1},{r.i2}] j=[{r.j1},{r.j2}] ({r.i2-r.i1+1}×{r.j2-r.j1+1})");

            // Build A-frame info for each covering rectangle
            var aframes = new List<(int ri1, int ri2, int rj1, int rj2, bool ridgeAlongI)>();
            foreach (var rect in coverRects)
            {
                int ri1 = rect.i1 + rMinI, ri2 = rect.i2 + rMinI;
                int rj1 = rect.j1 + rMinJ, rj2 = rect.j2 + rMinJ;
                int spanI = ri2 - ri1 + 1, spanJ = rj2 - rj1 + 1;
                bool ridgeAlongI = spanI >= spanJ;
                aframes.Add((ri1, ri2, rj1, rj2, ridgeAlongI));
            }

            // Generate and clip polygons
            var finalPolys = new List<List<Vector3>>();

            for (int a = 0; a < aframes.Count; a++)
            {
                var af = aframes[a];
                var polys = BuildAFramePolygons(af.ri1, af.ri2, af.rj1, af.rj2,
                    af.ridgeAlongI, roofY, cx, cz, pitchSlope, overhang);

                foreach (var poly in polys)
                {
                    var fragments = new List<List<Vector3>> { poly };

                    // Clip against every other A-frame
                    for (int b = 0; b < aframes.Count; b++)
                    {
                        if (b == a) continue;
                        var bf = aframes[b];
                        var next = new List<List<Vector3>>();
                        foreach (var frag in fragments)
                        {
                            var results = ClipPolygonAgainstAFrame(frag,
                            af.ri1, af.ri2, af.rj1, af.rj2, af.ridgeAlongI,
                            bf.ri1, bf.ri2, bf.rj1, bf.rj2, bf.ridgeAlongI,
                            roofY, pitchSlope, cx, cz, overhang, preferB: a > b);
                            next.AddRange(results);
                        }
                        fragments = next;
                        if (fragments.Count == 0) break;
                    }

                    finalPolys.AddRange(fragments);
                }
            }

            // Triangulate all polygons into a single mesh
            var verts = new List<Vector3>();
            var norms = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();

            foreach (var poly in finalPolys)
            {
                if (poly.Count < 3) continue;

                // Compute face normal from first triangle
                var faceNormal = Vector3.Cross(poly[1] - poly[0], poly[2] - poly[0]);
                float len = faceNormal.Length();
                faceNormal = len > 0.0001f ? -faceNormal / len : Vector3.UnitY;

                // Fan triangulation (polygons are convex after clipping)
                uint baseIdx = (uint)verts.Count;
                for (int i = 0; i < poly.Count; i++)
                {
                    verts.Add(poly[i]);
                    norms.Add(faceNormal);
                    uvs.Add(new Vector2(poly[i].X * 0.25f, poly[i].Z * 0.25f));
                }
                for (int i = 1; i < poly.Count - 1; i++)
                {
                    indices.Add(baseIdx);
                    indices.Add(baseIdx + (uint)i + 1);
                    indices.Add(baseIdx + (uint)i);
                }
            }

            // Smooth normals at shared vertices (valley lines between A-frames)
            SmoothSharedNormals(verts, norms);

            if (verts.Count == 0) return;

            // Create mesh
            var mesh = new Graphics.Mesh(Engine.Device,
                verts.ToArray(), norms.ToArray(), uvs.ToArray(), indices.ToArray());
            mesh.BoundingBox = new Vortice.Mathematics.BoundingBox(
                new Vector3(verts.Min(v => v.X), verts.Min(v => v.Y), verts.Min(v => v.Z)),
                new Vector3(verts.Max(v => v.X), verts.Max(v => v.Y), verts.Max(v => v.Z)));
            mesh.MeshParts.Add(new Graphics.MeshPart
            {
                NumIndices = indices.Count,
                BoundingBox = mesh.BoundingBox,
                BoundingSphere = mesh.LocalBoundingSphere
            });

            var roofEntity = new Entity($"Roof_{index}");
            roofEntity.Transform.Parent = parent.Transform;

            var renderer = roofEntity.AddComponent<MeshRenderer>();
            renderer.Mesh = mesh;
            renderer.Material = InternalAssets.DefaultMaterial;

            Debug.Log($"[Roof_{index}] A-frame roof: {verts.Count} verts, {indices.Count / 3} tris, {finalPolys.Count} faces");
        }

        /// <summary>
        /// Average normals at vertices that share the same position (within epsilon).
        /// Only averages normals within the angle threshold to preserve hard creases
        /// (e.g. gable edges) while smoothing valley-line seams between slopes.
        /// </summary>
        private static void SmoothSharedNormals(List<Vector3> verts, List<Vector3> norms,
            float cosAngleThreshold = 0.707f) // ~45°
        {
            const float gridSize = 0.01f;
            const float posEpsSq = 0.0001f;
            int n = verts.Count;

            // Bucket vertices by quantized position
            var groups = new Dictionary<(int, int, int), List<int>>();
            for (int i = 0; i < n; i++)
            {
                var key = (
                    (int)MathF.Round(verts[i].X / gridSize),
                    (int)MathF.Round(verts[i].Y / gridSize),
                    (int)MathF.Round(verts[i].Z / gridSize)
                );
                if (!groups.TryGetValue(key, out var list))
                {
                    list = new List<int>();
                    groups[key] = list;
                }
                list.Add(i);
            }

            foreach (var group in groups.Values)
            {
                if (group.Count <= 1) continue;

                for (int i = 0; i < group.Count; i++)
                {
                    int vi = group[i];
                    var sum = norms[vi];
                    int count = 1;

                    for (int j = 0; j < group.Count; j++)
                    {
                        if (j == i) continue;
                        int vj = group[j];
                        if ((verts[vi] - verts[vj]).LengthSquared() > posEpsSq) continue;
                        if (Vector3.Dot(norms[vi], norms[vj]) < cosAngleThreshold) continue;
                        sum += norms[vj];
                        count++;
                    }

                    if (count > 1)
                    {
                        float len = sum.Length();
                        if (len > 0.0001f) norms[vi] = sum / len;
                    }
                }
            }
        }

        /// <summary>
        /// Compute the A-frame surface height at a cell-space position (posI, posJ)
        /// for a given rectangle. Returns height in meters above roofY.
        /// Returns -∞ if outside the rectangle footprint (B has no surface there, so it never clips A).
        /// </summary>
        private float AFrameHeightAt(int ri1, int ri2, int rj1, int rj2,
            bool ridgeAlongI, float posI, float posJ, float pitchSlope, float overhang = 0f)
        {
            // Outside footprint (including overhang zone) → B has no surface here, never clips
            float ohCells = overhang / _cellSize;
            if (posI < ri1 - ohCells || posI > ri2 + 1 + ohCells ||
                posJ < rj1 - ohCells || posJ > rj2 + 1 + ohCells)
                return float.NegativeInfinity;

            // Height = perpendicular distance to nearest edge × pitch.
            // Negative in the overhang zone (below roofY), positive inside cells.
            if (ridgeAlongI)
            {
                float distW = (posJ - rj1) * _cellSize;
                float distE = (rj2 + 1 - posJ) * _cellSize;
                return MathF.Min(distW, distE) * pitchSlope;
            }
            else
            {
                float distN = (posI - ri1) * _cellSize;
                float distS = (ri2 + 1 - posI) * _cellSize;
                return MathF.Min(distN, distS) * pitchSlope;
            }
        }

        /// <summary>
        /// Build the 4 polygons (2 slopes + 2 gables) for an A-frame roof prism.
        /// Returns world-space polygons with CCW winding (outward-facing).
        /// </summary>
        private List<List<Vector3>> BuildAFramePolygons(int ri1, int ri2, int rj1, int rj2,
            bool ridgeAlongI, float roofY, float cx, float cz, float pitchSlope, float overhang)
        {
            var polys = new List<List<Vector3>>();
            float cs = _cellSize;
            float oh = overhang;

            Vector3 CellToWorld(float posI, float posJ, float h)
            {
                return new Vector3(cx - posJ * cs, roofY + h, posI * cs - cz);
            }

            if (ridgeAlongI)
            {
                // Ridge runs along I, slopes face east and west
                float centerJ = (rj1 + rj2 + 1) * 0.5f;
                float halfW = (rj2 + 1 - rj1) * cs * 0.5f;
                float ridgeH = halfW * pitchSlope;
                float eaveH = -oh * pitchSlope; // eave drops below roofY

                // 8 key points (with overhang)
                float oI1 = ri1 - oh / cs; // overhang north
                float oI2 = ri2 + 1 + oh / cs; // overhang south
                float oJ1 = rj1 - oh / cs; // overhang west
                float oJ2 = rj2 + 1 + oh / cs; // overhang east

                // West slope (low J side): from west eave to ridge
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI1, oJ1, eaveH),
                    CellToWorld(oI2, oJ1, eaveH),
                    CellToWorld(oI2, centerJ, ridgeH),
                    CellToWorld(oI1, centerJ, ridgeH),
                });

                // East slope (high J side): from ridge to east eave
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI2, oJ2, eaveH),
                    CellToWorld(oI1, oJ2, eaveH),
                    CellToWorld(oI1, centerJ, ridgeH),
                    CellToWorld(oI2, centerJ, ridgeH),
                });

                // North gable (i = oI1): triangle
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI1, oJ2, eaveH),
                    CellToWorld(oI1, oJ1, eaveH),
                    CellToWorld(oI1, centerJ, ridgeH),
                });

                // South gable (i = oI2): triangle
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI2, oJ1, eaveH),
                    CellToWorld(oI2, oJ2, eaveH),
                    CellToWorld(oI2, centerJ, ridgeH),
                });
            }
            else
            {
                // Ridge runs along J, slopes face north and south
                float centerI = (ri1 + ri2 + 1) * 0.5f;
                float halfW = (ri2 + 1 - ri1) * cs * 0.5f;
                float ridgeH = halfW * pitchSlope;
                float eaveH = -oh * pitchSlope;

                float oI1 = ri1 - oh / cs;
                float oI2 = ri2 + 1 + oh / cs;
                float oJ1 = rj1 - oh / cs;
                float oJ2 = rj2 + 1 + oh / cs;

                // North slope (low I side): from north eave to ridge
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI1, oJ2, eaveH),
                    CellToWorld(oI1, oJ1, eaveH),
                    CellToWorld(centerI, oJ1, ridgeH),
                    CellToWorld(centerI, oJ2, ridgeH),
                });

                // South slope (high I side): from ridge to south eave
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI2, oJ1, eaveH),
                    CellToWorld(oI2, oJ2, eaveH),
                    CellToWorld(centerI, oJ2, ridgeH),
                    CellToWorld(centerI, oJ1, ridgeH),
                });

                // West gable (j = oJ1): triangle
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI1, oJ1, eaveH),
                    CellToWorld(oI2, oJ1, eaveH),
                    CellToWorld(centerI, oJ1, ridgeH),
                });

                // East gable (j = oJ2): triangle
                polys.Add(new List<Vector3>
                {
                    CellToWorld(oI2, oJ2, eaveH),
                    CellToWorld(oI1, oJ2, eaveH),
                    CellToWorld(centerI, oJ2, ridgeH),
                });
            }

            return polys;
        }
        /// <summary>
        /// Clip a polygon from A-frame A against A-frame B.
        /// Returns multiple fragments: parts of A that are visible
        /// (outside B's footprint, or inside B's footprint where A >= B).
        /// </summary>
        private List<List<Vector3>> ClipPolygonAgainstAFrame(List<Vector3> polygon,
            int ari1, int ari2, int arj1, int arj2, bool aRidgeAlongI,
            int bri1, int bri2, int brj1, int brj2, bool bRidgeAlongI,
            float roofY, float pitchSlope, float cx, float cz, float overhang = 0f, bool preferB = false)
        {
            float cs = _cellSize;
            var results = new List<List<Vector3>>();

            // B's footprint boundaries in world space (extended by overhang)
            float oh = overhang;
            float zLo = bri1 * cs - cz - oh;
            float zHi = (bri2 + 1) * cs - cz + oh;
            float xHi = cx - brj1 * cs + oh;       // low J → high X
            float xLo = cx - (brj2 + 1) * cs - oh; // high J → low X

            // Step 1: Clip polygon to B's footprint rectangle → P_inside
            var pInside = polygon;
            pInside = ClipToHalfPlane(pInside, 0, xLo, true);  // X >= xLo
            pInside = ClipToHalfPlane(pInside, 0, xHi, false); // X <= xHi
            pInside = ClipToHalfPlane(pInside, 2, zLo, true);  // Z >= zLo
            pInside = ClipToHalfPlane(pInside, 2, zHi, false); // Z <= zHi

            if (pInside == null || pInside.Count < 3)
            {
                // Polygon is entirely outside B → return unchanged
                results.Add(polygon);
                return results;
            }

            // Step 2: Clip P_inside by height (keep where A's surface >= B's surface)
            void WorldToCell(Vector3 p, out float posI, out float posJ)
            {
                posJ = (cx - p.X) / cs;
                posI = (p.Z + cz) / cs;
            }

            // Subdivide P_inside edges at B's ridge line for linear sign within each segment
            if (bRidgeAlongI)
            {
                float centerJ = (brj1 + brj2 + 1) * 0.5f;
                float ridgeX = cx - centerJ * cs;
                pInside = SplitEdgesAtValue(pInside, 0, ridgeX);
            }
            else
            {
                float centerI = (bri1 + bri2 + 1) * 0.5f;
                float ridgeZ = centerI * cs - cz;
                pInside = SplitEdgesAtValue(pInside, 2, ridgeZ);
            }

            // When preferB is true, use strict comparison (hA > hB) so that at equal
            // heights only the lower-indexed A-frame keeps its face. Prevents z-fighting.
            float heightBias = preferB ? -0.01f : 0f;
            var pInsideVis = ClipBySign(pInside, v =>
            {
                WorldToCell(v, out float posI, out float posJ);
                float hA = v.Y - roofY;
                float hB = AFrameHeightAt(bri1, bri2, brj1, brj2, bRidgeAlongI,
                    posI, posJ, pitchSlope, overhang);
                if (float.IsNegativeInfinity(hB)) return 1f;
                return hA - hB + heightBias;
            });

            if (pInsideVis != null && pInsideVis.Count >= 3)
                results.Add(pInsideVis);

            // Step 3: Collect parts of polygon OUTSIDE B's footprint (up to 4 fragments)
            // Left strip: X < xLo
            var left = ClipToHalfPlane(polygon, 0, xLo, false);
            if (left != null && left.Count >= 3) results.Add(left);

            // Right strip: X > xHi
            var right = ClipToHalfPlane(polygon, 0, xHi, true);
            if (right != null && right.Count >= 3) results.Add(right);

            // Top strip: Z > zHi, but only within xLo..xHi (avoid double-counting corners)
            var topBand = ClipToHalfPlane(polygon, 0, xLo, true);
            topBand = ClipToHalfPlane(topBand, 0, xHi, false);
            topBand = ClipToHalfPlane(topBand, 2, zHi, true);
            if (topBand != null && topBand.Count >= 3) results.Add(topBand);

            // Bottom strip: Z < zLo, within xLo..xHi
            var botBand = ClipToHalfPlane(polygon, 0, xLo, true);
            botBand = ClipToHalfPlane(botBand, 0, xHi, false);
            botBand = ClipToHalfPlane(botBand, 2, zLo, false);
            if (botBand != null && botBand.Count >= 3) results.Add(botBand);

            return results;
        }

        /// <summary>
        /// Sutherland-Hodgman clip against a single axis-aligned half-plane.
        /// keepGreater=true: keep vertices where axis >= value.
        /// keepGreater=false: keep vertices where axis <= value.
        /// </summary>
        private List<Vector3> ClipToHalfPlane(List<Vector3> poly, int axis, float value, bool keepGreater)
        {
            if (poly == null || poly.Count < 3) return null;

            var output = new List<Vector3>();
            for (int i = 0; i < poly.Count; i++)
            {
                var cur = poly[i];
                var next = poly[(i + 1) % poly.Count];
                float cVal = axis == 0 ? cur.X : (axis == 1 ? cur.Y : cur.Z);
                float nVal = axis == 0 ? next.X : (axis == 1 ? next.Y : next.Z);

                bool cIn = keepGreater ? cVal >= value : cVal <= value;
                bool nIn = keepGreater ? nVal >= value : nVal <= value;

                if (cIn)
                {
                    output.Add(cur);
                    if (!nIn)
                    {
                        float t = (value - cVal) / (nVal - cVal);
                        output.Add(Vector3.Lerp(cur, next, t));
                    }
                }
                else if (nIn)
                {
                    float t = (value - cVal) / (nVal - cVal);
                    output.Add(Vector3.Lerp(cur, next, t));
                }
            }

            return output.Count >= 3 ? output : null;
        }

        /// <summary>
        /// Insert additional vertices wherever a polygon edge crosses an axis-aligned value.
        /// Does not remove any vertices — only adds split points.
        /// </summary>
        private List<Vector3> SplitEdgesAtValue(List<Vector3> poly, int axis, float value)
        {
            if (poly == null || poly.Count < 3) return poly;
            var result = new List<Vector3>(poly.Count + 4);

            for (int i = 0; i < poly.Count; i++)
            {
                var cur = poly[i];
                var next = poly[(i + 1) % poly.Count];
                result.Add(cur);

                float cVal = axis == 0 ? cur.X : (axis == 1 ? cur.Y : cur.Z);
                float nVal = axis == 0 ? next.X : (axis == 1 ? next.Y : next.Z);

                if ((cVal < value) != (nVal < value))
                {
                    float t = (value - cVal) / (nVal - cVal);
                    if (t > 0.001f && t < 0.999f)
                        result.Add(Vector3.Lerp(cur, next, t));
                }
            }
            return result;
        }

        /// <summary>
        /// Sutherland-Hodgman clip using a custom sign function.
        /// Keeps vertices where signFunc(v) >= 0.
        /// </summary>
        private List<Vector3> ClipBySign(List<Vector3> poly, Func<Vector3, float> signFunc)
        {
            if (poly == null || poly.Count < 3) return null;

            var output = new List<Vector3>();
            for (int i = 0; i < poly.Count; i++)
            {
                var cur = poly[i];
                var next = poly[(i + 1) % poly.Count];
                float sCur = signFunc(cur);
                float sNext = signFunc(next);

                if (sCur >= 0)
                {
                    output.Add(cur);
                    if (sNext < 0)
                    {
                        float t = sCur / (sCur - sNext);
                        output.Add(Vector3.Lerp(cur, next, t));
                    }
                }
                else if (sNext >= 0)
                {
                    float t = sCur / (sCur - sNext);
                    output.Add(Vector3.Lerp(cur, next, t));
                }
            }
            return output.Count >= 3 ? output : null;
        }

        /// <summary>
        /// Distance from a continuous position to the nearest boundary edge
        /// walking along the I axis (Z world axis). Returns distance in cell units.
        /// Positive = inside the region. Negative = outside.
        /// </summary>
        private float DistToBoundaryI(HashSet<CellCoord> region, float posI, float posJ, int dir)
        {
            int cellJ = (int)MathF.Floor(posJ);

            // Determine the first cell to check and the fractional distance to its far edge
            int startCellI;
            float firstDist;

            if (dir > 0) // walking south (increasing I)
            {
                float frac = posI - MathF.Floor(posI);
                startCellI = (int)MathF.Floor(posI);
                firstDist = (frac == 0f) ? 1.0f : (MathF.Ceiling(posI) - posI);
            }
            else // walking north (decreasing I)
            {
                float frac = posI - MathF.Floor(posI);
                if (frac == 0f)
                {
                    startCellI = (int)posI - 1;
                    firstDist = 1.0f;
                }
                else
                {
                    startCellI = (int)MathF.Floor(posI);
                    firstDist = frac;
                }
            }

            if (!region.Contains(new CellCoord(startCellI, cellJ)))
                return -firstDist;

            float dist = firstDist;
            int ci = startCellI + dir;
            while (region.Contains(new CellCoord(ci, cellJ)))
            {
                dist += 1.0f;
                ci += dir;
            }
            return dist;
        }

        /// <summary>
        /// Distance from a continuous position to the nearest boundary edge
        /// walking along the J axis (X world axis). Returns distance in cell units.
        /// </summary>
        private float DistToBoundaryJ(HashSet<CellCoord> region, float posI, float posJ, int dir)
        {
            int cellI = (int)MathF.Floor(posI);

            int startCellJ;
            float firstDist;

            if (dir > 0) // walking east (increasing J)
            {
                float frac = posJ - MathF.Floor(posJ);
                startCellJ = (int)MathF.Floor(posJ);
                firstDist = (frac == 0f) ? 1.0f : (MathF.Ceiling(posJ) - posJ);
            }
            else // walking west (decreasing J)
            {
                float frac = posJ - MathF.Floor(posJ);
                if (frac == 0f)
                {
                    startCellJ = (int)posJ - 1;
                    firstDist = 1.0f;
                }
                else
                {
                    startCellJ = (int)MathF.Floor(posJ);
                    firstDist = frac;
                }
            }

            if (!region.Contains(new CellCoord(cellI, startCellJ)))
                return -firstDist;

            float dist = firstDist;
            int cj = startCellJ + dir;
            while (region.Contains(new CellCoord(cellI, cj)))
            {
                dist += 1.0f;
                cj += dir;
            }
            return dist;
        }

        /// <summary>
        /// Find large maximal rectangles in a boolean grid.
        /// Filters by minimum size to eliminate tiny noise rectangles.
        /// </summary>
        private List<(int i1, int i2, int j1, int j2)> FindLargeMaximalRectangles(
            bool[,] grid, int rows, int cols, int minWidth = 1, int minHeight = 1, int minArea = 2)
        {
            var result = new List<(int i1, int i2, int j1, int j2)>();

            for (int i1 = 0; i1 < rows; i1++)
            {
                for (int i2 = i1 + minHeight - 1; i2 < rows; i2++)
                {
                    for (int j1 = 0; j1 < cols; j1++)
                    {
                        for (int j2 = j1 + minWidth - 1; j2 < cols; j2++)
                        {
                            int height = i2 - i1 + 1;
                            int width = j2 - j1 + 1;
                            if (height * width < minArea) continue;

                            // Check if rectangle is fully filled
                            bool filled = true;
                            for (int i = i1; i <= i2 && filled; i++)
                                for (int j = j1; j <= j2 && filled; j++)
                                    if (!grid[i, j]) filled = false;

                            if (!filled) continue;

                            // Check if maximal (cannot extend any side)
                            bool canExtend = false;

                            if (i1 > 0 && !canExtend)
                            {
                                bool ok = true;
                                for (int j = j1; j <= j2 && ok; j++)
                                    if (!grid[i1 - 1, j]) ok = false;
                                canExtend = ok;
                            }
                            if (i2 < rows - 1 && !canExtend)
                            {
                                bool ok = true;
                                for (int j = j1; j <= j2 && ok; j++)
                                    if (!grid[i2 + 1, j]) ok = false;
                                canExtend = ok;
                            }
                            if (j1 > 0 && !canExtend)
                            {
                                bool ok = true;
                                for (int i = i1; i <= i2 && ok; i++)
                                    if (!grid[i, j1 - 1]) ok = false;
                                canExtend = ok;
                            }
                            if (j2 < cols - 1 && !canExtend)
                            {
                                bool ok = true;
                                for (int i = i1; i <= i2 && ok; i++)
                                    if (!grid[i, j2 + 1]) ok = false;
                                canExtend = ok;
                            }

                            if (!canExtend)
                                result.Add((i1, i2, j1, j2));
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Find the minimum set of large maximal rectangles whose union covers every filled cell.
        /// Uses greedy set cover: each iteration picks the rectangle covering the most
        /// uncovered cells (with area as tie-break). Produces the fewest, largest A-frames.
        /// </summary>
        private List<(int i1, int i2, int j1, int j2)> MinimumCoveringRectangles(
            bool[,] grid, int rows, int cols)
        {
            var candidates = FindLargeMaximalRectangles(grid, rows, cols);

            var uncovered = new HashSet<(int, int)>();
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    if (grid[i, j]) uncovered.Add((i, j));

            if (uncovered.Count == 0) return new List<(int, int, int, int)>();

            // Precompute cells per candidate
            var coverSets = new Dictionary<(int, int, int, int), HashSet<(int, int)>>();
            foreach (var rect in candidates)
            {
                var cells = new HashSet<(int, int)>();
                for (int i = rect.i1; i <= rect.i2; i++)
                    for (int j = rect.j1; j <= rect.j2; j++)
                        cells.Add((i, j));
                coverSets[rect] = cells;
            }

            var selected = new List<(int i1, int i2, int j1, int j2)>();

            while (uncovered.Count > 0)
            {
                // Find rect with maximum new coverage (area as tie-break)
                (int i1, int i2, int j1, int j2) bestRect = default;
                int bestCount = -1;
                int bestArea = -1;
                bool found = false;

                foreach (var kvp in coverSets)
                {
                    int newCovered = 0;
                    foreach (var cell in kvp.Value)
                        if (uncovered.Contains(cell)) newCovered++;

                    if (newCovered == 0) continue;

                    int area = (kvp.Key.Item2 - kvp.Key.Item1 + 1) * (kvp.Key.Item4 - kvp.Key.Item3 + 1);

                    if (newCovered > bestCount || (newCovered == bestCount && area > bestArea))
                    {
                        bestCount = newCovered;
                        bestArea = area;
                        bestRect = kvp.Key;
                        found = true;
                    }
                }

                if (!found) break; // remaining cells can't be covered by qualifying rects

                selected.Add(bestRect);
                uncovered.ExceptWith(coverSets[bestRect]);
            }

            return selected;
        }


        /// <summary>
        /// Place a hip roof over a single rectangle of cells.
        /// Tiles LM_3M slope pieces at 3M intervals along the ridge,
        /// with hip end triangles at the ends.
        /// </summary>
        private void PlaceRoofRect(Entity parent, int rMinI, int rMaxI, int rMinJ, int rMaxJ,
            float roofY, float cx, float cz, int index)
        {
            int spanI = rMaxI - rMinI + 1; // depth (Z)
            int spanJ = rMaxJ - rMinJ + 1; // width (X)

            // Ridge along the longer axis
            bool ridgeAlongX = spanJ >= spanI;

            // Pick pitch based on half-width perpendicular to ridge:
            // LM covers 3M per side (3 cells wide), M covers 2M per side (2 cells), S covers 1M (1 cell)
            int perpSpan = ridgeAlongX ? spanI : spanJ;
            float halfWidth = perpSpan * _cellSize * 0.5f; // half-width in meters

            float ridgeRise;
            List<string> slopes3M, slopes2M, slopes1M;
            string pitchUsed;

            if (halfWidth >= 3f && (_roofSlopesLM_3M?.Count > 0 || _roofSlopesLM_2M?.Count > 0))
            {
                // 3+ cells wide → LM pitch (3M per side)
                slopes3M = _roofSlopesLM_3M;
                slopes2M = _roofSlopesLM_2M;
                slopes1M = _roofSlopesLM_1M;
                ridgeRise = 4.55f;
                pitchUsed = "LM";
            }
            else if (halfWidth >= 2f && (_roofSlopesM_3M?.Count > 0 || _roofSlopesM_2M?.Count > 0))
            {
                // 2 cells wide → M pitch (2M per side)
                slopes3M = _roofSlopesM_3M;
                slopes2M = _roofSlopesM_2M;
                slopes1M = _roofSlopesM_1M;
                ridgeRise = 3.0f;
                pitchUsed = "M";
            }
            else if (_roofSlopesS_3M?.Count > 0 || _roofSlopesS_2M?.Count > 0)
            {
                // 1 cell wide → S pitch (1M per side)
                slopes3M = _roofSlopesS_3M;
                slopes2M = _roofSlopesS_2M;
                slopes1M = _roofSlopesS_1M;
                ridgeRise = 1.5f;
                pitchUsed = "S";
            }
            else return;

            // Cross-pitch fallback for filler pieces (2M/1M often missing in LM)
            if (slopes2M == null || slopes2M.Count == 0)
                slopes2M = _roofSlopesM_2M ?? _roofSlopesS_2M;
            if (slopes1M == null || slopes1M.Count == 0)
                slopes1M = _roofSlopesM_1M ?? _roofSlopesS_1M;

            // Piece pivot is at the ridge peak; mesh extends downward to eaves.
            float slopeY = roofY + ridgeRise;

            Debug.Log($"[Roof_{index}] pitch={pitchUsed} perpSpan={perpSpan} halfW={halfWidth} " +
                $"rise={ridgeRise} roofY={roofY} slopeY={slopeY} " +
                $"ridgeAlongX={ridgeAlongX} I=[{rMinI},{rMaxI}] J=[{rMinJ},{rMaxJ}]");

            var roofEntity = new Entity($"Roof_{index}");
            roofEntity.Transform.Parent = parent.Transform;



            float eaveOverhang = 0.5f;

            if (ridgeAlongX)
            {
                // Ridge runs along X (J axis), slopes extend along Z
                float ridgeZ = (rMinI + rMaxI + 1) * 0.5f * _cellSize - cz + _cellSize;
                float westWall = cx - rMinJ * _cellSize + _cellSize * 0.5f;
                float eastWall = cx - (rMaxJ + 1) * _cellSize + _cellSize * 0.5f;
                float ridgeStartX = westWall + eaveOverhang;
                float ridgeEndX = eastWall - eaveOverhang;
                float ridgeLength = MathF.Abs(ridgeStartX - ridgeEndX);

                TileSlopes(roofEntity, slopeY, ridgeZ,
                    ridgeStartX, ridgeEndX, ridgeLength,
                    slopes3M, slopes2M, slopes1M,
                    isRidgeAlongX: true);

                TileRidgeCaps(roofEntity, slopeY, ridgeZ,
                    cx - rMinJ * _cellSize, cx - (rMaxJ + 1) * _cellSize,
                    MathF.Abs(cx - rMinJ * _cellSize - (cx - (rMaxJ + 1) * _cellSize)),
                    isRidgeAlongX: true);
            }
            else
            {
                // Ridge runs along Z (I axis), slopes extend along X
                float ridgeX = cx - (rMinJ + rMaxJ + 1) * 0.5f * _cellSize + _cellSize * 0.5f;
                float northWall = (rMaxI + 1) * _cellSize - cz + _cellSize;
                float southWall = rMinI * _cellSize - cz + _cellSize;
                float ridgeStartZ = northWall + eaveOverhang;
                float ridgeEndZ = southWall - eaveOverhang;
                float ridgeLength = MathF.Abs(ridgeStartZ - ridgeEndZ);

                TileSlopes(roofEntity, slopeY, ridgeX,
                    ridgeStartZ, ridgeEndZ, ridgeLength,
                    slopes3M, slopes2M, slopes1M,
                    isRidgeAlongX: false);

                // Caps: use raw wall positions, no slope correction
                float capNorth = (rMaxI + 1) * _cellSize - cz;
                float capSouth = rMinI * _cellSize - cz;
                float capLength = MathF.Abs(capNorth - capSouth);

                TileRidgeCaps(roofEntity, slopeY, ridgeX,
                    capNorth, capSouth, capLength,
                    isRidgeAlongX: false);
            }
        }

        /// <summary>
        /// Tile slope panels along the ridge line at 3M intervals.
        /// </summary>
        private void TileSlopes(Entity parent, float ridgeY, float ridgePerp,
            float start, float end, float length,
            List<string> slopes3M, List<string> slopes2M, List<string> slopes1M,
            bool isRidgeAlongX)
        {
            var slopeA = isRidgeAlongX ? RoofSlopeA_RidgeX : RoofSlopeA_RidgeZ;
            var slopeB = isRidgeAlongX ? RoofSlopeB_RidgeX : RoofSlopeB_RidgeZ;

            // Pieces extend toward +coord from origin (confirmed by overlap test).
            // Tile from low coord to high.
            float tileFrom = MathF.Min(start, end);
            float pos = tileFrom;
            float remaining = length;

            // Tile with 3M slope pieces from low coord to high
            while (remaining >= 2.5f && slopes3M?.Count > 0)
            {
                var guid = Pick(slopes3M);
                if (guid == null) break;

                Vector3 posA, posB;
                if (isRidgeAlongX)
                {
                    posA = new Vector3(pos, ridgeY, ridgePerp);
                    posB = new Vector3(pos, ridgeY, ridgePerp);
                }
                else
                {
                    posA = new Vector3(ridgePerp, ridgeY, pos);
                    posB = new Vector3(ridgePerp, ridgeY, pos);
                }

                PlaceRoofDirect(parent, posA, slopeA, guid);
                var guidB = Pick(slopes3M) ?? guid;
                PlaceRoofDirect(parent, posB, slopeB, guidB);

                pos += 3f;
                remaining -= 3f;
            }

            // Place 1M remainder piece: mesh starts 1M ahead of origin, so offset by -1
            if (remaining > 0.5f && slopes1M?.Count > 0)
            {
                float fillPos = pos - 1f;
                var guid = Pick(slopes1M);
                if (guid != null)
                {
                    Vector3 posA, posB;
                    if (isRidgeAlongX)
                    {
                        posA = new Vector3(fillPos, ridgeY, ridgePerp);
                        posB = new Vector3(fillPos, ridgeY, ridgePerp);
                    }
                    else
                    {
                        posA = new Vector3(ridgePerp, ridgeY, fillPos);
                        posB = new Vector3(ridgePerp, ridgeY, fillPos);
                    }

                    PlaceRoofDirect(parent, posA, slopeA, guid);
                    var guidB = Pick(slopes1M) ?? guid;
                    PlaceRoofDirect(parent, posB, slopeB, guidB);
                }
            }
        }

        /// <summary>
        /// Tile ridge caps along the ridge line.
        /// Uses 10M/6M/3M/2M/1M pieces for optimal coverage.
        /// </summary>
        private void TileRidgeCaps(Entity parent, float ridgeY, float ridgePerp,
            float start, float end, float length,
            bool isRidgeAlongX)
        {
            var capRot = isRidgeAlongX ? RidgeCapRot_RidgeX : RidgeCapRot_RidgeZ;
            // Caps extend backward from origin (toward -coord), so tile from max to min
            float tileFrom = MathF.Max(start, end);
            float pos = tileFrom;
            float remaining = length;

            // Available ridge cap sizes (prefer larger for fewer pieces)
            var caps = new (List<string> list, float size)[]
            {
                (_roofRidges10M, 10f), (_roofRidges6M, 6f),
                (_roofRidges3M, 3f), (_roofRidges2M, 2f), (_roofRidges1M, 1f)
            };

            while (remaining > 0.5f)
            {
                bool placed = false;
                foreach (var (list, size) in caps)
                {
                    if (remaining >= size && list?.Count > 0)
                    {
                        var guid = Pick(list);
                        if (guid == null) continue;

                        Vector3 p = isRidgeAlongX
                            ? new Vector3(pos, ridgeY, ridgePerp)
                            : new Vector3(ridgePerp, ridgeY, pos);
                        PlaceRoofDirect(parent, p, capRot, guid);

                        pos -= size;
                        remaining -= size;
                        placed = true;
                        break;
                    }
                }
                if (!placed) break;
            }
        }

        #endregion

        #region Flood Fill

        /// <summary>
        /// Decompose a set of cells into contiguous regions via 4-neighbor flood fill.
        /// </summary>
        private static List<HashSet<CellCoord>> FloodFillRegions(HashSet<CellCoord> cells)
        {
            var regions = new List<HashSet<CellCoord>>();
            var visited = new HashSet<CellCoord>();

            foreach (var seed in cells)
            {
                if (visited.Contains(seed)) continue;

                var region = new HashSet<CellCoord>();
                var queue = new Queue<CellCoord>();
                queue.Enqueue(seed);
                visited.Add(seed);

                while (queue.Count > 0)
                {
                    var c = queue.Dequeue();
                    region.Add(c);

                    foreach (var dir in new[] { "n", "s", "e", "w" })
                    {
                        var nb = c.Neighbor(dir);
                        if (cells.Contains(nb) && visited.Add(nb))
                            queue.Enqueue(nb);
                    }
                }

                regions.Add(region);
            }

            return regions;
        }

        #endregion

        #region Prefab Catalog

        private void ResolveCatalog()
        {
            if (_catalogResolved) return;

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

            // Walls
            _stoneWalls2M = FindByPrefix(prefabGuids, "SM_WallS_2X3M");
            _stoneWalls1M = FindByPrefix(prefabGuids, "SM_WallS_1X3M");
            _plasterWalls2M = FindByPrefix(prefabGuids, "SM_WallP_2X3M");
            _plasterWalls1M = FindByPrefix(prefabGuids, "SM_WallP_1X3M");

            // Window walls (the _Wall suffix = wall with window cutout, not the window insert)
            _stoneWindowWalls = FindByPrefix(prefabGuids, "SM_WallS_Window_2X3M_", "_XLarge", requiredSuffix: "_Wall");
            _plasterWindowWalls = FindByPrefix(prefabGuids, "SM_WallP_Window_2X3M_", "_XLarge", requiredSuffix: "_wall");

            // Door walls
            _stoneDoorWalls = FindByPrefix(prefabGuids, "SM_WallS_Door_2X3M");
            _plasterDoorWalls = FindByPrefix(prefabGuids, "SM_WallP_Door_2X3M");

            // Corners - only SM_Stone_Corner_3M (exclude V2=pillar, 120D=wrong angle)
            _stoneCorners = FindByPrefix(prefabGuids, "SM_Stone_Corner_3M", exclude: "_V2");
            _stoneCorners.RemoveAll(g =>
            {
                var prefab = Engine.Assets.LoadByGuid<Prefab>(g);
                return prefab?.Name?.Contains("120D") == true;
            });

            // Roof slopes — LM pitch (most common, from House02/03/04/06/07/09)
            _roofSlopesLM_1M = FindByPrefix(prefabGuids, "SM_RoofTiles_LM_1M", "_CutOff");
            _roofSlopesLM_2M = FindByPrefix(prefabGuids, "SM_RoofTiles_LM_2M");
            _roofSlopesLM_3M = FindByPrefix(prefabGuids, "SM_RoofTiles_LM_3M", "_cutoff");
            _roofSlopesLM_3M_cutoff = FindByPrefix(prefabGuids, "SM_RoofTiles_LM_3M_cutoff");

            // Roof slopes — S/M pitch (fallback)
            _roofSlopesS_1M = FindByPrefix(prefabGuids, "SM_RoofTiles_S_1M", "_CutOff");
            _roofSlopesS_2M = FindByPrefix(prefabGuids, "SM_RoofTiles_S_2M");
            _roofSlopesS_3M = FindByPrefix(prefabGuids, "SM_RoofTiles_S_3M");
            _roofSlopesM_1M = FindByPrefix(prefabGuids, "SM_RoofTiles_M_1M");
            _roofSlopesM_2M = FindByPrefix(prefabGuids, "SM_RoofTiles_M_2M");
            _roofSlopesM_3M = FindByPrefix(prefabGuids, "SM_RoofTiles_M_3M", "_Cutoff");

            // Ridge caps
            _roofRidges1M = FindByPrefix(prefabGuids, "SM_RoofStone_Top_1M");
            _roofRidges2M = FindByPrefix(prefabGuids, "SM_RoofStone_Top_2M");
            _roofRidges3M = FindByPrefix(prefabGuids, "SM_RoofStone_Top_3M");
            _roofRidges6M = FindByPrefix(prefabGuids, "SM_RoofStone_Top_6M");
            _roofRidges10M = FindByPrefix(prefabGuids, "SM_RoofStone_Top_10M");

            // Roof corners and hip ends
            _roofCorners = FindByPrefix(prefabGuids, "SM_RoofTiles_Corner_");
            _roofHipTri1M = FindByPrefix(prefabGuids, "SM_RoofTiles_Tri_1M");
            _roofHipTri2M = FindByPrefix(prefabGuids, "SM_RoofTiles_Tri_2M");

            // Gable wall triangles
            _gableTriStone = FindByPrefix(prefabGuids, "SM_WallS_Tri_");
            _gableTriPlaster = FindByPrefix(prefabGuids, "SM_WallP_Tri_");

            // Chimneys
            _chimneys = FindByPrefix(prefabGuids, "SM_Chimney");

            // Stairs
            _stairs = FindByPrefix(prefabGuids, "SM_Stair_Wood_3M");

            int totalWalls = (_stoneWalls2M?.Count ?? 0) + (_plasterWalls2M?.Count ?? 0);
            int totalWindows = (_stoneWindowWalls?.Count ?? 0) + (_plasterWindowWalls?.Count ?? 0);
            int totalDoors = (_stoneDoorWalls?.Count ?? 0) + (_plasterDoorWalls?.Count ?? 0);
            int slopeLM = (_roofSlopesLM_3M?.Count ?? 0) + (_roofSlopesLM_2M?.Count ?? 0) + (_roofSlopesLM_1M?.Count ?? 0);
            int slopeM = (_roofSlopesM_3M?.Count ?? 0) + (_roofSlopesM_2M?.Count ?? 0);
            int ridgeCaps = (_roofRidges10M?.Count ?? 0) + (_roofRidges6M?.Count ?? 0) +
                            (_roofRidges3M?.Count ?? 0) + (_roofRidges2M?.Count ?? 0) + (_roofRidges1M?.Count ?? 0);

            Debug.Log($"[DwellingsBuilder] Catalog: {totalWalls} walls, {totalWindows} windows, " +
                $"{totalDoors} doors, {slopeLM} LM slopes, {slopeM} M slopes, {ridgeCaps} ridge caps, " +
                $"{_roofCorners?.Count ?? 0} corners, {_roofHipTri2M?.Count ?? 0} hip tris, " +
                $"{_chimneys?.Count ?? 0} chimneys");

            _catalogResolved = true;
        }

        private static List<string> FindByPrefix(Dictionary<string, List<string>> prefabGuids,
            string namePrefix, string exclude = null, string requiredSuffix = null)
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

            // Scale normalization (same logic as PrefabBuildingPlacer):
            // Composite prefabs (with children): children have 100x for cm→m.
            //   If root also has 100x, it's double-scaled → reset root to 1.
            // Simple prefabs (no children): mesh is in cm → set root to 100.
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
