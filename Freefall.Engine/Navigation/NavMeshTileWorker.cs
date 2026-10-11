using System;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using DotRecast.Core;
using DotRecast.Core.Numerics;
using DotRecast.Detour;
using DotRecast.Detour.Io;
using DotRecast.Recast;

namespace Freefall.Navigation
{
    public static partial class NavMeshBuilder
    {
        /// <summary>
        /// One worker thread of a bake: scratch buffers and the build of a single grid cell.
        ///
        /// Triangles are voxelized here rather than by DotRecast's rasterizer. That one allocates a
        /// span object for every voxel column of every triangle, which for a scene with dense foliage
        /// (hundreds of millions of triangles) is over a hundred GB of garbage and one collection
        /// every few milliseconds. This one recycles its spans from cell to cell.
        /// </summary>
        private sealed class TileWorker
        {
            private readonly BakeContext _c;
            private readonly RcBuilder _builder = new();
            private readonly DtMeshDataWriter _writer = new();
            private readonly MemoryStream _scratch = new();

            public readonly BakeProfile Profile = new();

            // World-space vertices: the cell's terrain patches first, then one mesh chunk at a time
            private float[] _verts = new float[3 * 8192];
            private int _vertCount;

            // Terrain patches sampled into _verts, waiting for the heightfield
            private (int baseVertex, int rowWidth, int rows)[] _patches = new (int, int, int)[4];
            private int _patchCount;

            // ── Heightfield of the cell being built ──
            private RcHeightfield? _solid;
            private float _fieldMinX, _fieldMinY, _fieldMinZ;
            private float _fieldMaxX, _fieldMaxY, _fieldMaxZ;
            private int _walkableTriangles;

            // Spans not in use, linked through RcSpan.next
            private RcSpan? _freeSpans;

            // Clipping scratch: four polygons of up to 7 vertices, and the per-vertex plane distances
            private readonly float[] _clip = new float[7 * 3 * 4];
            private readonly float[] _clipDelta = new float[12];

            public TileWorker(BakeContext context)
            {
                _c = context;
            }

            public CellResult Build(int cell)
            {
                var c = _c;
                var tiles = c.Tiles;
                int tx = cell % tiles.TilesX, tz = cell / tiles.TilesX;

                // The tile's rect, grown by the border so its edges come out the same as its neighbours'
                float minX = tiles.Origin.X + tx * c.TileWorld - c.Pad;
                float minZ = tiles.Origin.Z + tz * c.TileWorld - c.Pad;
                float maxX = tiles.Origin.X + tx * c.TileWorld + c.TileWorld + c.Pad;
                float maxZ = tiles.Origin.Z + tz * c.TileWorld + c.TileWorld + c.Pad;

                _vertCount = 0;
                _patchCount = 0;
                long t0 = Stopwatch.GetTimestamp();

                // ── Inputs: hash them, find the height range ──
                ulong hash = NavHash.Mix(c.SettingsHash, ((ulong)(uint)tx << 32) | (uint)tz);
                float minY = float.MaxValue, maxY = float.MinValue;

                foreach (var terrain in c.Input.Terrains)
                    SampleTerrain(terrain, minX, minZ, maxX, maxZ, ref hash, ref minY, ref maxY);

                var instances = c.Input.Instances;
                int first = c.CellStart[cell], end = c.CellStart[cell + 1];

                // Summed, not chained: the order instances are listed in must not matter
                ulong instanceHash = 0;
                for (int i = first; i < end; i++)
                {
                    ref readonly var instance = ref instances[c.CellItems[i]];
                    instanceHash += instance.Hash;
                    minY = MathF.Min(minY, instance.Min.Y);
                    maxY = MathF.Max(maxY, instance.Max.Y);
                }

                hash = NavHash.Mix(NavHash.Mix(hash, instanceHash), (ulong)(end - first));
                if (hash == 0) hash = 1;
                tiles.Hashes[cell] = hash;

                long t1 = Stopwatch.GetTimestamp();
                Profile.Inputs += t1 - t0;

                if (c.Previous != null && c.Previous.Hashes[cell] == hash)
                {
                    tiles.Tiles[cell] = c.Previous.Tiles[cell];
                    return CellResult.Reused;
                }

                // Nothing here. If the last bake had a tile, losing it is a change like any other.
                if (minY > maxY)
                    return c.Previous?.Tiles[cell] != null ? CellResult.Rebuilt : CellResult.Empty;

                // ── Rasterize ──
                // The height range is the cell's own, snapped to the cell-height lattice: every tile
                // quantizes heights the same way whatever its neighbours contain.
                float ch = c.Settings.CellHeight;
                _fieldMinX = minX;
                _fieldMinY = MathF.Floor(minY / ch) * ch - ch;
                _fieldMinZ = minZ;
                _fieldMaxX = maxX;
                _fieldMaxY = maxY + ch;
                _fieldMaxZ = maxZ;
                _walkableTriangles = 0;

                var solid = new RcHeightfield(c.FieldSize, c.FieldSize,
                    new RcVec3f(_fieldMinX, _fieldMinY, _fieldMinZ), new RcVec3f(_fieldMaxX, _fieldMaxY, _fieldMaxZ),
                    c.Settings.CellSize, ch, c.Config.BorderSize);
                _solid = solid;

                RasterizeTerrain();
                for (int i = first; i < end; i++)
                    RasterizeInstance(in instances[c.CellItems[i]]);

                long t2 = Stopwatch.GetTimestamp();
                Profile.Rasterize += t2 - t1;

                // Nothing walkable went in, so nothing can come out: obstacles only ever take area away
                if (_walkableTriangles == 0)
                {
                    RecycleSpans(solid);
                    return CellResult.Rebuilt;
                }

                // ── Recast: filter, regions, contours, polygons, detail mesh ──
                var result = _builder.Build(new RcContext(), tx, tz, NoGeometry.Instance, c.Config, solid, false);
                RecycleSpans(solid);

                long t3 = Stopwatch.GetTimestamp();
                Profile.Recast += t3 - t2;

                var polyMesh = result.Mesh;
                if (polyMesh == null || polyMesh.npolys == 0)
                    return CellResult.Rebuilt;

                // Recast leaves poly flags at 0 — set the walkable flag so queries find them
                for (int i = 0; i < polyMesh.npolys; i++)
                    polyMesh.flags[i] = 1;

                var detail = result.MeshDetail;
                var create = new DtNavMeshCreateParams
                {
                    verts = polyMesh.verts,
                    vertCount = polyMesh.nverts,
                    polys = polyMesh.polys,
                    polyAreas = polyMesh.areas,
                    polyFlags = polyMesh.flags,
                    polyCount = polyMesh.npolys,
                    nvp = polyMesh.nvp,
                    walkableHeight = c.Settings.AgentHeight,
                    walkableRadius = c.Settings.AgentRadius,
                    walkableClimb = c.Settings.MaxClimb,
                    cs = c.Settings.CellSize,
                    ch = ch,
                    buildBvTree = true,
                    bmin = polyMesh.bmin,
                    bmax = polyMesh.bmax,
                    tileX = tx,
                    tileZ = tz,
                };
                if (detail != null)
                {
                    create.detailMeshes = detail.meshes;
                    create.detailVerts = detail.verts;
                    create.detailVertsCount = detail.nverts;
                    create.detailTris = detail.tris;
                    create.detailTriCount = detail.ntris;
                }

                var data = DtNavMeshBuilder.CreateNavMeshData(create);
                if (data != null)
                    tiles.Tiles[cell] = NavMeshTileSet.WriteTile(data, _writer, _scratch);

                Profile.Detour += Stopwatch.GetTimestamp() - t3;
                return CellResult.Rebuilt;
            }

            // ── Terrain ──

            /// <summary>
            /// Sample the terrain patch under the rect into the vertex buffer. Samples lie on the
            /// terrain's own step lattice, so neighbouring tiles see exactly the same triangles.
            /// </summary>
            private void SampleTerrain(TerrainSource terrain, float minX, float minZ, float maxX, float maxZ,
                ref ulong hash, ref float minY, ref float maxY)
            {
                float step = _c.Settings.TerrainStep;
                int stepsX = (int)MathF.Ceiling(terrain.Size.X / step);
                int stepsZ = (int)MathF.Ceiling(terrain.Size.Y / step);

                int x0 = Math.Max(0, (int)MathF.Floor((minX - terrain.Position.X) / step));
                int z0 = Math.Max(0, (int)MathF.Floor((minZ - terrain.Position.Z) / step));
                int x1 = Math.Min(stepsX, (int)MathF.Ceiling((maxX - terrain.Position.X) / step));
                int z1 = Math.Min(stepsZ, (int)MathF.Ceiling((maxZ - terrain.Position.Z) / step));
                if (x0 >= x1 || z0 >= z1) return;

                int rowWidth = x1 - x0 + 1;
                int rows = z1 - z0 + 1;
                ReserveVertices(rowWidth * rows);

                hash = NavHash.Mix(hash, terrain.Hash);
                hash = NavHash.Mix(hash, ((ulong)(uint)x0 << 32) | (uint)z0);
                hash = NavHash.Mix(hash, ((ulong)(uint)x1 << 32) | (uint)z1);

                if (_patchCount == _patches.Length)
                    Array.Resize(ref _patches, _patchCount * 2);
                _patches[_patchCount++] = (_vertCount, rowWidth, rows);

                var verts = _verts;
                int o = _vertCount * 3;
                for (int z = z0; z <= z1; z++)
                {
                    float localZ = MathF.Min(z * step, terrain.Size.Y);
                    for (int x = x0; x <= x1; x++)
                    {
                        float localX = MathF.Min(x * step, terrain.Size.X);
                        float y = terrain.SampleHeight(localX, localZ);

                        verts[o++] = terrain.Position.X + localX;
                        verts[o++] = y;
                        verts[o++] = terrain.Position.Z + localZ;

                        hash = NavHash.Mix(hash, y);
                        minY = MathF.Min(minY, y);
                        maxY = MathF.Max(maxY, y);
                    }
                }
                _vertCount += rowWidth * rows;
            }

            private void RasterizeTerrain()
            {
                for (int p = 0; p < _patchCount; p++)
                {
                    var (baseVertex, rowWidth, rows) = _patches[p];
                    for (int z = 0; z < rows - 1; z++)
                    {
                        for (int x = 0; x < rowWidth - 1; x++)
                        {
                            int i00 = baseVertex + z * rowWidth + x;
                            int i10 = i00 + 1;
                            int i01 = i00 + rowWidth;
                            int i11 = i01 + 1;

                            RasterizeTriangle(i00, i01, i10, false);
                            RasterizeTriangle(i10, i01, i11, false);
                        }
                    }
                }
                _vertCount = 0;
                _patchCount = 0;
            }

            // ── Meshes ──

            /// <summary>Rasterize the chunks of a mesh instance that reach into the cell's rect.</summary>
            private void RasterizeInstance(in MeshInstance instance)
            {
                var geometry = _c.Input.Geometries[instance.Geometry];
                var chunks = geometry.Chunks;
                var positions = geometry.Positions;
                var indices = geometry.Indices;

                for (int k = 0; k < chunks.Length; k++)
                {
                    ref readonly var chunk = ref chunks[k];

                    if (chunks.Length > 1)
                    {
                        MeshGeometry.TransformBounds(instance.World, chunk.Center, chunk.Extents, out var center, out var extents);
                        if (center.X + extents.X < _fieldMinX || center.X - extents.X > _fieldMaxX ||
                            center.Z + extents.Z < _fieldMinZ || center.Z - extents.Z > _fieldMaxZ)
                            continue;
                    }

                    _vertCount = 0;
                    ReserveVertices(chunk.VertexCount);

                    var verts = _verts;
                    int o = 0;
                    for (int v = 0; v < chunk.VertexCount; v++)
                    {
                        var world = Vector3.Transform(positions[chunk.VertexStart + v], instance.World);
                        verts[o++] = world.X;
                        verts[o++] = world.Y;
                        verts[o++] = world.Z;
                    }

                    int indexEnd = chunk.IndexStart + chunk.IndexCount;
                    for (int i = chunk.IndexStart; i < indexEnd; i += 3)
                        RasterizeTriangle(indices[i], indices[i + 1], indices[i + 2], instance.Mirrored);

                    Profile.Triangles += chunk.IndexCount / 3;
                }
            }

            private void ReserveVertices(int count)
            {
                int needed = (_vertCount + count) * 3;
                if (needed > _verts.Length)
                    Array.Resize(ref _verts, Math.Max(_verts.Length * 2, needed));
            }

            // ── Voxelization (Recast's rasterizeTri / addSpan) ──

            /// <summary>
            /// Voxelize one triangle of the vertex buffer into the cell's heightfield, walkable or not
            /// by its slope.
            /// </summary>
            private void RasterizeTriangle(int a, int b, int c, bool mirrored)
            {
                var verts = _verts;
                a *= 3; b *= 3; c *= 3;
                float ax = verts[a], ay = verts[a + 1], az = verts[a + 2];
                float bx = verts[b], by = verts[b + 1], bz = verts[b + 2];
                float cx = verts[c], cy = verts[c + 1], cz = verts[c + 2];

                float minX = MathF.Min(ax, MathF.Min(bx, cx)), maxX = MathF.Max(ax, MathF.Max(bx, cx));
                float minZ = MathF.Min(az, MathF.Min(bz, cz)), maxZ = MathF.Max(az, MathF.Max(bz, cz));
                if (minX > _fieldMaxX || maxX < _fieldMinX || minZ > _fieldMaxZ || maxZ < _fieldMinZ) return;

                float minY = MathF.Min(ay, MathF.Min(by, cy)), maxY = MathF.Max(ay, MathF.Max(by, cy));
                if (minY > _fieldMaxY || maxY < _fieldMinY) return;

                // Walkable if normal.Y / |normal| > cos(maxSlope), without the square root
                float e0x = bx - ax, e0y = by - ay, e0z = bz - az;
                float e1x = cx - ax, e1y = cy - ay, e1z = cz - az;
                float nx = e0y * e1z - e0z * e1y;
                float ny = e0z * e1x - e0x * e1z;
                float nz = e0x * e1y - e0y * e1x;
                if (mirrored) ny = -ny;

                int area = 0;
                if (ny > 0 && ny * ny > _c.WalkableThresholdSq * (nx * nx + ny * ny + nz * nz))
                {
                    area = _c.WalkableArea;
                    _walkableTriangles++;
                }

                float cs = _c.Settings.CellSize;
                float invCs = 1f / cs;
                int size = _c.FieldSize;
                float height = _fieldMaxY - _fieldMinY;

                int z0 = (int)((minZ - _fieldMinZ) * invCs);
                int z1 = (int)((maxZ - _fieldMinZ) * invCs);
                int x0 = (int)((minX - _fieldMinX) * invCs);
                int x1 = (int)((maxX - _fieldMinX) * invCs);

                // Inside a single column (most foliage triangles): nothing to clip
                if (x0 == x1 && z0 == z1 && minX >= _fieldMinX && minZ >= _fieldMinZ && x0 < size && z0 < size)
                {
                    AddSpan(x0 + z0 * size, minY - _fieldMinY, maxY - _fieldMinY, height, area);
                    return;
                }

                // -1 rather than 0, to cut the polygon properly at the start of the field
                z0 = Math.Clamp(z0, -1, size - 1);
                z1 = Math.Clamp(z1, 0, size - 1);

                // Four polygon buffers: the part still to be cut into rows, the current row, and two
                // for cutting the row into columns
                var clip = _clip;
                int input = 0, row = 21, p1 = 42, p2 = 63;
                clip[0] = ax; clip[1] = ay; clip[2] = az;
                clip[3] = bx; clip[4] = by; clip[5] = bz;
                clip[6] = cx; clip[7] = cy; clip[8] = cz;
                int inputCount = 3;

                for (int z = z0; z <= z1; z++)
                {
                    // Cut off the part of the polygon in this row, keep the rest for the next
                    float cellZ = _fieldMinZ + z * cs;
                    DividePoly(input, inputCount, row, out int rowCount, p1, out inputCount, cellZ + cs, 2);
                    (input, p1) = (p1, input);

                    if (rowCount < 3 || z < 0) continue;

                    float rowMinX = clip[row], rowMaxX = clip[row];
                    for (int v = 1; v < rowCount; v++)
                    {
                        float x = clip[row + v * 3];
                        if (x < rowMinX) rowMinX = x;
                        if (x > rowMaxX) rowMaxX = x;
                    }

                    int cx0 = (int)((rowMinX - _fieldMinX) * invCs);
                    int cx1 = (int)((rowMaxX - _fieldMinX) * invCs);
                    if (cx1 < 0 || cx0 >= size) continue;
                    cx0 = Math.Clamp(cx0, -1, size - 1);
                    cx1 = Math.Clamp(cx1, 0, size - 1);

                    int restCount = rowCount;
                    for (int x = cx0; x <= cx1; x++)
                    {
                        // Cut off the part of the row in this column, keep the rest for the next
                        float cellX = _fieldMinX + x * cs;
                        DividePoly(row, restCount, p1, out int count, p2, out restCount, cellX + cs, 0);
                        (row, p2) = (p2, row);

                        if (count < 3 || x < 0) continue;

                        float spanMin = clip[p1 + 1], spanMax = spanMin;
                        for (int v = 1; v < count; v++)
                        {
                            float y = clip[p1 + v * 3 + 1];
                            if (y < spanMin) spanMin = y;
                            if (y > spanMax) spanMax = y;
                        }

                        AddSpan(x + z * size, spanMin - _fieldMinY, spanMax - _fieldMinY, height, area);
                    }
                }
            }

            /// <summary>
            /// Split a convex polygon of the clip buffer along an axis-aligned plane: the part below
            /// axisOffset goes to out1, the part above to out2.
            /// </summary>
            private void DividePoly(int input, int inputCount, int out1, out int out1Count, int out2, out int out2Count, float axisOffset, int axis)
            {
                var clip = _clip;
                var delta = _clipDelta;
                for (int i = 0; i < inputCount; i++)
                    delta[i] = axisOffset - clip[input + i * 3 + axis];

                int n1 = 0, n2 = 0;
                for (int a = 0, b = inputCount - 1; a < inputCount; b = a, a++)
                {
                    int va = input + a * 3;
                    bool sameSide = (delta[a] >= 0) == (delta[b] >= 0);

                    if (!sameSide)
                    {
                        // The edge crosses the plane: the crossing point belongs to both parts
                        int vb = input + b * 3;
                        float s = delta[b] / (delta[b] - delta[a]);
                        float x = clip[vb] + (clip[va] - clip[vb]) * s;
                        float y = clip[vb + 1] + (clip[va + 1] - clip[vb + 1]) * s;
                        float z = clip[vb + 2] + (clip[va + 2] - clip[vb + 2]) * s;

                        int o1 = out1 + n1++ * 3;
                        clip[o1] = x; clip[o1 + 1] = y; clip[o1 + 2] = z;
                        int o2 = out2 + n2++ * 3;
                        clip[o2] = x; clip[o2 + 1] = y; clip[o2 + 2] = z;

                        // Then the vertex itself, unless it lies on the plane (already added above)
                        if (delta[a] > 0)
                        {
                            o1 = out1 + n1++ * 3;
                            clip[o1] = clip[va]; clip[o1 + 1] = clip[va + 1]; clip[o1 + 2] = clip[va + 2];
                        }
                        else if (delta[a] < 0)
                        {
                            o2 = out2 + n2++ * 3;
                            clip[o2] = clip[va]; clip[o2 + 1] = clip[va + 1]; clip[o2 + 2] = clip[va + 2];
                        }
                    }
                    else
                    {
                        // Same side: the vertex goes to its part, a vertex on the plane to both
                        if (delta[a] >= 0)
                        {
                            int o1 = out1 + n1++ * 3;
                            clip[o1] = clip[va]; clip[o1 + 1] = clip[va + 1]; clip[o1 + 2] = clip[va + 2];
                            if (delta[a] != 0) continue;
                        }
                        int o2 = out2 + n2++ * 3;
                        clip[o2] = clip[va]; clip[o2 + 1] = clip[va + 1]; clip[o2 + 2] = clip[va + 2];
                    }
                }

                out1Count = n1;
                out2Count = n2;
            }

            /// <summary>
            /// Add a solid span to a column, merging it with the spans it overlaps or touches.
            /// spanMin / spanMax are heights above the field's floor.
            /// </summary>
            private void AddSpan(int column, float spanMin, float spanMax, float fieldHeight, int area)
            {
                if (spanMax < 0f || spanMin > fieldHeight) return;
                if (spanMin < 0f) spanMin = 0f;
                if (spanMax > fieldHeight) spanMax = fieldHeight;

                // Snap to the height lattice
                float invCh = 1f / _c.Settings.CellHeight;
                int smin = Math.Clamp((int)MathF.Floor(spanMin * invCh), 0, RcRecast.RC_SPAN_MAX_HEIGHT);
                int smax = Math.Min(Math.Max((int)MathF.Ceiling(spanMax * invCh), smin + 1), RcRecast.RC_SPAN_MAX_HEIGHT);

                var spans = _solid!.spans;
                int mergeThreshold = _c.Config.WalkableClimb;

                RcSpan? previous = null;
                var current = spans[column];
                while (current != null)
                {
                    // Completely above the new span: it goes in before this one
                    if (current.smin > smax) break;

                    // Completely below: keep going
                    if (current.smax < smin)
                    {
                        previous = current;
                        current = current.next;
                        continue;
                    }

                    // The new span lies within an existing one (the usual case in dense geometry):
                    // only the area can change. Spans of a column never touch, so no other is affected.
                    if (current.smin <= smin && current.smax >= smax)
                    {
                        if (area > current.area) current.area = area;
                        return;
                    }

                    // Overlap: swallow the existing span
                    if (current.smin < smin) smin = current.smin;
                    if (current.smax > smax) smax = current.smax;

                    // Of two surfaces within a step of each other the walkable one wins
                    if (Math.Abs(smax - current.smax) <= mergeThreshold && current.area > area)
                        area = current.area;

                    var next = current.next;
                    if (previous != null) previous.next = next;
                    else spans[column] = next;

                    current.next = _freeSpans;
                    _freeSpans = current;
                    current = next;
                }

                var span = _freeSpans;
                if (span != null) _freeSpans = span.next;
                else span = new RcSpan();

                span.smin = smin;
                span.smax = smax;
                span.area = area;

                if (previous != null)
                {
                    span.next = previous.next;
                    previous.next = span;
                }
                else
                {
                    span.next = spans[column];
                    spans[column] = span;
                }
            }

            /// <summary>Take back the spans of a heightfield Recast is done with.</summary>
            private void RecycleSpans(RcHeightfield solid)
            {
                var spans = solid.spans;
                for (int i = 0; i < spans.Length; i++)
                {
                    var head = spans[i];
                    if (head == null) continue;

                    var last = head;
                    while (last.next != null) last = last.next;

                    last.next = _freeSpans;
                    _freeSpans = head;
                    spans[i] = null;
                }
                _solid = null;
            }
        }
    }
}
