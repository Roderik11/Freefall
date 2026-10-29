using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    /// <summary>
    /// The far look of a field of crops: a low mesh of the crop's top surface and its edge, sloping down to the ground
    /// as the plants there grow shorter and lean out over the headland, so a field keeps its height, its edge and the
    /// shadow that edge throws where the plants themselves have thinned out (the decorators thin by distance and stop at
    /// their range: crops painted on the ground read as bare from afar).
    ///
    /// Sits next to the DecoStamp (Op Add, closed spline) that sows the field and takes everything from it: the outline
    /// (the spline grown by the stamp's Radius and a share of its Falloff, where the plants stand dense), the height
    /// (the decorator's plants), the colour (the decorator's far colour, which the ground under it takes too) and where
    /// to rise. Near the camera the canopy lies below the ground; it rises to the crop's height as the plants thin out
    /// (crop_canopy.fx takes the spawn's thinning), so it stands where the plants stop covering the field and the ones
    /// still standing poke out of it. Nothing runs per frame: the mesh is rebuilt when the spline, the stamp, the
    /// ground under it or the terrain's decorators change.
    /// </summary>
    [Icon("icon_decostamp.png")]
    public class CropCanopy : PersistentRenderer
    {
        // m: the top surface's grid and the edge's longest step, the heightmap's own: at 1.5 m the canopy cut through the
        // crests of ridge and furrow (HeightStamp.RidgeSpacing) wherever it stood low, in bands of soil
        private const float CellSize = 1f;
        private const float Sink = 0.3f;            // m below the ground a top vertex goes where the plants still cover
        private const float EdgeTuck = 0.15f;       // m the edge begins inside the top's border, under it: no crack between them
        private const float EdgeReach = 0.7f;       // of the stamp's Falloff the edge reaches out past the outline (≥ 0.4 m)
        private const float HeightShare = 0.85f;    // of the plants' mean height: ears and blooms stand above the canopy
        private const float FalloffShare = 0.3f;    // of the stamp's Falloff past its edge: where its plants stand dense
        private const int SplineSamplesPerSpan = 8;
        private const float LodgedShare = 0.06f;    // of a grain field lying flat (crop_canopy.fx Lodging)
        private const int Shells = 4;               // layers of the top (crop_canopy.fx CANOPY_SHELLS)

        private static Material? _material;
        private Mesh? _mesh;

        protected override Mesh? RenderMesh => _mesh;

        protected override void Awake()
        {
            base.Awake();
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.AddListener(EngineMsg.StampChanged, OnStampChanged);
            MessageDispatcher.AddListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            TerrainRenderer.DecoratorsRebuilt += OnDecoratorsRebuilt;
            Rebuild();
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.RemoveListener(EngineMsg.StampChanged, OnStampChanged);
            MessageDispatcher.RemoveListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            TerrainRenderer.DecoratorsRebuilt -= OnDecoratorsRebuilt;
            base.Destroy();
            // Frames in flight still draw it
            Engine.Device.DeferDispose(_mesh);
            _mesh = null;
        }

        public override void OnMemberChanged() => Rebuild();

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is Spline spline && spline.Entity == Entity) Rebuild();
        }

        private void OnStampChanged(Message msg)
        {
            if (msg.Data is DecoStamp stamp && stamp.Entity == Entity) Rebuild();
        }

        private void OnTerrainHeightsChanged(Message msg)
        {
            if (msg.Data is TerrainHeightsChange { All: false } change && _mesh != null && Transform != null)
            {
                // Heights changed somewhere else: the canopy still stands on its ground
                var box = _mesh.BoundingBox;
                var a = Vector3.Transform(box.Min, Transform.WorldMatrix);
                var b = Vector3.Transform(box.Max, Transform.WorldMatrix);
                var min = new Vector2(MathF.Min(a.X, b.X), MathF.Min(a.Z, b.Z));
                var max = new Vector2(MathF.Max(a.X, b.X), MathF.Max(a.Z, b.Z));
                if (!change.Overlaps(min, max)) return;
            }
            Rebuild();
        }

        private void OnDecoratorsRebuilt(TerrainRenderer terrain) => Rebuild();

        protected override void AddDraws(DrawGroup group, Mesh mesh, int transformSlot)
        {
            _material ??= new Material(new Effect("crop_canopy"));
            var parts = mesh.DrawPartIndices;
            for (int i = 0; i < parts.Length; i++)
                CommandBuffer.AddPersistent(group, mesh, parts[i], _material, _noParams, transformSlot,
                                            lodManaged: true, instanceSlot: InstanceSlotFor(_material));
        }

        private static readonly MaterialBlock _noParams = new();

        // ── Building ──

        /// <summary>
        /// The canopy as the field's stamp, spline and decorator and the ground under it have it now: a new mesh, and
        /// the values crop_canopy.fx reads (its struct Instance).
        /// </summary>
        private void Rebuild()
        {
            var old = _mesh;
            _mesh = Build();
            Engine.Device.DeferDispose(old);
            Invalidate();
        }

        private Mesh? Build()
        {
            if (Entity == null || Transform == null) return null;
            var spline = Entity.GetComponent<Spline>();
            var stamp = Entity.GetComponent<DecoStamp>();
            var terrain = EntityManager.FindComponent<TerrainRenderer>();
            var decorator = stamp?.Decorator;
            if (spline is not { Closed: true } || spline.Points.Count < 3 || stamp is not { Op: DecoOp.Add }
                || decorator == null || terrain?.Terrain == null)
                return null;

            if (!CropOf(decorator, terrain.Terrain, out var crop, out float plantHeight, out float range, out float density))
                return null;
            float height = plantHeight * HeightShare;

            // The outline where the plants stand dense, counter-clockwise in (x, z)
            var outline = Outline(spline, stamp);
            if (outline.Count < 3) return null;

            Matrix4x4.Invert(Transform.WorldMatrix, out var toLocal);
            var min = new Vector2(float.MaxValue);
            var max = new Vector2(float.MinValue);
            foreach (var p in outline) { min = Vector2.Min(min, p); max = Vector2.Max(max, p); }
            var ground = new GroundGrid((IHeightProvider)terrain, min, max);

            var verts = new List<Vector3>();
            var norms = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();

            // The top: the outline cut into grid cells, each piece triangulated and stood at the canopy's height over
            // the ground's crests (GroundGrid), its vertices shared between cells. uv (-1 - how far the crests stand
            // above the ground there, 1 + shell): a top vertex, lowered below the ground near the camera. Its normal is
            // the ground's, taken per pixel from the heightmap (crop_canopy.fx), so ridge and furrow still shade it.
            var up = Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitY, toLocal));
            var piece = new List<Vector2>();
            var scratch = new List<Vector2>();
            var welded = new Dictionary<(long, long), uint>();
            var topIndices = new List<uint>();
            var corners = new uint[16];
            for (float cz = MathF.Floor(min.Y / CellSize) * CellSize; cz < max.Y; cz += CellSize)
            {
                for (float cx = MathF.Floor(min.X / CellSize) * CellSize; cx < max.X; cx += CellSize)
                {
                    piece.Clear();
                    piece.AddRange(outline);
                    ClipAxis(piece, scratch, 0, cx, true);
                    ClipAxis(scratch, piece, 0, cx + CellSize, false);
                    ClipAxis(piece, scratch, 1, cz, true);
                    ClipAxis(scratch, piece, 1, cz + CellSize, false);
                    DropDuplicates(piece);
                    if (piece.Count < 3) continue;

                    var triangles = RuntimeMesh.EarClipTriangulate(piece);
                    if (triangles.Count < 3) continue;
                    if (corners.Length < piece.Count) corners = new uint[piece.Count * 2];
                    for (int k = 0; k < piece.Count; k++)
                    {
                        var p = piece[k];
                        var key = ((long)MathF.Round(p.X * 1000f), (long)MathF.Round(p.Y * 1000f));   // millimetres
                        if (!welded.TryGetValue(key, out uint index))
                        {
                            index = (uint)verts.Count;
                            welded[key] = index;
                            ground.Sample(p, out float floor, out float crests);
                            verts.Add(Vector3.Transform(new Vector3(p.X, crests + height, p.Y), toLocal));
                            norms.Add(up);
                            uvs.Add(new Vector2(-1f - (crests - floor), 1f));
                        }
                        corners[k] = index;
                    }
                    for (int t = 0; t + 2 < triangles.Count; t += 3)
                        AddTriangle(topIndices, verts, corners[triangles[t]], corners[triangles[t + 1]], corners[triangles[t + 2]], up);
                }
            }

            // The edge: the plants growing shorter towards it and leaning out over the headland, a slope from just under
            // the top's border (EdgeTuck inside it: no crack between them) down to the ground EdgeReach of the stamp's
            // falloff past the outline, where its plants thin out. Part of the top, so it has the top's shells and gaps
            // and its tufts make its outline ragged. uv.y 1 = its inner ring, 1.5 = its foot on the ground (crop_canopy.fx);
            // its normal leans out by the slope (the top's is straight up).
            // (A vertical wall in its place, however ragged its top and shaded its side, read as a planter box.)
            int n = outline.Count;
            uint edgeBase = (uint)verts.Count;
            float reach = MathF.Max(0.4f, EdgeReach * stamp.Falloff);
            for (int i = 0; i < n; i++)
            {
                var p = outline[i];
                var e0 = p - outline[(i - 1 + n) % n];
                var e1 = outline[(i + 1) % n] - p;
                var outward = new Vector2(e0.Y, -e0.X) + new Vector2(e1.Y, -e1.X);   // counter-clockwise: (dz, -dx) is out
                outward = outward.LengthSquared() > 1e-10f ? Vector2.Normalize(outward) : Vector2.UnitX;
                var slope = Vector3.Normalize(new Vector3(outward.X * height, reach + EdgeTuck, outward.Y * height));
                var normal = Vector3.Normalize(Vector3.TransformNormal(slope, toLocal));

                var inner = p - outward * EdgeTuck;
                var foot = p + outward * reach;
                ground.Sample(inner, out float innerFloor, out float innerCrests);
                ground.Sample(foot, out float footFloor, out float footCrests);
                verts.Add(Vector3.Transform(new Vector3(inner.X, innerCrests + height, inner.Y), toLocal));
                verts.Add(Vector3.Transform(new Vector3(foot.X, footCrests + height, foot.Y), toLocal));
                norms.Add(normal);
                norms.Add(normal);
                uvs.Add(new Vector2(-1f - (innerCrests - innerFloor), 1f));
                uvs.Add(new Vector2(-1f - (footCrests - footFloor), 1.5f));
            }
            for (int i = 0; i < n; i++)
            {
                uint inner0 = edgeBase + (uint)(i * 2), foot0 = inner0 + 1;
                uint inner1 = edgeBase + (uint)(((i + 1) % n) * 2), foot1 = inner1 + 1;
                var facing = (norms[(int)inner0] + norms[(int)inner1]) * 0.5f;
                AddTriangle(topIndices, verts, inner0, foot0, inner1, facing);
                AddTriangle(topIndices, verts, inner1, foot0, foot1, facing);
            }

            // Shells: the top and its edge again Shells - 1 times, each a layer of the crop (uv.y 1 + shell: the lowest
            // solid; above it only where a tuft reaches that high: crop_canopy.fx), so near enough it has depth instead of
            // a skin
            int topCount = verts.Count;
            indices.AddRange(topIndices);
            for (int shell = 1; shell < Shells; shell++)
            {
                uint offset = (uint)verts.Count;
                for (int v = 0; v < topCount; v++)
                {
                    verts.Add(verts[v]);
                    norms.Add(norms[v]);
                    uvs.Add(new Vector2(uvs[v].X, uvs[v].Y + shell));
                }
                foreach (uint index in topIndices) indices.Add(index + offset);
            }

            var parts = new List<MeshPart> { new MeshPart { Name = "Canopy", NumIndices = indices.Count, MaterialSlot = 0 } };
            var mesh = RuntimeMesh.BuildMesh(verts, norms, uvs, indices, parts, "CropCanopy_");
            if (mesh == null) return null;

            var origin = terrain.Transform?.WorldPosition ?? Vector3.Zero;
            SetInstance("Lift", height + Sink);
            SetInstance("Density", density);
            SetInstance("Range", range);
            SetInstance("TintBuffer", (float)terrain.DecoratorTintSrv);
            SetInstance("Group", (float)Math.Max(0, terrain.DecoratorSlotOf(decorator)));
            // The colours its plants are drawn with, root to ears (crop_canopy.fx PlantColour)
            int variantSlot = terrain.VariantSlotOf(crop);
            SetInstance("BladeColors", variantSlot >= 0 ? (float)terrain.BladeColorsSrv : 0f);
            SetInstance("VariantSlot", (float)Math.Max(0, variantSlot));
            SetInstance("HealthyColor", new Vector3(crop.HealthyColor.X, crop.HealthyColor.Y, crop.HealthyColor.Z));
            SetInstance("TerrainCorner", new Vector2(origin.X, origin.Z));
            SetInstance("Heightmap", (float)(terrain.Terrain.Heightmap?.BindlessIndex ?? 0u));
            SetInstance("TerrainSize", terrain.Terrain.TerrainSize);
            SetInstance("MaxHeight", terrain.Terrain.MaxHeight);

            // The way the field was sown (its rows', or a broadcast crop's drill and plough direction: along its length,
            // turned by RowAngle), for the streaks along it; rows (DecoStamp.RowSpacing): stripes of plants over the
            // soil, each as wide as the band its plants stand in and a plant's width
            bool rows = stamp.RowSpacing > 0.05f;
            float plantWidth = 0.5f * (crop.WidthRange.X + crop.WidthRange.Y);
            SetInstance("RowAcross", TerrainBaker.RowNormal(stamp));
            SetInstance("RowSpacing", rows ? stamp.RowSpacing : 0f);
            SetInstance("RowCover", rows ? Math.Clamp((stamp.RowWidth + plantWidth) / stamp.RowSpacing, 0.15f, 1f) : 1f);

            // Grain lies flat in patches where rain and wind beat it down (lodging)
            SetInstance("Lodging", IsGrain(crop) ? LodgedShare : 0f);

            // How many dropped plants a kept one covers for, at most: blades widen by the whole growth (grass_blades.fx,
            // up to 16), grain, flowers and spikes their heads by its square root (grass_plants.fx PlantGrowth, up to 4).
            // Taken as 16 for all, wheat stopped covering the ground at half the distance the canopy rose at, and the
            // canopy came up as a hard edge beyond bare soil.
            SetInstance("KeptCover", IsBlades(crop) ? 16f : 4f);

            // The plants' tall and short patches (grass_compute.hlsl DecoHeightFactor), on the canopy too
            SetInstance("HeightNoiseScale", crop.HeightNoiseScale);
            SetInstance("HeightNoiseAmount", Math.Clamp(crop.HeightNoiseAmount, 0f, 1f));
            SetInstance("Seed", (float)crop.Seed);
            return mesh;
        }

        /// <summary>
        /// The ground under a field on a 1 m grid, and over it the crests the canopy rides on: the ground smoothed over
        /// ±SmoothReach plus the most it rises above that within ±CrestReach. Far off the terrain is drawn from coarser
        /// vertices that bridge the furrows of ridge and furrow higher than they lie; a canopy that dipped into each
        /// furrow let the ground through in dashes along every other one. On even ground and slopes the crests are the
        /// ground.
        /// </summary>
        private sealed class GroundGrid
        {
            private const int SmoothReach = 4;          // m: more than half a ridge spacing, so a ridge evens out
            private const int CrestReach = 3;           // m: about half a ridge spacing, so every point sees a crest
            private const int Margin = SmoothReach + CrestReach + 1;

            private readonly float[] _floor;
            private readonly float[] _crests;
            private readonly int _x0, _z0, _width, _height;

            public GroundGrid(IHeightProvider heights, Vector2 min, Vector2 max)
            {
                _x0 = (int)MathF.Floor(min.X) - Margin;
                _z0 = (int)MathF.Floor(min.Y) - Margin;
                _width = (int)MathF.Ceiling(max.X) + Margin - _x0 + 1;
                _height = (int)MathF.Ceiling(max.Y) + Margin - _z0 + 1;
                _floor = new float[_width * _height];
                for (int z = 0; z < _height; z++)
                    for (int x = 0; x < _width; x++)
                        _floor[z * _width + x] = heights.GetHeight(new Vector3(_x0 + x, 0, _z0 + z));

                var smooth = Filter(Filter(_floor, SmoothReach, 1, false), SmoothReach, _width, false);
                var above = new float[_floor.Length];
                for (int i = 0; i < above.Length; i++) above[i] = _floor[i] - smooth[i];
                var most = Filter(Filter(above, CrestReach, 1, true), CrestReach, _width, true);
                _crests = new float[_floor.Length];
                for (int i = 0; i < _crests.Length; i++) _crests[i] = MathF.Max(_floor[i], smooth[i] + most[i]);
            }

            /// <summary>The ground and the crests over it at a world (x, z), bilinear between the grid's nodes.</summary>
            public void Sample(Vector2 p, out float floor, out float crests)
            {
                float fx = Math.Clamp(p.X - _x0, 0f, _width - 1.001f);
                float fz = Math.Clamp(p.Y - _z0, 0f, _height - 1.001f);
                int x = (int)fx, z = (int)fz;
                float tx = fx - x, tz = fz - z;
                floor = Bilinear(_floor, x, z, tx, tz);
                crests = MathF.Max(floor, Bilinear(_crests, x, z, tx, tz));
            }

            private float Bilinear(float[] grid, int x, int z, float tx, float tz)
            {
                int i = z * _width + x;
                float a = grid[i] + (grid[i + 1] - grid[i]) * tx;
                float b = grid[i + _width] + (grid[i + _width + 1] - grid[i + _width]) * tx;
                return a + (b - a) * tz;
            }

            // One pass of a box mean or a running max over ±reach nodes along rows (stride 1) or columns (stride width),
            // clamped at the grid's edges
            private float[] Filter(float[] src, int reach, int stride, bool max)
            {
                var dst = new float[src.Length];
                for (int z = 0; z < _height; z++)
                {
                    for (int x = 0; x < _width; x++)
                    {
                        int along = stride == 1 ? x : z, length = stride == 1 ? _width : _height;
                        int row = z * _width + x - along * stride;
                        float sum = 0f, most = float.MinValue;
                        for (int k = -reach; k <= reach; k++)
                        {
                            float v = src[row + Math.Clamp(along + k, 0, length - 1) * stride];
                            sum += v;
                            most = MathF.Max(most, v);
                        }
                        dst[z * _width + x] = max ? most : sum / (2 * reach + 1);
                    }
                }
                return dst;
            }
        }

        /// <summary>
        /// What the canopy needs of a decorator's plants: the variant that stands for them (the tallest: a crop has one),
        /// their mean height, how far they are drawn (the farthest variant: its last band's end, never past the
        /// terrain's DecorationRadius) and how many stand on a square metre at full cover (the densest variant), as the
        /// spawn counts them (grass_compute.hlsl ThinFacing). False if none of its variants is drawn.
        /// </summary>
        private static bool CropOf(TerrainDecorator decorator, Terrain terrain, out DecoratorVariant crop, out float height,
                                   out float range, out float density)
        {
            crop = null!;
            height = range = density = 0f;
            int controlWidth = Math.Max(1, terrain.EffectiveDecorationMapResolution);
            float tile = terrain.TerrainSize.X / controlWidth;
            int count = Math.Min(decorator.Variants.Count, TerrainDecorator.MaxVariants);
            for (int v = 0; v < count; v++)
            {
                var variant = decorator.Variants[v];
                if (variant == null || !variant.IsRenderable) continue;
                float mean = 0.5f * (variant.HeightRange.X + variant.HeightRange.Y);
                if (mean > height) { height = mean; crop = variant; }
                float end = terrain.DecorationRadius;
                foreach (var band in variant.Bands)
                    if (band != null) end = band.MaxDistance;        // the last band's end
                range = MathF.Max(range, MathF.Min(end, terrain.DecorationRadius));
                density = MathF.Max(density, 64f * variant.Density * terrain.DecorationDensity / (tile * tile));
            }
            return height > 0f && range > 0f && density > 0f;
        }

        /// <summary>
        /// The spline sampled into a counter-clockwise outline in world (x, z), grown by the stamp's Radius (times the
        /// spline's width there) and FalloffShare of its Falloff: a coverage stamp's falloff lies outside its edge, and
        /// the plants stand dense to about a third of the way out (their density is the coverage cubed).
        /// </summary>
        private static List<Vector2> Outline(Spline spline, DecoStamp stamp)
        {
            int samples = Math.Max(3, spline.SpanCount * SplineSamplesPerSpan);
            var points = new List<Vector2>(samples);
            var grow = new List<float>(samples);
            for (int i = 0; i < samples; i++)
            {
                float t = (float)i / samples;
                var p = spline.GetWorldPoint(t);
                var q = new Vector2(p.X, p.Z);
                if (points.Count > 0 && Vector2.DistanceSquared(points[^1], q) < 1e-4f) continue;
                points.Add(q);
                grow.Add(stamp.Radius * spline.GetWidth(t) + FalloffShare * stamp.Falloff);
            }

            // Counter-clockwise (shoelace sum positive); the growth runs along the outward normal (dz, -dx)
            if (SignedArea(points) < 0f) { points.Reverse(); grow.Reverse(); }
            int n = points.Count;
            var grown = new List<Vector2>(n);
            for (int i = 0; i < n; i++)
            {
                var e0 = points[i] - points[(i - 1 + n) % n];
                var e1 = points[(i + 1) % n] - points[i];
                var normal = new Vector2(e0.Y, -e0.X) + new Vector2(e1.Y, -e1.X);
                grown.Add(normal.LengthSquared() > 1e-10f ? points[i] + Vector2.Normalize(normal) * grow[i] : points[i]);
            }

            // No step longer than a grid cell, so the wall's top follows the ground as the top surface's edge does
            var outline = new List<Vector2>(n * 2);
            for (int i = 0; i < n; i++)
            {
                var a = grown[i];
                var b = grown[(i + 1) % n];
                int steps = Math.Max(1, (int)MathF.Ceiling(Vector2.Distance(a, b) / CellSize));
                for (int k = 0; k < steps; k++)
                    outline.Add(Vector2.Lerp(a, b, (float)k / steps));
            }
            return outline;
        }

        // A variant drawn as grain in any of its bands (wheat, barley, oats)
        private static bool IsGrain(DecoratorVariant variant)
        {
            if (variant.Mode == DecoratorMode.Grain) return true;
            foreach (var band in variant.Bands)
                if (band?.Mode == DecoratorMode.Grain) return true;
            return false;
        }

        // A variant whose last band draws blades (onions, stubble): what it shows from afar
        private static bool IsBlades(DecoratorVariant variant)
        {
            var mode = variant.Mode;
            foreach (var band in variant.Bands)
                if (band != null) mode = band.Mode;
            return mode == DecoratorMode.Blades;
        }

        private static float SignedArea(List<Vector2> poly)
        {
            float sum = 0f;
            for (int i = 0; i < poly.Count; i++)
            {
                var a = poly[i];
                var b = poly[(i + 1) % poly.Count];
                sum += a.X * b.Y - b.X * a.Y;
            }
            return 0.5f * sum;
        }

        /// <summary>
        /// One edge of the Sutherland-Hodgman clip: keeps the part of <paramref name="src"/> on one side of the line
        /// axis = bound (axis 0 = x, 1 = z; above or below it). The window is convex, so a field's outline clipped by
        /// a grid cell's four edges is the piece of the field in that cell.
        /// </summary>
        private static void ClipAxis(List<Vector2> src, List<Vector2> dst, int axis, float bound, bool keepAbove)
        {
            dst.Clear();
            int n = src.Count;
            for (int i = 0; i < n; i++)
            {
                var p = src[i];
                var q = src[(i + 1) % n];
                float dp = (axis == 0 ? p.X : p.Y) - bound;
                float dq = (axis == 0 ? q.X : q.Y) - bound;
                if (!keepAbove) { dp = -dp; dq = -dq; }
                if (dp >= 0f) dst.Add(p);
                if ((dp >= 0f) != (dq >= 0f)) dst.Add(Vector2.Lerp(p, q, dp / (dp - dq)));
            }
        }

        private static void DropDuplicates(List<Vector2> poly)
        {
            for (int i = poly.Count - 1; i >= 0 && poly.Count > 0; i--)
            {
                int next = (i + 1) % poly.Count;
                if (next != i && Vector2.DistanceSquared(poly[i], poly[next]) < 1e-6f) poly.RemoveAt(i);
            }
        }

        // A triangle wound so that it faces 'facing' (front faces have cross(v1 - v0, v2 - v0) along their normal)
        private static void AddTriangle(List<uint> indices, List<Vector3> verts, uint a, uint b, uint c, Vector3 facing)
        {
            var normal = Vector3.Cross(verts[(int)b] - verts[(int)a], verts[(int)c] - verts[(int)a]);
            if (normal.LengthSquared() < 1e-12f) return;
            if (Vector3.Dot(normal, facing) >= 0f) { indices.Add(a); indices.Add(b); indices.Add(c); }
            else { indices.Add(a); indices.Add(c); indices.Add(b); }
        }
    }
}
