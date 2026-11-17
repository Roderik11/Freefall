using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;
using DefaultValue = System.ComponentModel.DefaultValueAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    public enum RuntimeMeshHeightMode
    {
        /// <summary>Use Y values from the spline control points.</summary>
        Spline,
        /// <summary>All vertices at Y=0 (entity-relative).</summary>
        Flat,
        /// <summary>Sample terrain height at each vertex.</summary>
        Surface
    }

    /// <summary>
    /// Generates a mesh from a sibling Spline and assigns it to a sibling MeshRenderer.
    /// Closed spline: triangulated polygon fill with planar UVs.
    /// Open spline: strip extruded along spline with UVs following the curve.
    /// Regenerates on wake and when the spline changes.
    /// </summary>
    [Icon("icon_runtime_mesh.png")]
    public class RuntimeMesh : Component
    {
        // ═══════════════════════════
        // ── Properties ──
        // ═══════════════════════════
        private Spline Spline;

        /// <summary>Half-width of the generated strip (open spline only).</summary>
        [Category("Shape")]
        [DefaultValue(2f)]
        [ValueRange(0.1f, 50f)]
        public float Width = 2f;

        /// <summary>Extrusion height. 0 = flat surface only.</summary>
        [DefaultValue(0f)]
        [ValueRange(0f, 50f)]
        public float Height = 0f;

        /// <summary>How vertex Y is determined.</summary>
        [DefaultValue(RuntimeMeshHeightMode.Spline)]
        public RuntimeMeshHeightMode HeightMode = RuntimeMeshHeightMode.Spline;

        /// <summary>
        /// Lifts the whole generated mesh above the spline (after HeightMode). A water surface over a
        /// riverbed whose spline carries the bed heights uses the water depth here.
        /// </summary>
        [DefaultValue(0f)]
        [ValueRange(-10f, 10f)]
        public float Offset = 0f;

        /// <summary>
        /// Closed spline only: grows the filled polygon outward by this many meters (negative shrinks),
        /// like a closed stamp's Radius. A lake's water reaches past the bed outline up the shore.
        /// </summary>
        [DefaultValue(0f)]
        [ValueRange(-20f, 50f)]
        public float Expand = 0f;

        /// <summary>Smoothness: segments per spline span.</summary>
        [DefaultValue(8)]
        [ValueRange(2, 32)]
        public int Smoothness = 8;

        /// <summary>UV tiling scale. Higher = more texture repeats.</summary>
        [DefaultValue(1f)]
        [ValueRange(0.01f, 3f)]
        public float UVScale = 1f;

        /// <summary>
        /// Open spline with Height: length (m) over which the slab tapers down to ground level at the start,
        /// so a road meets open ground flush instead of with a step. 0 = full height with a closed end face.
        /// </summary>
        [DefaultValue(0f)]
        [ValueRange(0f, 20f)]
        public float StartRamp = 0f;

        /// <summary>Same as StartRamp, for the end of the spline.</summary>
        [DefaultValue(0f)]
        [ValueRange(0f, 20f)]
        public float EndRamp = 0f;

        /// <summary>Enable raised curbs along the sides (open spline only).</summary>
        [Category("Curbs")]
        [DefaultValue(false)]
        public bool EnableCurbs = false;

        /// <summary>Curb height above the surface.</summary>
        [DefaultValue(0.15f)]
        [ValueRange(0.01f, 1f)]
        public float CurbHeight = 0.15f;

        /// <summary>Curb width (horizontal extent outward from the strip edge).</summary>
        [DefaultValue(0.2f)]
        [ValueRange(0.05f, 2f)]
        public float CurbWidth = 0.2f;

        /// <summary>
        /// Closed spline: wherever the ground under the edge lies below the surface's base, a wall runs from the edge
        /// down into the ground (in the curb material, slot 1), so paving on falling ground stands on masonry instead
        /// of floating over it: an island beside a ramp, a terrace edge. Rebuilt when the terrain's heights change.
        /// </summary>
        [Category("Foundation")]
        [DefaultValue(false)]
        [Description("Closed spline: wherever the ground under the edge lies below the base, a wall in the curb material runs from the edge down into it, so paving on falling ground stands on masonry instead of floating. Rebuilt when the terrain's heights change.")]
        public bool Foundation = false;

        /// <summary>With Foundation and curbs: where the ground falls more than this below the base, the curb rises to a parapet (0 = never).</summary>
        [DefaultValue(1.0f)]
        [ValueRange(0f, 20f)]
        [Description("With Foundation and curbs: where the ground falls more than this below the base (m), the curb rises to a parapet. 0 = never.")]
        public float ParapetDrop = 1.0f;

        /// <summary>The parapet's top above the base, counted as CurbHeight is.</summary>
        [DefaultValue(1.15f)]
        [ValueRange(0f, 3f)]
        [Description("The parapet's top above the base (m), counted as Curb Height is.")]
        public float ParapetHeight = 1.15f;

        /// <summary>The parapet's thickness out from the edge.</summary>
        [DefaultValue(0.35f)]
        [ValueRange(0.05f, 2f)]
        [Description("The parapet's thickness out from the edge (m).")]
        public float ParapetWidth = 0.35f;

        /// <summary>How far a foundation reaches below the ground, so no slit shows at its foot.</summary>
        private const float FoundationDepth = 0.3f;


        // ═══════════════════════════
        // ── Internals ──
        // ═══════════════════════════

        private MeshRenderer _renderer;
        private Mesh _generatedMesh;

        private static readonly ProfilerMarker _generateMarker = ProfilerMarker.Get("RuntimeMesh generate");
        private static readonly ProfilerMarker _uploadMarker = ProfilerMarker.Get("RuntimeMesh upload");

        protected override void Early()
        {
            Spline = Spline ?? Entity.GetComponent<Spline>();
            _renderer = Entity.GetComponent<MeshRenderer>();
            Generate();
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.AddListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            MessageDispatcher.RemoveListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            if (_generatedMesh != null)
            {
                if (_renderer != null && _renderer.Mesh == _generatedMesh)
                    _renderer.Mesh = null;

                // Deferred: a draw enqueued earlier this frame (WaterBody.Draw) and frames still in flight
                // reference the mesh. Disposing it here crashed the editor when a lake was deleted.
                Engine.Device.DeferDispose(_generatedMesh);
                _generatedMesh = null;
            }
        }

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is not Spline spline) return;
            if (spline != Spline && !IsHoleSpline(spline)) return;
            Generate();
        }

        // A child's closed spline cut out of this closed mesh (RuntimeMeshHole)
        private bool IsHoleSpline(Spline spline)
            => Spline is { Closed: true } && spline.Entity?.Transform?.Parent?.Entity == Entity
               && spline.Entity.GetComponent<RuntimeMeshHole>() is { Gone: false };

        // The holes the last closed build cut, so a hole waking after it knows whether it is in
        private readonly HashSet<RuntimeMeshHole> _cut = new();

        internal bool Cuts(RuntimeMeshHole hole) => _cut.Contains(hole);

        // Surface mode samples the CPU HeightField once; it can be stale at load (cached bake that no longer
        // matches the scene's stamps) until the GPU bake is read back, so rebuild whenever it is replaced.
        private void OnTerrainHeightsChanged(Message msg)
        {
            // Surface mode reads the heights; a closed mesh's Foundation reaches down to them
            if (HeightMode != RuntimeMeshHeightMode.Surface && !(Foundation && Spline is { Closed: true })) return;

            // Heights changed somewhere else on the terrain: the mesh still fits
            if (msg.Data is TerrainHeightsChange { All: false } change && Spline != null
                && Spline.TryGetWorldBoundsXZ(Width, out var min, out var max) && !change.Overlaps(min, max))
                return;

            Generate();
        }

        public override void OnMemberChanged()
        {
            Generate();
        }


        /// <summary>
        /// Regenerate the mesh from the current spline state.
        /// The previous mesh is released through the device's deferred disposal, once the
        /// frames still in flight have finished with it.
        /// </summary>
        public void Generate()
        {
            // A MeshRenderer is optional: components that draw the mesh themselves (WaterBody) read GeneratedMesh
            if (Spline == null) return;
            if (Spline.Points.Count < 2) return;
            using var _ = _generateMarker.Auto();

            // Frames in flight (and draws enqueued earlier this frame, WaterBody.Draw) still reference the current
            // mesh. A spline drag regenerates every frame, so "two generations ago" is only two frames old — fewer
            // than the frames in flight; disposing it directly removed the device while dragging a river point.
            Engine.Device.DeferDispose(_generatedMesh);

            _generatedMesh = Spline.Closed
                ? GenerateClosed()
                : GenerateOpen();

            if (_generatedMesh != null && _renderer != null)
                _renderer.Mesh = _generatedMesh;
        }

        /// <summary>The current mesh (null until a spline with at least two points exists).</summary>
        [System.ComponentModel.Browsable(false)]
        public Mesh GeneratedMesh => _generatedMesh;

        // ═══════════════════════════════════════
        // ── Open Spline: Strip Along Curve ──
        // ═══════════════════════════════════════

        private Mesh GenerateOpen()
        {
            // Cross-sections along the spline (Smoothness per span), narrowed on the inside of tight bends
            var strip = SplineStrip.Sample(Spline, Smoothness, Width);
            int samples = strip.Count;
            var points = strip.Points;
            var tangents = strip.Tangents;
            var halfWidths = strip.HalfWidths;   // Width / 2, scaled by the spline's per-point width
            var rights = strip.Rights;

            // Apply height mode
            ApplyHeightMode(points);

            // Compute per-edge positions
            var leftPositions = new Vector3[samples];
            var rightPositions = new Vector3[samples];

            for (int i = 0; i < samples; i++)
            {
                leftPositions[i] = strip.Left(i);
                rightPositions[i] = strip.Right(i);
            }

            // Center-line arc length for UV mapping
            // Using center-line for both edges avoids shearing on curves.
            strip.RecomputeArcLengths();
            var arcLengths = strip.ArcLengths;

            // ── Per-sample slab height (Height, tapered by Start/EndRamp) ──
            bool extruded = Height > 0.001f;
            float totalArc = arcLengths[samples - 1];
            var heights = new float[samples];
            for (int i = 0; i < samples; i++)
                heights[i] = extruded ? SlabHeightAt(arcLengths[i], totalArc) : 0f;

            var verts = new List<Vector3>();
            var norms = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();

            // ── Main surface: 2 verts per sample (MeshPart 0) ──
            // Extruded: the slab top (the only visible upward face). Flat: the strip itself.
            for (int i = 0; i < samples; i++)
            {
                // Tilt the normal with the ramp slope so ramps shade correctly
                var up = Vector3.UnitY;
                if (extruded)
                {
                    int a = Math.Max(0, i - 1), b = Math.Min(samples - 1, i + 1);
                    float ds = arcLengths[b] - arcLengths[a];
                    if (ds > 1e-4f)
                    {
                        var fwd = Vector3.Normalize(new Vector3(tangents[i].X, 0, tangents[i].Z));
                        up = Vector3.Normalize(Vector3.UnitY - fwd * ((heights[b] - heights[a]) / ds));
                    }
                }

                var lift = Vector3.UnitY * heights[i];
                verts.Add(leftPositions[i] + lift);
                verts.Add(rightPositions[i] + lift);
                norms.Add(up);
                norms.Add(up);

                float u = arcLengths[i] * UVScale;
                uvs.Add(new Vector2(u, 0));
                uvs.Add(new Vector2(u, halfWidths[i] * 2f * UVScale));
            }

            // Triangulate strip
            for (int i = 0; i < samples - 1; i++)
            {
                uint bl = (uint)(i * 2);
                uint br = bl + 1;
                uint tl = (uint)((i + 1) * 2);
                uint tr = tl + 1;

                indices.Add(bl); indices.Add(br); indices.Add(tl);
                indices.Add(br); indices.Add(tr); indices.Add(tl);
            }

            int surfaceIndexCount = indices.Count;

            // ── Side walls + end caps (if Height > 0) ──
            if (extruded)
            {
                int wallBase = verts.Count;
                for (int i = 0; i < samples; i++)
                {
                    var right = rights[i];
                    var lift = Vector3.UnitY * heights[i];

                    // Left wall: faces outward (-right)
                    float wu = arcLengths[i] * UVScale;
                    verts.Add(leftPositions[i]); norms.Add(-right); uvs.Add(new Vector2(wu, 0));
                    verts.Add(leftPositions[i] + lift); norms.Add(-right); uvs.Add(new Vector2(wu, heights[i] * UVScale));

                    // Right wall: faces outward (+right)
                    verts.Add(rightPositions[i]); norms.Add(right); uvs.Add(new Vector2(wu, 0));
                    verts.Add(rightPositions[i] + lift); norms.Add(right); uvs.Add(new Vector2(wu, heights[i] * UVScale));
                }

                // Left wall tris
                for (int i = 0; i < samples - 1; i++)
                {
                    uint lt0 = (uint)(wallBase + i * 4 + 0);
                    uint lb0 = (uint)(wallBase + i * 4 + 1);
                    uint lt1 = (uint)(wallBase + (i + 1) * 4 + 0);
                    uint lb1 = (uint)(wallBase + (i + 1) * 4 + 1);
                    indices.Add(lt0); indices.Add(lb0); indices.Add(lt1);
                    indices.Add(lt1); indices.Add(lb0); indices.Add(lb1);
                }

                // Right wall tris
                for (int i = 0; i < samples - 1; i++)
                {
                    uint rt0 = (uint)(wallBase + i * 4 + 2);
                    uint rb0 = (uint)(wallBase + i * 4 + 3);
                    uint rt1 = (uint)(wallBase + (i + 1) * 4 + 2);
                    uint rb1 = (uint)(wallBase + (i + 1) * 4 + 3);
                    indices.Add(rt0); indices.Add(rt1); indices.Add(rb0);
                    indices.Add(rb0); indices.Add(rt1); indices.Add(rb1);
                }

                // End caps close the slab at both ends (with a ramp they shrink to a sliver)
                AddEndCap(0, -1f);
                AddEndCap(samples - 1, 1f);

                void AddEndCap(int i, float dir)
                {
                    var fwd = Vector3.Normalize(new Vector3(tangents[i].X, 0, tangents[i].Z)) * dir;
                    var lift = Vector3.UnitY * heights[i];
                    uint c = (uint)verts.Count;
                    verts.Add(leftPositions[i]); verts.Add(rightPositions[i]);
                    verts.Add(leftPositions[i] + lift); verts.Add(rightPositions[i] + lift);
                    for (int k = 0; k < 4; k++) norms.Add(fwd);
                    uvs.Add(new Vector2(0, 0)); uvs.Add(new Vector2(Width * UVScale, 0));
                    uvs.Add(new Vector2(0, heights[i] * UVScale)); uvs.Add(new Vector2(Width * UVScale, heights[i] * UVScale));
                    // Front faces have cross(v1-v0, v2-v0) along the normal (same as the surface strip)
                    var n = Vector3.Cross(verts[(int)c + 1] - verts[(int)c], verts[(int)c + 2] - verts[(int)c]);
                    if (Vector3.Dot(n, fwd) > 0)
                    {
                        indices.Add(c); indices.Add(c + 1); indices.Add(c + 2);
                        indices.Add(c + 1); indices.Add(c + 3); indices.Add(c + 2);
                    }
                    else
                    {
                        indices.Add(c); indices.Add(c + 2); indices.Add(c + 1);
                        indices.Add(c + 1); indices.Add(c + 2); indices.Add(c + 3);
                    }
                }
            }

            // Build surface MeshPart (slot 0)
            var parts = new List<MeshPart>();
            parts.Add(new MeshPart
            {
                Name = "Surface",
                NumIndices = surfaceIndexCount,
                MaterialSlot = 0
            });

            if (indices.Count > surfaceIndexCount)
            {
                parts.Add(new MeshPart
                {
                    Name = "Walls",
                    BaseIndex = surfaceIndexCount,
                    NumIndices = indices.Count - surfaceIndexCount,
                    MaterialSlot = 0
                });
            }

            // ── Curbs (separate MeshPart, slot 1) ──
            int curbStartIndex = indices.Count;
            if (EnableCurbs)
                GenerateCurbs(strip, verts, norms, uvs, indices);

            if (indices.Count > curbStartIndex)
            {
                parts.Add(new MeshPart
                {
                    Name = "Curbs",
                    BaseIndex = curbStartIndex,
                    NumIndices = indices.Count - curbStartIndex,
                    MaterialSlot = 1
                });
            }

            return BuildMesh(verts, norms, uvs, indices, parts);
        }

        /// <summary>
        /// Generate curb geometry along both sides of the strip.
        /// Each curb is an L-shaped profile: inner wall, top ledge, outer wall.
        /// Curbs sit at surface level and extend upward.
        /// </summary>
        private void GenerateCurbs(SplineStrip strip,
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices)
        {
            int samples = strip.Count;
            var rights = strip.Rights;
            var arcLengths = strip.ArcLengths;

            // For each side (left=-1, right=+1)
            for (int side = -1; side <= 1; side += 2)
            {
                int curbBase = verts.Count;
                float s = side;
                var outward = Vector3.Zero; // will be set per-sample

                for (int i = 0; i < samples; i++)
                {
                    var right = rights[i];
                    outward = right * s; // direction away from road center

                    var innerBot = side > 0 ? strip.Right(i) : strip.Left(i);
                    var innerTop = innerBot + Vector3.UnitY * CurbHeight;
                    var outerTop = innerTop + right * (CurbWidth * s);
                    var outerBot = innerBot + right * (CurbWidth * s);

                    float u = arcLengths[i] * UVScale;

                    // 6 verts per sample — duplicated at hard edges for flat shading
                    // Inner wall (0, 1)
                    verts.Add(innerBot); norms.Add(-outward); uvs.Add(new Vector2(u, 0));
                    verts.Add(innerTop); norms.Add(-outward); uvs.Add(new Vector2(u, CurbHeight));
                    // Top ledge (2, 3) — duplicates innerTop and outerTop with +Y normal
                    verts.Add(innerTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(u, 0));
                    verts.Add(outerTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(u, CurbWidth));
                    // Outer wall (4, 5)
                    verts.Add(outerTop); norms.Add(outward); uvs.Add(new Vector2(u, CurbHeight));
                    verts.Add(outerBot); norms.Add(outward); uvs.Add(new Vector2(u, 0));
                }

                // Triangulate — 6 verts per sample
                for (int i = 0; i < samples - 1; i++)
                {
                    uint c0 = (uint)(curbBase + i * 6);
                    uint c1 = (uint)(curbBase + (i + 1) * 6);

                    if (side > 0)
                    {
                        // Right curb — inner wall
                        indices.Add(c0 + 0); indices.Add(c0 + 1); indices.Add(c1 + 0);
                        indices.Add(c1 + 0); indices.Add(c0 + 1); indices.Add(c1 + 1);
                        // Top ledge
                        indices.Add(c0 + 2); indices.Add(c0 + 3); indices.Add(c1 + 2);
                        indices.Add(c1 + 2); indices.Add(c0 + 3); indices.Add(c1 + 3);
                        // Outer wall
                        indices.Add(c0 + 4); indices.Add(c0 + 5); indices.Add(c1 + 4);
                        indices.Add(c1 + 4); indices.Add(c0 + 5); indices.Add(c1 + 5);
                    }
                    else
                    {
                        // Left curb
                        indices.Add(c0 + 0); indices.Add(c1 + 0); indices.Add(c0 + 1);
                        indices.Add(c1 + 0); indices.Add(c1 + 1); indices.Add(c0 + 1);

                        indices.Add(c0 + 2); indices.Add(c1 + 2); indices.Add(c0 + 3);
                        indices.Add(c1 + 2); indices.Add(c1 + 3); indices.Add(c0 + 3);

                        indices.Add(c0 + 4); indices.Add(c1 + 4); indices.Add(c0 + 5);
                        indices.Add(c1 + 4); indices.Add(c1 + 5); indices.Add(c0 + 5);
                    }
                }
            }
        }

        /// <summary>Lowest slab height at a ramped end: just above the ground so the tip doesn't z-fight it.</summary>
        private const float RampTipHeight = 0.01f;

        /// <summary>Slab height at arc length s: Height, eased down to RampTipHeight within Start/EndRamp of the ends.</summary>
        private float SlabHeightAt(float s, float totalArc)
        {
            float f = 1f;
            if (StartRamp > 0.001f) f = Math.Min(f, SmoothStep(s / StartRamp));
            if (EndRamp > 0.001f) f = Math.Min(f, SmoothStep((totalArc - s) / EndRamp));
            return RampTipHeight + (Height - RampTipHeight) * f;

            static float SmoothStep(float x)
            {
                x = Math.Clamp(x, 0f, 1f);
                return x * x * (3f - 2f * x);
            }
        }

        // ═══════════════════════════════════════════
        // ── Closed Spline: Polygon Fill ──
        // ═══════════════════════════════════════════

        private Mesh GenerateClosed()
        {
            _cut.Clear();

            // Sample the closed spline into a polygon
            int samples = Spline.SpanCount * Smoothness;
            if (samples < 3) samples = 3;

            var polygon2D = new List<Vector2>();
            var heights = new List<float>(samples);

            for (int i = 0; i < samples; i++)
            {
                float t = (float)i / samples; // don't include endpoint (closed = wraps)
                var p = Spline.GetPoint(t);
                polygon2D.Add(new Vector2(p.X, p.Z));
                heights.Add(p.Y);
            }

            // Ensure CCW winding
            if (GetSignedArea(polygon2D) > 0)
            {
                polygon2D.Reverse();
                heights.Reverse();
            }

            // Grow / shrink the outline along the averaged outward normal of its two edges
            if (MathF.Abs(Expand) > 0.001f)
            {
                int count = polygon2D.Count;
                var grown = new List<Vector2>(count);
                for (int i = 0; i < count; i++)
                {
                    var e0 = polygon2D[i] - polygon2D[(i - 1 + count) % count];
                    var e1 = polygon2D[(i + 1) % count] - polygon2D[i];

                    // CCW polygon: outward = (dz, -dx), same as the curbs below
                    var normal = new Vector2(e0.Y, -e0.X) + new Vector2(e1.Y, -e1.X);
                    grown.Add(normal.LengthSquared() > 1e-10f
                        ? polygon2D[i] + Vector2.Normalize(normal) * Expand
                        : polygon2D[i]);
                }
                polygon2D = grown;
            }

            // Rings: the outline, then each hole cut by a child's RuntimeMeshHole. A hole winds the other way round, so
            // every ring has the surface on its left: the walls' and curbs' (dz, -dx) faces out of the surface on each.
            var rings = new List<(int Start, int Count)> { (0, polygon2D.Count) };
            foreach (var (points, ys, hole) in CollectHoles(polygon2D))
            {
                rings.Add((polygon2D.Count, points.Count));
                polygon2D.AddRange(points);
                heights.AddRange(ys);
                _cut.Add(hole);
            }

            // One polygon for the ear clipper: each hole joined to the outline by a bridge there and back
            List<int> triangles;
            if (rings.Count == 1)
                triangles = EarClipTriangulate(polygon2D);
            else
            {
                var merged = BridgeHoles(polygon2D, rings);
                triangles = EarClipTriangulate(merged.ConvertAll(i => polygon2D[i]), rotate: true);
                for (int i = 0; i < triangles.Count; i++) triangles[i] = merged[triangles[i]];
            }
            if (triangles == null || triangles.Count < 3) return null;

            int n = polygon2D.Count;
            var verts = new List<Vector3>();
            var norms = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();
            var parts = new List<MeshPart>();

            // Apply height mode to Y values
            var yValues = new float[n];
            if (HeightMode == RuntimeMeshHeightMode.Flat)
            {
                // all zero
            }
            else if (HeightMode == RuntimeMeshHeightMode.Surface)
            {
                var terrain = EntityManager.FindComponent<TerrainRenderer>() as IHeightProvider;
                var worldMatrix = Transform?.Matrix ?? Matrix4x4.Identity;
                for (int i = 0; i < n; i++)
                {
                    var worldPos = Vector3.Transform(
                        new Vector3(polygon2D[i].X, 0, polygon2D[i].Y), worldMatrix);
                    yValues[i] = terrain?.GetHeight(worldPos) ?? 0;
                    // Convert back to local Y
                    yValues[i] -= Transform?.Position.Y ?? 0;
                }
            }
            else // Spline
            {
                for (int i = 0; i < n; i++)
                    yValues[i] = heights[i];
            }

            if (Offset != 0f)
            {
                for (int i = 0; i < n; i++)
                    yValues[i] += Offset;
            }

            // ── Base surface ──
            // With extrusion this is the underside (faces -Y; the top face is emitted with the walls below).
            // Without extrusion it is the only face, so it must face +Y or the fill is invisible from above.
            bool flatOnly = Height <= 0.001f;
            for (int i = 0; i < n; i++)
            {
                verts.Add(new Vector3(polygon2D[i].X, yValues[i], polygon2D[i].Y));
                norms.Add(flatOnly ? Vector3.UnitY : -Vector3.UnitY);
                uvs.Add(polygon2D[i] * UVScale);
            }

            for (int i = 0; i < triangles.Count / 3; i++)
            {
                if (flatOnly)
                {
                    indices.Add((uint)triangles[i * 3 + 2]);
                    indices.Add((uint)triangles[i * 3 + 1]);
                    indices.Add((uint)triangles[i * 3 + 0]);
                }
                else
                {
                    indices.Add((uint)triangles[i * 3 + 0]);
                    indices.Add((uint)triangles[i * 3 + 1]);
                    indices.Add((uint)triangles[i * 3 + 2]);
                }
            }

            int surfaceIndexCount = indices.Count;

            // ── Walls (if Height > 0): round the outline and round every hole ──
            if (Height > 0.001f)
            {
                foreach (var (start, count) in rings)
                {
                    for (int k = 0; k < count; k++)
                    {
                        int i = start + k;
                        int next = start + (k + 1) % count;
                        var p0 = polygon2D[i];
                        var p1 = polygon2D[next];

                        var edge = new Vector2(p1.X - p0.X, p1.Y - p0.Y);
                        var normal = Vector3.Normalize(new Vector3(edge.Y, 0, -edge.X));

                        float edgeLen = edge.Length();
                        uint vi = (uint)verts.Count;

                        verts.Add(new Vector3(p0.X, yValues[i], p0.Y));
                        verts.Add(new Vector3(p1.X, yValues[next], p1.Y));
                        verts.Add(new Vector3(p0.X, yValues[i] + Height, p0.Y));
                        verts.Add(new Vector3(p1.X, yValues[next] + Height, p1.Y));

                        norms.Add(normal); norms.Add(normal); norms.Add(normal); norms.Add(normal);

                        float uLen = edgeLen * UVScale;
                        float vH = Height * UVScale;
                        uvs.Add(new Vector2(0, 0));
                        uvs.Add(new Vector2(uLen, 0));
                        uvs.Add(new Vector2(0, vH));
                        uvs.Add(new Vector2(uLen, vH));

                        indices.Add(vi + 0); indices.Add(vi + 2); indices.Add(vi + 1);
                        indices.Add(vi + 1); indices.Add(vi + 2); indices.Add(vi + 3);
                    }
                }

                // Top face
                int topBase = verts.Count;
                for (int i = 0; i < n; i++)
                {
                    verts.Add(new Vector3(polygon2D[i].X, yValues[i] + Height, polygon2D[i].Y));
                    norms.Add(Vector3.UnitY);
                    uvs.Add(polygon2D[i] * UVScale);
                }
                for (int i = 0; i < triangles.Count / 3; i++)
                {
                    indices.Add((uint)(topBase + triangles[i * 3 + 2]));
                    indices.Add((uint)(topBase + triangles[i * 3 + 1]));
                    indices.Add((uint)(topBase + triangles[i * 3 + 0]));
                }
            }

            parts.Add(new MeshPart
            {
                Name = "Surface",
                NumIndices = surfaceIndexCount,
                MaterialSlot = 0
            });

            if (indices.Count > surfaceIndexCount)
            {
                parts.Add(new MeshPart
                {
                    Name = "Walls",
                    BaseIndex = surfaceIndexCount,
                    NumIndices = indices.Count - surfaceIndexCount,
                    MaterialSlot = 0
                });
            }

            // Outward normals per vertex of every ring (average of the adjacent edge normals): out of the surface,
            // into the hole on a hole's ring
            var outwardAt = new Vector3[n];
            foreach (var (start, count) in rings)
                for (int k = 0; k < count; k++)
                {
                    int i = start + k;
                    var e0 = polygon2D[i] - polygon2D[start + (k - 1 + count) % count];
                    var e1 = polygon2D[start + (k + 1) % count] - polygon2D[i];
                    // Outward normal in XZ: for CCW polygon, outward = (dz, -dx)
                    var avg = new Vector2(e0.Y, -e0.X) + new Vector2(e1.Y, -e1.X);
                    avg = avg.LengthSquared() > 1e-12f ? Vector2.Normalize(avg) : Vector2.UnitX;
                    outwardAt[i] = new Vector3(avg.X, 0, avg.Y);
                }

            // Foundation: the ground under the edge (at the curb's outer face); where it falls further than ParapetDrop
            // below the base, the curb there is a parapet
            float[] ground = Foundation ? GroundUnderEdge(polygon2D, outwardAt, EnableCurbs ? CurbWidth : 0f) : null;
            var curbHeight = new float[n];
            var curbWidth = new float[n];
            for (int i = 0; i < n; i++)
            {
                bool parapet = ground != null && ParapetHeight > 0f && ParapetDrop > 0f && yValues[i] - ground[i] > ParapetDrop;
                curbHeight[i] = parapet ? ParapetHeight : CurbHeight;
                curbWidth[i] = parapet ? ParapetWidth : CurbWidth;
            }

            // ── Perimeter curbs: round the outline and round every hole (into it) ──
            int curbStartIndex = indices.Count;
            if (EnableCurbs)
            {
                foreach (var (start, count) in rings)
                {
                    int curbBase = verts.Count;
                    var outwards = new Vector3[count];
                    for (int k = 0; k < count; k++) outwards[k] = outwardAt[start + k];

                    // Perimeter arc lengths
                    var perimArc = new float[count + 1]; // +1 for wrap
                    perimArc[0] = 0;
                    for (int k = 1; k <= count; k++)
                    {
                        int curr = start + k % count;
                        int prev = start + (k - 1) % count;
                        perimArc[k] = perimArc[k - 1] + Vector2.Distance(polygon2D[curr], polygon2D[prev]);
                    }

                    // Emit 6 verts per perimeter vertex (same L-profile)
                    for (int k = 0; k < count; k++)
                    {
                        int i = start + k;
                        var pos = new Vector3(polygon2D[i].X, yValues[i], polygon2D[i].Y);
                        var outward = outwards[k];

                        float height = curbHeight[i], width = curbWidth[i];
                        var innerBot = pos;
                        var innerTop = innerBot + Vector3.UnitY * height;
                        var outerTop = innerTop + outward * width;
                        var outerBot = innerBot + outward * width;

                        float u = perimArc[k] * UVScale;

                        // 6 verts per sample — duplicated at hard edges for flat shading
                        // Inner wall (0, 1)
                        verts.Add(innerBot); norms.Add(-outward); uvs.Add(new Vector2(u, 0));
                        verts.Add(innerTop); norms.Add(-outward); uvs.Add(new Vector2(u, height));
                        // Top ledge (2, 3)
                        verts.Add(innerTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(u, 0));
                        verts.Add(outerTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(u, width));
                        // Outer wall (4, 5)
                        verts.Add(outerTop); norms.Add(outward); uvs.Add(new Vector2(u, height));
                        verts.Add(outerBot); norms.Add(outward); uvs.Add(new Vector2(u, 0));
                    }

                    // Triangulate — wraps around, 6 verts per sample
                    for (int k = 0; k < count; k++)
                    {
                        uint c0 = (uint)(curbBase + k * 6);
                        uint c1 = (uint)(curbBase + ((k + 1) % count) * 6);

                        indices.Add(c0 + 0); indices.Add(c1 + 0); indices.Add(c0 + 1);
                        indices.Add(c1 + 0); indices.Add(c1 + 1); indices.Add(c0 + 1);

                        indices.Add(c0 + 2); indices.Add(c1 + 2); indices.Add(c0 + 3);
                        indices.Add(c1 + 2); indices.Add(c1 + 3); indices.Add(c0 + 3);

                        indices.Add(c0 + 4); indices.Add(c1 + 4); indices.Add(c0 + 5);
                        indices.Add(c1 + 4); indices.Add(c1 + 5); indices.Add(c0 + 5);
                    }
                }
            }

            if (indices.Count > curbStartIndex)
            {
                parts.Add(new MeshPart
                {
                    Name = "Curbs",
                    BaseIndex = curbStartIndex,
                    NumIndices = indices.Count - curbStartIndex,
                    MaterialSlot = 1
                });
            }

            // ── Foundation: the curb's outer face (the slab's edge without curbs) carried down into the ground ──
            int foundationStart = indices.Count;
            if (ground != null)
            {
                foreach (var (start, count) in rings)
                {
                    float arc = 0;
                    for (int k = 0; k < count; k++)
                    {
                        int i = start + k, j = start + (k + 1) % count;
                        float len = Vector2.Distance(polygon2D[i], polygon2D[j]);
                        float footI = MathF.Min(yValues[i], ground[i] - FoundationDepth);
                        float footJ = MathF.Min(yValues[j], ground[j] - FoundationDepth);
                        if (yValues[i] - footI < 0.01f && yValues[j] - footJ < 0.01f) { arc += len; continue; }

                        float wi = EnableCurbs ? curbWidth[i] : 0f, wj = EnableCurbs ? curbWidth[j] : 0f;
                        var topI = new Vector3(polygon2D[i].X, yValues[i], polygon2D[i].Y) + outwardAt[i] * wi;
                        var topJ = new Vector3(polygon2D[j].X, yValues[j], polygon2D[j].Y) + outwardAt[j] * wj;
                        var edge = polygon2D[j] - polygon2D[i];
                        var normal = Vector3.Normalize(new Vector3(edge.Y, 0, -edge.X));

                        uint v0 = (uint)verts.Count;
                        verts.Add(new Vector3(topI.X, footI, topI.Z));
                        verts.Add(new Vector3(topJ.X, footJ, topJ.Z));
                        verts.Add(topI);
                        verts.Add(topJ);
                        norms.Add(normal); norms.Add(normal); norms.Add(normal); norms.Add(normal);
                        uvs.Add(new Vector2(arc * UVScale, footI * UVScale));
                        uvs.Add(new Vector2((arc + len) * UVScale, footJ * UVScale));
                        uvs.Add(new Vector2(arc * UVScale, yValues[i] * UVScale));
                        uvs.Add(new Vector2((arc + len) * UVScale, yValues[j] * UVScale));
                        indices.Add(v0 + 0); indices.Add(v0 + 2); indices.Add(v0 + 1);
                        indices.Add(v0 + 1); indices.Add(v0 + 2); indices.Add(v0 + 3);
                        arc += len;
                    }
                }
            }

            if (indices.Count > foundationStart)
            {
                parts.Add(new MeshPart
                {
                    Name = "Foundation",
                    BaseIndex = foundationStart,
                    NumIndices = indices.Count - foundationStart,
                    MaterialSlot = 1
                });
            }

            return BuildMesh(verts, norms, uvs, indices, parts);
        }

        /// <summary>
        /// The ground's height, in this entity's space, under each outline vertex moved out by <paramref name="offset"/>;
        /// null until the terrain's heights are in (the mesh is rebuilt when they arrive).
        /// </summary>
        private float[] GroundUnderEdge(List<Vector2> polygon, Vector3[] outward, float offset)
        {
            var terrain = EntityManager.FindComponent<TerrainRenderer>();
            if (terrain?.Terrain?.Pages is not { HasHeights: true } || Transform == null) return null;
            var toWorld = Transform.Matrix;
            if (!Matrix4x4.Invert(toWorld, out var toLocal)) return null;
            var result = new float[polygon.Count];
            for (int i = 0; i < polygon.Count; i++)
            {
                var world = Vector3.Transform(new Vector3(polygon[i].X, 0f, polygon[i].Y) + outward[i] * offset, toWorld);
                float height = terrain.GetHeight(world);
                result[i] = Vector3.Transform(new Vector3(world.X, height, world.Z), toLocal).Y;
            }
            return result;
        }

        // ═══════════════════════════
        // ── Holes ──
        // ═══════════════════════════

        /// <summary>
        /// The closed splines of the direct children that carry a RuntimeMeshHole, sampled like the outline into this
        /// entity's space and wound clockwise. A hole that is not wholly inside the outline, or that overlaps a hole
        /// taken before it, is left out with a warning.
        /// </summary>
        private List<(List<Vector2> Points, List<float> Heights, RuntimeMeshHole Hole)> CollectHoles(List<Vector2> outline)
        {
            var holes = new List<(List<Vector2> Points, List<float> Heights, RuntimeMeshHole Hole)>();
            var self = Transform;
            if (self == null || self.Count == 0) return holes;
            if (!Matrix4x4.Invert(self.Matrix, out var toLocal)) return holes;

            foreach (Transform child in self)
            {
                var hole = child.Entity?.GetComponent<RuntimeMeshHole>();
                if (hole == null || hole.Gone || !hole.Enabled) continue;
                var spline = child.Entity.GetComponent<Spline>();
                if (spline == null || !spline.Closed || spline.Points.Count < 3) continue;

                var toThis = child.Matrix * toLocal;
                int count = Math.Max(3, spline.SpanCount * Smoothness);
                var points = new List<Vector2>(count);
                var ys = new List<float>(count);
                for (int i = 0; i < count; i++)
                {
                    var p = Vector3.Transform(spline.GetPoint((float)i / count), toThis);
                    points.Add(new Vector2(p.X, p.Z));
                    ys.Add(p.Y);
                }
                if (GetSignedArea(points) < 0)       // the outline's opposite (see GenerateClosed)
                {
                    points.Reverse();
                    ys.Reverse();
                }

                bool fits = points.TrueForAll(p => Inside(p, outline));
                foreach (var other in holes)
                    if (points.Exists(p => Inside(p, other.Points)) || other.Points.Exists(p => Inside(p, points)))
                        fits = false;
                if (!fits)
                {
                    Debug.LogWarning("RuntimeMesh", $"{Entity?.Name}: the hole '{child.Entity.Name}' is not wholly inside the outline or overlaps another hole; it is left out");
                    continue;
                }
                holes.Add((points, ys, hole));
            }
            return holes;
        }

        private static bool Inside(Vector2 p, List<Vector2> polygon)
        {
            bool inside = false;
            for (int i = 0, j = polygon.Count - 1; i < polygon.Count; j = i++)
            {
                var a = polygon[i]; var b = polygon[j];
                if ((a.Y > p.Y) != (b.Y > p.Y) && p.X < (b.X - a.X) * (p.Y - a.Y) / (b.Y - a.Y) + a.X)
                    inside = !inside;
            }
            return inside;
        }

        /// <summary>
        /// The outline (ring 0) with every hole joined into it by a bridge there and back, as one list of indices into
        /// <paramref name="points"/> for the ear clipper (Eberly, "Triangulation by Ear Clipping", §3). Holes go in from
        /// the rightmost, each bridged from its rightmost vertex to a vertex of the polygon so far that it sees along +X.
        /// </summary>
        private static List<int> BridgeHoles(List<Vector2> points, List<(int Start, int Count)> rings)
        {
            var polygon = new List<int>(points.Count + 2 * rings.Count);
            for (int i = 0; i < rings[0].Count; i++) polygon.Add(rings[0].Start + i);

            int Rightmost(int r)
            {
                var (s, c) = rings[r];
                int m = s;
                for (int i = s + 1; i < s + c; i++) if (points[i].X > points[m].X) m = i;
                return m;
            }
            var order = new List<int>();
            for (int r = 1; r < rings.Count; r++) order.Add(r);
            order.Sort((a, b) => points[Rightmost(b)].X.CompareTo(points[Rightmost(a)].X));

            foreach (int r in order)
            {
                var (s, c) = rings[r];
                int mi = Rightmost(r);
                var M = points[mi];

                // The nearest edge straight to the right of M
                int edge = -1;
                float nearest = float.MaxValue;
                for (int k = 0; k < polygon.Count; k++)
                {
                    var a = points[polygon[k]];
                    var b = points[polygon[(k + 1) % polygon.Count]];
                    if ((a.Y > M.Y) == (b.Y > M.Y)) continue;
                    float x = a.X + (M.Y - a.Y) * (b.X - a.X) / (b.Y - a.Y);
                    if (x < M.X || x >= nearest) continue;
                    nearest = x;
                    edge = k;
                }
                if (edge < 0) continue;      // not inside after all: CollectHoles keeps such holes out

                var I = new Vector2(nearest, M.Y);
                int ka = edge, kb = (edge + 1) % polygon.Count;
                int pick = points[polygon[ka]].X > points[polygon[kb]].X ? ka : kb;
                var P = points[polygon[pick]];

                // A reflex vertex inside the triangle M, I, P hides P from M: bridge to the one closest to the ray instead
                float bestAngle = float.MaxValue, bestDist = float.MaxValue;
                int choice = pick;
                for (int k = 0; k < polygon.Count; k++)
                {
                    if (k == pick) continue;
                    var R = points[polygon[k]];
                    if (R.X < M.X || R == P) continue;
                    var prev = points[polygon[(k - 1 + polygon.Count) % polygon.Count]];
                    var next = points[polygon[(k + 1) % polygon.Count]];
                    if (Cross2D(R - prev, next - R) >= 0) continue;          // convex or straight
                    if (!PointInTriangle(R, M, I, P)) continue;
                    var d = R - M;
                    float angle = MathF.Abs(MathF.Atan2(d.Y, d.X)), dist = d.LengthSquared();
                    if (angle < bestAngle || (angle == bestAngle && dist < bestDist))
                    {
                        bestAngle = angle;
                        bestDist = dist;
                        choice = k;
                    }
                }

                // ..., V, M, round the hole back to M, V, ...
                var splice = new List<int>(c + 2);
                for (int i = 0; i < c; i++) splice.Add(s + (mi - s + i) % c);
                splice.Add(mi);
                splice.Add(polygon[choice]);
                polygon.InsertRange(choice + 1, splice);
            }
            return polygon;
        }

        // ═══════════════════════════
        // ── Height Mode ──
        // ═══════════════════════════

        private void ApplyHeightMode(Vector3[] points)
        {
            switch (HeightMode)
            {
                case RuntimeMeshHeightMode.Flat:
                    for (int i = 0; i < points.Length; i++)
                        points[i] = new Vector3(points[i].X, 0, points[i].Z);
                    break;

                case RuntimeMeshHeightMode.Surface:
                    var terrain = EntityManager.FindComponent<TerrainRenderer>() as IHeightProvider;
                    if (terrain == null) break;
                    // The world transform, not the local one: under a parent that stands somewhere (a place's group)
                    // the local matrix sampled the terrain at the entity's coordinates in its parent's space, and
                    // the local Y left the mesh the parent's height too high (TestWorld's Spurwatch walls stood at
                    // y 225 over ground at 125).
                    var worldMatrix = Transform?.WorldMatrix ?? Matrix4x4.Identity;
                    float localY = Transform?.WorldPosition.Y ?? 0;
                    for (int i = 0; i < points.Length; i++)
                    {
                        var worldPos = Vector3.Transform(points[i], worldMatrix);
                        float h = terrain.GetHeight(worldPos) - localY;
                        points[i] = new Vector3(points[i].X, h, points[i].Z);
                    }
                    break;

                // Spline: keep Y as-is
            }

            if (Offset != 0f)
            {
                for (int i = 0; i < points.Length; i++)
                    points[i].Y += Offset;
            }
        }

        // ═══════════════════════════════════════
        // ── Mesh Construction ──
        // ═══════════════════════════════════════

        /// <param name="boundsPerPart">Bound each part by the vertices its indices reach (WaterBody's chunks: the culling
        /// and the shadow receivers test each part's sphere); false = every part gets the whole mesh's bounds.</param>
        internal static Mesh BuildMesh(List<Vector3> verts, List<Vector3> norms,
            List<Vector2> uvs, List<uint> indices, List<MeshPart> parts, string namePrefix = "RuntimeMesh_",
            bool boundsPerPart = false, bool pushNormals = true)
        {
            if (verts.Count == 0 || indices.Count == 0) return null;

            Mesh mesh;
            using (_uploadMarker.Auto())
                mesh = new Mesh(Engine.Device, verts.ToArray(), norms.ToArray(), uvs.ToArray(), indices.ToArray(), pushNormals);

            mesh.BoundingBox = ComputeBounds(verts);
            mesh.Guid = Guid.NewGuid().ToString("N");
            mesh.Name = namePrefix + mesh.Guid;
            mesh.IsDynamic = true;

            foreach (var part in parts)
            {
                if (boundsPerPart)
                    SetPartBounds(part, verts, indices);
                else
                {
                    part.BoundingBox = mesh.BoundingBox;
                    part.BoundingSphere = mesh.LocalBoundingSphere;
                }
                mesh.MeshParts.Add(part);
            }

            return mesh;
        }

        // The part's box over the vertices its indices reach, and the sphere about the box's centre that holds them
        private static void SetPartBounds(MeshPart part, List<Vector3> verts, List<uint> indices)
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            int end = Math.Min(indices.Count, part.BaseIndex + part.NumIndices);
            for (int i = part.BaseIndex; i < end; i++)
            {
                var v = verts[(int)indices[i] + part.BaseVertex];
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            if (end <= part.BaseIndex) min = max = Vector3.Zero;

            var center = (min + max) * 0.5f;
            float radiusSq = 0f;
            for (int i = part.BaseIndex; i < end; i++)
                radiusSq = MathF.Max(radiusSq, Vector3.DistanceSquared(verts[(int)indices[i] + part.BaseVertex], center));
            part.BoundingBox = new BoundingBox(min, max);
            part.BoundingSphere = new Vector4(center, MathF.Sqrt(radiusSq));
        }

        // ═══════════════════════════════════════
        // ── Ear-Clipping Triangulation ──
        // ═══════════════════════════════════════

        /// <param name="rotate">Look for the next ear where the last one was cut instead of from the start: needed for
        /// the long bridged polygons of a mesh with holes; off otherwise, so meshes without holes keep their triangles.</param>
        internal static List<int> EarClipTriangulate(List<Vector2> polygon, bool rotate = false)
        {
            var result = new List<int>();
            if (polygon.Count < 3) return result;

            var idx = new List<int>();
            for (int i = 0; i < polygon.Count; i++)
                idx.Add(i);

            int safety = polygon.Count * 3;
            int from = 0;
            while (idx.Count > 2 && safety-- > 0)
            {
                bool earFound = false;
                for (int step = 0; step < idx.Count; step++)
                {
                    int i = rotate ? (from + step) % idx.Count : step;
                    int prev = idx[(i - 1 + idx.Count) % idx.Count];
                    int curr = idx[i];
                    int next = idx[(i + 1) % idx.Count];

                    var a = polygon[prev];
                    var b = polygon[curr];
                    var c = polygon[next];

                    float cross = Cross2D(b - a, c - b);
                    if (cross <= 0) continue;

                    bool isEar = true;
                    for (int j = 0; j < idx.Count; j++)
                    {
                        int vi = idx[j];
                        if (vi == prev || vi == curr || vi == next) continue;
                        var p = polygon[vi];
                        // A hole's bridge visits two vertices twice: their copies are the triangle's own corners
                        if (p == a || p == b || p == c) continue;
                        if (PointInTriangle(p, a, b, c))
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
                        idx.RemoveAt(i);
                        from = idx.Count > 0 ? Math.Max(0, i - 1) % idx.Count : 0;
                        earFound = true;
                        break;
                    }
                }
                if (!earFound) break;
            }

            return result;
        }


        // ═══════════════════════════
        // ── Math Helpers ──
        // ═══════════════════════════

        private static float Cross2D(Vector2 a, Vector2 b) => a.X * b.Y - a.Y * b.X;

        internal static float GetSignedArea(List<Vector2> poly)
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
