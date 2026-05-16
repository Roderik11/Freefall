using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;
using DefaultValue = System.ComponentModel.DefaultValueAttribute;

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
    public class RuntimeMesh : Component
    {
        // ═══════════════════════════
        // ── Properties ──
        // ═══════════════════════════
        private Spline Spline;

        /// <summary>Half-width of the generated strip (open spline only).</summary>
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

        /// <summary>Smoothness: segments per spline span.</summary>
        [DefaultValue(8)]
        [ValueRange(2, 32)]
        public int Smoothness = 8;

        /// <summary>Enable raised curbs along the sides (open spline only).</summary>
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

        /// <summary>UV tiling scale. Higher = more texture repeats.</summary>
        [DefaultValue(1f)]
        [ValueRange(0.01f, 20f)]
        public float UVScale = 1f;


        // ═══════════════════════════
        // ── Internals ──
        // ═══════════════════════════

        private MeshRenderer _renderer;
        private Mesh _generatedMesh;
        private Mesh _retiring; // previous mesh kept alive until GPU is done with it

        protected override void Early()
        {
            Spline = Spline ?? Entity.GetComponent<Spline>();
            _renderer = Entity.GetComponent<MeshRenderer>();
            Generate();
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            _retiring?.Dispose();
            _retiring = null;
            if (_generatedMesh != null)
            {
                if (_renderer != null && _renderer.Mesh == _generatedMesh)
                    _renderer.Mesh = null;
                _generatedMesh.Dispose();
                _generatedMesh = null;
            }
        }

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is not Spline spline) return;
            if (spline != Spline) return;
            Generate();
        }

        public override void OnMemberChanged()
        {
            Generate();
        }


        /// <summary>
        /// Regenerate the mesh from the current spline state.
        /// Double-buffers: the old mesh is kept alive for one cycle so the GPU
        /// can finish rendering with it before its resources are freed.
        /// </summary>
        public void Generate()
        {
            if (Spline == null || _renderer == null) return;
            if (Spline.Points.Count < 2) return;

            // Dispose the mesh from two generations ago (GPU is definitely done)
            _retiring?.Dispose();

            // Current mesh retires — GPU may still be referencing it this frame
            _retiring = _generatedMesh;

            _generatedMesh = Spline.Closed
                ? GenerateClosed()
                : GenerateOpen();

            if (_generatedMesh != null)
                _renderer.Mesh = _generatedMesh;
        }

        // ═══════════════════════════════════════
        // ── Open Spline: Strip Along Curve ──
        // ═══════════════════════════════════════

        private Mesh GenerateOpen()
        {
            int totalSamples = Spline.TotalSegments + 1;
            if (totalSamples < 2) totalSamples = 2;

            // Override resolution with our Smoothness
            int samples = Spline.SpanCount * Smoothness + 1;
            if (samples < 2) samples = 2;

            // Sample spline
            var points = new Vector3[samples];
            var tangents = new Vector3[samples];
            for (int i = 0; i < samples; i++)
            {
                float t = (float)i / (samples - 1);
                points[i] = Spline.GetPoint(t);
                tangents[i] = Spline.GetTangent(t);
            }

            // Apply height mode
            ApplyHeightMode(points);

            float halfWidth = Width * 0.5f;

            // Compute per-edge positions
            var leftPositions = new Vector3[samples];
            var rightPositions = new Vector3[samples];
            var rights = new Vector3[samples];

            for (int i = 0; i < samples; i++)
            {
                var fwd = tangents[i];
                var right = Vector3.Normalize(new Vector3(-fwd.Z, 0, fwd.X));
                rights[i] = right;
                leftPositions[i] = points[i] - right * halfWidth;
                rightPositions[i] = points[i] + right * halfWidth;
            }

            // Center-line arc length for UV mapping
            // Using center-line for both edges avoids shearing on curves.
            var arcLengths = new float[samples];
            arcLengths[0] = 0;
            for (int i = 1; i < samples; i++)
                arcLengths[i] = arcLengths[i - 1] + Vector3.Distance(points[i], points[i - 1]);

            // ── Main surface: 2 verts per sample ──
            // MeshPart 0: surface
            var verts = new List<Vector3>();
            var norms = new List<Vector3>();
            var uvs = new List<Vector2>();
            var indices = new List<uint>();

            for (int i = 0; i < samples; i++)
            {
                var up = Vector3.UnitY;

                verts.Add(leftPositions[i]);
                verts.Add(rightPositions[i]);
                norms.Add(up);
                norms.Add(up);

                float u = arcLengths[i] * UVScale;
                uvs.Add(new Vector2(u, 0));
                uvs.Add(new Vector2(u, Width * UVScale));
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

            // ── Walls (if Height > 0) ──
            if (Height > 0.001f)
            {
                // Add top verts (duplicate strip shifted up by Height)
                int wallBase = verts.Count;
                for (int i = 0; i < samples; i++)
                {
                    var right = rights[i];
                    var leftTop = leftPositions[i] + Vector3.UnitY * Height;
                    var rightTop = rightPositions[i] + Vector3.UnitY * Height;

                    // Left wall: faces outward (-right)
                    float wu = arcLengths[i] * UVScale;
                    verts.Add(leftPositions[i]); norms.Add(-right); uvs.Add(new Vector2(wu, 0));
                    verts.Add(leftTop); norms.Add(-right); uvs.Add(new Vector2(wu, Height * UVScale));

                    // Right wall: faces outward (+right)
                    verts.Add(rightPositions[i]); norms.Add(right); uvs.Add(new Vector2(wu, 0));
                    verts.Add(rightTop); norms.Add(right); uvs.Add(new Vector2(wu, Height * UVScale));
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

                // Top face
                int topBase = verts.Count;
                for (int i = 0; i < samples; i++)
                {
                    var leftTop = leftPositions[i] + Vector3.UnitY * Height;
                    var rightTop = rightPositions[i] + Vector3.UnitY * Height;

                    float bu = arcLengths[i] * UVScale;
                    verts.Add(leftTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(bu, 0));
                    verts.Add(rightTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(bu, Width * UVScale));
                }

                for (int i = 0; i < samples - 1; i++)
                {
                    uint bl = (uint)(topBase + i * 2);
                    uint br = bl + 1;
                    uint tl = (uint)(topBase + (i + 1) * 2);
                    uint tr = tl + 1;
                    indices.Add(bl); indices.Add(br); indices.Add(tl);
                    indices.Add(br); indices.Add(tr); indices.Add(tl);
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
                GenerateCurbs(points, rights, arcLengths, halfWidth, verts, norms, uvs, indices);

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
        private void GenerateCurbs(Vector3[] points, Vector3[] rights, float[] arcLengths,
            float halfWidth,
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices)
        {
            int samples = points.Length;

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

                    var innerBot = points[i] + right * (halfWidth * s);
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

        // ═══════════════════════════════════════════
        // ── Closed Spline: Polygon Fill ──
        // ═══════════════════════════════════════════

        private Mesh GenerateClosed()
        {
            // Sample the closed spline into a polygon
            int samples = Spline.SpanCount * Smoothness;
            if (samples < 3) samples = 3;

            var polygon2D = new List<Vector2>();
            var heights = new float[samples];

            for (int i = 0; i < samples; i++)
            {
                float t = (float)i / samples; // don't include endpoint (closed = wraps)
                var p = Spline.GetPoint(t);
                polygon2D.Add(new Vector2(p.X, p.Z));
                heights[i] = p.Y;
            }

            // Ensure CCW winding
            if (GetSignedArea(polygon2D) > 0)
            {
                polygon2D.Reverse();
                Array.Reverse(heights);
            }

            var triangles = EarClipTriangulate(polygon2D);
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

            // ── Top surface ──
            for (int i = 0; i < n; i++)
            {
                verts.Add(new Vector3(polygon2D[i].X, yValues[i], polygon2D[i].Y));
                norms.Add(-Vector3.UnitY);
                uvs.Add(polygon2D[i] * UVScale);
            }

            // Top triangles (reversed winding for +Y facing)
            for (int i = 0; i < triangles.Count / 3; i++)
            {
                indices.Add((uint)triangles[i * 3 + 0]);
                indices.Add((uint)triangles[i * 3 + 1]);
                indices.Add((uint)triangles[i * 3 + 2]);
            }

            int surfaceIndexCount = indices.Count;

            // ── Walls (if Height > 0) ──
            if (Height > 0.001f)
            {
                int wallBase = verts.Count;

                for (int i = 0; i < n; i++)
                {
                    int next = (i + 1) % n;
                    var p0 = polygon2D[i];
                    var p1 = polygon2D[next];

                    var edge = new Vector2(p1.X - p0.X, p1.Y - p0.Y);
                    var normal = Vector3.Normalize(new Vector3(edge.Y, 0, -edge.X));

                    float edgeLen = edge.Length();
                    int vi = wallBase + i * 4;

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
                }

                for (int i = 0; i < n; i++)
                {
                    uint vi = (uint)(wallBase + i * 4);
                    indices.Add(vi + 0); indices.Add(vi + 2); indices.Add(vi + 1);
                    indices.Add(vi + 1); indices.Add(vi + 2); indices.Add(vi + 3);
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

            // ── Perimeter curbs (closed spline) ──
            int curbStartIndex = indices.Count;
            if (EnableCurbs)
            {
                int curbBase = verts.Count;

                // Compute outward normals per vertex (average of adjacent edge normals)
                var outwards = new Vector3[n];
                for (int i = 0; i < n; i++)
                {
                    int prev = (i - 1 + n) % n;
                    int next = (i + 1) % n;

                    var e0 = polygon2D[i] - polygon2D[prev];
                    var e1 = polygon2D[next] - polygon2D[i];

                    // Outward normal in XZ: for CCW polygon, outward = (dz, -dx)
                    var n0 = new Vector2(e0.Y, -e0.X);
                    var n1 = new Vector2(e1.Y, -e1.X);
                    var avg = Vector2.Normalize(n0 + n1);
                    outwards[i] = new Vector3(avg.X, 0, avg.Y);
                }

                // Perimeter arc lengths
                var perimArc = new float[n + 1]; // +1 for wrap
                perimArc[0] = 0;
                for (int i = 1; i <= n; i++)
                {
                    int curr = i % n;
                    int prev = (i - 1) % n;
                    perimArc[i] = perimArc[i - 1] + Vector2.Distance(polygon2D[curr], polygon2D[prev]);
                }

                // Emit 4 verts per perimeter vertex (same L-profile)
                for (int i = 0; i < n; i++)
                {
                    var pos = new Vector3(polygon2D[i].X, yValues[i], polygon2D[i].Y);
                    var outward = outwards[i];

                    var innerBot = pos;
                    var innerTop = innerBot + Vector3.UnitY * CurbHeight;
                    var outerTop = innerTop + outward * CurbWidth;
                    var outerBot = innerBot + outward * CurbWidth;

                    float u = perimArc[i] * UVScale;

                    // 6 verts per sample — duplicated at hard edges for flat shading
                    // Inner wall (0, 1)
                    verts.Add(innerBot); norms.Add(-outward); uvs.Add(new Vector2(u, 0));
                    verts.Add(innerTop); norms.Add(-outward); uvs.Add(new Vector2(u, CurbHeight));
                    // Top ledge (2, 3)
                    verts.Add(innerTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(u, 0));
                    verts.Add(outerTop); norms.Add(Vector3.UnitY); uvs.Add(new Vector2(u, CurbWidth));
                    // Outer wall (4, 5)
                    verts.Add(outerTop); norms.Add(outward); uvs.Add(new Vector2(u, CurbHeight));
                    verts.Add(outerBot); norms.Add(outward); uvs.Add(new Vector2(u, 0));
                }

                // Triangulate — wraps around, 6 verts per sample
                for (int i = 0; i < n; i++)
                {
                    uint c0 = (uint)(curbBase + i * 6);
                    uint c1 = (uint)(curbBase + ((i + 1) % n) * 6);

                    indices.Add(c0 + 0); indices.Add(c1 + 0); indices.Add(c0 + 1);
                    indices.Add(c1 + 0); indices.Add(c1 + 1); indices.Add(c0 + 1);

                    indices.Add(c0 + 2); indices.Add(c1 + 2); indices.Add(c0 + 3);
                    indices.Add(c1 + 2); indices.Add(c1 + 3); indices.Add(c0 + 3);

                    indices.Add(c0 + 4); indices.Add(c1 + 4); indices.Add(c0 + 5);
                    indices.Add(c1 + 4); indices.Add(c1 + 5); indices.Add(c0 + 5);
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

            return BuildMesh(verts, norms, uvs, indices, parts);
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
                    var worldMatrix = Transform?.Matrix ?? Matrix4x4.Identity;
                    float localY = Transform?.Position.Y ?? 0;
                    for (int i = 0; i < points.Length; i++)
                    {
                        var worldPos = Vector3.Transform(points[i], worldMatrix);
                        float h = terrain.GetHeight(worldPos) - localY;
                        points[i] = new Vector3(points[i].X, h, points[i].Z);
                    }
                    break;

                // Spline: keep Y as-is
            }
        }

        // ═══════════════════════════════════════
        // ── Mesh Construction ──
        // ═══════════════════════════════════════

        private static Mesh BuildMesh(List<Vector3> verts, List<Vector3> norms,
            List<Vector2> uvs, List<uint> indices, List<MeshPart> parts)
        {
            if (verts.Count == 0 || indices.Count == 0) return null;

            var mesh = new Mesh(Engine.Device,
                verts.ToArray(), norms.ToArray(), uvs.ToArray(), indices.ToArray());

            mesh.BoundingBox = ComputeBounds(verts);
            mesh.Guid = Guid.NewGuid().ToString("N");
            mesh.Name = "RuntimeMesh_" + mesh.Guid;
            mesh.IsDynamic = true;

            foreach (var part in parts)
            {
                part.BoundingBox = mesh.BoundingBox;
                part.BoundingSphere = mesh.LocalBoundingSphere;
                mesh.MeshParts.Add(part);
            }

            return mesh;
        }

        // ═══════════════════════════════════════
        // ── Ear-Clipping Triangulation ──
        // ═══════════════════════════════════════

        private static List<int> EarClipTriangulate(List<Vector2> polygon)
        {
            var result = new List<int>();
            if (polygon.Count < 3) return result;

            var idx = new List<int>();
            for (int i = 0; i < polygon.Count; i++)
                idx.Add(i);

            int safety = polygon.Count * 3;
            while (idx.Count > 2 && safety-- > 0)
            {
                bool earFound = false;
                for (int i = 0; i < idx.Count; i++)
                {
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
                        if (PointInTriangle(polygon[vi], a, b, c))
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
