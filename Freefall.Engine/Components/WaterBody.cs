using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// A lake or river: builds a water surface from the entity's Spline and draws it with the water shader
    /// (water.fx). Closed spline = still water (lake, pond), open spline = flowing water whose current
    /// follows the spline from its first point to its last.
    ///
    /// Setup: Spline + WaterBody. A river usually shares the entity with the HeightStamp that carves its
    /// bed: the spline carries the bed heights and Depth lifts the surface above them.
    ///
    /// The river surface is its own mesh rather than a RuntimeMesh strip: it is subdivided across its width
    /// and carries the current's direction and the steepness per vertex (in the normal stream), so the
    /// shader gets them smoothly instead of per triangle.
    ///
    /// Shading is shared with the ocean (water_common.fx), so all water follows time of day, sky and
    /// clouds the same way. Ripples borrow the ocean simulation's slope maps when the scene has an
    /// OceanRenderer; without one the surface is flat.
    ///
    /// In the cold it freezes (Graphics.Weather.Ice): water.fx draws the ice and the snow on it where the water is
    /// shallow and slow enough (Weather.IceThickness: from the banks in, a fast deep channel stays open), and in play a
    /// static collider over the ice that holds (Weather.WalkableIce) carries walkers on it, made again when the ice
    /// moves a step (Weather.IceStepChanged), not every frame.
    /// </summary>
    [Icon("icon_ocean.png")]
    public class WaterBody : Component, IDraw, IUpdate
    {
        [Category("Shape")]
        [ValueRange(0.5f, 100f)]
        [Description(@"River: width of the surface in meters where the spline's width is 1.
         Make it reach the banks (about twice the bed stamp's Radius + Falloff); the excess hides under the terrain")]
        public float Width = 12f;

        [ValueRange(0f, 10f)]
        [Description(@"Height of the surface above the spline, i.e. the water depth when the spline carries the bed heights")]
        public float Depth = 0.9f;

        [ValueRange(-20f, 50f)]
        [Description(@"Lake: grows the surface outward from the spline by this many meters,
         so the water reaches up the shore past the bed outline")]
        public float Expand = 0f;

        [Category("Color")]
        [Description(@"Color of deep water (light scattered back from the water itself)")]
        public Color3 WaterColor = new Color3(0.004f, 0.02f, 0.018f);

        [Description(@"Tint the bed takes on through shallow water")]
        public Color3 ShallowColor = new Color3(0.45f, 0.7f, 0.55f);

        [ValueRange(0.3f, 30f)]
        [Description(@"Underwater visibility in meters.
         At this much water the bed is tinted by ShallowColor and mostly faded out")]
        public float Visibility = 3.5f;

        [ValueRange(0f, 0.1f)]
        [Description(@"How much the ripples bend the view of the bed")]
        public float RefractionStrength = 0.02f;

        [Category("Surface")]
        [ValueRange(0f, 2f)]
        [Description(@"Ripple height. Lakes and slow rivers want far less than the open sea")]
        public float RippleStrength = 0.35f;

        [ValueRange(0.2f, 5f)]
        [Description(@"Ripple size multiplier (1 = the ocean's finest wave bands as they are)")]
        public float RippleSize = 1f;

        [ValueRange(0f, 6f)]
        [Description(@"Current speed in m/s along the spline (open splines only).
         Steep stretches run faster on their own")]
        public float FlowSpeed = 1.2f;

        [Category("Foam")]
        [ValueRange(0f, 1f)]
        [Description(@"Foam along the banks and around anything standing in the water")]
        public float EdgeFoam = 0.4f;

        [ValueRange(0f, 2f)]
        [Description(@"White water on steep stretches (open splines only)")]
        public float RapidsFoam = 1f;

        private Material _material = null!;
        private readonly MaterialBlock _params = new MaterialBlock();
        private Mesh? _mesh;
        private bool _meshDirty = true;
        private Spline? _spline;

        /// <summary>Cross-sections per spline span, and quads across the width, of a river's surface.</summary>
        private const int SegmentsPerSpan = 12;
        private const int Columns = 8;
        private OceanRenderer? _ocean;
        private DirectionalLight? _sunLight;

        // The surface as built (mesh-local), for the ice collider: its points, how steep the river falls at each, its
        // triangles. A lake's are the outline's (a few big triangles): its ice is laid on a grid of IceCell metres
        private Vector3[] _surfacePoints = Array.Empty<Vector3>();
        private float[] _surfaceSteep = Array.Empty<float>();
        private int[] _surfaceTriangles = Array.Empty<int>();
        private const float IceCell = 2f;

        // How far each place of the water lies from its shore, for where the ice forms first (Weather.IceThickness): a
        // grid over the surface's bounds in world xz (ShoreCellsMax cells a side at most, none finer than a metre), each
        // cell the distance to the nearest cell the water does not cover (the bed rises over the surface there, or there
        // is no surface) over the farthest any cell lies: 0 at the shore, 1 in the middle of the open water. By the depth
        // instead, a lake whose bed is flat froze everywhere at once, in blotches where the edge's noise tipped it
        // (user, 2026-10-23: "looks roughed in"). Made on a worker when the surface is built; water.fx reads it as a
        // texture (WaterData.ShoreField), the ice collider the array.
        private const int ShoreCellsMax = 512;

        private sealed class ShoreField
        {
            public required float[] Share;
            public int Width, Height;
            public Vector2 Origin;
            public float Cell;

            // Bilinear, 0 outside
            public float At(float x, float z)
            {
                float fx = (x - Origin.X) / Cell - 0.5f, fz = (z - Origin.Y) / Cell - 0.5f;
                int x0 = (int)MathF.Floor(fx), z0 = (int)MathF.Floor(fz);
                float tx = fx - x0, tz = fz - z0;
                float Get(int i, int j) => i < 0 || j < 0 || i >= Width || j >= Height ? 0f : Share[j * Width + i];
                return (Get(x0, z0) * (1f - tx) + Get(x0 + 1, z0) * tx) * (1f - tz)
                     + (Get(x0, z0 + 1) * (1f - tx) + Get(x0 + 1, z0 + 1) * tx) * tz;
            }
        }

        private ShoreField? _shore;
        private System.Threading.Tasks.Task<ShoreField?>? _shoreTask;
        private Texture? _shoreTexture;
        private PhysX.RigidStatic? _iceActor;
        private PhysX.Shape? _iceShape;
        private PhysX.TriangleMesh? _iceMesh;
        private bool _iceDirty;

        [StructLayout(LayoutKind.Sequential)]
        public struct WaterData
        {
            public float Time;
            public float FlowSpeed;
            public float RippleStrength;
            public float RippleSize;
            public Vector3 WaterColor;
            public float Visibility;
            public Vector3 ShallowColor;
            public float RefractionStrength;
            public Vector3 SunDirection;
            public float SunIntensity;
            public Vector3 SunColor;
            public float EdgeFoam;
            public float RapidsFoam;
            public uint SlopeSRV;
            public uint NoiseSRV;
            public uint DepthGBufferSRV;
            public uint CompositeSRV;
            public float InvViewportWidth;
            public float InvViewportHeight;
            public float TileScaleA;
            public float TileScaleB;
            public uint BandA;
            public uint BandB;
            public float Flowing;       // 1 = open spline (river), 0 = closed (lake)
            public float Ice;           // Weather.Ice
            public float SnowCover;     // Weather.SnowCover: snow lies on the ice
            public uint ShoreField;     // Texture2D<float>: the share of the way from the shore to the middle (0 = none yet)
            public float ShoreOriginX;  // world xz of the field's corner, and the metres it spans
            public float ShoreOriginZ;
            public float ShoreSizeX;
            public float ShoreSizeZ;
        }

        protected override void Awake()
        {
            _material = new Material(Assets.InternalAssets.WaterEffect);
            _spline = Entity.GetComponent<Spline>();
            _sunLight = EntityManager.FindComponent<DirectionalLight>();
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
            Weather.IceStepChanged += OnIceStepChanged;
            _iceDirty = true;
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);
            Weather.IceStepChanged -= OnIceStepChanged;
            ReleaseIce();
            Engine.Device.DeferDispose(_shoreTexture);
            _shoreTexture = null;

            // Deferred: a draw enqueued earlier this frame and frames still in flight reference the mesh
            Engine.Device.DeferDispose(_mesh);
            _mesh = null;
        }

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is Spline spline && spline.Entity == Entity)
                _meshDirty = true;
        }

        private void OnIceStepChanged() => _iceDirty = true;

        /// <summary>Inspector / command-server edits (Width, Depth, Expand) rebuild the surface.</summary>
        public override void OnMemberChanged() => _meshDirty = true;

        // ═══════════════════════════
        // ── Ice ──
        // ═══════════════════════════

        /// <summary>In play: the ice collider made again when the ice moved a step or the surface changed.</summary>
        public void Update()
        {
            if (!_iceDirty || Engine.IsEditor) return;
            if (_meshDirty) RebuildMesh();
            TakeShoreField();
            if (_shore == null && _shoreTask != null) return;   // the field is still being made: then
            _iceDirty = false;
            bool had = _iceActor != null;
            BuildIce();
            // The walkers' controllers remember the shapes round them: a new or vanished sheet of ice is not seen
            // otherwise (they walked through ice turned off and on again)
            if (had || _iceActor != null)
                PhysicsWorld.InvalidateControllerCaches();
        }

        /// <summary>
        /// A static triangle mesh over the ice that holds a walker (world space, at the water's surface): a river's
        /// surface triangles whose corners all freeze that far, a lake's grid cells. Nothing while it is all open water.
        /// </summary>
        private void BuildIce()
        {
            ReleaseIce();
            float ice = Weather.Ice;
            if (ice <= 0f || _surfaceTriangles.Length == 0) return;

            var world = Transform.WorldMatrix;
            bool flowing = _spline != null && !_spline.Closed;
            var points = new List<Vector3>();
            var triangles = new List<int>();

            // As far out as the ice has come; under the banks too (the surface runs on under them), so no gap opens
            // between the shore and the ice
            var shore = _shore;
            bool Holds(Vector3 p, float steep)
            {
                float speed = flowing ? FlowSpeed * (1f + 4f * steep) : 0f;
                return Weather.IceThickness(ice, shore?.At(p.X, p.Z) ?? 1f, speed) >= Weather.WalkableIce;
            }

            if (flowing)
            {
                var worldPoints = Array.ConvertAll(_surfacePoints, p => Vector3.Transform(p, world));
                var holds = new bool[worldPoints.Length];
                for (int i = 0; i < worldPoints.Length; i++)
                    holds[i] = Holds(worldPoints[i], _surfaceSteep[i]);
                points.AddRange(worldPoints);
                for (int t = 0; t + 2 < _surfaceTriangles.Length; t += 3)
                {
                    int a = _surfaceTriangles[t], b = _surfaceTriangles[t + 1], c = _surfaceTriangles[t + 2];
                    if (holds[a] && holds[b] && holds[c])
                    {
                        triangles.Add(a); triangles.Add(b); triangles.Add(c);
                    }
                }
            }
            else
            {
                // Every cell whose centre lies in a surface triangle, at the surface's height there (once: a centre on
                // an edge two triangles share)
                var laid = new HashSet<(int, int)>();
                for (int t = 0; t + 2 < _surfaceTriangles.Length; t += 3)
                {
                    var a = _surfacePoints[_surfaceTriangles[t]];
                    var b = _surfacePoints[_surfaceTriangles[t + 1]];
                    var c = _surfacePoints[_surfaceTriangles[t + 2]];
                    float minX = MathF.Min(a.X, MathF.Min(b.X, c.X)), maxX = MathF.Max(a.X, MathF.Max(b.X, c.X));
                    float minZ = MathF.Min(a.Z, MathF.Min(b.Z, c.Z)), maxZ = MathF.Max(a.Z, MathF.Max(b.Z, c.Z));
                    for (float x = MathF.Floor(minX / IceCell) * IceCell; x < maxX; x += IceCell)
                    for (float z = MathF.Floor(minZ / IceCell) * IceCell; z < maxZ; z += IceCell)
                    {
                        var centre = new Vector2(x + IceCell * 0.5f, z + IceCell * 0.5f);
                        if (!Barycentric(centre, a, b, c, out var w)) continue;
                        if (!laid.Add(((int)MathF.Round(x / IceCell), (int)MathF.Round(z / IceCell)))) continue;
                        float y = a.Y * w.X + b.Y * w.Y + c.Y * w.Z;
                        if (!Holds(Vector3.Transform(new Vector3(centre.X, y, centre.Y), world), 0f)) continue;
                        int first = points.Count;
                        points.Add(Vector3.Transform(new Vector3(x, y, z), world));
                        points.Add(Vector3.Transform(new Vector3(x + IceCell, y, z), world));
                        points.Add(Vector3.Transform(new Vector3(x + IceCell, y, z + IceCell), world));
                        points.Add(Vector3.Transform(new Vector3(x, y, z + IceCell), world));
                        triangles.Add(first); triangles.Add(first + 2); triangles.Add(first + 1);
                        triangles.Add(first); triangles.Add(first + 3); triangles.Add(first + 2);
                    }
                }
            }
            if (triangles.Count == 0) return;

            // Every triangle facing up ((b − a) × (c − a) along +y): the character's sweeps leave out the back of a mesh,
            // and a river's strip turns whichever way its spline runs (it is drawn from both sides)
            for (int t = 0; t + 2 < triangles.Count; t += 3)
            {
                var a = points[triangles[t]];
                var ab = points[triangles[t + 1]] - a;
                var ac = points[triangles[t + 2]] - a;
                if (ab.Z * ac.X - ab.X * ac.Z < 0f)
                    (triangles[t + 1], triangles[t + 2]) = (triangles[t + 2], triangles[t + 1]);
            }

            var cooking = PhysicsWorld.Physics.CreateCooking();
            try
            {
                using var stream = new System.IO.MemoryStream();
                cooking.CookTriangleMesh(new PhysX.TriangleMeshDesc { Flags = 0, Triangles = triangles.ToArray(), Points = points.ToArray() }, stream);
                stream.Position = 0;
                _iceMesh = PhysicsWorld.Physics.CreateTriangleMesh(stream);
            }
            finally
            {
                PhysXRelease.Release(cooking);
            }
            _iceActor = PhysicsWorld.Physics.CreateRigidStatic();
            _iceActor.UserData = Entity;
            _iceActor.Flags &= ~PhysX.ActorFlag.Visualization;
            _iceShape = PhysX.RigidActorExt.CreateExclusiveShape(_iceActor, new PhysX.TriangleMeshGeometry(_iceMesh), PhysicsWorld.DefaultMaterial);
            _iceShape.QueryFilterData = new PhysX.FilterData(Collider.Solid, 0, 0, 0);
            PhysicsWorld.Scene.AddActor(_iceActor);
        }

        /// <summary>The shore field of the surface just built, made on a worker (the terrain's heights are only read).</summary>
        private void StartShoreField()
        {
            _shore = null;
            _shoreTask = null;
            if (_surfaceTriangles.Length < 3) return;
            var world = Transform.WorldMatrix;
            var points = Array.ConvertAll(_surfacePoints, p => Vector3.Transform(p, world));
            var triangles = _surfaceTriangles;
            var terrain = EntityManager.FindComponent<TerrainRenderer>();
            _shoreTask = System.Threading.Tasks.Task.Run(() => MakeShoreField(points, triangles, terrain));
        }

        /// <summary>Main thread: the finished field into its texture (water.fx), and the collider made again over it.</summary>
        private void TakeShoreField()
        {
            if (_shoreTask is not { IsCompleted: true } task) return;
            _shoreTask = null;
            _shore = task.IsCompletedSuccessfully ? task.Result : null;
            Engine.Device.DeferDispose(_shoreTexture);
            _shoreTexture = null;
            if (_shore != null)
            {
                var bytes = new byte[_shore.Share.Length * 4];
                Buffer.BlockCopy(_shore.Share, 0, bytes, 0, bytes.Length);
                _shoreTexture = Texture.CreateFromData(Engine.Device, _shore.Width, _shore.Height, bytes, Vortice.DXGI.Format.R32_Float);
            }
            _iceDirty = true;
        }

        // Any thread. The grid's cells whose centre lies in a surface triangle and over the bed are water, the rest shore;
        // the distance to the shore by a two-pass chamfer (1 and √2), over its largest
        private static ShoreField? MakeShoreField(Vector3[] points, int[] triangles, TerrainRenderer? terrain)
        {
            float minX = float.MaxValue, minZ = float.MaxValue, maxX = float.MinValue, maxZ = float.MinValue;
            foreach (var p in points)
            {
                minX = MathF.Min(minX, p.X); maxX = MathF.Max(maxX, p.X);
                minZ = MathF.Min(minZ, p.Z); maxZ = MathF.Max(maxZ, p.Z);
            }
            float cell = MathF.Max(1f, MathF.Max(maxX - minX, maxZ - minZ) / (ShoreCellsMax - 2));
            var origin = new Vector2(minX - cell, minZ - cell);
            int w = (int)MathF.Ceiling((maxX - minX) / cell) + 2, h = (int)MathF.Ceiling((maxZ - minZ) / cell) + 2;
            var d = new float[w * h];

            for (int t = 0; t + 2 < triangles.Length; t += 3)
            {
                var a = points[triangles[t]];
                var b = points[triangles[t + 1]];
                var c = points[triangles[t + 2]];
                int x0 = Math.Max(0, (int)((MathF.Min(a.X, MathF.Min(b.X, c.X)) - origin.X) / cell));
                int x1 = Math.Min(w - 1, (int)((MathF.Max(a.X, MathF.Max(b.X, c.X)) - origin.X) / cell));
                int z0 = Math.Max(0, (int)((MathF.Min(a.Z, MathF.Min(b.Z, c.Z)) - origin.Y) / cell));
                int z1 = Math.Min(h - 1, (int)((MathF.Max(a.Z, MathF.Max(b.Z, c.Z)) - origin.Y) / cell));
                for (int j = z0; j <= z1; j++)
                for (int i = x0; i <= x1; i++)
                {
                    var centre = origin + new Vector2(i + 0.5f, j + 0.5f) * cell;
                    if (!Barycentric(centre, a, b, c, out var weight)) continue;
                    float surface = a.Y * weight.X + b.Y * weight.Y + c.Y * weight.Z;
                    float bed = terrain?.GetHeight(new Vector3(centre.X, surface, centre.Y)) ?? surface - 10f;
                    if (bed < surface) d[j * w + i] = float.MaxValue;
                }
            }

            const float Diagonal = 1.41421356f;
            float At(int i, int j) => i < 0 || j < 0 || i >= w || j >= h ? 0f : d[j * w + i];
            for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++)
            {
                ref float v = ref d[j * w + i];
                if (v == 0f) continue;
                v = MathF.Min(v, MathF.Min(MathF.Min(At(i - 1, j) + 1f, At(i, j - 1) + 1f),
                                           MathF.Min(At(i - 1, j - 1) + Diagonal, At(i + 1, j - 1) + Diagonal)));
            }
            float most = 0f;
            for (int j = h - 1; j >= 0; j--)
            for (int i = w - 1; i >= 0; i--)
            {
                ref float v = ref d[j * w + i];
                if (v == 0f) continue;
                v = MathF.Min(v, MathF.Min(MathF.Min(At(i + 1, j) + 1f, At(i, j + 1) + 1f),
                                           MathF.Min(At(i + 1, j + 1) + Diagonal, At(i - 1, j + 1) + Diagonal)));
                most = MathF.Max(most, v);
            }
            if (most <= 0f) return null;
            for (int k = 0; k < d.Length; k++)
                d[k] /= most;
            return new ShoreField { Share = d, Width = w, Height = h, Origin = origin, Cell = cell };
        }

        // The weights of a point (xz) in a triangle, false when it lies outside
        private static bool Barycentric(Vector2 p, Vector3 a, Vector3 b, Vector3 c, out Vector3 w)
        {
            var v0 = new Vector2(b.X - a.X, b.Z - a.Z);
            var v1 = new Vector2(c.X - a.X, c.Z - a.Z);
            var v2 = new Vector2(p.X - a.X, p.Y - a.Z);
            float d = v0.X * v1.Y - v1.X * v0.Y;
            w = default;
            if (MathF.Abs(d) < 1e-8f) return false;
            float u = (v2.X * v1.Y - v1.X * v2.Y) / d;
            float v = (v0.X * v2.Y - v2.X * v0.Y) / d;
            w = new Vector3(1f - u - v, u, v);
            return u >= 0f && v >= 0f && u + v <= 1f;
        }

        private void ReleaseIce()
        {
            if (_iceActor != null)
            {
                PhysicsWorld.Scene.RemoveActor(_iceActor);
                // The shape goes with its actor natively, but its wrapper stays in PhysX.Net's object table unless let go
                // first: the next sheet's shape, made at the same address, then threw ("already a managed instance")
                PhysXRelease.Release(_iceShape, releaseNative: false);
                _iceShape = null;
                PhysXRelease.Release(_iceActor);
                _iceActor = null;
            }
            if (_iceMesh != null)
            {
                PhysXRelease.Release(_iceMesh);
                _iceMesh = null;
            }
        }

        // ═══════════════════════════
        // ── Surface mesh ──
        // ═══════════════════════════

        private void RebuildMesh()
        {
            _meshDirty = false;
            _iceDirty = true;   // over the new surface
            Engine.Device.DeferDispose(_mesh);
            _mesh = null;
            _surfacePoints = Array.Empty<Vector3>();
            _surfaceSteep = Array.Empty<float>();
            _surfaceTriangles = Array.Empty<int>();

            if (_spline == null || _spline.Points.Count < 2) return;
            _mesh = _spline.Closed ? BuildLake(_spline) : BuildRiver(_spline);
            StartShoreField();
        }

        /// <summary>
        /// A grid along the spline. Positions and UVs (x = meters along, y = meters across) as a strip; the
        /// normal stream holds what the shader needs instead of a normal: xz = direction of the current,
        /// y = how steeply the surface falls there (drop per meter).
        /// </summary>
        private Mesh BuildRiver(Spline spline)
        {
            var strip = SplineStrip.Sample(spline, SegmentsPerSpan, Width);
            int n = strip.Count;

            var verts = new List<Vector3>(n * (Columns + 1));
            var flow = new List<Vector3>(n * (Columns + 1));
            var uvs = new List<Vector2>(n * (Columns + 1));
            var indices = new List<uint>((n - 1) * Columns * 6);

            for (int i = 0; i < n; i++)
            {
                // Fall of the centre line across the two neighbouring cross-sections
                int a = Math.Max(0, i - 1), b = Math.Min(n - 1, i + 1);
                var run = strip.Points[b] - strip.Points[a];
                float ground = MathF.Sqrt(run.X * run.X + run.Z * run.Z);
                float steep = ground > 1e-4f ? MathF.Abs(run.Y) / ground : 0f;

                var fwd = strip.Tangents[i];
                var dir = new Vector2(fwd.X, fwd.Z);
                dir = dir.LengthSquared() > 1e-12f ? Vector2.Normalize(dir) : Vector2.UnitX;

                var left = strip.Left(i);
                var right = strip.Right(i);
                float across = strip.HalfWidths[i] * 2f;

                for (int c = 0; c <= Columns; c++)
                {
                    float f = (float)c / Columns;
                    var p = Vector3.Lerp(left, right, f);
                    p.Y += Depth;
                    verts.Add(p);
                    flow.Add(new Vector3(dir.X, steep, dir.Y));
                    uvs.Add(new Vector2(strip.ArcLengths[i], f * across));
                }
            }

            for (int i = 0; i < n - 1; i++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    uint bl = (uint)(i * (Columns + 1) + c);
                    uint br = bl + 1;
                    uint tl = bl + (uint)(Columns + 1);
                    uint tr = tl + 1;

                    indices.Add(bl); indices.Add(br); indices.Add(tl);
                    indices.Add(br); indices.Add(tr); indices.Add(tl);
                }
            }

            _surfacePoints = verts.ToArray();
            _surfaceSteep = flow.ConvertAll(f => f.Y).ToArray();
            _surfaceTriangles = indices.ConvertAll(i => (int)i).ToArray();
            return BuildSurface(verts, flow, uvs, indices);
        }

        /// <summary>The spline's outline, grown by Expand, filled and lifted by Depth.</summary>
        private Mesh BuildLake(Spline spline)
        {
            int samples = Math.Max(3, spline.SpanCount * SegmentsPerSpan);
            var polygon = new List<Vector2>(samples);
            var heights = new float[samples];
            for (int i = 0; i < samples; i++)
            {
                var p = spline.GetPoint((float)i / samples);   // no end point: the outline wraps
                polygon.Add(new Vector2(p.X, p.Z));
                heights[i] = p.Y;
            }

            // Counter-clockwise, as the triangulation expects
            if (RuntimeMesh.GetSignedArea(polygon) > 0)
            {
                polygon.Reverse();
                Array.Reverse(heights);
            }

            if (MathF.Abs(Expand) > 0.001f)
            {
                var grown = new List<Vector2>(samples);
                for (int i = 0; i < samples; i++)
                {
                    var e0 = polygon[i] - polygon[(i - 1 + samples) % samples];
                    var e1 = polygon[(i + 1) % samples] - polygon[i];
                    var outward = new Vector2(e0.Y, -e0.X) + new Vector2(e1.Y, -e1.X);
                    grown.Add(outward.LengthSquared() > 1e-10f
                        ? polygon[i] + Vector2.Normalize(outward) * Expand
                        : polygon[i]);
                }
                polygon = grown;
            }

            var triangles = RuntimeMesh.EarClipTriangulate(polygon);
            if (triangles == null || triangles.Count < 3) return null!;

            var verts = new List<Vector3>(samples);
            var normals = new List<Vector3>(samples);
            var uvs = new List<Vector2>(samples);
            var indices = new List<uint>(triangles.Count);
            for (int i = 0; i < samples; i++)
            {
                verts.Add(new Vector3(polygon[i].X, heights[i] + Depth, polygon[i].Y));
                normals.Add(Vector3.UnitY);
                uvs.Add(polygon[i]);
            }
            for (int i = triangles.Count - 1; i >= 0; i--)   // reversed: faces up
                indices.Add((uint)triangles[i]);

            _surfacePoints = verts.ToArray();
            _surfaceSteep = new float[verts.Count];
            _surfaceTriangles = indices.ConvertAll(i => (int)i).ToArray();
            return BuildSurface(verts, normals, uvs, indices);
        }

        private static Mesh BuildSurface(List<Vector3> verts, List<Vector3> normals, List<Vector2> uvs, List<uint> indices)
        {
            var parts = new List<MeshPart> { new MeshPart { Name = "Surface", NumIndices = indices.Count, MaterialSlot = 0 } };
            return RuntimeMesh.BuildMesh(verts, normals, uvs, indices, parts, "WaterBody_");
        }

        public void Draw()
        {
            if (!Enabled) return;
            if (_material == null) return;

            // Lazy lookups: the ocean (ripple textures) may be created after this component
            _ocean ??= EntityManager.FindComponent<OceanRenderer>();
            _sunLight ??= EntityManager.FindComponent<DirectionalLight>();
            _spline ??= Entity.GetComponent<Spline>();

            if (_meshDirty) RebuildMesh();
            TakeShoreField();
            var mesh = _mesh;
            if (mesh == null) return;

            var sunDir = Vector3.UnitY;
            var sunColor = Vector3.One;
            float sunIntensity = 1.0f;
            if (_sunLight != null)
            {
                sunDir = Vector3.Transform(Vector3.UnitZ, _sunLight.Entity.Transform.Rotation);
                sunColor = new Vector3(_sunLight.Color.R, _sunLight.Color.G, _sunLight.Color.B);
                sunIntensity = _sunLight.Intensity;
            }

            // Ripples: the two finest bands of the ocean simulation
            var fft = _ocean?.FFT;
            int bandCount = _ocean?.Bands.Count ?? 0;
            uint bandA = 0, bandB = 0;
            float tileA = 0f, tileB = 0f;
            if (fft != null && bandCount > 0)
            {
                bandB = (uint)(bandCount - 1);
                bandA = (uint)Math.Max(0, bandCount - 2);
                tileA = 1.0f / _ocean!.Bands[(int)bandA].LengthScale;
                tileB = 1.0f / _ocean.Bands[(int)bandB].LengthScale;
            }

            var depth = DeferredRenderer.Current?.DepthGBuffer;

            _params.SetParameter("WaterData", new WaterData
            {
                Time = Time.TotalTime,
                FlowSpeed = FlowSpeed,
                RippleStrength = RippleStrength,
                RippleSize = RippleSize,
                WaterColor = new Vector3(WaterColor.R, WaterColor.G, WaterColor.B),
                Visibility = Visibility,
                ShallowColor = new Vector3(ShallowColor.R, ShallowColor.G, ShallowColor.B),
                RefractionStrength = RefractionStrength,
                SunDirection = sunDir,
                SunIntensity = sunIntensity,
                SunColor = sunColor,
                EdgeFoam = EdgeFoam,
                RapidsFoam = RapidsFoam,
                SlopeSRV = fft?.SlopeSRV ?? 0,
                NoiseSRV = fft?.NoiseSRV ?? 0,
                DepthGBufferSRV = depth?.BindlessIndex ?? 0,
                CompositeSRV = DeferredRenderer.Current?.CompositeSnapshot?.BindlessIndex ?? 0,
                InvViewportWidth = 1.0f / MathF.Max(1, depth?.Native.Description.Width ?? 1920),
                InvViewportHeight = 1.0f / MathF.Max(1, depth?.Native.Description.Height ?? 1080),
                TileScaleA = tileA,
                TileScaleB = tileB,
                BandA = bandA,
                BandB = bandB,
                Flowing = _spline != null && !_spline.Closed ? 1f : 0f,
                Ice = Weather.Ice,
                SnowCover = Weather.SnowCover,
                ShoreField = _shoreTexture?.BindlessIndex ?? 0u,
                ShoreOriginX = _shore?.Origin.X ?? 0f,
                ShoreOriginZ = _shore?.Origin.Y ?? 0f,
                ShoreSizeX = _shore != null ? _shore.Width * _shore.Cell : 1f,
                ShoreSizeZ = _shore != null ? _shore.Height * _shore.Cell : 1f,
            });

            CommandBuffer.Enqueue(mesh, _material, _params, Transform.TransformSlot);
        }
    }
}
