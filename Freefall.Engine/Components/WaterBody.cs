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
    /// </summary>
    [Icon("icon_ocean.png")]
    public class WaterBody : Component, IDraw
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
            public Vector3 CloudColor;
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
            public float _pad0;
        }

        protected override void Awake()
        {
            _material = new Material(Assets.InternalAssets.WaterEffect);
            _spline = Entity.GetComponent<Spline>();
            _sunLight = EntityManager.FindComponent<DirectionalLight>();
            MessageDispatcher.AddListener(EngineMsg.SplineChanged, OnSplineChanged);
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.SplineChanged, OnSplineChanged);

            // Deferred: a draw enqueued earlier this frame and frames still in flight reference the mesh
            Engine.Device.DeferDispose(_mesh);
            _mesh = null;
        }

        private void OnSplineChanged(Message msg)
        {
            if (msg.Data is Spline spline && spline.Entity == Entity)
                _meshDirty = true;
        }

        /// <summary>Inspector / command-server edits (Width, Depth, Expand) rebuild the surface.</summary>
        public override void OnMemberChanged() => _meshDirty = true;

        // ═══════════════════════════
        // ── Surface mesh ──
        // ═══════════════════════════

        private void RebuildMesh()
        {
            _meshDirty = false;
            Engine.Device.DeferDispose(_mesh);
            _mesh = null;

            if (_spline == null || _spline.Points.Count < 2) return;
            _mesh = _spline.Closed ? BuildLake(_spline) : BuildRiver(_spline);
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
                CloudColor = SkyboxRenderer.CurrentCloudColor,
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
            });

            CommandBuffer.Enqueue(mesh, _material, _params, Transform.TransformSlot);
        }
    }
}
