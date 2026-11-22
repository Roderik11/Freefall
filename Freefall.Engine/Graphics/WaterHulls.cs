using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Base;

namespace Freefall.Graphics
{
    /// <summary>
    /// The inside of a hull, where there is no water: a ship's hold, an open boat's bottom. The sea and the lakes are not
    /// drawn there (ocean.fx, water.fx: water_hulls.hlsli), and <see cref="Components.Water.SurfaceAt"/> finds no water
    /// there, so nobody swims in a hold under the waterline and the camera in it is not under water.
    ///
    /// The shape is the hull's own, sliced from its mesh (<see cref="Fit"/>): its half-breadth at <see cref="Stations"/>
    /// places along its length and <see cref="Levels"/> heights from its keel up to a little over its waterline, each the
    /// outside of the planking at exactly that place and height, less <see cref="Inset"/>. Between them the shape is
    /// interpolated; a hull is convex round its bilge and its bow, so the interpolation lies inside it, never outside
    /// (where the water would stop short of the hull and leave a dry moat). Its frame is the hull's: x from stern to bow,
    /// y up from the waterline, z across.
    /// Docs: .agent/knowledge/engine_internals/ships.md.
    /// </summary>
    public sealed class WaterHull
    {
        public const int Stations = 32, Levels = 8;

        /// <summary>How far inside the planking's outer face the water stops (m): the planking is ~0.1 m thick.</summary>
        public const float Inset = 0.05f;

        /// <summary>How far over its waterline (local y 0) the shape reaches at most (m): water higher than this inside is drawn.</summary>
        public const float Reach = 3f;

        /// <summary>The shape's extent in the hull's frame: x from stern to bow, y from the keel up.</summary>
        public float X0, X1, Bottom, Top;

        /// <summary>Half-breadths (m), level by level: [level * Stations + station]; 0 where the hull is not at that height.</summary>
        public readonly float[] Breadth = new float[Stations * Levels];

        /// <summary>Off, the water is drawn through it again.</summary>
        public bool Enabled = true;

        /// <summary>
        /// Where the sea stands at each station against the waterline (m, in the hull's frame: 0 at its waterline, below
        /// it negative), as game code measured it along the hull this frame; 0 where it did not. Only the ships' waves
        /// (Graphics.WakeWaves) read it, and only where it is below the waterline: the hull presses and churns the water
        /// where it is in it, so a bow lifted clear over a trough no longer churns the sea under it.
        /// </summary>
        public readonly float[] Sea = new float[Stations];

        /// <summary>
        /// How high the water has wetted its sides at each station (m, in the hull's frame: 0 at its waterline), port's
        /// stations first, then starboard's: game code keeps it, raised by the waves and spray it meets and drying down.
        /// The decal pass shades its planking under it wet (decals_cs.hlsl HullWetness); under the water it is wet anyway.
        /// </summary>
        public readonly float[] Wet = new float[2 * Stations];

        /// <summary>
        /// How high the water has just been on its sides at each station (m, as <see cref="Wet"/>, port's first): what it
        /// left there is still running off, a glossy film streaked with rivulets; game code keeps it, falling within
        /// seconds. Under <see cref="Wet"/>'s line and over this one the hull is only damp.
        /// </summary>
        public readonly float[] Fresh = new float[2 * Stations];

        /// <summary>
        /// How far the water has run along the hull from bow to stern (m, wrapped): the way it has made along its own
        /// length, summed as it is placed. The light the sea throws up onto it (light_directional_cs.hlsl HullWaterLight)
        /// moves with the water, so it streams aft under way.
        /// </summary>
        public float Travel { get; private set; }

        internal Matrix4x4 WorldToLocal = Matrix4x4.Identity;
        internal Vector3 Centre;
        internal float Radius;
        private bool _placed;

        /// <summary>Where the hull is now: its frame in the world. Main thread, before the water is drawn (a ship's system).</summary>
        public void Place(Matrix4x4 localToWorld)
        {
            if (!Matrix4x4.Invert(localToWorld, out var inverse)) return;
            WorldToLocal = inverse;
            var mid = new Vector3((X0 + X1) * 0.5f, (Bottom + Top) * 0.5f, 0f);
            var centre = Vector3.Transform(mid, localToWorld);
            // The way made along its length since it was last placed; a jump (put somewhere else) is not sailed
            var along = new Vector3(localToWorld.M11, localToWorld.M12, localToWorld.M13);
            float moved = Vector3.DistanceSquared(centre, Centre);
            if (_placed && moved < 20f * 20f && along.LengthSquared() > 1e-8f)
                Travel = (Travel + Vector3.Dot(centre - Centre, Vector3.Normalize(along))) % 10000f;
            _placed = true;
            Centre = centre;
            float half = MathF.Max(X1 - X0, Top - Bottom) * 0.5f;
            float wide = 0f;
            foreach (var b in Breadth) wide = MathF.Max(wide, b);
            Radius = MathF.Sqrt(half * half + wide * wide + ((Top - Bottom) * 0.5f) * ((Top - Bottom) * 0.5f));
        }

        /// <summary>The half-breadth at a place in the hull's frame, as the shaders interpolate it; 0 outside its extent.</summary>
        public float BreadthAt(float x, float y)
        {
            if (x <= X0 || x >= X1 || y <= Bottom || y >= Top) return 0f;
            float fx = (x - X0) / (X1 - X0) * (Stations - 1);
            float fy = (y - Bottom) / (Top - Bottom) * (Levels - 1);
            int ix = Math.Min((int)fx, Stations - 2), iy = Math.Min((int)fy, Levels - 2);
            float tx = fx - ix, ty = fy - iy;
            float b00 = Breadth[iy * Stations + ix], b10 = Breadth[iy * Stations + ix + 1];
            float b01 = Breadth[(iy + 1) * Stations + ix], b11 = Breadth[(iy + 1) * Stations + ix + 1];
            return (b00 + (b10 - b00) * tx) * (1f - ty) + (b01 + (b11 - b01) * tx) * ty;
        }

        /// <summary>Whether a world point is inside the hull (where its water is kept out).</summary>
        public bool Contains(Vector3 world)
        {
            if (!Enabled || Vector3.DistanceSquared(world, Centre) > Radius * Radius) return false;
            var p = Vector3.Transform(world, WorldToLocal);
            return MathF.Abs(p.Z) < BreadthAt(p.X, p.Y);
        }

        /// <summary>
        /// The shape of a hull from its mesh: the triangles in the hull's frame (positions already put there), sliced at
        /// every level, and the slices' outer half-breadth found at every station. Null when the mesh does not reach the
        /// waterline.
        /// </summary>
        public static WaterHull? Fit(ReadOnlySpan<Vector3> positions, ReadOnlySpan<uint> indices)
        {
            if (positions.Length == 0 || indices.Length < 3) return null;
            float x0 = float.MaxValue, x1 = float.MinValue, y0 = float.MaxValue, y1 = float.MinValue;
            foreach (var p in positions)
            {
                x0 = MathF.Min(x0, p.X); x1 = MathF.Max(x1, p.X);
                y0 = MathF.Min(y0, p.Y); y1 = MathF.Max(y1, p.Y);
            }
            if (y0 >= 0f || x1 <= x0) return null;
            var hull = new WaterHull { X0 = x0, X1 = x1, Bottom = y0, Top = MathF.Min(y1, Reach) };
            if (hull.Top <= hull.Bottom) return null;

            float dx = (x1 - x0) / (Stations - 1), dy = (hull.Top - y0) / (Levels - 1);
            Span<float> levelY = stackalloc float[Levels];
            for (int k = 0; k < Levels; k++)
                levelY[k] = y0 + dy * k + (k == 0 ? 0.01f : k == Levels - 1 ? -0.01f : 0f);   // not exactly on the keel or the top

            Span<Vector3> cut = stackalloc Vector3[3];
            for (int t = 0; t + 2 < indices.Length; t += 3)
            {
                var a = positions[(int)indices[t]];
                var b = positions[(int)indices[t + 1]];
                var c = positions[(int)indices[t + 2]];
                float lo = MathF.Min(a.Y, MathF.Min(b.Y, c.Y)), hi = MathF.Max(a.Y, MathF.Max(b.Y, c.Y));
                for (int k = 0; k < Levels; k++)
                {
                    float y = levelY[k];
                    if (y < lo || y > hi) continue;
                    // Where the plane y cuts the triangle: a segment between two of its edges
                    int n = 0;
                    Cross(a, b, y, cut, ref n);
                    Cross(b, c, y, cut, ref n);
                    Cross(c, a, y, cut, ref n);
                    if (n < 2) continue;
                    var s0 = cut[0];
                    var s1 = cut[1];
                    // The stations the segment passes over: its |z| at each
                    float sx0 = MathF.Min(s0.X, s1.X), sx1 = MathF.Max(s0.X, s1.X);
                    int i0 = Math.Max(0, (int)MathF.Ceiling((sx0 - x0) / dx)), i1 = Math.Min(Stations - 1, (int)MathF.Floor((sx1 - x0) / dx));
                    for (int i = i0; i <= i1; i++)
                    {
                        float x = x0 + dx * i;
                        float f = sx1 - sx0 > 1e-6f ? (x - s0.X) / (s1.X - s0.X) : 0f;
                        float z = MathF.Abs(s0.Z + (s1.Z - s0.Z) * Math.Clamp(f, 0f, 1f));
                        ref float slot = ref hull.Breadth[k * Stations + i];
                        if (z > slot) slot = z;
                    }
                }
            }
            for (int i = 0; i < hull.Breadth.Length; i++)
                hull.Breadth[i] = MathF.Max(0f, hull.Breadth[i] - Inset);
            return hull;
        }

        private static void Cross(Vector3 a, Vector3 b, float y, Span<Vector3> into, ref int n)
        {
            if (n >= 2 || (a.Y < y) == (b.Y < y)) return;
            float t = (y - a.Y) / (b.Y - a.Y);
            into[n++] = a + (b - a) * t;
        }
    }

    /// <summary>
    /// The hulls that keep the water out (<see cref="WaterHull"/>): game code adds a ship's and places it whenever the ship
    /// moves; the water's shaders get the nearest <see cref="MaxDrawn"/> to the camera each frame (<see cref="Srv"/>,
    /// <see cref="Count"/>), and <see cref="Inside"/> answers for the CPU. Docs: .agent/knowledge/engine_internals/ships.md.
    /// </summary>
    public static class WaterHulls
    {
        /// <summary>Hulls the shaders test at most: the nearest to the camera.</summary>
        public const int MaxDrawn = 16;

        /// <summary>Hulls farther than this from the camera are not drawn as such (m).</summary>
        public const float DrawnWithin = 3000f;

        /// <summary>water_hulls.hlsli WaterHull: the world-to-hull rows, a bounding sphere, the extent, the half-breadths.</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private unsafe struct Gpu
        {
            public Vector4 Row0, Row1, Row2;
            public Vector4 Sphere;
            public float X0, X1, Bottom, Top;
            public fixed float Breadth[WaterHull.Stations * WaterHull.Levels];
        }

        private const int FrameCount = 3;

        // Copied on change: readers on any thread walk the array they took
        private static WaterHull[] _all = Array.Empty<WaterHull>();
        private static readonly object _lock = new();
        private static readonly GraphicsBuffer?[] _buffers = new GraphicsBuffer?[FrameCount];
        private static readonly GraphicsBuffer?[] _seaBuffers = new GraphicsBuffer?[FrameCount];
        private static readonly GraphicsBuffer?[] _wetBuffers = new GraphicsBuffer?[FrameCount];
        private static int _uploadedTick = -1;
        private static uint _srv, _count, _seaSrv, _wetSrv;
        private static readonly List<(float Distance, WaterHull Hull)> _nearest = new();

        /// <summary>Where the camera was when the water was last drawn: the hulls nearest it are uploaded.</summary>
        internal static Vector3 Eye;

        public static void Add(WaterHull hull)
        {
            lock (_lock)
            {
                if (Array.IndexOf(_all, hull) >= 0) return;
                var next = new WaterHull[_all.Length + 1];
                _all.CopyTo(next, 0);
                next[^1] = hull;
                _all = next;
            }
        }

        public static void Remove(WaterHull hull)
        {
            lock (_lock)
            {
                int at = Array.IndexOf(_all, hull);
                if (at < 0) return;
                var next = new WaterHull[_all.Length - 1];
                for (int i = 0, j = 0; i < _all.Length; i++)
                    if (i != at) next[j++] = _all[i];
                _all = next;
            }
        }

        /// <summary>Whether a world point is inside any hull. Any thread.</summary>
        public static bool Inside(Vector3 world)
        {
            var all = _all;
            for (int i = 0; i < all.Length; i++)
                if (all[i].Contains(world)) return true;
            return false;
        }

        /// <summary>Whether any hull reaches within <paramref name="distance"/> of a point, across the water. Any thread.</summary>
        public static bool AnyWithin(Vector3 at, float distance)
        {
            var all = _all;
            for (int i = 0; i < all.Length; i++)
            {
                var hull = all[i];
                if (!hull.Enabled) continue;
                float dx = hull.Centre.X - at.X, dz = hull.Centre.Z - at.Z, r = distance + hull.Radius;
                if (dx * dx + dz * dz < r * r) return true;
            }
            return false;
        }

        /// <summary>The hulls the shaders test this frame (StructuredBuffer&lt;WaterHull&gt;), 0 when there are none.</summary>
        public static uint Srv
        {
            get
            {
                Upload();
                return _srv;
            }
        }

        /// <summary>How many hulls are in <see cref="Srv"/>.</summary>
        public static uint Count
        {
            get
            {
                Upload();
                return _count;
            }
        }

        /// <summary>
        /// Where the sea stands along each hull of <see cref="Srv"/>, in the same order (StructuredBuffer&lt;float&gt;,
        /// <see cref="WaterHull.Stations"/> a hull: <see cref="WaterHull.Sea"/>), then each one's
        /// <see cref="WaterHull.Travel"/> (at <see cref="Count"/> × Stations + its index); 0 when there are no hulls.
        /// </summary>
        public static uint SeaSrv
        {
            get
            {
                Upload();
                return _seaSrv;
            }
        }

        /// <summary>
        /// How high the water has wetted each hull of <see cref="Srv"/>, in the same order (StructuredBuffer&lt;float&gt;,
        /// 4 × <see cref="WaterHull.Stations"/> a hull: <see cref="WaterHull.Wet"/>, then <see cref="WaterHull.Fresh"/>),
        /// 0 when there are no hulls.
        /// </summary>
        public static uint WetSrv
        {
            get
            {
                Upload();
                return _wetSrv;
            }
        }

        // Once a tick, when the first water asks: the nearest hulls into this frame's upload buffer
        private static unsafe void Upload()
        {
            if (_uploadedTick == Engine.TickCount) return;
            _uploadedTick = Engine.TickCount;
            _srv = 0;
            _count = 0;
            _seaSrv = 0;
            _wetSrv = 0;
            var all = _all;
            if (all.Length == 0 || Engine.Device == null) return;

            _nearest.Clear();
            foreach (var hull in all)
            {
                if (!hull.Enabled) continue;
                float d = Vector3.Distance(hull.Centre, Eye) - hull.Radius;
                if (d < DrawnWithin) _nearest.Add((d, hull));
            }
            if (_nearest.Count == 0) return;
            _nearest.Sort((a, b) => a.Distance.CompareTo(b.Distance));
            int n = Math.Min(_nearest.Count, MaxDrawn);

            int slot = Engine.FrameIndex % FrameCount;   // a buffer per frame in flight, as the ocean's own data
            var buffer = _buffers[slot] ??= GraphicsBuffer.CreateUpload<Gpu>(MaxDrawn, mapped: true);
            var write = buffer.WritePtr<Gpu>();
            var seaBuffer = _seaBuffers[slot] ??= GraphicsBuffer.CreateUpload<float>(MaxDrawn * (WaterHull.Stations + 1), mapped: true);
            var sea = seaBuffer.WritePtr<float>();
            var wetBuffer = _wetBuffers[slot] ??= GraphicsBuffer.CreateUpload<float>(MaxDrawn * 4 * WaterHull.Stations, mapped: true);
            var wet = wetBuffer.WritePtr<float>();
            for (int i = 0; i < n; i++)
            {
                var hull = _nearest[i].Hull;
                var m = hull.WorldToLocal;
                ref var g = ref write[i];
                // local = world × M (row vectors): local.x = dot(column 0 of M, (world, 1))
                g.Row0 = new Vector4(m.M11, m.M21, m.M31, m.M41);
                g.Row1 = new Vector4(m.M12, m.M22, m.M32, m.M42);
                g.Row2 = new Vector4(m.M13, m.M23, m.M33, m.M43);
                g.Sphere = new Vector4(hull.Centre, hull.Radius);
                g.X0 = hull.X0;
                g.X1 = hull.X1;
                g.Bottom = hull.Bottom;
                g.Top = hull.Top;
                fixed (float* b = g.Breadth)
                    for (int k = 0; k < hull.Breadth.Length; k++) b[k] = hull.Breadth[k];
                for (int k = 0; k < WaterHull.Stations; k++)
                    sea[i * WaterHull.Stations + k] = hull.Sea[k];
                sea[n * WaterHull.Stations + i] = hull.Travel;
                for (int k = 0; k < 2 * WaterHull.Stations; k++)
                {
                    wet[i * 4 * WaterHull.Stations + k] = hull.Wet[k];
                    wet[i * 4 * WaterHull.Stations + 2 * WaterHull.Stations + k] = hull.Fresh[k];
                }
            }
            _srv = buffer.SrvIndex;
            _count = (uint)n;
            _seaSrv = seaBuffer.SrvIndex;
            _wetSrv = wetBuffer.SrvIndex;
        }
    }
}
