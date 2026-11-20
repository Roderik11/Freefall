using System;
using System.Numerics;
using System.Threading;
using Freefall.Base;

namespace Freefall.Components
{
    /// <summary>What a water surface belongs to.</summary>
    public enum WaterKind
    {
        None,
        /// <summary>The open sea (OceanRenderer), where its sea mask lets it in.</summary>
        Sea,
        /// <summary>A WaterBody on a closed spline.</summary>
        Lake,
        /// <summary>A WaterBody on an open spline: its surface falls along the river and carries a current.</summary>
        River,
    }

    /// <summary>The water over a place (<see cref="Water.SurfaceAt"/>).</summary>
    public struct WaterSurface
    {
        public WaterKind Kind;

        /// <summary>
        /// World height of the surface over the place: a lake's level, a river's surface there, the sea with its waves where
        /// they have been measured near the place lately (<see cref="Water.WatchSea"/>), else the sea at rest.
        /// </summary>
        public float Height;

        /// <summary>How much the surface rises per metre of x and of z there (a river's fall, a wave's flank).</summary>
        public Vector2 Slope;

        /// <summary>The current at the surface, metres a second over x and z (rivers; zero on still water and the sea).</summary>
        public Vector2 Flow;

        /// <summary>How frozen the water is, 0..1 (WaterBody.Ice; 0 for the sea).</summary>
        public float Ice;

        /// <summary>The lake or river; null for the sea.</summary>
        public WaterBody? Body;

        /// <summary>The ocean; null for a lake or river.</summary>
        public OceanRenderer? Ocean;

        /// <summary>How far under the surface a point is (negative: above it).</summary>
        public readonly float DepthOf(Vector3 point) => Height - point.Y;
    }

    /// <summary>
    /// Where the water is: the surface of the sea, the lakes and the rivers over a place, for whatever needs to know it on
    /// the CPU (a swimmer, the camera going under). Lakes and rivers answer from their built surface (WaterBody's lookup, made
    /// with the mesh); the sea from its sea mask, at rest, or with its waves where the underwater pass has measured them for
    /// someone who asked (<see cref="WatchSea"/>). Any thread: what it reads is replaced, never written, while it is read.
    /// Docs: .agent/knowledge/ocean_system/underwater.md.
    /// </summary>
    public static class Water
    {
        /// <summary>
        /// The water whose surface is over <paramref name="point"/>, or at most <paramref name="above"/> metres under it
        /// (a point just over the water still finds it). Where surfaces lie over one another (a lake in the hills over the
        /// sea's level, one river over another) the lowest that qualifies: the one the point is in or just over.
        /// </summary>
        public static bool SurfaceAt(Vector3 point, float above, out WaterSurface surface)
            => SurfaceAt(point, above, out surface, false);

        /// <summary>
        /// As <see cref="SurfaceAt(Vector3, float, out WaterSurface)"/>; inside a ship's hull (Graphics.WaterHulls) there is
        /// no water, unless <paramref name="throughHulls"/>: the water the hull floats in (its own buoyancy).
        /// </summary>
        public static bool SurfaceAt(Vector3 point, float above, out WaterSurface surface, bool throughHulls)
        {
            surface = default;
            if (!throughHulls && Graphics.WaterHulls.Inside(point)) return false;
            float best = float.MaxValue;
            float floor = point.Y - above;

            var bodies = ComponentCache<WaterBody>.All;
            for (int i = 0; i < bodies.Count; i++)
            {
                var body = bodies[i];
                if (body == null || !body.Enabled || body.Lookup is not { } lookup) continue;
                if (!lookup.HeightAt(point.X, point.Z, out float y, out var slope, out var flow)) continue;
                if (y < floor || y >= best) continue;
                best = y;
                surface = new WaterSurface
                {
                    Kind = lookup.Flowing ? WaterKind.River : WaterKind.Lake,
                    Height = y, Slope = slope, Flow = flow * body.FlowSpeed, Ice = body.Ice, Body = body,
                };
            }

            var oceans = ComponentCache<OceanRenderer>.All;
            for (int i = 0; i < oceans.Count; i++)
            {
                var ocean = oceans[i];
                if (ocean == null || !ocean.Enabled || !ocean.IsSea(point.X, point.Z)) continue;
                float y = ocean.PlaneY;
                var slope = Vector2.Zero;
                if (SeaWaveAt(new Vector2(point.X, point.Z), out float wave, out var waveSlope))
                {
                    y = wave;
                    slope = waveSlope;
                }
                if (y < floor || y >= best) continue;
                best = y;
                surface = new WaterSurface { Kind = WaterKind.Sea, Height = y, Slope = slope, Ocean = ocean };
            }
            return surface.Kind != WaterKind.None;
        }

        /// <summary>How far under water <paramref name="point"/> is (0 when it is not in any).</summary>
        public static float DepthAt(Vector3 point)
            => SurfaceAt(point, 0f, out var surface) ? MathF.Max(0f, surface.Height - point.Y) : 0f;

        // ── The sea's waves where somebody is: measured on the GPU by the underwater pass (Graphics.Underwater), read back
        // a few frames later. A place asked for is measured from the next frame on for as long as it keeps being asked for ──

        /// <summary>Places the sea's height is measured at, at most: swimmers and the points ships float on.</summary>
        public const int SeaProbes = 64;

        private static readonly object _watchLock = new();
        private static readonly Vector2[] _watched = new Vector2[SeaProbes];
        private static int _watchedCount;

        // The last read back: place, height, slope, and the tick it was measured at
        private static readonly Vector4[] _measured = new Vector4[SeaProbes + 1];   // x, z, height, (unused)
        private static readonly Vector2[] _measuredSlope = new Vector2[SeaProbes + 1];
        private static int _measuredCount;
        private static int _measuredTick = -1000;
        private static readonly object _measuredLock = new();

        /// <summary>
        /// Asks for the sea's waves to be measured at a place this frame and the next few (a swimmer at sea): the answer
        /// comes back through <see cref="SurfaceAt"/> a few frames later. Places within a metre of one asked for already
        /// share it; past <see cref="SeaProbes"/> places, the rest are not measured.
        /// </summary>
        public static void WatchSea(Vector2 xz)
        {
            lock (_watchLock)
            {
                for (int i = 0; i < _watchedCount; i++)
                    if (Vector2.DistanceSquared(_watched[i], xz) < 1f) return;
                if (_watchedCount < SeaProbes) _watched[_watchedCount++] = xz;
            }
        }

        /// <summary>The underwater pass takes this frame's places to measure (the camera's comes first, its own) and starts over.</summary>
        internal static int TakeWatched(Span<Vector2> into)
        {
            lock (_watchLock)
            {
                int n = Math.Min(_watchedCount, into.Length);
                for (int i = 0; i < n; i++) into[i] = _watched[i];
                _watchedCount = 0;
                return n;
            }
        }

        /// <summary>The underwater pass hands back what it read: per place its xz, the surface's height and slope.</summary>
        internal static void TakeMeasured(ReadOnlySpan<Vector2> places, ReadOnlySpan<Vector4> results)
        {
            lock (_measuredLock)
            {
                int n = Math.Min(places.Length, SeaProbes + 1);   // the camera's and every place asked for
                for (int i = 0; i < n; i++)
                {
                    _measured[i] = new Vector4(places[i].X, places[i].Y, results[i].X, 0f);
                    _measuredSlope[i] = new Vector2(results[i].Y, results[i].Z);
                }
                _measuredCount = n;
                _measuredTick = Engine.TickCount;
            }
        }

        // The sea's surface at a place measured lately (within 2 m of it, at most 20 ticks ago)
        private static bool SeaWaveAt(Vector2 xz, out float height, out Vector2 slope)
        {
            height = 0f;
            slope = Vector2.Zero;
            lock (_measuredLock)
            {
                if (Engine.TickCount - _measuredTick > 20) return false;
                float bestD = 4f;
                int best = -1;
                for (int i = 0; i < _measuredCount; i++)
                {
                    float d = Vector2.DistanceSquared(new Vector2(_measured[i].X, _measured[i].Y), xz);
                    if (d < bestD) { bestD = d; best = i; }
                }
                if (best < 0) return false;
                height = _measured[best].Z;
                slope = _measuredSlope[best];
                return true;
            }
        }
    }
}
