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

        /// <summary>
        /// Places the sea's height is measured at, at most, per frame: swimmers, the points ships float on, the line round the
        /// hull of the ship nearest the camera. It was 64, and with Low Town's moored ships and the NPC ships of the sea lanes
        /// near, a ship summoned last got none and rode the still sea level under the waves it was seen on (2026-11-22). The
        /// GPU measures a thousand places in the same pass for next to nothing; what kept it small was the CPU's lookup, a
        /// scan of every place, now a hash of cells.
        /// </summary>
        public const int SeaProbes = 1024;

        // Hashes of places by cell (power of two buckets, a chain through Next): the places asked for this frame by 1 m
        // cells (one within a metre of another shares it), the last read back by 2 m cells (an answer within 2 m counts)
        private const int Buckets = 4096;

        private static int Bucket(int cx, int cz) => (int)(((uint)cx * 73856093u) ^ ((uint)cz * 19349663u)) & (Buckets - 1);

        private static readonly object _watchLock = new();
        private static readonly Vector2[] _watched = new Vector2[SeaProbes];
        private static readonly int[] _watchedHead = NewHeads();
        private static readonly int[] _watchedNext = new int[SeaProbes];
        private static int _watchedCount;

        // The last read back: place, height, slope, and the tick it was measured at
        private static readonly Vector4[] _measured = new Vector4[SeaProbes + 1];   // x, z, height, (unused)
        private static readonly Vector2[] _measuredSlope = new Vector2[SeaProbes + 1];
        private static readonly int[] _measuredHead = NewHeads();
        private static readonly int[] _measuredNext = new int[SeaProbes + 1];
        private static int _measuredCount;
        private static int _measuredTick = -1000;
        private static readonly object _measuredLock = new();

        private static int[] NewHeads()
        {
            var heads = new int[Buckets];
            Array.Fill(heads, -1);
            return heads;
        }

        /// <summary>
        /// Asks for the sea's waves to be measured at a place this frame and the next few (a swimmer at sea): the answer
        /// comes back through <see cref="SurfaceAt"/> a few frames later. Places within a metre of one asked for already
        /// share it; past <see cref="SeaProbes"/> places, the rest are not measured, so whoever matters most asks first.
        /// </summary>
        public static void WatchSea(Vector2 xz)
        {
            int cx = (int)MathF.Floor(xz.X), cz = (int)MathF.Floor(xz.Y);
            lock (_watchLock)
            {
                for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                        for (int i = _watchedHead[Bucket(cx + dx, cz + dz)]; i >= 0; i = _watchedNext[i])
                            if (Vector2.DistanceSquared(_watched[i], xz) < 1f) return;
                if (_watchedCount >= SeaProbes) return;
                int at = _watchedCount++;
                int bucket = Bucket(cx, cz);
                _watched[at] = xz;
                _watchedNext[at] = _watchedHead[bucket];
                _watchedHead[bucket] = at;
            }
        }

        /// <summary>The underwater pass takes this frame's places to measure (the camera's comes first, its own) and starts over.</summary>
        internal static int TakeWatched(Span<Vector2> into)
        {
            lock (_watchLock)
            {
                int n = Math.Min(_watchedCount, into.Length);
                for (int i = 0; i < n; i++) into[i] = _watched[i];
                for (int i = 0; i < _watchedCount; i++)
                    _watchedHead[Bucket((int)MathF.Floor(_watched[i].X), (int)MathF.Floor(_watched[i].Y))] = -1;
                _watchedCount = 0;
                return n;
            }
        }

        /// <summary>The underwater pass hands back what it read: per place its xz, the surface's height and slope.</summary>
        internal static void TakeMeasured(ReadOnlySpan<Vector2> places, ReadOnlySpan<Vector4> results)
        {
            lock (_measuredLock)
            {
                for (int i = 0; i < _measuredCount; i++)
                    _measuredHead[MeasuredBucket(_measured[i].X, _measured[i].Y)] = -1;
                int n = Math.Min(places.Length, SeaProbes + 1);   // the camera's and every place asked for
                for (int i = 0; i < n; i++)
                {
                    _measured[i] = new Vector4(places[i].X, places[i].Y, results[i].X, 0f);
                    _measuredSlope[i] = new Vector2(results[i].Y, results[i].Z);
                    int bucket = MeasuredBucket(places[i].X, places[i].Y);
                    _measuredNext[i] = _measuredHead[bucket];
                    _measuredHead[bucket] = i;
                }
                _measuredCount = n;
                _measuredTick = Engine.TickCount;
            }
        }

        private static int MeasuredBucket(float x, float z) => Bucket((int)MathF.Floor(x * 0.5f), (int)MathF.Floor(z * 0.5f));

        // The sea's surface at a place measured lately (within 2 m of it, at most 20 ticks ago): the nearest of the places
        // read back in its 2 m cell and the eight round it
        private static bool SeaWaveAt(Vector2 xz, out float height, out Vector2 slope)
        {
            height = 0f;
            slope = Vector2.Zero;
            int cx = (int)MathF.Floor(xz.X * 0.5f), cz = (int)MathF.Floor(xz.Y * 0.5f);
            lock (_measuredLock)
            {
                if (Engine.TickCount - _measuredTick > 20) return false;
                float bestD = 4f;
                int best = -1;
                for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                        for (int i = _measuredHead[Bucket(cx + dx, cz + dz)]; i >= 0; i = _measuredNext[i])
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
