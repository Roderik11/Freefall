using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// The surf: the open sea's own waves carried to the beach. Near the coast the ocean samples a warped band not where a
    /// point is but where its wave was out at sea (u): the band's waves travel from deep water at the speed the depth
    /// allows, so their crests turn toward the beach (refraction), crowd together (shorter waves) and grow (shoaling, from
    /// how the energy between two rays is squeezed). What arrives is the sea's own field: a set out at sea comes in as a
    /// set. The FFT alone is deep water, endless and blind to the seabed: its crests would slide along a beach at whatever
    /// angle the wind blows.
    ///
    /// Baked on a worker from the terrain's height field and the sea mask, per warped band (its heading and frequency),
    /// by fast marching the wave's travel time from deep water, where it is the plane wave's own. ocean.fx
    /// (SampleBandWarp) reads two texture arrays, a slice per warped band, laid out like the heightmap (row = Z):
    /// RGBA32F (u − x, amplitude, up the beach how far below the ridge before it a spot lies) and RGBA16F (∂u/∂x:
    /// ∂ux/∂x, ∂ux/∂z, ∂uz/∂x, ∂uz/∂z).
    /// See .agent/knowledge/ocean_system/surf_plan.md.
    /// </summary>
    public sealed class ShoreWarp : IDisposable
    {
        /// <summary>Most cells along a side: 4 m on a 4 km terrain (a 4097² height field read at every 4th sample).</summary>
        public const int MaxSize = 1025;

        /// <summary>Warped bands, the texture arrays' slices (the swell and the wind sea's coarsest band).</summary>
        public const int MaxBands = 2;

        private const float Gravity = 9.81f;

        /// <summary>Over the beach (the sea mask's reach above the waterline) the rays run on as if over this much water (m).</summary>
        private const float MinDepth = 0.3f;

        /// <summary>The warp fades to nothing over this distance inside the terrain's border (m): past it the seabed is unknown.</summary>
        private const float BorderFade = 200f;

        /// <summary>The share of the swell's height that wraps round an island to its lee shore.</summary>
        private const float LeeShare = 0.3f;

        public struct Band
        {
            public int Index;            // the ocean band
            public Vector2 Direction;    // world xz unit vector its waves travel along in deep water
            public float Omega;          // its representative angular frequency (rad/s): the spectrum's peak
            /// <summary>
            /// The depth (m) from which the band is warped: deeper, its waves run as the open sea's (the march starts there
            /// as the plane wave), and the warp is whole from half of it. 0 = where its peak begins to feel the bottom
            /// (2/k0). The wind sea's coarsest band gives its shortest waves' length (2026-11-23): it holds waves a quarter
            /// of its peak's length and longer, and is warped as a whole by its peak's shortening. Since storms grew long
            /// (WindSea.StormLength) an 18 m/s peak of 157 m felt TestWorld's whole 40 m floor, and the band's short waves,
            /// which feel none of it, were squeezed as much: crests stacked a few metres apart, folding over.
            /// </summary>
            public float WarpDepth;

            /// <summary>The depth from which the band is warped (<see cref="WarpDepth"/>, at most where its peak feels the bottom).</summary>
            public float SourceDepth(float k0) => WarpDepth > 0f ? MathF.Min(WarpDepth, 2f / k0) : 2f / k0;

            public bool SameAs(Band o) =>
                Index == o.Index && Vector2.Dot(Direction, o.Direction) > 0.9994f      // 2°
                && MathF.Abs(Omega - o.Omega) <= 0.02f * o.Omega && WarpDepth == o.WarpDepth;
        }

        /// <summary>A baked warp, ready to upload.</summary>
        public sealed class Result
        {
            public int Size;
            public Band[] Bands = Array.Empty<Band>();
            public float[] Warp = Array.Empty<float>();        // slice-major, Size² × (Δu.x, Δu.z, amplitude, hollow m)
            public Half[] Jacobian = Array.Empty<Half>();      // slice-major, Size² × (∂ux/∂x, ∂ux/∂z, ∂uz/∂x, ∂uz/∂z)
            public double Milliseconds;
            /// <summary>What the surf's pages are marched from (<see cref="BakePage"/>): the waves' travel time over the whole
            /// terrain at this bake's cells.</summary>
            public Seeds? Seeds;
        }

        /// <summary>
        /// The whole terrain's march, kept for the surf's pages (<see cref="SurfPages"/>): a page takes its travel time and
        /// height, and marches only its coast strip from that travel time, so neighbouring pages, and a page and this coarse
        /// warp beside it, carry the same waves. Never changed once built (any thread reads it).
        /// </summary>
        public sealed class Seeds
        {
            public int Size;
            public float Hx, Hz;
            public Vector2 Origin;                  // world xz of cell 0
            public Band[] Bands = Array.Empty<Band>();
            /// <summary>Per cell: 0 land, 1 beach (the sea mask's reach above the waterline), 2 water, 3 water deeper than
            /// <see cref="PlainDepth"/> (the bed's detail does not matter there).</summary>
            public byte[] Class = Array.Empty<byte>();
            /// <summary>Per slice, per cell: the travel time (s); NaN where the march did not reach.</summary>
            public float[][] Tau = Array.Empty<float[]>();
            /// <summary>Per slice, per cell: the wave's height over the open sea's, as this bake gives it (shoaling, the rays'
            /// crowding, the island's shadow, smooth over a quarter wavelength), and spread a few cells into the land beside
            /// the sea, where a page's finer water lies. A page takes it as it is: from each page's own rays it was noise at
            /// fine cells and jumped from page to page, and with the shoaling of the page's own depth unblurred a narrow
            /// shallow inlet raised breakers the whole terrain's bake never had.</summary>
            public float[][] Amp = Array.Empty<float[]>();
        }

        /// <summary>Water deeper than this (m) is plain for the surf: a page over nothing else keeps the coarse warp.</summary>
        public const float PlainDepth = 25f;

        private ID3D12Resource? _warpTex, _jacobianTex;
        private uint _warpSrv, _jacobianSrv;
        private int _size;
        private Band[] _bands = Array.Empty<Band>();

        public uint WarpSrv => _warpSrv;
        public uint JacobianSrv => _jacobianSrv;

        /// <summary>The bands the uploaded warp was baked for, slice by slice.</summary>
        public Band[] Bands => _bands;

        /// <summary>
        /// Bake the warp of each band over the terrain's heights (the CPU store: values 0..1 of <paramref name="maxHeight"/>
        /// above the terrain's base). <paramref name="mask"/>: the sea mask (where the sea is, beaches included); null =
        /// wherever the ground is below sea level. Any thread: a store is never changed.
        /// </summary>
        public static Result Build(Assets.TerrainPages heights, SeaMask.Result? mask, float seaLevel01, float maxHeight,
                                   Vector2 terrainSize, Vector2 origin, Band[] bands)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            int res = heights.Resolution;
            int step = 1;
            while ((res - 1) / step + 1 > MaxSize) step *= 2;
            int n = (res - 1) / step + 1;
            int count = n * n;
            float hx = terrainSize.X / (n - 1), hz = terrainSize.Y / (n - 1);

            // The bake's scratch lives in native memory (NativeGrid): as arrays it sent ~470 MiB of large objects through
            // the managed heap per bake of a 1025² warp (ram.md). Only the result is managed
            using var depth = new NativeGrid<float>(count);
            using var sea = new NativeGrid<bool>(count);
            using var borderDist = new NativeGrid<float>(count);
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                depth[i] = (seaLevel01 - heights.Sample01(x * step, z * step)) * maxHeight;
                if (mask != null && mask.Size > 1)
                {
                    int mx = (int)MathF.Round(x * (mask.Size - 1) / (float)(n - 1));
                    int mz = (int)MathF.Round(z * (mask.Size - 1) / (float)(n - 1));
                    sea[i] = mask.Cells[mz * mask.Size + mx] != 0;
                }
                else sea[i] = depth[i] > 0f;
                borderDist[i] = MathF.Min(MathF.Min(x, n - 1 - x) * hx, MathF.Min(z, n - 1 - z) * hz);
            }

            int slices = Math.Min(bands.Length, MaxBands);
            var seeds = new Seeds
            {
                Size = n, Hx = hx, Hz = hz, Origin = origin, Bands = bands[..slices],
                Class = new byte[count],
                Tau = new float[slices][], Amp = new float[slices][],
            };
            for (int i = 0; i < count; i++)
                seeds.Class[i] = (byte)(!sea[i] ? 0 : depth[i] <= 0f ? 1 : depth[i] < PlainDepth ? 2 : 3);
            var result = new Result
            {
                Size = n,
                Bands = bands[..slices],
                Warp = new float[count * 4 * MaxBands],
                Jacobian = new Half[count * 4 * MaxBands],
                Seeds = seeds,
            };

            // Unused slices: no warp
            for (int s = slices; s < MaxBands; s++)
                FillIdentity(result, s, count);

            // One queue for every march of the bake: it grows once
            var queue = new PriorityQueue<int, float>();
            using var hollow = BeachHollows(n, depth, sea, queue);
            for (int s = 0; s < slices; s++)
                BakeBand(result, s, bands[s], n, hx, hz, origin, depth, sea, borderDist, hollow, queue);

            result.Milliseconds = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            return result;
        }

        // Up the beach (the sea mask's reach above the waterline): how far below the ridge between it and the sea each cell
        // lies (m), 0 on a plain slope. The spill level of a priority flood from the water: the lowest level at which the
        // swash reaches the cell, over any path; minus the cell's own height. ocean.fx keeps a hollow dry until the run-up
        // tops its ridge. (Carried along the marched rays instead, the 4-neighbour grid lifted plain slope cells to a
        // neighbour's height and the waterline frayed into grid stairs.)
        private static NativeGrid<float> BeachHollows(int n, NativeGrid<float> depth, NativeGrid<bool> sea, PriorityQueue<int, float> queue)
        {
            int count = n * n;
            using var spill = new NativeGrid<float>(count);
            queue.Clear();
            for (int i = 0; i < count; i++)
            {
                spill[i] = float.PositiveInfinity;
                if (sea[i] && depth[i] > 0f) { spill[i] = 0f; queue.Enqueue(i, 0f); }
            }
            while (queue.TryDequeue(out int i, out float level))
            {
                if (level > spill[i]) continue;
                int x = i % n, z = i / n;
                for (int dz = -1; dz <= 1; dz++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int xx = x + dx, zz = z + dz;
                    if ((dx | dz) == 0 || xx < 0 || zz < 0 || xx >= n || zz >= n) continue;
                    int j = zz * n + xx;
                    if (!sea[j] || depth[j] > 0f) continue;
                    float l = MathF.Max(level, -depth[j]);
                    if (l < spill[j]) { spill[j] = l; queue.Enqueue(j, l); }
                }
            }
            var hollow = new NativeGrid<float>(count);
            for (int i = 0; i < count; i++)
                if (sea[i] && depth[i] <= 0f && spill[i] < float.PositiveInfinity)
                    hollow[i] = MathF.Max(spill[i] + depth[i], 0f);
            return hollow;
        }

        private static void FillIdentity(Result r, int slice, int count)
        {
            int o = slice * count * 4;
            for (int i = 0; i < count; i++)
            {
                r.Warp[o + i * 4 + 2] = 1f;
                r.Jacobian[o + i * 4 + 0] = (Half)1f;
                r.Jacobian[o + i * 4 + 3] = (Half)1f;
            }
        }

        // kh of a wave of angular frequency ω over water h deep (linear dispersion ω² = g k tanh(kh)): Eckart's
        // approximation, then two Newton steps
        private static float WaveNumberDepth(float omega, float h)
        {
            float x0 = omega * omega * h / Gravity;                  // k0·h
            float y = x0 / MathF.Sqrt(MathF.Tanh(MathF.Max(x0, 1e-6f)));
            for (int it = 0; it < 2; it++)
            {
                float t = MathF.Tanh(y);
                float f = y * t - x0;
                float df = t + y * (1f - t * t);
                y -= f / MathF.Max(df, 1e-6f);
            }
            return MathF.Max(y, 1e-6f);
        }

        // Group over phase speed: 1/2 in deep water, 1 in shallow
        private static float GroupRatio(float kh) =>
            kh > 10f ? 0.5f : 0.5f * (1f + 2f * kh / MathF.Sinh(2f * kh));

        // Fast marching of a band's travel time over a square grid of n² cells: cells become known in order of arrival time,
        // each from its known neighbours (first-order upwind). Along with the time each cell takes from where its ray came:
        // the ray's along-crest coordinate (it keeps it all the way) and the exposure. Cells known at the start (state 2)
        // are where the march begins: the deep sea and the border (sources), or a page's edge seeded from the whole
        // terrain's march
        private sealed class Marcher
        {
            private readonly int _n;
            private readonly float _hx, _hz;
            private readonly Vector2 _dir;
            private readonly NativeGrid<float> _slowness, _tau, _side, _exposure;
            private readonly NativeGrid<byte> _state;          // 0 far, 1 trial, 2 known
            private readonly NativeGrid<bool> _source;
            private readonly PriorityQueue<int, float> _queue;

            public Marcher(int n, float hx, float hz, Vector2 dir, NativeGrid<float> slowness, NativeGrid<float> tau,
                           NativeGrid<float> side, NativeGrid<float> exposure, NativeGrid<byte> state, NativeGrid<bool> source,
                           PriorityQueue<int, float> queue)
            {
                _n = n; _hx = hx; _hz = hz; _dir = dir;
                _slowness = slowness; _tau = tau; _side = side; _exposure = exposure; _state = state; _source = source;
                _queue = queue;
            }

            private bool Relax(int i)
            {
                int n = _n, x = i % n, z = i / n;
                var tau = _tau; var state = _state;
                int nxI = -1, nzI = -1;
                float a = float.PositiveInfinity, b = float.PositiveInfinity;
                float sx = 0f, sz = 0f;
                if (x > 0 && state[i - 1] == 2 && tau[i - 1] < a) { a = tau[i - 1]; nxI = i - 1; sx = 1f; }
                if (x < n - 1 && state[i + 1] == 2 && tau[i + 1] < a) { a = tau[i + 1]; nxI = i + 1; sx = -1f; }
                if (z > 0 && state[i - n] == 2 && tau[i - n] < b) { b = tau[i - n]; nzI = i - n; sz = 1f; }
                if (z < n - 1 && state[i + n] == 2 && tau[i + n] < b) { b = tau[i + n]; nzI = i + n; sz = -1f; }
                if (nxI < 0 && nzI < 0) return false;

                float hx = _hx, hz = _hz;
                float s = _slowness[i];
                float t;
                float gx = 0f, gz = 0f;
                bool both = false;
                if (nxI >= 0 && nzI >= 0)
                {
                    // (t − a)²/hx² + (t − b)²/hz² = s²
                    float ix = 1f / (hx * hx), iz = 1f / (hz * hz);
                    float A = ix + iz;
                    float B = -2f * (a * ix + b * iz);
                    float C = a * a * ix + b * b * iz - s * s;
                    float disc = B * B - 4f * A * C;
                    t = disc >= 0f ? (-B + MathF.Sqrt(disc)) / (2f * A) : float.NegativeInfinity;
                    both = t >= MathF.Max(a, b);
                }
                else t = float.NegativeInfinity;
                if (!both)
                {
                    float ta = nxI >= 0 ? a + s * hx : float.PositiveInfinity;
                    float tb = nzI >= 0 ? b + s * hz : float.PositiveInfinity;
                    if (ta <= tb) { t = ta; nzI = -1; } else { t = tb; nxI = -1; }
                }
                if (t >= tau[i]) return false;

                // The ray comes in along the gradient: it crosses the segment between the two neighbours at weights in
                // proportion to the gradient's components (over the cell sizes)
                if (nxI >= 0) gx = sx * (t - a) / hx;
                if (nzI >= 0) gz = sz * (t - b) / hz;
                float wx = nxI >= 0 ? MathF.Abs(gx) * hz : 0f;
                float wz = nzI >= 0 ? MathF.Abs(gz) * hx : 0f;
                float wsum = wx + wz;
                if (wsum <= 0f) { wx = nxI >= 0 ? 1f : 0f; wz = 1f - wx; wsum = 1f; }
                wx /= wsum; wz /= wsum;

                float sideUp = 0f, eUp = 0f;
                float gl = MathF.Sqrt(gx * gx + gz * gz);
                float along = gl > 0f ? (gx * _dir.X + gz * _dir.Y) / gl : 1f;
                // A ray leaving the deep sea against the wave reaches a lee shore: the island's shadow, where only the swell
                // that wraps round it arrives (LeeShare of its height; shoreward, as the lee rays run). With none, the
                // open sea off a lee coast showed the full swell and the beach no surf at all (user, 2026-10-19)
                float leave = LeeShare + (1f - LeeShare) * Smooth(-0.1f, 0.4f, along);
                if (nxI >= 0)
                {
                    sideUp += wx * _side[nxI];
                    eUp += wx * _exposure[nxI] * (_source[nxI] ? leave : 1f);
                }
                if (nzI >= 0)
                {
                    sideUp += wz * _side[nzI];
                    eUp += wz * _exposure[nzI] * (_source[nzI] ? leave : 1f);
                }

                tau[i] = t;
                _side[i] = sideUp;
                _exposure[i] = eUp;
                return true;
            }

            private void Push(int i)
            {
                if (Relax(i))
                {
                    _state[i] = 1;
                    _queue.Enqueue(i, _tau[i]);
                }
            }

            // From every known cell to its neighbours in the phase's cells
            private void PushNeighbours(int i, NativeGrid<bool> allowed)
            {
                int n = _n, x = i % n, z = i / n;
                var state = _state;
                if (x > 0 && allowed[i - 1] && state[i - 1] != 2) Push(i - 1);
                if (x < n - 1 && allowed[i + 1] && state[i + 1] != 2) Push(i + 1);
                if (z > 0 && allowed[i - n] && state[i - n] != 2) Push(i - n);
                if (z < n - 1 && allowed[i + n] && state[i + n] != 2) Push(i + n);
            }

            /// <summary>March over the allowed cells from every cell known so far, until none is left.</summary>
            public void March(NativeGrid<bool> allowed)
            {
                _queue.Clear();
                int count = _n * _n;
                for (int i = 0; i < count; i++)
                    if (_state[i] == 2) PushNeighbours(i, allowed);
                while (_queue.TryDequeue(out int i, out float t))
                {
                    if (_state[i] == 2 || t > _tau[i]) continue;
                    _state[i] = 2;
                    PushNeighbours(i, allowed);
                }
            }
        }

        private static void BakeBand(Result r, int slice, Band band, int n, float hx, float hz, Vector2 origin,
                                     NativeGrid<float> depth, NativeGrid<bool> sea, NativeGrid<float> borderDist,
                                     NativeGrid<float> hollow, PriorityQueue<int, float> queue)
        {
            int count = n * n;
            float omega = MathF.Max(band.Omega, 0.05f);
            float k0 = omega * omega / Gravity;
            float c0 = Gravity / omega;
            Vector2 dir = band.Direction.LengthSquared() > 1e-6f ? Vector2.Normalize(band.Direction) : Vector2.UnitX;

            // Deep enough that the wave hardly feels the bottom (k0·h ≥ 2: 98% of its deep-water speed), or for the wind sea
            // than its shortest waves feel it (Band.WarpDepth): the plane wave there, and on the terrain's border (past it
            // the seabed is unknown)
            float sourceDepth = band.SourceDepth(k0);

            // Per cell: the travel time, the ray's along-crest coordinate where it left the deep sea (the deep heading turned
            // left, d⊥·x there) and the exposure
            var dPerp = new Vector2(-dir.Y, dir.X);
            using var slowness = new NativeGrid<float>(count);
            using var tau = new NativeGrid<float>(count);
            using var side = new NativeGrid<float>(count);
            using var exposure = new NativeGrid<float>(count);
            using var state = new NativeGrid<byte>(count);          // 0 far, 1 trial, 2 known
            using var source = new NativeGrid<bool>(count);

            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                tau[i] = float.PositiveInfinity;
                if (!sea[i]) continue;
                float h = MathF.Max(depth[i], MinDepth);
                float kh = WaveNumberDepth(omega, h);
                // Up the beach (the sea mask's reach above the waterline) the warp holds still: the swash there runs on
                // the phase of the wave arriving at the waterline (ocean.fx)
                slowness[i] = depth[i] > 0f ? kh / h / omega : 1f / c0;  // 1 / c = k / ω

                bool border = x == 0 || z == 0 || x == n - 1 || z == n - 1;
                if (depth[i] >= sourceDepth || border)
                {
                    var p = origin + new Vector2(x * hx, z * hz);
                    tau[i] = Vector2.Dot(dir, p) / c0;
                    side[i] = Vector2.Dot(dPerp, p);
                    exposure[i] = 1f;
                    state[i] = 2;
                    source[i] = true;
                }
            }

            // The water first, then up the beach from it. Up the beach the rays run at the deep speed (the warp holds the
            // waterline's phase there), and marched together with the water that fast strip carried the waves along the
            // sand ahead of the shallows and back into them: foam surging out to sea against the incoming waves and white
            // hooks racing along the shore (user, 2026-10-19). Marched after the water, the beach only takes from it
            using var water = new NativeGrid<bool>(count);
            using var beach = new NativeGrid<bool>(count);
            for (int i = 0; i < count; i++)
            {
                water[i] = sea[i] && depth[i] > 0f;
                beach[i] = sea[i] && !water[i];
            }
            var marcher = new Marcher(n, hx, hz, dir, slowness, tau, side, exposure, state, source, queue);
            marcher.March(water);
            marcher.March(beach);

            using var known = new NativeGrid<bool>(count);
            for (int i = 0; i < count; i++) known[i] = sea[i] && state[i] == 2;

            // Smooth what first-order marching leaves: the travel time's corners where the rays from two sides meet (a crease
            // in the crests) a little, in symmetric windows only (on a straight slope that changes nothing); the rays'
            // along-crest coordinate, which marching copies from one neighbour or the other in steps, over a quarter
            // wavelength like the height (it only sets how far the rays spread)
            int radius = Math.Clamp((int)MathF.Round(0.25f * (2f * MathF.PI / k0) / MathF.Max(hx, hz)), 1, 24);
            SmoothSymmetric(tau, n, known, 2);
            for (int pass = 0; pass < 2; pass++)
                BoxBlurWide(side, n, known, radius);

            // What the pages take away from the coast and march their coast strip from: this travel time
            if (r.Seeds is { } seeds && slice < seeds.Tau.Length)
            {
                var st = seeds.Tau[slice] = new float[count];
                for (int i = 0; i < count; i++)
                    st[i] = known[i] ? tau[i] : float.NaN;
            }

            // The sample position u: along the deep heading as far as the wave would have run in deep water in the same
            // time (c0·τ, exact and smooth), across it where the point is: u = d·c0·τ + d⊥·(d⊥·x). Until 2026-10-19 u was
            // carried along the rays as a whole, and the copied cross part stepped the breakers' crests sideways in stairs
            // (user screenshots). J = d ⊗ c0∇τ + d⊥ ⊗ d⊥ turns the slopes. The height comes from the rays themselves:
            // the energy flux between two rays is kept, det(∂(c0τ, side)/∂x) = c0·(∇τ × ∇side) = (c0/c)·(b0/b), so
            // H/H0 = √(det · ½ / n), n the group over phase speed. Then the blend back to no warp toward deep water and the
            // terrain's border
            using var jac = new NativeGrid<float>(count * 4);
            using var amp = new NativeGrid<float>(count);
            using var blend = new NativeGrid<float>(count);
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                jac[i * 4 + 0] = 1f; jac[i * 4 + 3] = 1f;
                amp[i] = 1f; blend[i] = 1f;
                if (!known[i]) continue;

                // Central differences where both neighbours are sea, one-sided at the coast
                bool l = x > 0 && known[i - 1], rt = x < n - 1 && known[i + 1];
                bool d = z > 0 && known[i - n], u = z < n - 1 && known[i + n];
                int x0 = l ? i - 1 : i, x1 = rt ? i + 1 : i;
                int z0 = d ? i - n : i, z1 = u ? i + n : i;
                float dxs = (x1 - x0) * hx, dzs = ((z1 - z0) / n) * hz;
                float tx = dxs > 0f ? (tau[x1] - tau[x0]) / dxs : dir.X / c0;
                float tz = dzs > 0f ? (tau[z1] - tau[z0]) / dzs : dir.Y / c0;
                float sx = dxs > 0f ? (side[x1] - side[x0]) / dxs : dPerp.X;
                float sz = dzs > 0f ? (side[z1] - side[z0]) / dzs : dPerp.Y;

                jac[i * 4 + 0] = dir.X * c0 * tx + dPerp.X * dPerp.X;
                jac[i * 4 + 1] = dir.X * c0 * tz + dPerp.X * dPerp.Y;
                jac[i * 4 + 2] = dir.Y * c0 * tx + dPerp.Y * dPerp.X;
                jac[i * 4 + 3] = dir.Y * c0 * tz + dPerp.Y * dPerp.Y;

                float h = MathF.Max(depth[i], MinDepth);
                float kh = WaveNumberDepth(omega, h);
                // (its size: on a lee shore the rays run the other way round and the determinant is negative)
                float det = MathF.Abs(c0 * (tx * sz - tz * sx));
                float k = MathF.Sqrt(Math.Clamp(det * 0.5f / GroupRatio(kh), 0f, 6f));
                amp[i] = k * exposure[i];

                blend[i] = MathF.Max(Smooth(0.5f * sourceDepth, sourceDepth, depth[i]),
                                     1f - Smooth(0f, BorderFade, borderDist[i]));
            }

            // The Jacobian a little (the slopes' grid noise), the height over about a quarter wavelength: a wave's height
            // cannot change faster than that (diffraction blurs the edge of an island's shadow, where the rays from its two
            // sides meet and the exposure jumps)
            BoxBlur(jac, 4, n, sea);
            for (int pass = 0; pass < 3; pass++)
                BoxBlurWide(amp, n, sea, radius);

            // The pages' height: this one, spread into the land beside the sea (DilateSeed)
            if (r.Seeds is { } ampSeeds && slice < ampSeeds.Amp.Length)
            {
                var sa = ampSeeds.Amp[slice] = new float[count];
                for (int i = 0; i < count; i++)
                    sa[i] = known[i] ? amp[i] : float.NaN;
                DilateSeed(sa, n, 4);
            }

            int o = slice * count * 4;
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                float w = blend[i];
                var p = origin + new Vector2(x * hx, z * hz);
                bool ok = known[i];
                // u − x = d·(c0·τ − d·x): the warp moves a point along the deep heading only
                float ahead = ok ? (c0 * tau[i] - Vector2.Dot(dir, p)) * (1f - w) : 0f;
                r.Warp[o + i * 4 + 0] = dir.X * ahead;
                r.Warp[o + i * 4 + 1] = dir.Y * ahead;
                r.Warp[o + i * 4 + 2] = ok ? amp[i] + (1f - amp[i]) * w : 1f;
                // How far below the ridge before it a beach cell lies (BeachHollows: 0 on a plain slope): stored as that,
                // not as the ridge's height, so interpolating between cells cannot lift a ridge above the real sand
                r.Warp[o + i * 4 + 3] = hollow[i];
                for (int c = 0; c < 4; c++)
                {
                    float id = c == 0 || c == 3 ? 1f : 0f;
                    r.Jacobian[o + i * 4 + c] = (Half)(ok ? jac[i * 4 + c] + (id - jac[i * 4 + c]) * w : id);
                }
            }

            // Land next to the sea takes its neighbours' warp, so bilinear filtering at the coast blends warp with warp
            DilateIntoLand(r, o, n, sea, state);

            DebugDump(r, o, slice, n, hx, hz, origin, k0, dir, sea, exposure, depth);
        }

        // ── The surf's pages (SurfPages, 2026-11-23) ──
        // The whole terrain's warp above is 1025² cells at most: 4 m on a 4 km terrain, 16 m on TestWorld's 16 km, where its
        // stairs showed in the swash's terraces and in the breakers along the shore. Round the camera the surf is baked again
        // in pages on the terrain's page grid (256 texels a side, level 0 at the height samples' own spacing, each level
        // twice the last). A page takes the whole terrain's travel time and height as they are, the travel time bicubic, and
        // marches at its own cells only a strip along the coast (StripCells), from the strip's seaward edge: what a strip
        // cell gets depends on the strip alone, which the page's margin covers, so pages meet each other and the coarse
        // warp without a step (ocean_system/overview.md §3f-2).

        /// <summary>Texels a page holds a side, the border round it in its slot, and the margin marched round that.</summary>
        public const int PageTexels = 256, PageBorder = 4, SlotTexels = PageTexels + 2 * PageBorder, PageMargin = 32;

        /// <summary>One page to bake: where it is and what it is marched from. Every input is never changed once handed out.</summary>
        public sealed class PageJob
        {
            public int Level, Px, Py;
            public Seeds Seeds = null!;
            public Assets.TerrainPages Heights = null!;
            public SeaMask.Result? Mask;
            public float SeaLevel01, MaxHeight;
            public Vector2 TerrainSize, Origin;
        }

        /// <summary>A baked page, ready to upload into its slot (<see cref="SlotTexels"/>² texels, border included).</summary>
        public sealed class Page
        {
            public int Level, Px, Py;
            public Seeds Seeds = null!;
            /// <summary>Slice-major, per slice (warped band) SlotTexels² × (lag − <see cref="Base"/>, amplitude, g.x, g.z):
            /// the warp moves a point only along the band's deep heading d, by the lag; ∂u/∂x = d ⊗ g + d⊥ ⊗ d⊥.</summary>
            public Half[] Data = Array.Empty<Half>();
            /// <summary>SlotTexels²: up the beach, how far below the ridge before it a spot lies (m).</summary>
            public Half[] Hollow = Array.Empty<Half>();
            /// <summary>Per slice: the lag the page's values are kept relative to (half floats near 0 keep their precision).</summary>
            public float[] Base = new float[MaxBands];
            public double Milliseconds;
        }

        /// <summary>
        /// Whether a page (with its margin) needs its own surf: it holds a beach, water shallower than <see cref="PlainDepth"/>,
        /// or land beside deep water. Elsewhere the coarse warp is as good. From the whole terrain's cells.
        /// </summary>
        public static bool PageWanted(Seeds seeds, int level, int px, int py, int samples, Vector2 terrainSize)
        {
            int n = seeds.Size;
            float cell0 = terrainSize.X / Math.Max(samples - 1, 1);
            float span = PageTexels * (1 << level) * cell0, margin = (PageBorder + PageMargin) * (1 << level) * cell0;
            int x0 = Math.Clamp((int)MathF.Floor((px * span - margin) / seeds.Hx), 0, n - 1);
            int x1 = Math.Clamp((int)MathF.Ceiling(((px + 1) * span + margin) / seeds.Hx), 0, n - 1);
            int z0 = Math.Clamp((int)MathF.Floor((py * span - margin) / seeds.Hz), 0, n - 1);
            int z1 = Math.Clamp((int)MathF.Ceiling(((py + 1) * span + margin) / seeds.Hz), 0, n - 1);
            bool land = false, deep = false;
            for (int z = z0; z <= z1; z++)
                for (int x = x0; x <= x1; x++)
                {
                    byte c = seeds.Class[z * n + x];
                    if (c == 1 || c == 2) return true;
                    if (c == 0) land = true; else deep = true;
                    if (land && deep) return true;
                }
            return false;
        }

        /// <summary>The strip along the coast a page marches at its own cells, in the whole terrain's cells (3: 48 m on a 16 km
        /// terrain, 12 m on a 4 km one). Beyond it the page takes the whole terrain's travel time.</summary>
        public const float StripCells = 3f;

        // Per cell: how far (m) the nearest cell that is not water lies (land, beach, or ground the sea does not reach); 0 on
        // those. Two chamfer passes (straight and diagonal steps)
        private static NativeGrid<float> CoastDistance(int n, float hx, float hz, NativeGrid<bool> sea, NativeGrid<float> depth)
        {
            var d = new NativeGrid<float>(n * n);
            float hd = MathF.Sqrt(hx * hx + hz * hz);
            for (int i = 0; i < n * n; i++) d[i] = sea[i] && depth[i] > 0f ? float.PositiveInfinity : 0f;
            for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    int i = z * n + x;
                    float v = d[i];
                    if (x > 0) v = MathF.Min(v, d[i - 1] + hx);
                    if (z > 0)
                    {
                        v = MathF.Min(v, d[i - n] + hz);
                        if (x > 0) v = MathF.Min(v, d[i - n - 1] + hd);
                        if (x < n - 1) v = MathF.Min(v, d[i - n + 1] + hd);
                    }
                    d[i] = v;
                }
            for (int z = n - 1; z >= 0; z--)
                for (int x = n - 1; x >= 0; x--)
                {
                    int i = z * n + x;
                    float v = d[i];
                    if (x < n - 1) v = MathF.Min(v, d[i + 1] + hx);
                    if (z < n - 1)
                    {
                        v = MathF.Min(v, d[i + n] + hz);
                        if (x < n - 1) v = MathF.Min(v, d[i + n + 1] + hd);
                        if (x > 0) v = MathF.Min(v, d[i + n - 1] + hd);
                    }
                    d[i] = v;
                }
            return d;
        }

        // Catmull-Rom over the 4 × 4 cells round a world point when all of them were reached (its gradient continuous, where
        // bilinear steps at every cell), else bilinear over those that were
        private static bool SampleSeedCubic(Seeds sd, float[] field, Vector2 p, out float v)
        {
            int n = sd.Size;
            float fx = (p.X - sd.Origin.X) / sd.Hx, fz = (p.Y - sd.Origin.Y) / sd.Hz;
            int x1 = (int)MathF.Floor(fx), z1 = (int)MathF.Floor(fz);
            if (x1 >= 1 && z1 >= 1 && x1 + 2 <= n - 1 && z1 + 2 <= n - 1)
            {
                float tx = fx - x1, tz = fz - z1;
                Span<float> wx = stackalloc float[4], wz = stackalloc float[4];
                CatmullRom(tx, wx);
                CatmullRom(tz, wz);
                float sum = 0f;
                bool all = true;
                for (int j = 0; j < 4 && all; j++)
                {
                    float row = 0f;
                    for (int k = 0; k < 4; k++)
                    {
                        float f = field[(z1 - 1 + j) * n + x1 - 1 + k];
                        if (float.IsNaN(f)) { all = false; break; }
                        row += wx[k] * f;
                    }
                    sum += wz[j] * row;
                }
                if (all) { v = sum; return true; }
            }
            return SampleSeed(sd, field, p, out v);
        }

        private static void CatmullRom(float t, Span<float> w)
        {
            float t2 = t * t, t3 = t2 * t;
            w[0] = 0.5f * (-t3 + 2f * t2 - t);
            w[1] = 0.5f * (3f * t3 - 5f * t2 + 2f);
            w[2] = 0.5f * (-3f * t3 + 4f * t2 + t);
            w[3] = 0.5f * (t3 - t2);
        }

        // NaN cells next to valued ones take their mean, a ring a pass
        private static void DilateSeed(float[] f, int n, int passes)
        {
            var next = new float[f.Length];
            for (int pass = 0; pass < passes; pass++)
            {
                Array.Copy(f, next, f.Length);
                for (int z = 0; z < n; z++)
                    for (int x = 0; x < n; x++)
                    {
                        int i = z * n + x;
                        if (!float.IsNaN(f[i])) continue;
                        float sum = 0f; int cnt = 0;
                        for (int dz = -1; dz <= 1; dz++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                int xx = x + dx, zz = z + dz;
                                if (xx < 0 || zz < 0 || xx >= n || zz >= n) continue;
                                float v = f[zz * n + xx];
                                if (!float.IsNaN(v)) { sum += v; cnt++; }
                            }
                        if (cnt > 0) next[i] = sum / cnt;
                    }
                Array.Copy(next, f, f.Length);
            }
        }

        // A field of the whole terrain's march at a world point, bilinear over the cells it reached; false where it reached none
        private static bool SampleSeed(Seeds sd, float[] field, Vector2 p, out float v)
        {
            int n = sd.Size;
            float fx = Math.Clamp((p.X - sd.Origin.X) / sd.Hx, 0f, n - 1), fz = Math.Clamp((p.Y - sd.Origin.Y) / sd.Hz, 0f, n - 1);
            int x0 = Math.Min((int)fx, n - 2), z0 = Math.Min((int)fz, n - 2);
            float tx = fx - x0, tz = fz - z0;
            float sum = 0f, wsum = 0f;
            for (int k = 0; k < 4; k++)
            {
                int dx = k & 1, dz = k >> 1;
                float f = field[(z0 + dz) * n + x0 + dx];
                if (float.IsNaN(f)) continue;
                float w = (dx == 1 ? tx : 1f - tx) * (dz == 1 ? tz : 1f - tz);
                sum += w * f; wsum += w;
            }
            v = wsum > 1e-4f ? sum / wsum : 0f;
            return wsum > 1e-4f;
        }

        /// <summary>Bake one page of the surf (any thread).</summary>
        public static Page BakePage(PageJob job)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            var heights = job.Heights;
            var seeds = job.Seeds;
            int res = heights.Resolution;
            int scale = 1 << job.Level;
            float hx0 = job.TerrainSize.X / (res - 1), hz0 = job.TerrainSize.Y / (res - 1);
            float hx = hx0 * scale, hz = hz0 * scale;
            // The margin marched round the page: at least PageMargin, and the coast strip's width with room to spare (a strip
            // cell's travel time comes from its seaward edge, which must lie inside the grid)
            int margin = Math.Max(PageMargin, (int)MathF.Ceiling(StripCells * MathF.Max(seeds.Hx, seeds.Hz) / MathF.Min(hx, hz)) + 16);
            int m = SlotTexels + 2 * margin;
            int count = m * m;
            // The page's cells at its level: grid cell 0 is level texel px·256 − border − margin; texel k at the height
            // sample k·2^level
            int firstX = job.Px * PageTexels - PageBorder - margin, firstZ = job.Py * PageTexels - PageBorder - margin;

            using var depth = new NativeGrid<float>(count);
            using var sea = new NativeGrid<bool>(count);
            using var outside = new NativeGrid<bool>(count);
            using var borderDist = new NativeGrid<float>(count);
            var mask = job.Mask;
            for (int z = 0; z < m; z++)
            for (int x = 0; x < m; x++)
            {
                int i = z * m + x;
                int sx = (firstX + x) * scale, sz = (firstZ + z) * scale;
                bool off = sx < 0 || sz < 0 || sx > res - 1 || sz > res - 1;
                int cx = Math.Clamp(sx, 0, res - 1), cz = Math.Clamp(sz, 0, res - 1);
                float d = (job.SeaLevel01 - heights.Sample01(cx, cz)) * job.MaxHeight;
                depth[i] = d;
                outside[i] = off;
                borderDist[i] = MathF.Max(0f, MathF.Min(MathF.Min(sx, res - 1 - sx) * hx0, MathF.Min(sz, res - 1 - sz) * hz0));
                if (off) { sea[i] = d > 0f; continue; }
                if (mask != null && mask.Size > 1)
                {
                    // The mask says which low ground the open sea reaches (and its beaches); the depth here says where its
                    // edge is: water wherever any mask cell round it is sea, beach where the nearest one is
                    float fx = cx * (mask.Size - 1) / (float)(res - 1), fz = cz * (mask.Size - 1) / (float)(res - 1);
                    int mx = (int)MathF.Round(fx), mz = (int)MathF.Round(fz);
                    if (d > 0f)
                    {
                        bool any = false;
                        for (int dz = -1; dz <= 1 && !any; dz++)
                        for (int dx = -1; dx <= 1 && !any; dx++)
                        {
                            int xx = Math.Clamp(mx + dx, 0, mask.Size - 1), zz = Math.Clamp(mz + dz, 0, mask.Size - 1);
                            any = mask.Cells[zz * mask.Size + xx] != 0;
                        }
                        sea[i] = any;
                    }
                    else sea[i] = mask.Cells[mz * mask.Size + mx] != 0;
                }
                else sea[i] = d > 0f;
            }

            var page = new Page
            {
                Level = job.Level, Px = job.Px, Py = job.Py, Seeds = seeds,
                Data = new Half[SlotTexels * SlotTexels * 4 * MaxBands],
                Hollow = new Half[SlotTexels * SlotTexels],
            };

            var queue = new PriorityQueue<int, float>();
            using (var hollow = BeachHollows(m, depth, sea, queue))
                for (int z = 0; z < SlotTexels; z++)
                    for (int x = 0; x < SlotTexels; x++)
                        page.Hollow[z * SlotTexels + x] = (Half)hollow[(z + margin) * m + x + margin];

            int slices = Math.Min(seeds.Bands.Length, MaxBands);
            for (int s = 0; s < MaxBands; s++)
            {
                if (s >= slices)
                {
                    // No band: no warp
                    for (int t = 0; t < SlotTexels * SlotTexels; t++)
                    {
                        int o = (s * SlotTexels * SlotTexels + t) * 4;
                        page.Data[o + 1] = (Half)1f;
                    }
                    continue;
                }
                BakePageBand(page, s, seeds.Bands[s], seeds, m, hx, hz, job.Origin, firstX, firstZ, scale, hx0, hz0,
                             depth, sea, outside, borderDist, queue);
            }

            page.Milliseconds = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            return page;
        }

        private static void BakePageBand(Page page, int slice, Band band, Seeds seeds, int m, float hx, float hz, Vector2 origin,
                                         int firstX, int firstZ, int scale, float hx0, float hz0,
                                         NativeGrid<float> depth, NativeGrid<bool> sea, NativeGrid<bool> outside,
                                         NativeGrid<float> borderDist, PriorityQueue<int, float> queue)
        {
            int count = m * m;
            int margin = (m - SlotTexels) / 2;
            float omega = MathF.Max(band.Omega, 0.05f);
            float k0 = omega * omega / Gravity;
            float c0 = Gravity / omega;
            Vector2 dir = band.Direction.LengthSquared() > 1e-6f ? Vector2.Normalize(band.Direction) : Vector2.UnitX;
            var dPerp = new Vector2(-dir.Y, dir.X);
            float sourceDepth = band.SourceDepth(k0);
            Vector2 At(int x, int z) => origin + new Vector2((firstX + x) * scale * hx0, (firstZ + z) * scale * hz0);

            using var slowness = new NativeGrid<float>(count);
            using var tau = new NativeGrid<float>(count);
            using var side = new NativeGrid<float>(count);
            using var exposure = new NativeGrid<float>(count);
            using var state = new NativeGrid<byte>(count);
            using var source = new NativeGrid<bool>(count);
            using var water = new NativeGrid<bool>(count);
            using var beach = new NativeGrid<bool>(count);

            // Away from the coast the page takes the whole terrain's travel time as it is (bicubic: its gradient, the crests'
            // spacing, without the 16 m cells' steps); only a strip along the coast (StripCells of the whole terrain's cells)
            // is marched at the page's own cells, from that strip's seaward edge. Marched from its margin's edge instead, a
            // page carried the first-order march's errors of its own over hundreds of metres (the waves' local length 5-13 %
            // off, the whole terrain's too, differently) and neighbouring pages' crests met with steps of tens of metres. The
            // strip's march reaches no further than its own width, which the margin covers: pages agree, and with the coarse
            // warp beside them
            var seedTau = seeds.Tau[slice];
            float strip = StripCells * MathF.Max(seeds.Hx, seeds.Hz);
            using var coast = CoastDistance(m, hx, hz, sea, depth);
            using var open = new NativeGrid<float>(count);   // the whole terrain's travel time where the strip begins (NaN: none)
            for (int z = 0; z < m; z++)
            for (int x = 0; x < m; x++)
            {
                int i = z * m + x;
                tau[i] = float.PositiveInfinity;
                exposure[i] = 1f;
                open[i] = float.NaN;
                water[i] = sea[i] && depth[i] > 0f;
                beach[i] = sea[i] && !water[i];
                if (!sea[i]) continue;
                float h = MathF.Max(depth[i], MinDepth);
                float kh = WaveNumberDepth(omega, h);
                slowness[i] = depth[i] > 0f ? kh / h / omega : 1f / c0;

                var p = At(x, z);
                // The deep sea and past the terrain's border: the plane wave, as the whole terrain's march has it
                if (outside[i] || (water[i] && depth[i] >= sourceDepth))
                {
                    tau[i] = Vector2.Dot(dir, p) / c0;
                    side[i] = Vector2.Dot(dPerp, p);
                    state[i] = 2;
                    source[i] = true;
                    continue;
                }
                if (water[i] && coast[i] >= 0.5f * strip && SampleSeedCubic(seeds, seedTau, p, out float t0))
                {
                    open[i] = t0;
                    // Beyond the strip: known from the start, the march's seeds
                    if (coast[i] >= strip)
                    {
                        tau[i] = t0;
                        side[i] = Vector2.Dot(dPerp, p);
                        state[i] = 2;
                    }
                }
            }

            var marcher = new Marcher(m, hx, hz, dir, slowness, tau, side, exposure, state, source, queue);
            marcher.March(water);

            // Water the march did not reach: a narrow inlet or a pool with no water as far as the strip from both its banks,
            // the waves come in from beyond the grid. Its cells a whole terrain's cell from the coast take that bake's travel
            // time and the march goes on from them; what is still left takes it as it is. (Left unreached, an inlet kept no
            // warp: the open sea's long waves at full height in two metres of water, a breaker across it)
            for (int pass = 0; pass < 2; pass++)
            {
                bool any = false;
                float from = pass == 0 ? MathF.Max(seeds.Hx, seeds.Hz) : 0f;
                for (int z = 0; z < m; z++)
                    for (int x = 0; x < m; x++)
                    {
                        int i = z * m + x;
                        if (!water[i] || state[i] == 2 || coast[i] < from) continue;
                        if (!SampleSeedCubic(seeds, seedTau, At(x, z), out float t1)) continue;
                        tau[i] = t1;
                        open[i] = t1;
                        side[i] = Vector2.Dot(dPerp, At(x, z));
                        state[i] = 2;
                        any = true;
                    }
                if (any) marcher.March(water);
            }
            marcher.March(beach);

            using var known = new NativeGrid<bool>(count);
            using var marched = new NativeGrid<bool>(count);
            for (int i = 0; i < count; i++)
            {
                known[i] = sea[i] && state[i] == 2;
                marched[i] = known[i] && !(water[i] && coast[i] >= strip);
            }

            // The strip's travel time: its creases a little, then eased into the whole terrain's over its outer half, so the
            // crests do not bend where it begins
            SmoothSymmetric(tau, m, marched, 2);
            for (int i = 0; i < count; i++)
                if (marched[i] && !float.IsNaN(open[i]))
                {
                    float t = Smooth(0.5f * strip, strip, coast[i]);
                    tau[i] += (open[i] - tau[i]) * t;
                }

            // Per cell: the lag and g = c0·∇τ (∂u/∂x = d ⊗ g + d⊥ ⊗ d⊥) from this march; the height the whole terrain's bake
            // gives (Seeds.Amp: a function of the point alone, so pages and levels agree on it)
            var seedAmp = seeds.Amp[slice];
            using var lag = new NativeGrid<float>(count);
            using var amp = new NativeGrid<float>(count);
            using var g = new NativeGrid<float>(count * 2);
            using var blend = new NativeGrid<float>(count);
            for (int z = 0; z < m; z++)
            for (int x = 0; x < m; x++)
            {
                int i = z * m + x;
                amp[i] = 1f; blend[i] = 1f;
                g[i * 2] = dir.X; g[i * 2 + 1] = dir.Y;
                if (!known[i]) continue;

                bool l = x > 0 && known[i - 1], rt = x < m - 1 && known[i + 1];
                bool dn = z > 0 && known[i - m], up = z < m - 1 && known[i + m];
                int x0 = l ? i - 1 : i, x1 = rt ? i + 1 : i;
                int z0 = dn ? i - m : i, z1 = up ? i + m : i;
                float dxs = (x1 - x0) * hx, dzs = ((z1 - z0) / m) * hz;
                float tx = dxs > 0f ? (tau[x1] - tau[x0]) / dxs : dir.X / c0;
                float tz = dzs > 0f ? (tau[z1] - tau[z0]) / dzs : dir.Y / c0;
                g[i * 2] = c0 * tx; g[i * 2 + 1] = c0 * tz;

                var p = At(x, z);
                amp[i] = SampleSeed(seeds, seedAmp, p, out float av) ? av : 1f;
                lag[i] = c0 * tau[i] - Vector2.Dot(dir, p);
                blend[i] = MathF.Max(Smooth(0.5f * sourceDepth, sourceDepth, depth[i]),
                                     1f - Smooth(0f, BorderFade, borderDist[i]));
            }
            BoxBlur(g, 2, m, sea);

            // Blended back to no warp toward deep water and the terrain's border, then into the land beside the sea, so
            // bilinear filtering at the coast blends warp with warp
            using var v = new NativeGrid<float>(count * 4);
            using var filled = new NativeGrid<bool>(count);
            for (int i = 0; i < count; i++)
            {
                bool ok = known[i];
                float w = blend[i];
                v[i * 4 + 0] = ok ? lag[i] * (1f - w) : 0f;
                v[i * 4 + 1] = ok ? amp[i] + (1f - amp[i]) * w : 1f;
                v[i * 4 + 2] = ok ? g[i * 2] + (dir.X - g[i * 2]) * w : dir.X;
                v[i * 4 + 3] = ok ? g[i * 2 + 1] + (dir.Y - g[i * 2 + 1]) * w : dir.Y;
                filled[i] = ok;
            }
            Dilate4(v, filled, m, 3);

            // Kept relative to the page's mean lag: half floats near zero hold a fraction of a metre where the lag runs to
            // hundreds over the terrain
            double sum = 0; int cnt = 0;
            for (int z = margin; z < margin + SlotTexels; z++)
                for (int x = margin; x < margin + SlotTexels; x++)
                {
                    int i = z * m + x;
                    if (filled[i]) { sum += v[i * 4]; cnt++; }
                }
            float baseLag = cnt > 0 ? (float)(sum / cnt) : 0f;
            page.Base[slice] = baseLag;
            int so = slice * SlotTexels * SlotTexels;
            for (int z = 0; z < SlotTexels; z++)
                for (int x = 0; x < SlotTexels; x++)
                {
                    int i = (z + margin) * m + x + margin;
                    int o = (so + z * SlotTexels + x) * 4;
                    page.Data[o + 0] = (Half)(v[i * 4] - baseLag);
                    page.Data[o + 1] = (Half)v[i * 4 + 1];
                    page.Data[o + 2] = (Half)v[i * 4 + 2];
                    page.Data[o + 3] = (Half)v[i * 4 + 3];
                }
        }

        // Cells not filled take the mean of their filled neighbours, a ring a pass (4 values a cell)
        private static void Dilate4(NativeGrid<float> v, NativeGrid<bool> filled, int n, int passes)
        {
            int count = n * n;
            using var next = new NativeGrid<bool>(count);
            Span<float> acc = stackalloc float[4];
            for (int pass = 0; pass < passes; pass++)
            {
                filled.CopyTo(next);
                for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    int i = z * n + x;
                    if (filled[i]) continue;
                    acc.Clear(); int cnt = 0;
                    for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int xx = x + dx, zz = z + dz;
                        if (xx < 0 || zz < 0 || xx >= n || zz >= n) continue;
                        int j = zz * n + xx;
                        if (!filled[j]) continue;
                        for (int c = 0; c < 4; c++) acc[c] += v[j * 4 + c];
                        cnt++;
                    }
                    if (cnt == 0) continue;
                    for (int c = 0; c < 4; c++) v[i * 4 + c] = acc[c] / cnt;
                    next[i] = true;
                }
                next.CopyTo(filled);
            }
        }

        // Diagnostic: with the environment variable FREEFALL_SHOREWARP_DUMP=<folder> set when the editor starts, every bake
        // writes per slice the crest lines the shader sees (cos of the deep phase at u), the amplitude, the exposure and
        // the depth as 512² PGMs (+Z up). How the bake was checked against the seabed (surf_plan.md)
        private static void DebugDump(Result r, int o, int slice, int n, float hx, float hz, Vector2 origin, float k0,
                                      Vector2 dir, NativeGrid<bool> sea, NativeGrid<float> exposure, NativeGrid<float> depth)
        {
            string? folder = Environment.GetEnvironmentVariable("FREEFALL_SHOREWARP_DUMP");
            if (string.IsNullOrEmpty(folder)) return;
            const int m = 512;
            var crest = new byte[m * m]; var ampB = new byte[m * m]; var expB = new byte[m * m]; var depB = new byte[m * m];
            for (int y = 0; y < m; y++)
            for (int x = 0; x < m; x++)
            {
                int cx = x * (n - 1) / (m - 1), cz = (m - 1 - y) * (n - 1) / (m - 1);   // +Z up
                int i = cz * n + cx;
                int j = y * m + x;
                if (!sea[i]) { crest[j] = ampB[j] = expB[j] = 40; depB[j] = 0; continue; }
                var p = origin + new Vector2(cx * hx, cz * hz);
                float ux = p.X + r.Warp[o + i * 4], uz = p.Y + r.Warp[o + i * 4 + 1];
                crest[j] = (byte)(127.5f + 127.5f * MathF.Cos(k0 * (dir.X * ux + dir.Y * uz)));
                ampB[j] = (byte)Math.Clamp(r.Warp[o + i * 4 + 2] / 2.5f * 255f, 0f, 255f);
                expB[j] = (byte)Math.Clamp(exposure[i] * 255f, 0f, 255f);
                depB[j] = (byte)Math.Clamp(depth[i] / 120f * 255f, 0f, 255f);
            }
            void Write(string name, byte[] px)
            {
                using var f = System.IO.File.Create(System.IO.Path.Combine(folder, $"warp{slice}_{name}.pgm"));
                var header = System.Text.Encoding.ASCII.GetBytes($"P5\n{m} {m}\n255\n");
                f.Write(header, 0, header.Length);
                f.Write(px, 0, px.Length);
            }
            System.IO.Directory.CreateDirectory(folder);
            Write("crest", crest); Write("amp", ampB); Write("exposure", expB); Write("depth", depB);
        }

        private static float Smooth(float e0, float e1, float v)
        {
            float t = Math.Clamp((v - e0) / (e1 - e0), 0f, 1f);
            return t * t * (3f - 2f * t);
        }

        // 3×3 box over the sea cells only
        private static void BoxBlur(NativeGrid<float> data, int stride, int n, NativeGrid<bool> sea)
        {
            using var copy = new NativeGrid<float>(data);
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                if (!sea[i]) continue;
                for (int c = 0; c < stride; c++)
                {
                    float sum = 0f; int cnt = 0;
                    for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int xx = x + dx, zz = z + dz;
                        if (xx < 0 || zz < 0 || xx >= n || zz >= n) continue;
                        int j = zz * n + xx;
                        if (!sea[j]) continue;
                        sum += copy[j * stride + c]; cnt++;
                    }
                    data[i * stride + c] = sum / cnt;
                }
            }
        }

        // Average over ±r cells along rows, then columns, only where the whole window is usable (else the cell keeps its
        // value): symmetric windows leave a straight slope as it is, so a steep field like the travel time near the
        // coast is not pulled toward its land side
        private static void SmoothSymmetric(NativeGrid<float> data, int n, NativeGrid<bool> ok, int r)
        {
            using var tmp = new NativeGrid<float>(data);
            for (int pass = 0; pass < 2; pass++)
            {
                bool rows = pass == 0;
                var src = rows ? data : tmp;
                var dst = rows ? tmp : data;
                for (int line = 0; line < n; line++)
                for (int k = 0; k < n; k++)
                {
                    int i = rows ? line * n + k : k * n + line;
                    dst[i] = src[i];
                    if (!ok[i] || k < r || k >= n - r) continue;
                    float sum = 0f;
                    bool full = true;
                    for (int m = -r; m <= r && full; m++)
                    {
                        int j = rows ? line * n + k + m : (k + m) * n + line;
                        full = ok[j];
                        sum += src[j];
                    }
                    if (full) dst[i] = sum / (2 * r + 1);
                }
            }
        }

        // A (2r+1)² box over the sea cells only, separable with running sums
        private static void BoxBlurWide(NativeGrid<float> data, int n, NativeGrid<bool> sea, int r)
        {
            using var tmp = new NativeGrid<float>(data.Length);
            for (int pass = 0; pass < 2; pass++)
            {
                bool rows = pass == 0;
                var src = rows ? data : tmp;
                var dst = rows ? tmp : data;
                for (int line = 0; line < n; line++)
                {
                    float sum = 0f; int cnt = 0;
                    int Index(int k) => rows ? line * n + k : k * n + line;
                    for (int k = 0; k <= Math.Min(r, n - 1); k++)
                    {
                        int j = Index(k);
                        if (sea[j]) { sum += src[j]; cnt++; }
                    }
                    for (int k = 0; k < n; k++)
                    {
                        int i = Index(k);
                        dst[i] = sea[i] && cnt > 0 ? sum / cnt : src[i];
                        int add = k + r + 1, sub = k - r;
                        if (add < n) { int j = Index(add); if (sea[j]) { sum += src[j]; cnt++; } }
                        if (sub >= 0) { int j = Index(sub); if (sea[j]) { sum -= src[j]; cnt--; } }
                    }
                }
            }
        }

        private static void DilateIntoLand(Result r, int o, int n, NativeGrid<bool> sea, NativeGrid<byte> state)
        {
            int count = n * n;
            // This pass's and the next's filled cells, swapped after each pass (the next one starts as a copy, as the clone
            // did); the sums in stack memory (they were two new float[4] per land cell and pass)
            using var a = new NativeGrid<bool>(count);
            using var b = new NativeGrid<bool>(count);
            var filled = a;
            var next = b;
            Span<float> w = stackalloc float[4];
            Span<float> j4 = stackalloc float[4];
            for (int i = 0; i < count; i++) filled[i] = sea[i] && state[i] == 2;
            for (int pass = 0; pass < 3; pass++)
            {
                filled.CopyTo(next);
                for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    int i = z * n + x;
                    if (filled[i]) continue;
                    w.Clear(); j4.Clear(); int cnt = 0;
                    for (int dz = -1; dz <= 1; dz++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int xx = x + dx, zz = z + dz;
                        if (xx < 0 || zz < 0 || xx >= n || zz >= n) continue;
                        int j = zz * n + xx;
                        if (!filled[j]) continue;
                        for (int c = 0; c < 4; c++) { w[c] += r.Warp[o + j * 4 + c]; j4[c] += (float)r.Jacobian[o + j * 4 + c]; }
                        cnt++;
                    }
                    if (cnt == 0) continue;
                    for (int c = 0; c < 4; c++)
                    {
                        r.Warp[o + i * 4 + c] = w[c] / cnt;
                        r.Jacobian[o + i * 4 + c] = (Half)(j4[c] / cnt);
                    }
                    next[i] = true;
                }
                (filled, next) = (next, filled);
            }
        }

        /// <summary>Copy a baked warp into the textures (made anew when the size changed). Main thread; the copy runs at the
        /// start of the next frame's opaque pass, before the forward pass draws the sea.</summary>
        public unsafe void Upload(Result warp)
        {
            var device = Engine.Device;
            int n = warp.Size;

            if (_warpTex == null || _size != n)
            {
                Dispose();
                _warpTex = device.CreateTexture2D(Format.R32G32B32A32_Float, n, n, MaxBands, 1, ResourceFlags.None, ResourceStates.Common);
                _jacobianTex = device.CreateTexture2D(Format.R16G16B16A16_Float, n, n, MaxBands, 1, ResourceFlags.None, ResourceStates.Common);
                _warpSrv = device.AllocateBindlessIndex();
                _jacobianSrv = device.AllocateBindlessIndex();
                CreateArraySrv(device, _warpTex, Format.R32G32B32A32_Float, _warpSrv);
                CreateArraySrv(device, _jacobianTex, Format.R16G16B16A16_Float, _jacobianSrv);
                _size = n;
            }

            var warpUpload = Stage(device, warp.Warp.AsSpan(), n, 16, out int warpPitch);
            var jacUpload = Stage(device, warp.Jacobian.AsSpan(), n, 8, out int jacPitch);

            var warpTex = _warpTex!;
            var jacTex = _jacobianTex!;
            CommandBuffer.Enqueue(RenderPass.Opaque, list =>
            {
                list.ResourceBarrierTransition(warpTex, ResourceStates.Common, ResourceStates.CopyDest);
                list.ResourceBarrierTransition(jacTex, ResourceStates.Common, ResourceStates.CopyDest);
                for (int s = 0; s < MaxBands; s++)
                {
                    list.CopyTextureRegion(new TextureCopyLocation(warpTex, (uint)s), 0, 0, 0, new TextureCopyLocation(warpUpload,
                        new PlacedSubresourceFootPrint
                        {
                            Offset = (ulong)s * (ulong)warpPitch * (ulong)n,
                            Footprint = new SubresourceFootPrint(Format.R32G32B32A32_Float, (uint)n, (uint)n, 1, (uint)warpPitch)
                        }));
                    list.CopyTextureRegion(new TextureCopyLocation(jacTex, (uint)s), 0, 0, 0, new TextureCopyLocation(jacUpload,
                        new PlacedSubresourceFootPrint
                        {
                            Offset = (ulong)s * (ulong)jacPitch * (ulong)n,
                            Footprint = new SubresourceFootPrint(Format.R16G16B16A16_Float, (uint)n, (uint)n, 1, (uint)jacPitch)
                        }));
                }
                list.ResourceBarrierTransition(warpTex, ResourceStates.CopyDest, ResourceStates.Common);
                list.ResourceBarrierTransition(jacTex, ResourceStates.CopyDest, ResourceStates.Common);
            });
            device.DeferDispose(warpUpload);
            device.DeferDispose(jacUpload);
            _bands = warp.Bands;

            Debug.LogAlways($"[ShoreWarp] {n}² cells, {warp.Bands.Length} band(s), baked in {warp.Milliseconds:0} ms");
        }

        // An upload buffer holding every slice, rows padded to the copy pitch
        private static unsafe ID3D12Resource Stage<T>(GraphicsDevice device, Span<T> data, int n, int texelBytes, out int pitch)
            where T : unmanaged
        {
            int rowBytes = n * texelBytes;
            pitch = (rowBytes + 255) & ~255;
            var upload = device.CreateCommittedResource(
                new HeapProperties(HeapType.Upload), HeapFlags.None,
                ResourceDescription.Buffer((ulong)pitch * (ulong)n * MaxBands), ResourceStates.GenericRead, null);
            void* mapped;
            upload.Map(0, null, &mapped);
            fixed (T* src = data)
            {
                byte* s = (byte*)src;
                for (int row = 0; row < n * MaxBands; row++)
                    Buffer.MemoryCopy(s + (long)row * rowBytes, (byte*)mapped + (long)row * pitch, rowBytes, rowBytes);
            }
            upload.Unmap(0);
            return upload;
        }

        private static void CreateArraySrv(GraphicsDevice device, ID3D12Resource tex, Format format, uint srv)
        {
            device.NativeDevice.CreateShaderResourceView(tex, new ShaderResourceViewDescription
            {
                Format = format,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView { MostDetailedMip = 0, MipLevels = 1, FirstArraySlice = 0, ArraySize = MaxBands }
            }, device.GetCpuHandle(srv));
        }

        public void Dispose()
        {
            var device = Engine.Device;
            device.DeferDispose(_warpTex);
            device.DeferDispose(_jacobianTex);
            if (_warpSrv != 0) device.DeferReleaseBindlessIndex(_warpSrv);
            if (_jacobianSrv != 0) device.DeferReleaseBindlessIndex(_jacobianSrv);
            _warpTex = _jacobianTex = null;
            _warpSrv = _jacobianSrv = 0;
            _size = 0;
        }
    }
}
