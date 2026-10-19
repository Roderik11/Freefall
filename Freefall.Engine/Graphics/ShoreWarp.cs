using System;
using System.Collections.Generic;
using System.Numerics;
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
    /// RGBA32F (u − x, amplitude, wave speed / deep speed) and RGBA16F (∂u/∂x: ∂ux/∂x, ∂ux/∂z, ∂uz/∂x, ∂uz/∂z).
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

        public struct Band
        {
            public int Index;            // the ocean band
            public Vector2 Direction;    // world xz unit vector its waves travel along in deep water
            public float Omega;          // its representative angular frequency (rad/s): the spectrum's peak

            public bool SameAs(Band o) =>
                Index == o.Index && Vector2.Dot(Direction, o.Direction) > 0.9994f      // 2°
                && MathF.Abs(Omega - o.Omega) <= 0.02f * o.Omega;
        }

        /// <summary>A baked warp, ready to upload.</summary>
        public sealed class Result
        {
            public int Size;
            public Band[] Bands = Array.Empty<Band>();
            public float[] Warp = Array.Empty<float>();        // slice-major, Size² × (Δu.x, Δu.z, amplitude, c/c0)
            public Half[] Jacobian = Array.Empty<Half>();      // slice-major, Size² × (∂ux/∂x, ∂ux/∂z, ∂uz/∂x, ∂uz/∂z)
            public double Milliseconds;
        }

        private ID3D12Resource? _warpTex, _jacobianTex;
        private uint _warpSrv, _jacobianSrv;
        private int _size;
        private Band[] _bands = Array.Empty<Band>();

        public uint WarpSrv => _warpSrv;
        public uint JacobianSrv => _jacobianSrv;

        /// <summary>The bands the uploaded warp was baked for, slice by slice.</summary>
        public Band[] Bands => _bands;

        /// <summary>
        /// Bake the warp of each band over the height field (values 0..1 of <paramref name="maxHeight"/> above the terrain's
        /// base, [x, z]). <paramref name="mask"/>: the sea mask (where the sea is, beaches included); null = wherever the
        /// ground is below sea level. Any thread.
        /// </summary>
        public static Result Build(float[,] heights, SeaMask.Result? mask, float seaLevel01, float maxHeight,
                                   Vector2 terrainSize, Vector2 origin, Band[] bands)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            int res = Math.Min(heights.GetLength(0), heights.GetLength(1));
            int step = 1;
            while ((res - 1) / step + 1 > MaxSize) step *= 2;
            int n = (res - 1) / step + 1;
            int count = n * n;
            float hx = terrainSize.X / (n - 1), hz = terrainSize.Y / (n - 1);

            var depth = new float[count];
            var sea = new bool[count];
            var borderDist = new float[count];
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                depth[i] = (seaLevel01 - heights[x * step, z * step]) * maxHeight;
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
            var result = new Result
            {
                Size = n,
                Bands = bands[..slices],
                Warp = new float[count * 4 * MaxBands],
                Jacobian = new Half[count * 4 * MaxBands],
            };

            // Unused slices: no warp
            for (int s = slices; s < MaxBands; s++)
                FillIdentity(result, s, count);

            for (int s = 0; s < slices; s++)
                BakeBand(result, s, bands[s], n, hx, hz, origin, depth, sea, borderDist);

            result.Milliseconds = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            return result;
        }

        private static void FillIdentity(Result r, int slice, int count)
        {
            int o = slice * count * 4;
            for (int i = 0; i < count; i++)
            {
                r.Warp[o + i * 4 + 2] = 1f;
                r.Warp[o + i * 4 + 3] = 1f;
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

        private static void BakeBand(Result r, int slice, Band band, int n, float hx, float hz, Vector2 origin,
                                     float[] depth, bool[] sea, float[] borderDist)
        {
            int count = n * n;
            float omega = MathF.Max(band.Omega, 0.05f);
            float k0 = omega * omega / Gravity;
            float c0 = Gravity / omega;
            Vector2 dir = band.Direction.LengthSquared() > 1e-6f ? Vector2.Normalize(band.Direction) : Vector2.UnitX;

            // Deep enough that the wave hardly feels the bottom (k0·h ≥ 2: 98% of its deep-water speed): the plane wave
            // there, and on the terrain's border (past it the seabed is unknown)
            float sourceDepth = 2f / k0;

            var slowness = new float[count];
            var tau = new float[count];
            var ux = new float[count];
            var uz = new float[count];
            var exposure = new float[count];
            var state = new byte[count];          // 0 far, 1 trial, 2 known
            var source = new bool[count];

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
                    ux[i] = p.X;
                    uz[i] = p.Y;
                    exposure[i] = 1f;
                    state[i] = 2;
                    source[i] = true;
                }
            }

            // Fast marching: cells become known in order of arrival time, each from its known neighbours (first-order
            // upwind). Along with the time each cell takes where its ray came from: the deep sea position u (moved on
            // along the deep heading as far as the wave would have travelled there in the same time) and the exposure
            var queue = new PriorityQueue<int, float>();

            bool Relax(int i)
            {
                int x = i % n, z = i / n;
                int nxI = -1, nzI = -1;
                float a = float.PositiveInfinity, b = float.PositiveInfinity;
                float sx = 0f, sz = 0f;
                if (x > 0 && state[i - 1] == 2 && tau[i - 1] < a) { a = tau[i - 1]; nxI = i - 1; sx = 1f; }
                if (x < n - 1 && state[i + 1] == 2 && tau[i + 1] < a) { a = tau[i + 1]; nxI = i + 1; sx = -1f; }
                if (z > 0 && state[i - n] == 2 && tau[i - n] < b) { b = tau[i - n]; nzI = i - n; sz = 1f; }
                if (z < n - 1 && state[i + n] == 2 && tau[i + n] < b) { b = tau[i + n]; nzI = i + n; sz = -1f; }
                if (nxI < 0 && nzI < 0) return false;

                float s = slowness[i];
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

                float tauUp = 0f, uxUp = 0f, uzUp = 0f, eUp = 0f;
                float gl = MathF.Sqrt(gx * gx + gz * gz);
                float along = gl > 0f ? (gx * dir.X + gz * dir.Y) / gl : 1f;
                // A ray leaving the deep sea has to run with the wave: on a lee shore it would run against it, and
                // the wave never gets there (the island's shadow)
                float leave = Smooth(-0.1f, 0.4f, along);
                if (nxI >= 0)
                {
                    tauUp += wx * a; uxUp += wx * ux[nxI]; uzUp += wx * uz[nxI];
                    eUp += wx * exposure[nxI] * (source[nxI] ? leave : 1f);
                }
                if (nzI >= 0)
                {
                    tauUp += wz * b; uxUp += wz * ux[nzI]; uzUp += wz * uz[nzI];
                    eUp += wz * exposure[nzI] * (source[nzI] ? leave : 1f);
                }

                tau[i] = t;
                float travelled = c0 * (t - tauUp);
                ux[i] = uxUp + dir.X * travelled;
                uz[i] = uzUp + dir.Y * travelled;
                exposure[i] = eUp;
                return true;
            }

            void Push(int i)
            {
                if (Relax(i))
                {
                    state[i] = 1;
                    queue.Enqueue(i, tau[i]);
                }
            }

            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                if (!source[i]) continue;
                if (x > 0 && sea[i - 1] && state[i - 1] != 2) Push(i - 1);
                if (x < n - 1 && sea[i + 1] && state[i + 1] != 2) Push(i + 1);
                if (z > 0 && sea[i - n] && state[i - n] != 2) Push(i - n);
                if (z < n - 1 && sea[i + n] && state[i + n] != 2) Push(i + n);
            }
            while (queue.TryDequeue(out int i, out float t))
            {
                if (state[i] == 2 || t > tau[i]) continue;
                state[i] = 2;
                int x = i % n, z = i / n;
                if (x > 0 && sea[i - 1] && state[i - 1] != 2) Push(i - 1);
                if (x < n - 1 && sea[i + 1] && state[i + 1] != 2) Push(i + 1);
                if (z > 0 && sea[i - n] && state[i - n] != 2) Push(i - n);
                if (z < n - 1 && sea[i + n] && state[i + n] != 2) Push(i + n);
            }

            // From u: its Jacobian, the wave height (energy flux kept between rays: det J = (c0/c)·(b0/b), so
            // H/H0 = √(det J · ½ / n), n the group over phase speed), and the blend back to no warp toward deep water
            // and the terrain's border
            var jac = new float[count * 4];
            var amp = new float[count];
            var speed = new float[count];
            var blend = new float[count];
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                jac[i * 4 + 0] = 1f; jac[i * 4 + 3] = 1f;
                amp[i] = 1f; speed[i] = 1f; blend[i] = 1f;
                if (!sea[i] || state[i] != 2) continue;

                // Central differences where both neighbours are sea, one-sided at the coast
                bool l = x > 0 && sea[i - 1] && state[i - 1] == 2, rt = x < n - 1 && sea[i + 1] && state[i + 1] == 2;
                bool d = z > 0 && sea[i - n] && state[i - n] == 2, u = z < n - 1 && sea[i + n] && state[i + n] == 2;
                int x0 = l ? i - 1 : i, x1 = rt ? i + 1 : i;
                int z0 = d ? i - n : i, z1 = u ? i + n : i;
                float dxs = (x1 - x0) * hx, dzs = ((z1 - z0) / n) * hz;
                if (dxs > 0f) { jac[i * 4 + 0] = (ux[x1] - ux[x0]) / dxs; jac[i * 4 + 2] = (uz[x1] - uz[x0]) / dxs; }
                if (dzs > 0f) { jac[i * 4 + 1] = (ux[z1] - ux[z0]) / dzs; jac[i * 4 + 3] = (uz[z1] - uz[z0]) / dzs; }

                float h = MathF.Max(depth[i], MinDepth);
                float kh = WaveNumberDepth(omega, h);
                float det = jac[i * 4 + 0] * jac[i * 4 + 3] - jac[i * 4 + 1] * jac[i * 4 + 2];
                float k = MathF.Sqrt(Math.Clamp(det * 0.5f / GroupRatio(kh), 0f, 6f));
                amp[i] = k * exposure[i];
                speed[i] = (omega * h / kh) / c0;                       // c / c0

                blend[i] = MathF.Max(Smooth(0.5f * sourceDepth, sourceDepth, depth[i]),
                                     1f - Smooth(0f, BorderFade, borderDist[i]));
            }

            // Smooth the Jacobian (first-order marching leaves grid noise in its derivatives), and the height over about a
            // quarter wavelength: a wave's height cannot change faster than that (diffraction blurs the edge of an island's
            // shadow, where the rays from its two sides meet and the exposure jumps)
            for (int pass = 0; pass < 2; pass++)
                BoxBlur(jac, 4, n, sea);
            int radius = Math.Clamp((int)MathF.Round(0.25f * (2f * MathF.PI / k0) / MathF.Max(hx, hz)), 1, 24);
            for (int pass = 0; pass < 3; pass++)
                BoxBlurWide(amp, n, sea, radius);

            int o = slice * count * 4;
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
            {
                int i = z * n + x;
                float w = blend[i];
                var p = origin + new Vector2(x * hx, z * hz);
                bool known = sea[i] && state[i] == 2;
                r.Warp[o + i * 4 + 0] = known ? (ux[i] - p.X) * (1f - w) : 0f;
                r.Warp[o + i * 4 + 1] = known ? (uz[i] - p.Y) * (1f - w) : 0f;
                r.Warp[o + i * 4 + 2] = known ? amp[i] + (1f - amp[i]) * w : 1f;
                r.Warp[o + i * 4 + 3] = known ? speed[i] + (1f - speed[i]) * w : 1f;
                for (int c = 0; c < 4; c++)
                {
                    float id = c == 0 || c == 3 ? 1f : 0f;
                    r.Jacobian[o + i * 4 + c] = (Half)(known ? jac[i * 4 + c] + (id - jac[i * 4 + c]) * w : id);
                }
            }

            // Land next to the sea takes its neighbours' warp, so bilinear filtering at the coast blends warp with warp
            DilateIntoLand(r, o, n, sea, state);

            DebugDump(r, o, slice, n, hx, hz, origin, k0, dir, sea, exposure, depth);
        }

        // Diagnostic: with the environment variable FREEFALL_SHOREWARP_DUMP=<folder> set when the editor starts, every bake
        // writes per slice the crest lines the shader sees (cos of the deep phase at u), the amplitude, the exposure and
        // the depth as 512² PGMs (+Z up). How the bake was checked against the seabed (surf_plan.md)
        private static void DebugDump(Result r, int o, int slice, int n, float hx, float hz, Vector2 origin, float k0,
                                      Vector2 dir, bool[] sea, float[] exposure, float[] depth)
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
        private static void BoxBlur(float[] data, int stride, int n, bool[] sea)
        {
            var copy = (float[])data.Clone();
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

        // A (2r+1)² box over the sea cells only, separable with running sums
        private static void BoxBlurWide(float[] data, int n, bool[] sea, int r)
        {
            var tmp = new float[data.Length];
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

        private static void DilateIntoLand(Result r, int o, int n, bool[] sea, byte[] state)
        {
            int count = n * n;
            var filled = new bool[count];
            for (int i = 0; i < count; i++) filled[i] = sea[i] && state[i] == 2;
            for (int pass = 0; pass < 3; pass++)
            {
                var next = (bool[])filled.Clone();
                for (int z = 0; z < n; z++)
                for (int x = 0; x < n; x++)
                {
                    int i = z * n + x;
                    if (filled[i]) continue;
                    var w = new float[4]; var j4 = new float[4]; int cnt = 0;
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
                filled = next;
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

            Debug.Log($"[ShoreWarp] {n}² cells, {warp.Bands.Length} band(s), baked in {warp.Milliseconds:0} ms");
        }

        // An upload buffer holding every slice, rows padded to the copy pitch
        private static unsafe ID3D12Resource Stage<T>(GraphicsDevice device, Span<T> data, int n, int texelBytes, out int pitch)
            where T : unmanaged
        {
            int rowBytes = n * texelBytes;
            pitch = (rowBytes + 255) & ~255;
            var upload = device.NativeDevice.CreateCommittedResource(
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
