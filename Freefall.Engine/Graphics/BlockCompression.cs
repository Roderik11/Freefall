using System;
using System.Numerics;
using System.Threading.Tasks;

namespace Freefall.Graphics
{
    /// <summary>
    /// BC1, BC4 and BC5 written in C#, for pictures made at run of the editor in large numbers (the world map's tiles),
    /// where a texconv process per picture would cost more than the compression. The quality is that of a range fit:
    /// BC1 along the colours' principal axis, BC4 between a block's lowest and highest value. Also the decoders, for
    /// checks and previews. Widths and heights are multiples of 4.
    /// </summary>
    public static class BlockCompression
    {
        // ── BC4 / BC5 ──

        /// <summary>One channel (0..255 per pixel, row by row) as BC4: 8 bytes per 4×4 block.</summary>
        public static byte[] EncodeBC4(ReadOnlySpan<byte> values, int width, int height)
        {
            var output = new byte[width / 4 * (height / 4) * 8];
            EncodeBC4Into(values, 1, 0, width, height, output, 8, 0);
            return output;
        }

        /// <summary>Two channels (interleaved RG, 0..255) as BC5: a BC4 block of R, then one of G, 16 bytes per block.</summary>
        public static byte[] EncodeBC5(ReadOnlySpan<byte> rg, int width, int height)
        {
            var output = new byte[width / 4 * (height / 4) * 16];
            EncodeBC4Into(rg, 2, 0, width, height, output, 16, 0);
            EncodeBC4Into(rg, 2, 1, width, height, output, 16, 8);
            return output;
        }

        private static void EncodeBC4Into(ReadOnlySpan<byte> values, int stride, int channel, int width, int height,
                                          byte[] output, int blockBytes, int blockOffset)
        {
            int bw = width / 4, bh = height / 4;
            Span<byte> block = stackalloc byte[16];
            for (int by = 0; by < bh; by++)
                for (int bx = 0; bx < bw; bx++)
                {
                    for (int y = 0; y < 4; y++)
                        for (int x = 0; x < 4; x++)
                            block[y * 4 + x] = values[((by * 4 + y) * width + bx * 4 + x) * stride + channel];
                    EncodeBC4Block(block, output.AsSpan((by * bw + bx) * blockBytes + blockOffset, 8));
                }
        }

        /// <summary>
        /// A BC4 block in the eight-value mode (red0 > red1): the block's highest and lowest value and six steps between,
        /// each pixel the nearest. A flat block keeps both ends equal (every index 0).
        /// </summary>
        private static void EncodeBC4Block(ReadOnlySpan<byte> block, Span<byte> dst)
        {
            byte lo = 255, hi = 0;
            for (int i = 0; i < 16; i++)
            {
                if (block[i] < lo) lo = block[i];
                if (block[i] > hi) hi = block[i];
            }
            dst[0] = hi;
            dst[1] = lo;
            ulong bits = 0;
            if (hi != lo)
            {
                // Palette: 0 = hi, 1 = lo, 2..7 = hi→lo in sevenths. A value's position t from hi (0) to lo (7)
                float scale = 7f / (hi - lo);
                for (int i = 0; i < 16; i++)
                {
                    int t = (int)MathF.Round((hi - block[i]) * scale);
                    int index = t == 0 ? 0 : t == 7 ? 1 : t + 1;
                    bits |= (ulong)index << (3 * i);
                }
            }
            for (int i = 0; i < 6; i++) dst[2 + i] = (byte)(bits >> (8 * i));
        }

        /// <summary>BC4 back to one byte per pixel.</summary>
        public static byte[] DecodeBC4(ReadOnlySpan<byte> data, int width, int height)
        {
            var output = new byte[width * height];
            DecodeBC4Into(data, 8, 0, width, height, output, 1, 0);
            return output;
        }

        /// <summary>BC5 back to interleaved RG bytes.</summary>
        public static byte[] DecodeBC5(ReadOnlySpan<byte> data, int width, int height)
        {
            var output = new byte[width * height * 2];
            DecodeBC4Into(data, 16, 0, width, height, output, 2, 0);
            DecodeBC4Into(data, 16, 8, width, height, output, 2, 1);
            return output;
        }

        private static void DecodeBC4Into(ReadOnlySpan<byte> data, int blockBytes, int blockOffset, int width, int height,
                                          byte[] output, int stride, int channel)
        {
            int bw = width / 4, bh = height / 4;
            Span<byte> palette = stackalloc byte[8];
            for (int by = 0; by < bh; by++)
                for (int bx = 0; bx < bw; bx++)
                {
                    var src = data.Slice((by * bw + bx) * blockBytes + blockOffset, 8);
                    int r0 = src[0], r1 = src[1];
                    palette[0] = (byte)r0;
                    palette[1] = (byte)r1;
                    if (r0 > r1)
                        for (int k = 1; k <= 6; k++) palette[k + 1] = (byte)(((7 - k) * r0 + k * r1 + 3) / 7);
                    else
                    {
                        for (int k = 1; k <= 4; k++) palette[k + 1] = (byte)(((5 - k) * r0 + k * r1 + 2) / 5);
                        palette[6] = 0;
                        palette[7] = 255;
                    }
                    ulong bits = 0;
                    for (int i = 0; i < 6; i++) bits |= (ulong)src[2 + i] << (8 * i);
                    for (int i = 0; i < 16; i++)
                    {
                        int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                        output[(y * width + x) * stride + channel] = palette[(int)((bits >> (3 * i)) & 7)];
                    }
                }
        }

        // ── BC1 ──

        /// <summary>
        /// RGB (3 bytes a pixel, row by row) as BC1 in its four-colour mode: 8 bytes per block. The ends are the block's
        /// colours furthest apart along their principal axis, pulled in by a sixteenth of the span (the inset of the usual
        /// range fit), each pixel the nearest of the four. Blocks in parallel.
        /// </summary>
        public static byte[] EncodeBC1(byte[] rgb, int width, int height)
        {
            int bw = width / 4, bh = height / 4;
            var output = new byte[bw * bh * 8];
            Parallel.For(0, bh, by =>
            {
                Span<Vector3> block = stackalloc Vector3[16];
                for (int bx = 0; bx < bw; bx++)
                {
                    for (int y = 0; y < 4; y++)
                        for (int x = 0; x < 4; x++)
                        {
                            int p = ((by * 4 + y) * width + bx * 4 + x) * 3;
                            block[y * 4 + x] = new Vector3(rgb[p], rgb[p + 1], rgb[p + 2]);
                        }
                    EncodeBC1Block(block, output.AsSpan((by * bw + bx) * 8, 8));
                }
            });
            return output;
        }

        private static void EncodeBC1Block(ReadOnlySpan<Vector3> block, Span<byte> dst)
        {
            var mean = Vector3.Zero;
            for (int i = 0; i < 16; i++) mean += block[i];
            mean /= 16f;

            // The principal axis by a few rounds of power iteration on the covariance
            float xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
            for (int i = 0; i < 16; i++)
            {
                var d = block[i] - mean;
                xx += d.X * d.X; xy += d.X * d.Y; xz += d.X * d.Z;
                yy += d.Y * d.Y; yz += d.Y * d.Z; zz += d.Z * d.Z;
            }
            var axis = new Vector3(1f, 1f, 1f);
            for (int k = 0; k < 6; k++)
            {
                var next = new Vector3(xx * axis.X + xy * axis.Y + xz * axis.Z,
                                       xy * axis.X + yy * axis.Y + yz * axis.Z,
                                       xz * axis.X + yz * axis.Y + zz * axis.Z);
                float len = next.Length();
                if (len < 1e-6f) break;
                axis = next / len;
            }

            float tMin = float.MaxValue, tMax = float.MinValue;
            for (int i = 0; i < 16; i++)
            {
                float t = Vector3.Dot(block[i] - mean, axis);
                if (t < tMin) tMin = t;
                if (t > tMax) tMax = t;
            }
            float inset = (tMax - tMin) / 16f;
            var a = Vector3.Clamp(mean + axis * (tMax - inset), Vector3.Zero, new Vector3(255f));
            var b = Vector3.Clamp(mean + axis * (tMin + inset), Vector3.Zero, new Vector3(255f));

            ushort c0 = To565(a), c1 = To565(b);
            if (c0 < c1) (c0, c1) = (c1, c0);
            dst[0] = (byte)c0; dst[1] = (byte)(c0 >> 8);
            dst[2] = (byte)c1; dst[3] = (byte)(c1 >> 8);

            uint bits = 0;
            if (c0 != c1)
            {
                Span<Vector3> palette = stackalloc Vector3[4];
                palette[0] = From565(c0);
                palette[1] = From565(c1);
                palette[2] = (2f * palette[0] + palette[1]) / 3f;
                palette[3] = (palette[0] + 2f * palette[1]) / 3f;
                for (int i = 0; i < 16; i++)
                {
                    int best = 0;
                    float bestD = float.MaxValue;
                    for (int k = 0; k < 4; k++)
                    {
                        float d = Vector3.DistanceSquared(block[i], palette[k]);
                        if (d < bestD) { bestD = d; best = k; }
                    }
                    bits |= (uint)best << (2 * i);
                }
            }
            dst[4] = (byte)bits; dst[5] = (byte)(bits >> 8); dst[6] = (byte)(bits >> 16); dst[7] = (byte)(bits >> 24);
        }

        private static ushort To565(Vector3 c)
        {
            int r = (int)MathF.Round(c.X * 31f / 255f), g = (int)MathF.Round(c.Y * 63f / 255f), b = (int)MathF.Round(c.Z * 31f / 255f);
            return (ushort)((Math.Clamp(r, 0, 31) << 11) | (Math.Clamp(g, 0, 63) << 5) | Math.Clamp(b, 0, 31));
        }

        private static Vector3 From565(ushort c)
        {
            int r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
            return new Vector3((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2));
        }

        /// <summary>BC1 back to RGB bytes.</summary>
        public static byte[] DecodeBC1(ReadOnlySpan<byte> data, int width, int height)
        {
            int bw = width / 4, bh = height / 4;
            var output = new byte[width * height * 3];
            Span<Vector3> palette = stackalloc Vector3[4];
            for (int by = 0; by < bh; by++)
                for (int bx = 0; bx < bw; bx++)
                {
                    var src = data.Slice((by * bw + bx) * 8, 8);
                    ushort c0 = (ushort)(src[0] | (src[1] << 8)), c1 = (ushort)(src[2] | (src[3] << 8));
                    palette[0] = From565(c0);
                    palette[1] = From565(c1);
                    if (c0 > c1)
                    {
                        palette[2] = (2f * palette[0] + palette[1]) / 3f;
                        palette[3] = (palette[0] + 2f * palette[1]) / 3f;
                    }
                    else
                    {
                        palette[2] = (palette[0] + palette[1]) / 2f;
                        palette[3] = Vector3.Zero;
                    }
                    uint bits = (uint)(src[4] | (src[5] << 8) | (src[6] << 16) | (src[7] << 24));
                    for (int i = 0; i < 16; i++)
                    {
                        var c = palette[(int)((bits >> (2 * i)) & 3)];
                        int p = ((by * 4 + (i >> 2)) * width + bx * 4 + (i & 3)) * 3;
                        output[p] = (byte)c.X; output[p + 1] = (byte)c.Y; output[p + 2] = (byte)c.Z;
                    }
                }
            return output;
        }
    }
}
