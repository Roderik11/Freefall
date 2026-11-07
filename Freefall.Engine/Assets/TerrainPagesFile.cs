using System;
using System.IO;
using System.IO.Compression;
using System.Threading.Tasks;

namespace Freefall.Assets
{
    /// <summary>
    /// The terrain's heights and holes on disk, page by page (TERRAIN_PAGES_PLAN.md 4c): what a load reads in place of
    /// a bake, and what streaming (stage 5) will read a page at a time. Until 4c a save wrote the whole map as one flat
    /// R16 DDS, a flat R8 hole mask and one cooked PhysX height field of the whole terrain: 537 MB, and 1.07 GB of cook,
    /// for 16 km at 1 m per vertex.
    ///
    /// <code>
    /// header     magic "FFTP", version, resolution (samples a side), pages across, page samples (257), flags (1 = heights)
    /// directory  pages across² entries, row by row along +Z: height blob offset (8 bytes), its length, hole blob length
    ///            (4 bytes each; 0 = no hole cuts the page); the hole blob follows its height blob
    /// blobs      zlib streams
    /// </code>
    ///
    /// A height blob is the page's 257² samples (the R16 values the GPU pool holds) as differences from the plane
    /// through the left, lower and lower-left samples (left only on the first row, lower only in the first column),
    /// zig-zagged so a small difference either way is a small number, low bytes then high bytes, then zlib. A hole blob is
    /// the page's hole bytes under zlib. Everything is exact: a load gives back the store that was saved, to the bit.
    ///
    /// Lives in the Cache root as {guid}.pages (Terrain.PagesGuid), read through <see cref="Artifacts"/> so a published
    /// game finds it in its packs (TerrainLoader.GetImplicitDependencies declares it).
    /// </summary>
    public static class TerrainPagesFile
    {
        private const uint Magic = 0x50544646;   // "FFTP"
        private const int Version = 1;
        private const int HeaderBytes = 24;
        private const int EntryBytes = 16;

        /// <summary>The file a terrain's pages are saved in: no meta lists it, it sits in the Cache root.</summary>
        public static string PathOf(string guid) => Path.Combine(AssetDatabase.Project.CacheDirectory, $"{guid}.pages");

        // ── Writing ──

        /// <summary>
        /// Write <paramref name="pages"/> to <paramref name="path"/>: the pages compressed side by side, then written in
        /// one go to a file beside it that replaces the old one only once it is whole. Returns the bytes written.
        /// </summary>
        public static long Write(string path, TerrainPages pages)
        {
            int across = pages.PagesAcross, count = across * across;
            var heights = new byte[count][];
            var holes = new byte[count][];
            Parallel.For(0, count, i =>
            {
                int px = i % across, pz = i / across;
                heights[i] = pages.HasHeights ? EncodeHeights(pages.HeightPage(px, pz)) : Array.Empty<byte>();
                var holePage = pages.HolePage(px, pz);
                holes[i] = holePage.IsEmpty ? Array.Empty<byte>() : Deflate(holePage);
            });

            var temp = path + ".writing";
            using (var file = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
            using (var w = new BinaryWriter(file))
            {
                w.Write(Magic);
                w.Write(Version);
                w.Write(pages.Resolution);
                w.Write(across);
                w.Write(TerrainPages.PageSamples);
                w.Write(pages.HasHeights ? 1 : 0);

                long offset = HeaderBytes + (long)count * EntryBytes;
                for (int i = 0; i < count; i++)
                {
                    w.Write(offset);
                    w.Write(heights[i].Length);
                    w.Write(holes[i].Length);
                    offset += heights[i].Length + holes[i].Length;
                }
                for (int i = 0; i < count; i++)
                {
                    w.Write(heights[i]);
                    w.Write(holes[i]);
                }
            }
            File.Move(temp, path, overwrite: true);
            return new FileInfo(path).Length;
        }

        // ── Reading ──

        /// <summary>
        /// Read a whole pages file into a store (every page: nothing streams before stage 5), the pages decompressed side
        /// by side. Throws on a file that is not one, or of another layout.
        /// </summary>
        public static TerrainPages Read(Stream stream)
        {
            var r = new BinaryReader(stream);
            if (r.ReadUInt32() != Magic) throw new InvalidDataException("not a terrain pages file");
            int version = r.ReadInt32();
            if (version != Version) throw new InvalidDataException($"terrain pages file version {version}, this engine reads {Version}");
            int resolution = r.ReadInt32();
            int across = r.ReadInt32();
            int pageSamples = r.ReadInt32();
            bool hasHeights = (r.ReadInt32() & 1) != 0;
            if (pageSamples != TerrainPages.PageSamples || across != TerrainPages.PagesFor(resolution))
                throw new InvalidDataException($"pages of {pageSamples} samples, {across} across for {resolution} samples: not this engine's layout");

            int count = across * across;
            var offsets = new long[count];
            var heightLengths = new int[count];
            var holeLengths = new int[count];
            for (int i = 0; i < count; i++)
            {
                offsets[i] = r.ReadInt64();
                heightLengths[i] = r.ReadInt32();
                holeLengths[i] = r.ReadInt32();
            }

            // The blobs lie one after the other: read them in one go, then take them apart in parallel
            long first = HeaderBytes + (long)count * EntryBytes;
            long end = count > 0 ? offsets[count - 1] + heightLengths[count - 1] + holeLengths[count - 1] : first;
            var blobs = new byte[end - first];
            stream.Position = first;
            stream.ReadExactly(blobs);

            var heights = new ushort[]?[count];
            var holes = new byte[]?[count];
            Parallel.For(0, count, i =>
            {
                int at = (int)(offsets[i] - first);
                if (heightLengths[i] > 0)
                    heights[i] = DecodeHeights(blobs.AsSpan(at, heightLengths[i]));
                if (holeLengths[i] > 0)
                {
                    var page = new byte[TerrainPages.PageLength];
                    Inflate(blobs.AsSpan(at + heightLengths[i], holeLengths[i]), page);
                    holes[i] = page;
                }
            });
            return TerrainPages.FromPages(resolution, hasHeights, heights, holes);
        }

        // ── Heights ──

        private const int N = TerrainPages.PageSamples;

        // The plane through the left, lower and lower-left neighbours, in 16-bit arithmetic as the decoder does it
        private static ushort Predict(ReadOnlySpan<ushort> s, int x, int z)
        {
            if (z == 0) return x == 0 ? (ushort)0 : s[x - 1];
            if (x == 0) return s[(z - 1) * N];
            int i = z * N + x;
            return (ushort)(s[i - 1] + s[i - N] - s[i - N - 1]);
        }

        private static byte[] EncodeHeights(ReadOnlySpan<ushort> page)
        {
            var planes = new byte[2 * TerrainPages.PageLength];
            for (int z = 0; z < N; z++)
                for (int x = 0; x < N; x++)
                {
                    int i = z * N + x;
                    short d = (short)(page[i] - Predict(page, x, z));
                    ushort zig = (ushort)((d << 1) ^ (d >> 15));
                    planes[i] = (byte)zig;
                    planes[TerrainPages.PageLength + i] = (byte)(zig >> 8);
                }
            return Deflate(planes);
        }

        private static ushort[] DecodeHeights(ReadOnlySpan<byte> blob)
        {
            var planes = new byte[2 * TerrainPages.PageLength];
            Inflate(blob, planes);
            var page = new ushort[TerrainPages.PageLength];
            for (int z = 0; z < N; z++)
                for (int x = 0; x < N; x++)
                {
                    int i = z * N + x;
                    int zig = planes[i] | (planes[TerrainPages.PageLength + i] << 8);
                    ushort d = (ushort)((zig >> 1) ^ -(zig & 1));
                    page[i] = (ushort)(Predict(page, x, z) + d);
                }
            return page;
        }

        // ── zlib ──

        private static byte[] Deflate(ReadOnlySpan<byte> data)
        {
            var output = new MemoryStream(data.Length / 4 + 64);
            using (var z = new ZLibStream(output, CompressionLevel.Fastest, leaveOpen: true))
                z.Write(data);
            return output.ToArray();
        }

        private static void Inflate(ReadOnlySpan<byte> blob, Span<byte> into)
        {
            using var z = new ZLibStream(new MemoryStream(blob.ToArray()), CompressionMode.Decompress);
            z.ReadExactly(into);
        }
    }
}
