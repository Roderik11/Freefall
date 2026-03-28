using System;
using System.IO;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Minimal PSD file reader that extracts the flattened composite image
    /// with proper alpha channel preservation.
    /// 
    /// The PSD format stores RGBA channels separately in the image data section.
    /// texconv destroys alpha when converting PSDs — this reader handles it correctly
    /// by reading the raw channel data and outputting a BMP that can be fed to texconv.
    /// 
    /// Supports: RGB/RGBA, 8-bit/16-bit, raw and RLE (PackBits) compression.
    /// </summary>
    public static class PsdReader
    {
        private const int PSD_SIGNATURE = 0x38425053; // "8BPS"

        /// <summary>
        /// Read a PSD file and return the flattened composite as RGBA bytes (8-bit per channel).
        /// </summary>
        /// <param name="psdPath">Path to the .psd file</param>
        /// <param name="width">Output image width</param>
        /// <param name="height">Output image height</param>
        /// <param name="hasAlpha">Whether the PSD has an alpha channel</param>
        /// <returns>RGBA pixel data (width * height * 4 bytes), or null on failure</returns>
        public static byte[] ReadRgba(string psdPath, out int width, out int height, out bool hasAlpha)
        {
            width = height = 0;
            hasAlpha = false;

            using var stream = File.OpenRead(psdPath);
            using var reader = new BinaryReader(stream);

            // ── Header (26 bytes) ──
            var sig = ReadInt32BE(reader);
            if (sig != PSD_SIGNATURE)
                return null;

            var version = ReadInt16BE(reader);
            if (version != 1)
                return null;

            reader.ReadBytes(6); // reserved

            var channels = ReadInt16BE(reader);
            height = ReadInt32BE(reader);
            width = ReadInt32BE(reader);
            var depth = ReadInt16BE(reader);
            var colorMode = ReadInt16BE(reader);

            if (depth != 8 && depth != 16)
                return null;

            if (colorMode != 3) // 3 = RGB
                return null;

            hasAlpha = channels >= 4;

            // ── Color mode data section ──
            var colorDataLen = ReadInt32BE(reader);
            if (colorDataLen > 0) stream.Seek(colorDataLen, SeekOrigin.Current);

            // ── Image resources section ──
            var imageResourcesLen = ReadInt32BE(reader);
            if (imageResourcesLen > 0) stream.Seek(imageResourcesLen, SeekOrigin.Current);

            // ── Layer and mask section ──
            var layerMaskLen = ReadInt32BE(reader);
            if (layerMaskLen > 0) stream.Seek(layerMaskLen, SeekOrigin.Current);

            // ── Image data section (flattened composite) ──
            var compression = ReadInt16BE(reader);

            int pixelCount = width * height;
            var rgba = new byte[pixelCount * 4];

            // Initialize alpha to opaque
            for (int i = 0; i < pixelCount; i++)
                rgba[i * 4 + 3] = 255;

            if (compression == 0)
                ReadRawChannels(reader, rgba, width, height, channels, depth);
            else if (compression == 1)
                ReadRleChannels(reader, rgba, width, height, channels, depth);
            else
                return null; // ZIP compression not supported

            return rgba;
        }

        /// <summary>
        /// Convert a PSD to a temp BMP file suitable for texconv.
        /// Returns the path to the temp BMP, or null on failure.
        /// Caller is responsible for cleanup.
        /// </summary>
        public static string ConvertToTempBmp(string psdPath)
        {
            var rgba = ReadRgba(psdPath, out int width, out int height, out _);
            if (rgba == null)
                return null;

            var tempPath = Path.Combine(
                Path.GetTempPath(),
                $"freefall_psd_{Path.GetFileNameWithoutExtension(psdPath)}_{Guid.NewGuid():N}.bmp"
            );

            WriteBmp32(tempPath, rgba, width, height);
            return tempPath;
        }

        /// <summary>
        /// Quick check whether a PSD file has an alpha channel.
        /// Reads only the header (26 bytes).
        /// </summary>
        public static bool HasAlpha(string psdPath)
        {
            try
            {
                using var stream = File.OpenRead(psdPath);
                using var reader = new BinaryReader(stream);

                if (ReadInt32BE(reader) != PSD_SIGNATURE) return false;
                ReadInt16BE(reader); // version
                reader.ReadBytes(6); // reserved
                var channels = ReadInt16BE(reader);
                return channels >= 4;
            }
            catch { return true; } // assume alpha on error
        }

        // ── Channel readers ──

        private static void ReadRawChannels(BinaryReader reader, byte[] rgba, int width, int height, int channels, int depth)
        {
            int pixelCount = width * height;
            int channelsToRead = Math.Min(channels, 4);
            bool is16bit = depth == 16;

            for (int ch = 0; ch < channelsToRead; ch++)
            {
                int offset = ch; // R=0, G=1, B=2, A=3 in our RGBA layout
                for (int i = 0; i < pixelCount; i++)
                {
                    if (is16bit)
                    {
                        // Read 16-bit, scale to 8-bit
                        rgba[i * 4 + offset] = (byte)(ReadInt16BE(reader) >> 8);
                    }
                    else
                    {
                        rgba[i * 4 + offset] = reader.ReadByte();
                    }
                }
            }

            // Skip extra channels beyond RGBA
            int bytesPerPixel = is16bit ? 2 : 1;
            for (int ch = channelsToRead; ch < channels; ch++)
                reader.BaseStream.Seek((long)pixelCount * bytesPerPixel, SeekOrigin.Current);
        }

        private static void ReadRleChannels(BinaryReader reader, byte[] rgba, int width, int height, int channels, int depth)
        {
            int pixelCount = width * height;
            int channelsToRead = Math.Min(channels, 4);
            int bytesPerRow = width * (depth == 16 ? 2 : 1);

            // RLE: first come byte-count-per-scanline for ALL channels × ALL rows
            int totalScanlines = channels * height;
            var scanlineLengths = new int[totalScanlines];
            for (int i = 0; i < totalScanlines; i++)
                scanlineLengths[i] = ReadInt16BE(reader);

            // Then compressed data for each channel, each scanline
            int scanlineIdx = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                var channelData = new byte[pixelCount * (depth == 16 ? 2 : 1)];
                int pos = 0;

                for (int row = 0; row < height; row++)
                {
                    int bytesForLine = scanlineLengths[scanlineIdx++];
                    var compressedLine = reader.ReadBytes(bytesForLine);
                    pos = UnpackBits(compressedLine, channelData, pos);
                }

                // Map to RGBA if within first 4 channels
                if (ch < channelsToRead)
                {
                    if (depth == 16)
                    {
                        for (int i = 0; i < pixelCount; i++)
                            rgba[i * 4 + ch] = channelData[i * 2]; // high byte
                    }
                    else
                    {
                        for (int i = 0; i < pixelCount; i++)
                            rgba[i * 4 + ch] = channelData[i];
                    }
                }
            }
        }

        /// <summary>
        /// PackBits decompression (Apple format).
        /// </summary>
        private static int UnpackBits(byte[] src, byte[] dst, int dstPos)
        {
            int srcPos = 0;

            while (srcPos < src.Length)
            {
                sbyte n = (sbyte)src[srcPos++];

                if (n >= 0)
                {
                    // Copy next n+1 bytes literally
                    int count = n + 1;
                    for (int i = 0; i < count && srcPos < src.Length; i++)
                        dst[dstPos++] = src[srcPos++];
                }
                else if (n > -128)
                {
                    // Repeat next byte 1-n+1 times
                    int count = -n + 1;
                    byte value = srcPos < src.Length ? src[srcPos++] : (byte)0;
                    for (int i = 0; i < count; i++)
                        dst[dstPos++] = value;
                }
                // n == -128: no-op
            }

            return dstPos;
        }

        // ── BMP writer (32-bit BGRA) ──

        /// <summary>
        /// Write RGBA pixel data to a 32-bit BMP file with alpha channel preserved.
        /// Uses BITMAPV4HEADER (108 bytes) with BI_BITFIELDS so tools like texconv
        /// correctly recognize the alpha channel instead of treating it as padding.
        /// BMP uses BGRA bottom-up scanlines.
        /// </summary>
        private static void WriteBmp32(string path, byte[] rgba, int width, int height)
        {
            int rowBytes = width * 4;
            int imageSize = rowBytes * height;
            int headerSize = 108; // BITMAPV4HEADER
            int fileHeaderSize = 14;
            int pixelOffset = fileHeaderSize + headerSize;
            int fileSize = pixelOffset + imageSize;

            using var fs = File.Create(path);
            using var w = new BinaryWriter(fs);

            // BMP file header (14 bytes)
            w.Write((byte)'B'); w.Write((byte)'M');
            w.Write(fileSize);
            w.Write(0); // reserved
            w.Write(pixelOffset); // pixel data offset

            // BITMAPV4HEADER (108 bytes)
            w.Write(headerSize);
            w.Write(width);
            w.Write(height); // positive = bottom-up
            w.Write((short)1); // planes
            w.Write((short)32); // bpp
            w.Write(3); // compression = BI_BITFIELDS
            w.Write(imageSize);
            w.Write(2835); // X ppi (~72 dpi)
            w.Write(2835); // Y ppi
            w.Write(0); // colors used
            w.Write(0); // important colors

            // Channel masks: BGRA in memory
            w.Write(0x00FF0000); // R mask
            w.Write(0x0000FF00); // G mask
            w.Write(0x000000FF); // B mask
            w.Write(unchecked((int)0xFF000000)); // A mask

            // V4 fields: color space type + endpoints + gamma (56 bytes of zeros)
            w.Write(0x73524742); // LCS_sRGB
            for (int i = 0; i < 12; i++) w.Write(0); // CIEXYZTRIPLE endpoints (36 bytes) + gamma RGB (12 bytes)

            // Pixel data: BMP is bottom-up and BGRA
            for (int y = height - 1; y >= 0; y--)
            {
                int rowStart = y * width * 4;
                for (int x = 0; x < width; x++)
                {
                    int i = rowStart + x * 4;
                    w.Write(rgba[i + 2]); // B
                    w.Write(rgba[i + 1]); // G
                    w.Write(rgba[i + 0]); // R
                    w.Write(rgba[i + 3]); // A
                }
            }
        }

        // ── Big-endian readers (PSD is big-endian) ──

        private static short ReadInt16BE(BinaryReader reader)
        {
            var b = reader.ReadBytes(2);
            return (short)((b[0] << 8) | b[1]);
        }

        private static int ReadInt32BE(BinaryReader reader)
        {
            var b = reader.ReadBytes(4);
            return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
        }
    }
}
