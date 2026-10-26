using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Compression formats supported by texconv.
    /// Values map directly to DXGI format names used by the -f argument.
    /// </summary>
    public enum TextureFormat
    {
        BC7_UNORM,
        BC7_UNORM_SRGB,
        BC5_UNORM,
        BC4_UNORM,
        BC3_UNORM,
        BC1_UNORM,
        BC1_UNORM_SRGB,
        R8G8B8A8_UNORM,
        R8_UNORM,
        R16_UNORM,
    }

    /// <summary>
    /// Imports texture files (.png, .jpg, .jpeg, .tga, .bmp) by converting them to
    /// BC7-compressed DDS via texconv.exe. DDS files are passed through as-is.
    ///
    /// Configurable fields (shown in editor import settings UI):
    ///   Format       – compression format (default: BC7_UNORM)
    ///   GenerateMips – generate full mipmap chain
    ///   sRGB         – treat source as sRGB color data
    ///   MaxSize      – largest width or height; a larger source is scaled down before it is compressed
    /// </summary>
    [AssetImporter(".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds", ".psd", ".tif")]
    public class TextureImporter : IImporter
    {
        [System.Text.Json.Serialization.JsonIgnore]
        public Type AssetType => typeof(Graphics.Texture);
       
        // ── Configurable import settings (editable in inspector) ──
        public TextureFormat Format = TextureFormat.BC7_UNORM;

        /// <summary>
        /// Largest width or height in pixels: a larger source is scaled down to it (aspect kept) before it is
        /// compressed, so the artifact, the packs and the GPU all hold the smaller one. 0 = the source's size.
        /// The project's guideline is 1024 (asset_pipeline overview §1.1); the engine takes any size. DDS sources
        /// pass through unscaled.
        /// </summary>
        [System.ComponentModel.Description("Largest width or height; a larger source is scaled down before it is compressed. Source = its own size")]
        [Reflection.ValueSelect(typeof(TextureMaxSizes))]
        public int MaxSize = 0;

        public bool sRGB = true;
        public bool GenerateMips = true;
        public bool IsNormalMap = false;
        public bool AlphaFromGrayscale = false;
        public bool IgnoreAlpha = false;
        public bool PremultiplyAlpha = false;
        public bool PreserveAlphaCoverage = false;
        public float AlphaCoverageThreshold = 0.2f;

        /// <summary>
        /// Set when the user manually saves settings. Prevents auto-detection
        /// from overwriting the configured Format/sRGB/IsNormalMap on reimport.
        /// </summary>
        [System.Text.Json.Serialization.JsonInclude]
        public bool UserConfigured = false;

        // Suffixes that indicate linear (non-color) data maps
        private static readonly string[] LinearSuffixes = {
            "_Nor", "Normal", "Norm", "Nrm", "_N",
            "_Spec", "_Specular", "_SpecGloss",
            "_Roughness", "_Rough", "_Smoothness", "_Gloss",
            "_Metal", "_Metallic", "_MetallicGloss",
            "AO", "Aoc", "Occlusion",
            "_Depth", "_Height", "_Parallax",
            "_Emissive", "_Emission",
            "_DetailMask", "_DetailNormal", "_Mask",
            "_BumpMap", "_Bump",
            "_ShadowOffset", "_TranslucencyMap",
        };

        // Suffixes that indicate normal maps — use BC5 (two high-quality channels)
        private static readonly string[] NormalSuffixes = {
            "_Nor", "Normal","Norm", "Nrm", "_N",
            "_DetailNormal",
            "_BumpMap", "_Bump",
        };

        // Single-channel data maps (shaders read .r only) — use BC4: one channel with 8-bit endpoints and
        // 8 levels per block. BC1's 5:6:5 RGB terraces smooth gradients, which screen-space displacement
        // turns into visible steps. Packed maps (_MetallicGloss, _SpecGloss, _Mask) are excluded.
        private static readonly string[] SingleChannelSuffixes = {
            "_Roughness", "_Rough", "_Smoothness", "_Gloss",
            "_Metal", "_Metallic",
            "AO", "Aoc", "Occlusion",
            "_Depth", "_Height", "_Parallax", "_DetailMask",
        };

        public ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var ext = Path.GetExtension(filepath).ToLowerInvariant();

            byte[] ddsBytes;

            if (ext == ".dds")
            {
                // DDS files are already in the correct format — pass through as-is
                // This preserves R16 heightmaps, splatmaps, etc.
                ddsBytes = File.ReadAllBytes(filepath);
            }
            else
            {
                // PSD: decode to temp BMP first (texconv destroys alpha on PSDs)
                string inputPath = filepath;
                string tempBmp = null;

                if (ext == ".psd")
                {
                    tempBmp = PsdReader.ConvertToTempBmp(filepath);
                    if (tempBmp == null)
                        throw new Exception($"Failed to decode PSD: {Path.GetFileName(filepath)}");
                    // Check alpha on the ORIGINAL PSD, not the temp BMP
                    // (HasAlphaChannel returns false for .bmp files)
                    if (PsdReader.HasAlpha(filepath))
                        Format = TextureFormat.BC7_UNORM;
                    inputPath = tempBmp;
                }

                try
                {
                    GuessSettings(filepath);
                    ddsBytes = ConvertToDds(inputPath, filepath);
                }
                finally
                {
                    // Cleanup temp BMP
                    if (tempBmp != null)
                        try { File.Delete(tempBmp); } catch { }
                }
            }

            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = nameof(DdsTextureData),
                Data = new DdsTextureData(ddsBytes)
            });

            return result;
        }

//        public object GetInspectionTarget(MetaFile meta) => this;

        /// <summary>
        /// Unless someone set them (UserConfigured): what the file's name says. Data maps (normals, roughness, ...) are
        /// linear, normal maps BC5 (two good channels, Z rebuilt in the shader), single-channel maps BC4. Import applies
        /// it, and AssetDatabase.GetImporter does for a meta with no settings saved yet.
        /// </summary>
        public void GuessSettings(string sourcePath)
        {
            if (UserConfigured) return;
            var name = Path.GetFileNameWithoutExtension(sourcePath);

            sRGB = !LinearSuffixes.Any(s => name.EndsWith(s, StringComparison.OrdinalIgnoreCase));
            if (NormalSuffixes.Any(s => name.EndsWith(s, StringComparison.OrdinalIgnoreCase)))
            {
                Format = TextureFormat.BC5_UNORM;
                IsNormalMap = true;
            }
            else if (SingleChannelSuffixes.Any(s => name.EndsWith(s, StringComparison.OrdinalIgnoreCase)))
            {
                Format = TextureFormat.BC4_UNORM;
            }
        }

        /// <summary>The Max Size choices: Source (0), then the powers of two from 8192 down.</summary>
        public sealed class TextureMaxSizes : Reflection.IValuesProvider
        {
            private static readonly Reflection.ProviderValue[] Values =
                new[] { 0, 8192, 4096, 2048, 1024, 512, 256, 128, 64 }
                    .Select(v => new Reflection.ProviderValue { Name = v == 0 ? "Source" : v.ToString(), Value = v }).ToArray();

            public System.Collections.Generic.IList<Reflection.ProviderValue> GetValues(object context) => Values;
        }

        private byte[] ConvertToDds(string sourcePath, string originalPath)
        {
            // Resolve texconv.exe next to the running executable
            var exeDir = AppDomain.CurrentDomain.BaseDirectory;
            var texconvPath = Path.Combine(exeDir, "texconv.exe");

            if (!File.Exists(texconvPath))
                throw new FileNotFoundException(
                    $"texconv.exe not found at '{texconvPath}'. " +
                    "Ensure it is included in the editor project and copied to output.");

            // Use a temp directory for output
            var tempDir = Path.Combine(Path.GetTempPath(), "freefall_texconv_" + Guid.NewGuid().ToString("N")[..8]);
            Directory.CreateDirectory(tempDir);

            try
            {
                bool hasAlpha = !IgnoreAlpha && HasAlphaChannel(sourcePath);

                // Auto-select format: BC5 is already set for normals, otherwise BC1/BC7
                var format = Format;
                if (format != TextureFormat.BC5_UNORM && format != TextureFormat.BC4_UNORM
                    && format != TextureFormat.R8_UNORM && format != TextureFormat.R16_UNORM)
                {
                    if (format == TextureFormat.BC7_UNORM || format == TextureFormat.BC7_UNORM_SRGB)
                    {
                        if (!hasAlpha && !IgnoreAlpha)
                            format = TextureFormat.BC1_UNORM;
                    }
                }

                // Build texconv arguments
                // When sRGB is enabled, use the _SRGB format variant so the DDS header
                // tells the GPU to auto-linearize during sampling
                if (sRGB)
                {
                    format = format switch
                    {
                        TextureFormat.BC7_UNORM => TextureFormat.BC7_UNORM_SRGB,
                        TextureFormat.BC1_UNORM => TextureFormat.BC1_UNORM_SRGB,
                        _ => format
                    };
                }

                var args = $"-y -gpu 0 -f {format}";

                if (GenerateMips)
                    args += " -m 0";

                // Premultiplied alpha: multiplies RGB×A so transparent pixels stay
                // black (0,0,0,0), fixing white fringing at alpha edges.
                // Off by default — enable per-texture for alpha-clip materials
                // (leaves, grass) where fringing is visible. Enabling this on
                // textures with empty or packed alpha will produce black RGB.
                if (PremultiplyAlpha && hasAlpha)
                    args += " -pmalpha";

                if (PreserveAlphaCoverage)
                    args += $" -keepcoverage {AlphaCoverageThreshold.ToString(System.Globalization.CultureInfo.InvariantCulture)}";

                // Always "sRGB in, sRGB out", also for linear data maps: the pair cancels, so texconv
                // never colour-converts. Without it a source that is tagged sRGB (PNG sRGB chunk — Blender
                // writes one into every PNG — or EXIF colour space) loads as *_UNORM_SRGB and is linearised
                // into the UNORM target (normal 0.5 → 0.21). A no-op for untagged sources.
                args += " -srgbi -srgbo";

                // The flags keep the top mip intact, but texconv still filters the mips of a tagged image in
                // decoded space. For linear data hand it a copy without the colour-space chunks instead.
                if (!sRGB)
                    sourcePath = StripPngColorSpace(sourcePath, tempDir) ?? sourcePath;

                // Force alpha to 1.0 — strips alpha data so mips don't bleed black.
                // texconv resizes (Max Size) before it swizzles, and its resize weights colour by alpha:
                // every pixel whose alpha is 0 would come out black (Head01's jaw). -sepalpha resizes
                // colour on its own; the mips are built after the swizzle, so it changes nothing there.
                if (IgnoreAlpha)
                    args += " -swizzle rgb1 -sepalpha";

                // Scaled down before compression (texconv resizes first, then builds the mips from the result)
                if (MaxSize > 0)
                {
                    var size = ImageSize(sourcePath);
                    if (size == null)
                        Debug.LogWarning("TextureImporter", $"{Path.GetFileName(originalPath)}: size unknown, Max Size {MaxSize} not applied");
                    else if (Math.Max(size.Value.Width, size.Value.Height) > MaxSize)
                    {
                        var (w, h) = ScaledSize(size.Value.Width, size.Value.Height, MaxSize);
                        args += $" -w {w} -h {h}";
                    }
                }

                args += $" -o \"{tempDir}\" \"{sourcePath}\"";

                var psi = new ProcessStartInfo(texconvPath)
                {
                    Arguments = args,
                    CreateNoWindow = true,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };

                using var process = Process.Start(psi);
                var stdout = process.StandardOutput.ReadToEnd();
                var stderr = process.StandardError.ReadToEnd();
                process.WaitForExit();

                if (process.ExitCode != 0)
                    throw new Exception(
                        $"texconv failed (exit {process.ExitCode}):\n{stderr}\n{stdout}");

                // Find the output DDS file
                var baseName = Path.GetFileNameWithoutExtension(sourcePath);
                var outputPath = Path.Combine(tempDir, baseName + ".dds");

                if (!File.Exists(outputPath))
                    throw new FileNotFoundException(
                        $"texconv did not produce expected output: {outputPath}\n{stdout}");

                return File.ReadAllBytes(outputPath);
            }
            finally
            {
                // Cleanup temp directory
                try { Directory.Delete(tempDir, true); } catch { }
            }
        }

        /// <summary>
        /// <paramref name="width"/> x <paramref name="height"/> scaled so the larger side is <paramref name="max"/>, the other
        /// in proportion: each side a multiple of 4 (block compression wants whole 4x4 blocks), at least 4.
        /// </summary>
        public static (int Width, int Height) ScaledSize(int width, int height, int max)
        {
            double scale = max / (double)Math.Max(width, height);
            static int Side(double v) => Math.Max(4, (int)Math.Round(v / 4) * 4);
            return (Side(width * scale), Side(height * scale));
        }

        /// <summary>The pixel size from the file's header (PNG, BMP, TGA, JPEG), or null for another or an unreadable file.</summary>
        public static (int Width, int Height)? ImageSize(string path)
        {
            try
            {
                using var fs = File.OpenRead(path);
                var head = new byte[32];
                if (fs.Read(head, 0, head.Length) < 26) return null;

                static int BigEndian(byte[] b, int i) => b[i] << 24 | b[i + 1] << 16 | b[i + 2] << 8 | b[i + 3];

                // PNG: IHDR width and height, big endian, right after the signature and the chunk header
                if (head[0] == 0x89 && head[1] == 'P' && head[2] == 'N' && head[3] == 'G')
                    return (BigEndian(head, 16), BigEndian(head, 20));

                // BMP: BITMAPINFOHEADER width and height (negative height = top-down)
                if (head[0] == 'B' && head[1] == 'M')
                    return (BitConverter.ToInt32(head, 18), Math.Abs(BitConverter.ToInt32(head, 22)));

                // JPEG: the first start-of-frame marker holds height and width
                if (head[0] == 0xFF && head[1] == 0xD8)
                {
                    fs.Position = 2;
                    var marker = new byte[4];
                    while (fs.Read(marker, 0, 4) == 4 && marker[0] == 0xFF)
                    {
                        int length = marker[2] << 8 | marker[3];
                        byte kind = marker[1];
                        bool frame = kind >= 0xC0 && kind <= 0xCF && kind != 0xC4 && kind != 0xC8 && kind != 0xCC;
                        if (frame)
                        {
                            var sof = new byte[5];
                            if (fs.Read(sof, 0, 5) < 5) return null;
                            return (sof[3] << 8 | sof[4], sof[1] << 8 | sof[2]);
                        }
                        fs.Position += length - 2;
                    }
                    return null;
                }

                // TGA (no signature): by extension, width and height little endian at 12 and 14
                if (Path.GetExtension(path).Equals(".tga", StringComparison.OrdinalIgnoreCase))
                    return (BitConverter.ToUInt16(head, 12), BitConverter.ToUInt16(head, 14));
            }
            catch (IOException) { }
            return null;
        }

        /// <summary>
        /// Writes a copy of a PNG without its colour-space chunks (sRGB, gAMA, cHRM, iCCP) into a
        /// subfolder of <paramref name="tempDir"/>, keeping the file name. Returns null when the file
        /// is not a PNG or carries none of them.
        /// </summary>
        private static string StripPngColorSpace(string path, string tempDir)
        {
            var data = File.ReadAllBytes(path);
            if (data.Length < 8 || data[0] != 0x89 || data[1] != 'P' || data[2] != 'N' || data[3] != 'G')
                return null;

            using var stripped = new MemoryStream(data.Length);
            stripped.Write(data, 0, 8);
            bool found = false;

            // Chunk: 4 length (big endian) + 4 type + data + 4 CRC — copied whole, so CRCs stay valid
            int pos = 8;
            while (pos + 12 <= data.Length)
            {
                long size = 12L + ((uint)data[pos] << 24 | (uint)data[pos + 1] << 16 | (uint)data[pos + 2] << 8 | data[pos + 3]);
                if (pos + size > data.Length)
                    return null; // truncated — let texconv report it on the original

                var type = System.Text.Encoding.ASCII.GetString(data, pos + 4, 4);
                if (type == "sRGB" || type == "gAMA" || type == "cHRM" || type == "iCCP")
                    found = true;
                else
                    stripped.Write(data, pos, (int)size);
                pos += (int)size;
            }

            if (!found)
                return null;

            var dir = Path.Combine(tempDir, "src");
            Directory.CreateDirectory(dir);
            var copy = Path.Combine(dir, Path.GetFileName(path));
            File.WriteAllBytes(copy, stripped.ToArray());
            return copy;
        }

        /// <summary>
        /// Quick alpha channel detection without loading the full image.
        /// Reads PNG IHDR color type; JPG/BMP never have alpha.
        /// </summary>
        private static bool HasAlphaChannel(string path)
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();

            // Formats that never have alpha
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
                return false;

            // PSD: quick header check (reads channel count)
            if (ext == ".psd")
                return PsdReader.HasAlpha(path);

            // PNG: read IHDR chunk to check color type
            if (ext == ".png")
            {
                try
                {
                    using var fs = File.OpenRead(path);
                    var header = new byte[26]; // 8 signature + 4 length + 4 type + 13 IHDR data + 1
                    if (fs.Read(header, 0, 26) < 26) return true; // assume alpha if can't read

                    // PNG color type is at byte 25 (offset 9 into IHDR data)
                    // 0=Grayscale, 2=RGB, 3=Indexed, 4=Grayscale+Alpha, 6=RGBA
                    byte colorType = header[25];
                    return colorType == 4 || colorType == 6; // has alpha component
                }
                catch { return true; } // assume alpha on error
            }

            // TGA: check image type and alpha bits in header
            if (ext == ".tga")
            {
                try
                {
                    using var fs = File.OpenRead(path);
                    var header = new byte[18];
                    if (fs.Read(header, 0, 18) < 18) return true;

                    byte bitsPerPixel = header[16];
                    byte descriptor = header[17];
                    int alphaBits = descriptor & 0x0F;
                    return bitsPerPixel == 32 && alphaBits > 0;
                }
                catch { return true; }
            }

            // DDS and unknown: assume alpha (conservative)
            return true;
        }
    }
}
