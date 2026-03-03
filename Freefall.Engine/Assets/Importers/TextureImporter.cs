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
    }

    /// <summary>
    /// Imports texture files (.png, .jpg, .jpeg, .tga, .bmp) by converting them to
    /// BC7-compressed DDS via texconv.exe. DDS files are passed through as-is.
    ///
    /// Configurable fields (shown in editor import settings UI):
    ///   Format       – compression format (default: BC7_UNORM)
    ///   GenerateMips – generate full mipmap chain
    ///   sRGB         – treat source as sRGB color data
    /// </summary>
    [AssetImporter(".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds")]
    public class TextureImporter : IImporter
    {
        public Type AssetType => typeof(Graphics.Texture);
        // ── Configurable import settings (editable in inspector) ──
        public TextureFormat Format = TextureFormat.BC7_UNORM;
        public bool GenerateMips = true;
        public bool sRGB = true;

        // Suffixes that indicate linear (non-color) data maps
        private static readonly string[] LinearSuffixes = {
            "_Nor", "_Normal", "_Nrm",
            "_Spec", "_Specular", "_SpecGloss",
            "_Roughness", "_Rough",
            "_Metal", "_Metallic", "_MetallicGloss",
            "_AO", "_Aoc", "_Occlusion",
            "_Depth", "_Height", "_Parallax",
            "_Emissive", "_Emission",
            "_DetailMask", "_DetailNormal", "_Mask",
            "_BumpMap", "_Bump",
            "_ShadowOffset", "_TranslucencyMap",
        };

        // Suffixes that indicate normal maps — use BC5 (two high-quality channels)
        private static readonly string[] NormalSuffixes = {
            "_Nor", "_Normal", "_Nrm",
            "_DetailNormal",
            "_BumpMap", "_Bump",
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
                // Auto-detect: data maps (normals, roughness, etc.) should be linear, not sRGB
                bool isLinearData = LinearSuffixes.Any(s =>
                    name.EndsWith(s, StringComparison.OrdinalIgnoreCase));
                sRGB = !isLinearData;

                // Normal maps → BC5_UNORM (two high-quality RG channels, Z reconstructed in shader)
                bool isNormalMap = NormalSuffixes.Any(s =>
                    name.EndsWith(s, StringComparison.OrdinalIgnoreCase));
                if (isNormalMap)
                    Format = TextureFormat.BC5_UNORM;

                ddsBytes = ConvertToDds(filepath);
            }

            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = nameof(DdsTextureData),
                Data = new DdsTextureData(ddsBytes)
            });

            return result;
        }

        private byte[] ConvertToDds(string sourcePath)
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
                // Auto-select format: BC5 is already set for normals, otherwise BC1/BC7
                var format = Format;
                if (format != TextureFormat.BC5_UNORM && format != TextureFormat.BC4_UNORM)
                {
                    if (format == TextureFormat.BC7_UNORM || format == TextureFormat.BC7_UNORM_SRGB)
                    {
                        if (!HasAlphaChannel(sourcePath))
                        {
                            format = TextureFormat.BC1_UNORM;
                        }
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
                    args += " -m 0"; // full mip chain

                if (sRGB)
                    args += " -srgbi -srgbo";

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
        /// Quick alpha channel detection without loading the full image.
        /// Reads PNG IHDR color type; JPG/BMP never have alpha.
        /// </summary>
        private static bool HasAlphaChannel(string path)
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();

            // Formats that never have alpha
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
                return false;

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
