using System;
using System.Collections.Generic;
using System.IO;
using Freefall;
using Freefall.Graphics;
using Freefall.UI;

namespace Freefall.Editor
{
    /// <summary>
    /// The editor's Squid renderer: the engine's <see cref="SquidRenderer"/> (drawing, and the game UI's skin textures
    /// and Typeface fonts) plus what only the editor has: its images and cursors found by file name under Resources, its
    /// pre-baked Roboto atlases (Resources/Fonts, DDS+XML), fonts rasterized from TTFs under Resources/Fonts at sizes
    /// without an atlas (landing page headings, zoomed controls). None of these are a game's: a game's text and pictures
    /// come from its assets.
    /// </summary>
    public class EditorSquidRenderer : SquidRenderer
    {
        private readonly Dictionary<string, Squid.Font> _fontTypes = new();

        public EditorSquidRenderer(GraphicsDevice device) : base(device)
        {
            // Register font types used by the editor
            _fontTypes.Add(Squid.Font.Default, new Squid.Font { Name = "roboto_medium_10", Family = "Roboto", Size = 10, Bold = false, International = true });
            _fontTypes.Add("roboto_regular_10", new Squid.Font { Name = "roboto_regular_10", Family = "Roboto", Size = 10, Bold = false, International = true });
            _fontTypes.Add("roboto_medium_10", new Squid.Font { Name = "roboto_medium_10", Family = "Roboto", Size = 10, Bold = false, International = true });
            _fontTypes.Add("roboto_regular_9", new Squid.Font { Name = "roboto_regular_9", Family = "Roboto", Size = 9, Bold = false, International = true });
            _fontTypes.Add("roboto_medium_9", new Squid.Font { Name = "roboto_medium_9", Family = "Roboto", Size = 9, Bold = false, International = true });
            _fontTypes.Add("roboto_bold_9", new Squid.Font { Name = "roboto_bold_9", Family = "Roboto", Size = 9, Bold = true, International = true });

            Squid.Gui.ScaledFontResolver = GetScaledFont;
        }

        private static readonly string[] TextureSearchDirs = ["Resources", "Resources/Images", "Resources/Cursors", "Resources/Fonts"];

        /// <summary>The editor's images and cursors: a file of that name in one of its Resources folders.</summary>
        protected override Texture? ResolveTexture(string name)
        {
            foreach (var dir in TextureSearchDirs)
            {
                var path = Path.Combine(Engine.RootDirectory, dir, name);
                if (File.Exists(path))
                    return Texture.LoadFromFile(Engine.Device, path);
            }
            return null;
        }

        private readonly Dictionary<string, (string TtfFile, int PixelSize, int LetterSpacing)> _runtimeFonts = new();

        // Runtime fonts being rasterized on workers since their registration (GetFont waits for its one, then uploads)
        private readonly Dictionary<string, System.Threading.Tasks.Task<RuntimeFont.Raster>> _rasterizing = new();

        /// <summary>
        /// Register a font that is rasterized from a TTF under Resources/Fonts, for sizes without a pre-baked
        /// DDS+XML atlas. Rasterizing starts on a worker now; the texture is made on first use.
        /// </summary>
        public void RegisterRuntimeFont(string name, string ttfFile, int pixelSize, int letterSpacing = 0)
        {
            _runtimeFonts[name] = (ttfFile, pixelSize, letterSpacing);

            string ttfPath = RuntimeFontPath(ttfFile);
            if (!HasFont(name) && File.Exists(ttfPath))
                _rasterizing[name] = RuntimeFont.TakePrefetched(ttfPath, pixelSize, letterSpacing)
                    ?? System.Threading.Tasks.Task.Run(() => RuntimeFont.Rasterize(ttfPath, pixelSize, letterSpacing));
        }

        /// <summary>Full path of a TTF under Resources/Fonts (valid before the engine starts).</summary>
        public static string RuntimeFontPath(string ttfFile) =>
            Path.Combine(AppContext.BaseDirectory, "Resources", "Fonts", ttfFile);

        // TTF and em size (px) behind each pre-baked atlas, so they can be re-rasterized at other sizes
        private static readonly Dictionary<string, (string TtfFile, int PixelSize)> BakedFontSources = new()
        {
            { "roboto_regular_10", ("Roboto/Roboto-Regular.ttf", 11) },
            { "roboto_medium_10", ("Roboto/Roboto-Medium.ttf", 11) },
            { "roboto_regular_9", ("Roboto/Roboto-Regular.ttf", 10) },
            { "roboto_medium_9", ("Roboto/Roboto-Medium.ttf", 10) },
            { "roboto_bold_9", ("Roboto/Roboto-Bold.ttf", 10) },
        };

        /// <summary>
        /// Font to draw with under a zoomed control (Gui.ScaledFontResolver): the same face
        /// rasterized at the scaled size. Falls back to the unscaled font when the source is unknown.
        /// </summary>
        public int GetScaledFont(string name, float scale)
        {
            if (string.IsNullOrEmpty(name) || name == Squid.Font.Default)
                name = "roboto_medium_10";

            string ttfFile;
            int basePixels, letterSpacing = 0;
            if (_runtimeFonts.TryGetValue(name, out var runtime))
                (ttfFile, basePixels, letterSpacing) = runtime;
            else if (BakedFontSources.TryGetValue(name, out var baked))
                (ttfFile, basePixels) = baked;
            else
                return GetFont(name);

            int pixels = Math.Max(4, (int)MathF.Round(basePixels * scale));
            if (pixels == basePixels)
                return GetFont(name);

            string scaledName = $"{name}@{pixels}";
            if (!_runtimeFonts.ContainsKey(scaledName))
                RegisterRuntimeFont(scaledName, ttfFile, pixels, (int)MathF.Round(letterSpacing * scale));

            return GetFont(scaledName);
        }

        /// <summary>The editor's fonts: rasterized from a registered TTF, or one of its pre-baked atlases.</summary>
        protected override Freefall.Graphics.Font? ResolveFont(string name)
        {
            if (_runtimeFonts.TryGetValue(name, out var runtime))
            {
                string ttfPath = RuntimeFontPath(runtime.TtfFile);
                if (File.Exists(ttfPath))
                {
                    var raster = _rasterizing.Remove(name, out var pending)
                        ? pending.GetAwaiter().GetResult()
                        : RuntimeFont.Rasterize(ttfPath, runtime.PixelSize, runtime.LetterSpacing);
                    return RuntimeFont.Upload(raster);
                }

                Debug.LogWarning("SquidRenderer", $"Font file not found: {runtime.TtfFile}");
                return null;
            }

            if (!_fontTypes.TryGetValue(name, out var type))
                return null;

            // Load bitmap font from DDS+XML
            string fontPath = Path.Combine(Engine.RootDirectory, "Resources", "Fonts", type.Name);
            if (File.Exists(fontPath + ".dds") && File.Exists(fontPath + "_data.xml"))
                return Freefall.Graphics.Font.LoadFont(fontPath);

            Debug.LogWarning("SquidRenderer", $"Font not found for: {type.Name}");
            return null;
        }
    }
}
