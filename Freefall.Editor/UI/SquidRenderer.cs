using System;
using System.Collections.Generic;
using System.IO;
using Vortice.Direct3D12;
using Vortice.Mathematics;
using Freefall;
using Freefall.Graphics;
using System.Drawing;

namespace Freefall.Editor
{
    /// <summary>
    /// Squid GUI renderer implementation for Freefall's DX12 bindless engine.
    /// Bridges Squid's draw calls to SpriteBatch + Font.
    /// </summary>
    public class SquidRenderer : Squid.ISquidRenderer, Freefall.UI.IGameUIRenderer, IDisposable
    {
        public SpriteBatch SpriteBatch { get; private set; }

        private int _fontIndex;
        private int _textureIndex;

        private readonly Dictionary<int, Freefall.Graphics.Font> _fonts = new();
        private readonly Dictionary<string, int> _fontLookup = new();
        private readonly Dictionary<int, TextureInfo> _textures = new();
        private readonly Dictionary<string, int> _textureLookup = new();
        private readonly Dictionary<string, Squid.Font> _fontTypes = new();

        // Command list set during StartBatch for use in EndBatch
        private ID3D12GraphicsCommandList _commandList = null!;
        private int _screenWidth;
        private int _screenHeight;
        private bool _isBatching;

        /// <summary>
        /// Stores bindless index + dimensions for a loaded texture.
        /// </summary>
        private struct TextureInfo
        {
            public uint BindlessIndex;
            public int Width;
            public int Height;
            public bool Smooth;   // linear sampling: art drawn at another size than painted (game UI)
        }

        public SquidRenderer(GraphicsDevice device)
        {
            SpriteBatch = new SpriteBatch(device);

            // Register font types used by the editor
            _fontTypes.Add(Squid.Font.Default, new Squid.Font { Name = "roboto_medium_10", Family = "Roboto", Size = 10, Bold = false, International = true });
            _fontTypes.Add("roboto_regular_10", new Squid.Font { Name = "roboto_regular_10", Family = "Roboto", Size = 10, Bold = false, International = true });
            _fontTypes.Add("roboto_medium_10", new Squid.Font { Name = "roboto_medium_10", Family = "Roboto", Size = 10, Bold = false, International = true });
            _fontTypes.Add("roboto_regular_9", new Squid.Font { Name = "roboto_regular_9", Family = "Roboto", Size = 9, Bold = false, International = true });
            _fontTypes.Add("roboto_medium_9", new Squid.Font { Name = "roboto_medium_9", Family = "Roboto", Size = 9, Bold = false, International = true });
            _fontTypes.Add("roboto_bold_9", new Squid.Font { Name = "roboto_bold_9", Family = "Roboto", Size = 9, Bold = true, International = true });

            Squid.Gui.AlwaysScissor = true;
            Squid.Gui.ScaledFontResolver = GetScaledFont;
        }

        /// <summary>
        /// Set the rendering context for the current frame. Call before Gui.Draw().
        /// </summary>
        public void SetContext(ID3D12GraphicsCommandList commandList, int screenWidth, int screenHeight)
        {
            _commandList = commandList;
            _screenWidth = screenWidth;
            _screenHeight = screenHeight;
        }

        private static readonly string[] TextureSearchDirs = ["Resources", "Resources/Images", "Resources/Cursors", "Resources/Fonts"];

        public int GetTexture(string name)
        {
            if (_textureLookup.TryGetValue(name, out var id))
                return id;

            // Search multiple subdirectories for the texture
            string path = "";
            bool found = false;
            foreach (var dir in TextureSearchDirs)
            {
                path = Path.Combine(Engine.RootDirectory, dir, name);
                if (File.Exists(path))
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                Debug.LogWarning("SquidRenderer", $"Texture not found: {name}");
                // Cache the miss with SpriteBatch's built-in white pixel as fallback
                _textureIndex++;
                _textureLookup.Add(name, _textureIndex);
                _textures.Add(_textureIndex, new TextureInfo
                {
                    BindlessIndex = SpriteBatch.WhiteTextureIndex,
                    Width = 1,
                    Height = 1
                });
                return _textureIndex;
            }

            var texture = Texture.LoadFromFile(Engine.Device, path);
            if (texture == null) return -1;

            var desc = texture.Native.Description;
            _textureIndex++;
            _textureLookup.Add(name, _textureIndex);
            _textures.Add(_textureIndex, new TextureInfo
            {
                BindlessIndex = texture.BindlessIndex,
                Width = (int)desc.Width,
                Height = (int)desc.Height
            });

            return _textureIndex;
        }

        /// <summary>
        /// Insert a pre-loaded texture by name (for programmatic texture injection).
        /// </summary>
        public void InsertTexture(string name, Texture texture)
        {
            var desc = texture.Native.Description;

            if (_textureLookup.TryGetValue(name, out var index))
            {
                _textures[index] = new TextureInfo
                {
                    BindlessIndex = texture.BindlessIndex,
                    Width = (int)desc.Width,
                    Height = (int)desc.Height
                };
                return;
            }

            _textureIndex++;
            _textureLookup.Add(name, _textureIndex);
            _textures.Add(_textureIndex, new TextureInfo
            {
                BindlessIndex = texture.BindlessIndex,
                Width = (int)desc.Width,
                Height = (int)desc.Height
            });
        }

        /// <summary>
        /// A texture of the game's UI skin: drawn smoothly, and known to Squid as <paramref name="scale"/>
        /// times its size, so texture rects in those units draw the art at that scale.
        /// </summary>
        public void InsertTexture(string name, Texture texture, float scale)
        {
            var desc = texture.Native.Description;
            var info = new TextureInfo
            {
                BindlessIndex = texture.BindlessIndex,
                Width = Math.Max(1, (int)MathF.Round(desc.Width * scale)),
                Height = Math.Max(1, (int)MathF.Round(desc.Height * scale)),
                Smooth = true,
            };

            if (_textureLookup.TryGetValue(name, out var index))
            {
                _textures[index] = info;
                return;
            }

            _textureIndex++;
            _textureLookup.Add(name, _textureIndex);
            _textures.Add(_textureIndex, info);
        }

        /// <summary>
        /// Insert a texture by bindless index and dimensions (for RenderTarget injection).
        /// </summary>
        public void InsertTexture(string name, uint bindlessIndex, int width, int height)
        {
            if (_textureLookup.TryGetValue(name, out var index))
            {
                _textures[index] = new TextureInfo
                {
                    BindlessIndex = bindlessIndex,
                    Width = width,
                    Height = height
                };
                return;
            }

            _textureIndex++;
            _textureLookup.Add(name, _textureIndex);
            _textures.Add(_textureIndex, new TextureInfo
            {
                BindlessIndex = bindlessIndex,
                Width = width,
                Height = height
            });
        }

        /// <summary>
        /// Update an existing texture entry's bindless index and dimensions.
        /// Used when a RenderTarget is resized.
        /// </summary>
        public void UpdateTexture(string name, uint bindlessIndex, int width, int height)
        {
            if (!_textureLookup.TryGetValue(name, out var id))
                return;

            _textures[id] = new TextureInfo
            {
                BindlessIndex = bindlessIndex,
                Width = width,
                Height = height
            };
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
            if (!_fontLookup.ContainsKey(name) && File.Exists(ttfPath))
                _rasterizing[name] = RuntimeFont.TakePrefetched(ttfPath, pixelSize, letterSpacing)
                    ?? System.Threading.Tasks.Task.Run(() => RuntimeFont.Rasterize(ttfPath, pixelSize, letterSpacing));
        }

        /// <summary>Full path of a TTF under Resources/Fonts (valid before the engine starts).</summary>
        public static string RuntimeFontPath(string ttfFile) =>
            Path.Combine(AppContext.BaseDirectory, "Resources", "Fonts", ttfFile);

        private readonly Dictionary<string, (Freefall.Assets.Typeface Face, int PixelSize)> _typefaceFonts = new();

        /// <summary>
        /// A font of the game's UI skin: rasterized from the typeface's bytes on first use. A name
        /// already built stays as it is (the name carries face and size).
        /// </summary>
        public void RegisterFont(string name, Freefall.Assets.Typeface face, int pixelSize)
        {
            if (face?.Data == null) return;
            _typefaceFonts[name] = (face, pixelSize);
        }

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

        public int GetFont(string name)
        {
            if (_fontLookup.TryGetValue(name, out var fontId))
                return fontId;

            if (_typefaceFonts.TryGetValue(name, out var typeface))
            {
                try
                {
                    var built = RuntimeFont.Build(typeface.Face.Data, typeface.PixelSize, 0);
                    _fontIndex++;
                    _fontLookup.Add(name, _fontIndex);
                    _fonts.Add(_fontIndex, built);
                    return _fontIndex;
                }
                catch (Exception e)
                {
                    Debug.LogWarning("SquidRenderer", $"Typeface {typeface.Face.Name} could not be read: {e.Message}");
                    _typefaceFonts.Remove(name);
                    return GetFont(Squid.Font.Default);
                }
            }

            if (_runtimeFonts.TryGetValue(name, out var runtime))
            {
                string ttfPath = RuntimeFontPath(runtime.TtfFile);
                if (File.Exists(ttfPath))
                {
                    var raster = _rasterizing.Remove(name, out var pending)
                        ? pending.GetAwaiter().GetResult()
                        : RuntimeFont.Rasterize(ttfPath, runtime.PixelSize, runtime.LetterSpacing);
                    var font = RuntimeFont.Upload(raster);
                    _fontIndex++;
                    _fontLookup.Add(name, _fontIndex);
                    _fonts.Add(_fontIndex, font);
                    return _fontIndex;
                }

                Debug.LogWarning("SquidRenderer", $"Font file not found: {runtime.TtfFile}");
                return -1;
            }

            if (!_fontTypes.ContainsKey(name))
                return -1;

            var type = _fontTypes[name];

            // Load bitmap font from DDS+XML
            string fontPath = Path.Combine(Engine.RootDirectory, "Resources", "Fonts", type.Name);
            if (File.Exists(fontPath + ".dds") && File.Exists(fontPath + "_data.xml"))
            {
                var font = Freefall.Graphics.Font.LoadFont(fontPath);
                _fontIndex++;
                _fontLookup.Add(name, _fontIndex);
                _fonts.Add(_fontIndex, font);
                return _fontIndex;
            }

            Debug.LogWarning("SquidRenderer", $"Font not found for: {type.Name}");
            return -1;
        }

        public Squid.Point GetTextSize(string text, int font)
        {
            if (string.IsNullOrEmpty(text))
                return new Squid.Point();

            if (!_fonts.TryGetValue(font, out var f))
                return new Squid.Point();

            var size = f.GetTextSize(text);
            return new Squid.Point(size.X, size.Y);
        }

        public Squid.Point GetTextureSize(int texture)
        {
            if (!_textures.TryGetValue(texture, out var info))
                return new Squid.Point();

            return new Squid.Point(info.Width, info.Height);
        }

        public void Scissor(int x, int y, int w, int h)
        {
            // RectI ctor is (x, y, width, height); Right/Bottom are computed as X+Width, Y+Height
            SpriteBatch.Scissor = new RectI(x, y, w, h);
        }

        public void DrawBox(int x, int y, int w, int h, int color)
        {
            SpriteBatch.Draw(x, y, w, h, new RectF(0, 0, 1, 1), new Color4(color),
                SpriteBatch.WhiteTextureIndex, 1, 1);
        }

        public void DrawText(string text, int x, int y, int font, int color)
        {
            if (_fonts.TryGetValue(font, out var f))
                f.DrawString(SpriteBatch, text, x, y, color);
        }

        public void DrawTexture(int texture, int x, int y, int w, int h, Squid.Rectangle rect, int color)
        {
            if (_textures.TryGetValue(texture, out var info))
            {
                SpriteBatch.Draw(x, y, w, h,
                    new RectF(rect.Left, rect.Top, rect.Width, rect.Height),
                    new Color4(color),
                    info.BindlessIndex, info.Width, info.Height, info.Smooth);
            }
        }

        public void StartBatch()
        {
            if (_isBatching) return;
            SpriteBatch.Begin();
            _isBatching = true;
        }

        public void EndBatch(bool final)
        {
            if (!final) return;

            SpriteBatch.End(_commandList, _screenWidth, _screenHeight);
            _isBatching = false;
        }

        public void Dispose()
        {
            SpriteBatch?.Dispose();
            foreach (var f in _fonts.Values)
                f.Dispose();
        }
    }
}
