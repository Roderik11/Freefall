using System;
using System.Collections.Generic;
using Vortice.Direct3D12;
using Vortice.Mathematics;
using Freefall;
using Freefall.Graphics;

namespace Freefall.UI
{
    /// <summary>
    /// Squid GUI renderer implementation for Freefall's DX12 bindless engine.
    /// Bridges Squid's draw calls to SpriteBatch + Font. This is what a game draws its UI with: textures and fonts
    /// come from its assets (a UI skin's atlas and Typeface, <see cref="IGameUIRenderer"/>). A name it was not given
    /// goes to <see cref="ResolveTexture"/> / <see cref="ResolveFont"/>, which the editor's renderer overrides with its
    /// own images and fonts; here they warn once and draw nothing (a texture: a white pixel).
    /// </summary>
    public class SquidRenderer : Squid.ISquidRenderer, IGameUIRenderer, IDisposable
    {
        public SpriteBatch SpriteBatch { get; private set; }

        private int _fontIndex;
        private int _textureIndex;

        private readonly Dictionary<int, Freefall.Graphics.Font> _fonts = new();
        private readonly Dictionary<string, int> _fontLookup = new();
        private readonly Dictionary<int, TextureInfo> _textures = new();
        private readonly Dictionary<string, int> _textureLookup = new();
        private readonly HashSet<string> _missingFonts = new();

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
            Squid.Gui.AlwaysScissor = true;
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

        public int GetTexture(string name)
        {
            if (_textureLookup.TryGetValue(name, out var id))
                return id;

            var texture = ResolveTexture(name);
            if (texture == null)
            {
                Debug.LogWarning("SquidRenderer", $"Texture not found: {name}");
                // Cache the miss with SpriteBatch's built-in white pixel as fallback
                return AddTexture(name, new TextureInfo { BindlessIndex = SpriteBatch.WhiteTextureIndex, Width = 1, Height = 1 });
            }

            var desc = texture.Native.Description;
            return AddTexture(name, new TextureInfo { BindlessIndex = texture.BindlessIndex, Width = (int)desc.Width, Height = (int)desc.Height });
        }

        /// <summary>A texture asked for by a name it was not given (InsertTexture), or null. The editor loads its images here.</summary>
        protected virtual Texture? ResolveTexture(string name) => null;

        private int AddTexture(string name, TextureInfo info)
        {
            if (_textureLookup.TryGetValue(name, out var index))
            {
                _textures[index] = info;
                return index;
            }
            _textureIndex++;
            _textureLookup.Add(name, _textureIndex);
            _textures.Add(_textureIndex, info);
            return _textureIndex;
        }

        /// <summary>
        /// Insert a pre-loaded texture by name (for programmatic texture injection).
        /// </summary>
        public void InsertTexture(string name, Texture texture)
        {
            var desc = texture.Native.Description;
            AddTexture(name, new TextureInfo { BindlessIndex = texture.BindlessIndex, Width = (int)desc.Width, Height = (int)desc.Height });
        }

        /// <summary>
        /// A texture of the game's UI skin: drawn smoothly, and known to Squid as <paramref name="scale"/>
        /// times its size, so texture rects in those units draw the art at that scale.
        /// </summary>
        public void InsertTexture(string name, Texture texture, float scale)
        {
            var desc = texture.Native.Description;
            AddTexture(name, new TextureInfo
            {
                BindlessIndex = texture.BindlessIndex,
                Width = Math.Max(1, (int)MathF.Round(desc.Width * scale)),
                Height = Math.Max(1, (int)MathF.Round(desc.Height * scale)),
                Smooth = true,
            });
        }

        /// <summary>
        /// Insert a texture by bindless index and dimensions (for RenderTarget injection).
        /// </summary>
        public void InsertTexture(string name, uint bindlessIndex, int width, int height)
        {
            AddTexture(name, new TextureInfo { BindlessIndex = bindlessIndex, Width = width, Height = height });
        }

        /// <summary>
        /// Update an existing texture entry's bindless index and dimensions.
        /// Used when a RenderTarget is resized.
        /// </summary>
        public void UpdateTexture(string name, uint bindlessIndex, int width, int height)
        {
            if (!_textureLookup.TryGetValue(name, out var id))
                return;

            _textures[id] = new TextureInfo { BindlessIndex = bindlessIndex, Width = width, Height = height };
        }

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

        public int GetFont(string name)
        {
            if (string.IsNullOrEmpty(name)) return -1;
            if (_fontLookup.TryGetValue(name, out var fontId))
                return fontId;

            if (_typefaceFonts.TryGetValue(name, out var typeface))
            {
                try
                {
                    return AddFont(name, RuntimeFont.Build(typeface.Face.Data, typeface.PixelSize, 0));
                }
                catch (Exception e)
                {
                    Debug.LogWarning("SquidRenderer", $"Typeface {typeface.Face.Name} could not be read: {e.Message}");
                    _typefaceFonts.Remove(name);
                    return -1;
                }
            }

            var font = ResolveFont(name);
            if (font != null)
                return AddFont(name, font);

            if (_missingFonts.Add(name))
                Debug.LogWarning("SquidRenderer", $"No font '{name}': a game's text comes from its UI skin's Font");
            return -1;
        }

        /// <summary>A font asked for by a name no Typeface was registered under, or null. The editor builds its own fonts here.</summary>
        protected virtual Freefall.Graphics.Font? ResolveFont(string name) => null;

        /// <summary>Whether a font of this name has been built already.</summary>
        protected bool HasFont(string name) => _fontLookup.ContainsKey(name);

        private int AddFont(string name, Freefall.Graphics.Font font)
        {
            _fontIndex++;
            _fontLookup.Add(name, _fontIndex);
            _fonts.Add(_fontIndex, font);
            return _fontIndex;
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

        public virtual void Dispose()
        {
            SpriteBatch?.Dispose();
            foreach (var f in _fonts.Values)
                f.Dispose();
        }
    }
}
