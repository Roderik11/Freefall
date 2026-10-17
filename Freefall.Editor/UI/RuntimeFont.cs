using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Linq;
using System.Runtime.InteropServices;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// Rasterizes a TTF into a bitmap-font atlas at startup, for sizes we don't ship pre-baked
    /// DDS+XML fonts for (headings on the landing page). Registered through
    /// SquidRenderer.RegisterRuntimeFont, which starts Rasterize on a worker right away (the landing page's
    /// seven fonts took 0.37 s of its first frame one after another, 2026-10-17); Upload makes the texture on
    /// first use.
    /// </summary>
    internal static class RuntimeFont
    {
        // ASCII + Latin-1 (umlauts in project paths) + the few typographic marks the UI uses.
        private static readonly char[] Charset =
            Enumerable.Range(32, 95).Concat(Enumerable.Range(161, 95)).Select(i => (char)i)
                .Concat("…·×—–’“”•").ToArray();

        /// <summary>A rasterized atlas before its texture exists (RGBA, coverage in alpha).</summary>
        internal sealed class Raster
        {
            public byte[] Pixels;
            public int Width, Height, LineHeight, LetterSpacing;
            public Dictionary<char, Freefall.Graphics.Font.Glyph> Glyphs;
        }

        // Rasterizations started before anyone asked (Prefetch), taken by the registration that needs them
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<(string, int, int), System.Threading.Tasks.Task<Raster>> _prefetched = new();

        /// <summary>Start rasterizing on a worker now; SquidRenderer.RegisterRuntimeFont takes the result.</summary>
        public static void Prefetch(string ttfPath, int pixelSize, int letterSpacing)
        {
            if (System.IO.File.Exists(ttfPath))
                _prefetched.GetOrAdd((ttfPath, pixelSize, letterSpacing),
                    _ => System.Threading.Tasks.Task.Run(() => Rasterize(ttfPath, pixelSize, letterSpacing)));
        }

        /// <summary>The prefetched rasterization of this font (and forget it), or null.</summary>
        public static System.Threading.Tasks.Task<Raster> TakePrefetched(string ttfPath, int pixelSize, int letterSpacing)
            => _prefetched.TryRemove((ttfPath, pixelSize, letterSpacing), out var task) ? task : null;

        /// <param name="letterSpacing">Extra pixels between glyphs (for small uppercase labels).</param>
        public static Freefall.Graphics.Font Build(string ttfPath, int pixelSize, int letterSpacing)
            => Upload(Rasterize(ttfPath, pixelSize, letterSpacing));

        /// <summary>From a font file's bytes (a <see cref="Freefall.Assets.Typeface"/>).</summary>
        public static Freefall.Graphics.Font Build(byte[] fontFile, int pixelSize, int letterSpacing)
            => Upload(Rasterize(fontFile, pixelSize, letterSpacing));

        /// <summary>Rasterize a TTF file's glyphs. Any thread (GDI+ objects of its own, no GPU).</summary>
        public static Raster Rasterize(string ttfPath, int pixelSize, int letterSpacing)
        {
            using var collection = new PrivateFontCollection();
            collection.AddFontFile(ttfPath);
            return Rasterize(collection.Families[0], pixelSize, letterSpacing);
        }

        /// <summary>Rasterize a font file's bytes. Any thread.</summary>
        public static Raster Rasterize(byte[] fontFile, int pixelSize, int letterSpacing)
        {
            // GDI+ reads the face from this memory for as long as the collection lives
            var pinned = GCHandle.Alloc(fontFile, GCHandleType.Pinned);
            try
            {
                using var collection = new PrivateFontCollection();
                collection.AddMemoryFont(pinned.AddrOfPinnedObject(), fontFile.Length);
                return Rasterize(collection.Families[0], pixelSize, letterSpacing);
            }
            finally
            {
                pinned.Free();
            }
        }

        /// <summary>The atlas texture and font. Main thread (creates a GPU texture).</summary>
        public static Freefall.Graphics.Font Upload(Raster raster)
        {
            var texture = Texture.CreateFromData(Engine.Device, raster.Width, raster.Height, raster.Pixels);
            var result = Freefall.Graphics.Font.FromGlyphs(texture, raster.Width, raster.Height, raster.Glyphs, raster.LineHeight);
            result.Spacing = -raster.LetterSpacing;
            return result;
        }

        private static Raster Rasterize(FontFamily family, int pixelSize, int letterSpacing)
        {
            // A single-weight file only answers to the style it was built as.
            var style = new[] { FontStyle.Regular, FontStyle.Bold, FontStyle.Italic, FontStyle.Bold | FontStyle.Italic }
                .First(family.IsStyleAvailable);

            using var font = new System.Drawing.Font(family, pixelSize, style, GraphicsUnit.Pixel);
            using var format = new StringFormat(StringFormat.GenericTypographic);
            format.FormatFlags |= StringFormatFlags.MeasureTrailingSpaces | StringFormatFlags.NoClip;

            // Grid-fitting snaps every glyph to whole pixels, which reads as letter-spacing at small sizes;
            // there, keep the true outlines and fractional advances instead.
            var hint = pixelSize < 12 ? TextRenderingHint.AntiAlias : TextRenderingHint.AntiAliasGridFit;

            const int pad = 1;
            int atlasWidth = pixelSize >= 20 ? 1024 : 512;
            int lineHeight;
            var cells = new List<(char c, int x, int y, int w, float advance)>();

            using (var scratch = new Bitmap(1, 1, PixelFormat.Format32bppArgb))
            using (var g = System.Drawing.Graphics.FromImage(scratch))
            {
                g.TextRenderingHint = hint;
                lineHeight = (int)Math.Ceiling(font.GetHeight(g));

                int x = 0, y = 0;
                foreach (char c in Charset)
                {
                    float width = g.MeasureString(c.ToString(), font, PointF.Empty, format).Width;
                    int cellWidth = (int)Math.Ceiling(width) + pad * 2;
                    if (x + cellWidth > atlasWidth)
                    {
                        x = 0;
                        y += lineHeight + pad;
                    }
                    cells.Add((c, x, y, cellWidth, width));
                    x += cellWidth + pad;
                }
            }

            int atlasHeight = cells[^1].y + lineHeight;

            using var atlas = new Bitmap(atlasWidth, atlasHeight, PixelFormat.Format32bppArgb);
            using (var g = System.Drawing.Graphics.FromImage(atlas))
            {
                g.Clear(System.Drawing.Color.Transparent);
                g.TextRenderingHint = hint;
                foreach (var cell in cells)
                    g.DrawString(cell.c.ToString(), font, Brushes.White, cell.x + pad, cell.y, format);
            }

            // Coverage goes into alpha; RGB is forced white so tinting in the sprite shader is exact.
            var data = atlas.LockBits(new System.Drawing.Rectangle(0, 0, atlasWidth, atlasHeight),
                ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var pixels = new byte[atlasWidth * atlasHeight * 4];
            for (int row = 0; row < atlasHeight; row++)
                Marshal.Copy(data.Scan0 + row * data.Stride, pixels, row * atlasWidth * 4, atlasWidth * 4);
            atlas.UnlockBits(data);
            for (int i = 0; i < pixels.Length; i += 4)
                pixels[i] = pixels[i + 1] = pixels[i + 2] = 255;

            var glyphs = new Dictionary<char, Freefall.Graphics.Font.Glyph>(cells.Count);
            foreach (var cell in cells)
            {
                glyphs[cell.c] = new Freefall.Graphics.Font.Glyph
                {
                    Rect = new RectF(cell.x, cell.y, cell.w, lineHeight),
                    Width = cell.w,
                    Height = lineHeight,
                    Advance = cell.advance,
                };
            }

            return new Raster
            {
                Pixels = pixels,
                Width = atlasWidth,
                Height = atlasHeight,
                LineHeight = lineHeight,
                LetterSpacing = letterSpacing,
                Glyphs = glyphs,
            };
        }
    }
}
