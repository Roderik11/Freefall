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
    /// SquidRenderer.RegisterRuntimeFont and built lazily on first use.
    /// </summary>
    internal static class RuntimeFont
    {
        // ASCII + Latin-1 (umlauts in project paths) + the few typographic marks the UI uses.
        private static readonly char[] Charset =
            Enumerable.Range(32, 95).Concat(Enumerable.Range(161, 95)).Select(i => (char)i)
                .Concat("…·×—–’“”•").ToArray();

        /// <param name="letterSpacing">Extra pixels between glyphs (for small uppercase labels).</param>
        public static Freefall.Graphics.Font Build(string ttfPath, int pixelSize, int letterSpacing)
        {
            using var collection = new PrivateFontCollection();
            collection.AddFontFile(ttfPath);
            var family = collection.Families[0];

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

            var texture = Texture.CreateFromData(Engine.Device, atlasWidth, atlasHeight, pixels);

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

            var result = Freefall.Graphics.Font.FromGlyphs(texture, atlasWidth, atlasHeight, glyphs, lineHeight);
            result.Spacing = -letterSpacing;
            return result;
        }
    }
}
