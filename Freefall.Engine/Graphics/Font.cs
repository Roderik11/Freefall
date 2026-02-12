using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Xml;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Bitmap font loaded from XML descriptor + texture atlas.
    /// Renders text through SpriteBatch using the font texture's bindless index.
    /// </summary>
    public class Font : IDisposable
    {
        private Texture _texture;
        private int _texWidth;
        private int _texHeight;
        private Dictionary<char, Glyph> _glyphs = new();

        public int Height;
        public int Spacing = 0;

        public struct Glyph
        {
            public RectF Rect;
            public int Height;
            public int Width;
            public int OffsetX;
            public int OffsetY;
            public int Advance;
        }

        const char SPACE_CHAR = ' ';
        const char NEWLINE_CHAR = '\n';

        /// <summary>
        /// Draw a string using the SpriteBatch.
        /// </summary>
        public void DrawString(SpriteBatch batch, string str, int x, int y, int color)
        {
            if (string.IsNullOrEmpty(str)) return;

            int currentX = x;
            int currentY = y;
            var col = new Color4(color);

            foreach (char c in str)
            {
                if (c == NEWLINE_CHAR)
                {
                    currentY += Height;
                    currentX = x;
                }
                else if (_glyphs.TryGetValue(c, out Glyph glyph))
                {
                    if (c != SPACE_CHAR)
                    {
                        batch.Draw(
                            currentX, currentY + glyph.OffsetY,
                            glyph.Width, glyph.Height,
                            glyph.Rect, col,
                            _texture.BindlessIndex,
                            _texWidth, _texHeight);
                    }

                    currentX += glyph.Advance - Spacing;
                }
            }
        }

        /// <summary>
        /// Measure text dimensions in pixels.
        /// </summary>
        public Point GetTextSize(string text)
        {
            var p = new Point(0, Height);

            foreach (char c in text)
            {
                if (c == NEWLINE_CHAR)
                    p.Y += Height;
                else if (_glyphs.TryGetValue(c, out Glyph glyph))
                    p.X += glyph.Advance - Spacing;
            }

            return p;
        }

        /// <summary>
        /// Load from a texture and XML descriptor string.
        /// </summary>
        public static Font LoadFont(Texture texture, string xml)
        {
            var font = new Font { _texture = texture };

            // Cache texture dimensions
            var desc = texture.Native.Description;
            font._texWidth = (int)desc.Width;
            font._texHeight = (int)desc.Height;

            var doc = new XmlDocument();
            doc.LoadXml(xml);

            int maxHeight = 0;
            int minOffsetY = int.MaxValue;
            var glyphs = new Dictionary<char, Glyph>();

            var nodes = doc.SelectNodes("descendant::Char");
            int fontHeight = Convert.ToInt32(doc.SelectSingleNode("Font")!.Attributes!["height"]!.Value);

            foreach (XmlNode node in nodes!)
            {
                string[] rect = node.Attributes!["rect"]!.Value.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                string[] offset = node.Attributes["offset"]!.Value.Split(' ', StringSplitOptions.RemoveEmptyEntries);

                int offx = Convert.ToInt32(offset[0]);
                int offy = Convert.ToInt32(offset[1]);

                int rx = Math.Abs(Convert.ToInt32(rect[0]));
                int ry = Math.Abs(Convert.ToInt32(rect[1]));
                int rw = Convert.ToInt32(Convert.ToSingle(rect[2]));
                int rh = Convert.ToInt32(Convert.ToSingle(rect[3]));
                int advance = Convert.ToInt32(node.Attributes["width"]!.Value);

                maxHeight = Math.Max(maxHeight, rh);
                minOffsetY = Math.Min(minOffsetY, offy);

                string code = UnescapeXML(node.Attributes["code"]!.Value);

                try
                {
                    char c = char.Parse(code);
                    glyphs.Add(c, new Glyph
                    {
                        Rect = new RectF(rx, ry, rw, rh),
                        Advance = advance,
                        OffsetX = offx,
                        OffsetY = offy,
                        Width = rw,
                        Height = rh
                    });
                }
                catch
                {
                    Debug.LogWarning("Font", $"Malformed glyph entry: '{code}'");
                }
            }

            font.Height = maxHeight;

            foreach (var kv in glyphs)
            {
                var g = kv.Value;
                g.OffsetY -= minOffsetY;
                font._glyphs[kv.Key] = g;
            }

            return font;
        }

        /// <summary>
        /// Load a font by name (searches for name.dds + name_data.xml in engine resources).
        /// </summary>
        public static Font LoadFont(string path)
        {
            var texture = Texture.LoadFromFile(Engine.Device, path + ".dds");
            var xml = File.ReadAllText(path + "_data.xml");
            return LoadFont(texture, xml);
        }

        private static string UnescapeXML(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            return s
                .Replace("&apos;", "'")
                .Replace("&quot;", "\"")
                .Replace("&gt;", ">")
                .Replace("&lt;", "<")
                .Replace("&amp;", "&");
        }

        public void Dispose()
        {
            _texture?.Dispose();
            _texture = null!;
        }
    }
}
