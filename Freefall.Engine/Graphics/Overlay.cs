using System.Collections.Generic;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Flat marks over the picture for one frame: a crosshair, a line of text. Game code calls
    /// <see cref="Box"/> and <see cref="Text"/> from its update, in every frame it wants them
    /// shown; whoever presents the picture draws them over it, and they are gone with the next
    /// frame. A position is an anchor in the picture (0..1 across and down) plus an offset in
    /// pixels, so a mark keeps its size and its place whatever the size of the view.
    ///
    /// This is not a UI: nothing here can be clicked or laid out. Whoever presents the picture calls
    /// <see cref="Draw"/>: the editor's viewport over the view the main camera renders to, the player
    /// over its window. Text is drawn in the game UI skin's Font (UISkin.MarkFontSize): a game's text
    /// comes from its assets.
    /// </summary>
    public static class Overlay
    {
        private static readonly List<Mark> _drawing = new();
        private static bool _warnedNoFont;

        /// <summary>
        /// Draw this frame's marks with a Squid renderer (inside its batch, as a control's draw is) over a
        /// picture at <paramref name="at"/> of <paramref name="size"/> pixels.
        /// </summary>
        public static void Draw(Squid.ISquidRenderer renderer, Squid.Point at, Squid.Point size, float opacity = 1f)
        {
            _drawing.Clear();
            CopyTo(_drawing);
            if (_drawing.Count == 0) return;

            int font = -2;   // not looked up yet
            foreach (var mark in _drawing)
            {
                int x = at.x + (int)MathF.Round(mark.Anchor.X * size.x + mark.Offset.X);
                int y = at.y + (int)MathF.Round(mark.Anchor.Y * size.y + mark.Offset.Y);
                int color = Squid.ColorInt.ARGB(mark.Color.W * opacity, mark.Color.X, mark.Color.Y, mark.Color.Z);

                if (mark.Text == null)
                {
                    renderer.DrawBox(x, y, Math.Max(1, (int)MathF.Round(mark.Size.X)), Math.Max(1, (int)MathF.Round(mark.Size.Y)), color);
                    continue;
                }

                if (font == -2)
                {
                    var name = UI.GameUI.Skin?.MarkFontName(renderer as UI.IGameUIRenderer);
                    font = name != null ? renderer.GetFont(name) : -1;
                    if (font < 0 && !_warnedNoFont)
                    {
                        _warnedNoFont = true;
                        Debug.LogWarning("Overlay", "Text marks need the game UI skin's Font (GameUI.Skin): not drawn");
                    }
                }
                if (font < 0) continue;

                // Centred on the point, with a dark copy underneath so it reads on any ground
                var extent = renderer.GetTextSize(mark.Text, font);
                int left = x - extent.x / 2;
                renderer.DrawText(mark.Text, left + 1, y + 1, font, Squid.ColorInt.ARGB(0.75f * mark.Color.W * opacity, 0f, 0f, 0f));
                renderer.DrawText(mark.Text, left, y, font, color);
            }
        }
        public struct Mark
        {
            /// <summary>Where in the picture, 0..1 across and down.</summary>
            public Vector2 Anchor;

            /// <summary>Pixels from there: to the top left corner of a box, to the top middle of a text.</summary>
            public Vector2 Offset;

            /// <summary>A box's size in pixels. Unused for text.</summary>
            public Vector2 Size;

            /// <summary>Red, green, blue, alpha, 0..1.</summary>
            public Vector4 Color;

            /// <summary>Null for a box.</summary>
            public string Text;
        }

        // Component updates can run side by side
        private static readonly object gate = new();
        private static readonly List<Mark> marks = new();

        public static void Box(Vector2 anchor, Vector2 offset, Vector2 size, Vector4 color)
        {
            lock (gate) marks.Add(new Mark { Anchor = anchor, Offset = offset, Size = size, Color = color });
        }

        /// <summary>A line of text, centred on the point.</summary>
        public static void Text(Vector2 anchor, Vector2 offset, string text, Vector4 color)
        {
            if (string.IsNullOrEmpty(text)) return;
            lock (gate) marks.Add(new Mark { Anchor = anchor, Offset = offset, Color = color, Text = text });
        }

        /// <summary>The marks asked for this frame, for whoever draws them.</summary>
        public static void CopyTo(List<Mark> target)
        {
            lock (gate) target.AddRange(marks);
        }

        /// <summary>A new frame: the engine calls this before anything updates.</summary>
        internal static void Clear()
        {
            lock (gate) marks.Clear();
        }
    }
}
