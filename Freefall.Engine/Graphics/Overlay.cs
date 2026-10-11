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
    /// This is not a UI: nothing here can be clicked or laid out. The editor's viewport draws
    /// the marks of the view the main camera renders to; a game without the editor has to draw
    /// them itself.
    /// </summary>
    public static class Overlay
    {
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
