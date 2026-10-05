using System;
using System.Collections.Generic;
using System.Numerics;
using Squid;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    /// <summary>
    /// Shared drawing and navigation for the zoomable node canvases (graph editor, animation editor).
    /// </summary>
    internal static class CanvasDrawing
    {
        private const int GridStep = 24;
        private const int GridMajorEvery = 5;

        /// <summary>
        /// Grid that pans and zooms with the canvas. Call from the canvas' DrawStyle, after the background.
        /// </summary>
        public static void DrawGrid(Control canvas)
        {
            if (canvas.Parent == null) return;

            float step = GridStep * canvas.UIScale;
            if (step < 6) return;

            // The canvas itself is huge; only the part inside the parent is visible
            Point origin = canvas.Location, view = canvas.Parent.Location, viewSize = canvas.Parent.Size;
            int minor = ColorInt.ARGB(.035f, 1f, 1f, 1f);
            int major = ColorInt.ARGB(.075f, 1f, 1f, 1f);

            int column = (int)MathF.Floor((view.x - origin.x) / step);
            for (float gx = origin.x + column * step; gx < view.x + viewSize.x; gx += step, column++)
                Gui.Renderer.DrawBox((int)gx, view.y, 1, viewSize.y, column % GridMajorEvery == 0 ? major : minor);

            int row = (int)MathF.Floor((view.y - origin.y) / step);
            for (float gy = origin.y + row * step; gy < view.y + viewSize.y; gy += step, row++)
                Gui.Renderer.DrawBox(view.x, (int)gy, viewSize.x, 1, row % GridMajorEvery == 0 ? major : minor);
        }

        /// <summary>
        /// Pan and zoom a canvas so the given node controls fit its parent, never zooming in past 100%.
        /// Returns false if the canvas has no size yet (try again next update).
        /// </summary>
        public static bool FrameNodes(Control canvas, Point canvasSize, IEnumerable<Control> nodes)
        {
            if (canvas.Parent == null || canvas.Parent.Size.x <= 0 || canvas.Parent.Size.y <= 0) return false;

            int minX = int.MaxValue, minY = int.MaxValue, maxX = int.MinValue, maxY = int.MinValue;
            foreach (var node in nodes)
            {
                minX = Math.Min(minX, node.Position.x);
                minY = Math.Min(minY, node.Position.y);
                maxX = Math.Max(maxX, node.Position.x + node.Size.x);
                maxY = Math.Max(maxY, node.Position.y + node.Size.y);
            }
            if (minX == int.MaxValue) return true;

            const int margin = 48;
            Point view = canvas.Parent.Size;
            float fit = Math.Min((view.x - margin * 2) / (float)Math.Max(1, maxX - minX),
                                 (view.y - margin * 2) / (float)Math.Max(1, maxY - minY));
            canvas.UIScale = MathF.Round(Math.Clamp(fit, 0.3f, 1f) / .05f) * .05f;

            // Free the canvas from its initial centring dock, then put the nodes' centre at the view's centre
            canvas.Dock = DockStyle.None;
            canvas.Size = canvasSize * canvas.UIScale;
            float centerX = (minX + maxX) / 2f, centerY = (minY + maxY) / 2f;
            canvas.Position = new Point((int)(view.x / 2f - centerX * canvas.UIScale), (int)(view.y / 2f - centerY * canvas.UIScale));
            return true;
        }

        /// <summary>
        /// Straight line with a chevron arrowhead at its end.
        /// </summary>
        public static void DrawArrow(Vector2 from, Vector2 to, Color4 color, float width, float headSize)
        {
            var batch = ((SquidRenderer)Gui.Renderer).SpriteBatch;
            var direction = to - from;
            if (direction.LengthSquared() < 1) return;
            direction = Vector2.Normalize(direction);
            var normal = new Vector2(-direction.Y, direction.X);

            float savedWidth = batch.LineWidth;
            batch.LineWidth = width;

            batch.DrawLine((int)from.X, (int)from.Y, (int)to.X, (int)to.Y, color);

            var back = to - direction * headSize;
            var left = back + normal * headSize * .6f;
            var right = back - normal * headSize * .6f;
            batch.DrawLine((int)to.X, (int)to.Y, (int)left.X, (int)left.Y, color);
            batch.DrawLine((int)to.X, (int)to.Y, (int)right.X, (int)right.Y, color);

            batch.LineWidth = savedWidth;
        }
    }
}
