using System;
using Freefall.Editor.Mcp;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Heads-up drawn while ScreenshotScheduler is about to capture: a pulsing border around the capture
    /// target plus a "hands off the mouse" pill with a draining countdown bar. Lives in the desktop's
    /// Elements (drawn over everything, no events); draws nothing outside the cue phase, and the
    /// scheduler hides it for a few frames before capturing, so it never shows up in the image.
    /// </summary>
    public class ScreenshotCueOverlay : Control
    {
        private const int Border = 4;
        private const string CueText = "Capturing screenshot - hands off the mouse";

        private static readonly (float r, float g, float b) Accent = (1f, 0.55f, 0.1f);

        public ScreenshotCueOverlay()
        {
            NoEvents = true;
        }

        protected override void OnUpdate()
        {
            // Cover the whole desktop; the cue itself picks its rectangle in DrawCustom.
            Position = Point.Zero;
            if (Desktop != null) Size = Desktop.Size;
        }

        protected override void DrawCustom()
        {
            if (ScreenshotScheduler.CueTarget is not { } target) return;

            var area = new Rectangle(Location, Size);
            if (target == CaptureTarget.Viewport && Program.EditorUI?.SceneViewport is { } vp)
                area = new Rectangle(vp.Location, vp.Size);

            float t = ScreenshotScheduler.CueProgress;
            float pulse = 0.65f + 0.35f * MathF.Cos(t * MathF.PI * 6f);
            int accent = ColorInt.ARGB(pulse, Accent.r, Accent.g, Accent.b);

            int x = area.Left, y = area.Top, w = area.Width, h = area.Height;
            var r = Gui.Renderer;
            r.DrawBox(x, y, w, Border, accent);
            r.DrawBox(x, y + h - Border, w, Border, accent);
            r.DrawBox(x, y + Border, Border, h - 2 * Border, accent);
            r.DrawBox(x + w - Border, y + Border, Border, h - 2 * Border, accent);

            // Pill at the top centre with a countdown bar underneath the text.
            int font = r.GetFont("roboto_medium_10");
            var ts = r.GetTextSize(CueText, font);
            int pw = ts.x + 32, ph = ts.y + 18;
            int px = x + (w - pw) / 2, py = y + Border + 12;

            r.DrawBox(px, py, pw, ph, ColorInt.ARGB(0.85f, 0.08f, 0.08f, 0.08f));
            r.DrawBox(px + 10, py + (ph - 4 - 6) / 2, 6, 6, ColorInt.ARGB(1f, Accent.r, Accent.g, Accent.b)); // "rec" dot
            r.DrawText(CueText, px + 22, py + (ph - 4 - ts.y) / 2, font, ColorInt.ARGB(1f, 1f, 1f, 1f));

            int barW = (int)((pw - 4) * (1f - t));
            r.DrawBox(px + 2, py + ph - 4, barW, 2, accent);
        }
    }
}
