using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Palette, generated textures (rounded panels, corner masks, scrim, logo) and immediate-mode
    /// drawing helpers for the landing page. Shapes are white so they can be tinted per draw.
    /// </summary>
    internal static class LandingArt
    {
        // ── Palette ──
        public static readonly int Background = Rgb(0x0c0e12);
        public static readonly int Card = Rgb(0x161a21);
        public static readonly int CardHot = Rgb(0x1c212a);
        public static readonly int Well = Rgb(0x10131a);
        public static readonly int Line = Rgb(0xffffff, .07f);
        public static readonly int Text = Rgb(0xece8e2);
        public static readonly int TextOnImage = Rgb(0xd7d3cc);
        public static readonly int Muted = Rgb(0x8d928c);
        public static readonly int Faint = Rgb(0x5c615c);
        public static readonly int Coral = Rgb(0xff775f);
        public static readonly int CoralHot = Rgb(0xff8f7a);
        public static readonly int OnCoral = Rgb(0x1a0d08);

        // ── Textures ──
        public const string Round = "landing_round";            // filled, radius 14
        public const string Outline = "landing_outline";        // 1px ring, radius 14
        public const string Mask = "landing_mask";              // everything outside radius 14
        public const string RoundSmall = "landing_round_sm";    // filled, radius 8
        public const string OutlineSmall = "landing_outline_sm";
        public const string Scrim = "landing_scrim";            // vertical alpha ramp
        public const string Logo = "landing_logo";

        public const int Radius = 16;       // 9-slice inset for the radius-14 set
        public const int RadiusSmall = 10;  // 9-slice inset for the radius-8 set

        // ── Fonts (rasterized at runtime, see SquidRenderer.RegisterRuntimeFont) ──
        public const string FontHeadline = "landing_headline";
        public const string FontHeading = "landing_heading";
        public const string FontTitle = "landing_title";
        public const string FontBody = "landing_body";
        public const string FontBodyMedium = "landing_body_medium";
        public const string FontSmall = "landing_small";
        public const string FontEyebrow = "landing_eyebrow";

        private static bool _ready;

        public static int Rgb(int rgb, float alpha = 1f)
        {
            return ColorInt.ARGB(alpha, ((rgb >> 16) & 0xff) / 255f, ((rgb >> 8) & 0xff) / 255f, (rgb & 0xff) / 255f);
        }

        public static void Ensure()
        {
            if (_ready || Gui.Renderer is not SquidRenderer renderer) return;
            _ready = true;

            renderer.RegisterRuntimeFont(FontHeadline, "Roboto/Roboto-Medium.ttf", 32);
            renderer.RegisterRuntimeFont(FontHeading, "Roboto/Roboto-Medium.ttf", 22);
            renderer.RegisterRuntimeFont(FontTitle, "Roboto/Roboto-Medium.ttf", 16);
            renderer.RegisterRuntimeFont(FontBody, "Roboto/Roboto-Regular.ttf", 14);
            renderer.RegisterRuntimeFont(FontBodyMedium, "Roboto/Roboto-Medium.ttf", 14);
            renderer.RegisterRuntimeFont(FontSmall, "Roboto/Roboto-Regular.ttf", 12);
            renderer.RegisterRuntimeFont(FontEyebrow, "Roboto/Roboto-Medium.ttf", 11, letterSpacing: 2);

            Insert(renderer, Round, 48, 48, (g, r) => { using var p = RoundedRect(r, 14); g.FillPath(Brushes.White, p); });
            Insert(renderer, Outline, 48, 48, (g, r) =>
            {
                r.Inflate(-.5f, -.5f);
                using var p = RoundedRect(r, 13.5f);
                using var pen = new Pen(System.Drawing.Color.White, 1f);
                g.DrawPath(pen, p);
            });
            Insert(renderer, Mask, 48, 48, (g, r) =>
            {
                using var p = RoundedRect(r, 14);
                p.AddRectangle(r);          // alternate fill: the rect minus the rounded rect
                g.FillPath(Brushes.White, p);
            });
            Insert(renderer, RoundSmall, 32, 32, (g, r) => { using var p = RoundedRect(r, 8); g.FillPath(Brushes.White, p); });
            Insert(renderer, OutlineSmall, 32, 32, (g, r) =>
            {
                r.Inflate(-.5f, -.5f);
                using var p = RoundedRect(r, 7.5f);
                using var pen = new Pen(System.Drawing.Color.White, 1f);
                g.DrawPath(pen, p);
            });
            Insert(renderer, Scrim, 4, 128, (g, r) =>
            {
                using var brush = new LinearGradientBrush(new PointF(0, 0), new PointF(0, 128),
                    System.Drawing.Color.FromArgb(0, 255, 255, 255), System.Drawing.Color.FromArgb(255, 255, 255, 255));
                g.FillRectangle(brush, r);
            });

            using (var stream = typeof(Program).Assembly.GetManifestResourceStream("Freefall.Editor.icon.ico"))
            {
                if (stream != null)
                {
                    using var icon = new Icon(stream, 48, 48);
                    using var bitmap = icon.ToBitmap();
                    renderer.InsertTexture(Logo, ProjectThumbnails.ToTexture(bitmap));
                }
            }
        }

        internal static void Insert(SquidRenderer renderer, string name, int width, int height, Action<System.Drawing.Graphics, RectangleF> paint)
        {
            using var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            using (var g = System.Drawing.Graphics.FromImage(bitmap))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.Clear(System.Drawing.Color.Transparent);
                paint(g, new RectangleF(0, 0, width, height));
            }
            renderer.InsertTexture(name, ProjectThumbnails.ToTexture(bitmap));
        }

        internal static GraphicsPath RoundedRect(RectangleF r, float radius)
        {
            float d = radius * 2;
            var path = new GraphicsPath(FillMode.Alternate);
            path.AddArc(r.X, r.Y, d, d, 180, 90);
            path.AddArc(r.Right - d, r.Y, d, d, 270, 90);
            path.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
            path.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
            path.CloseFigure();
            return path;
        }

        // ── Drawing helpers ──

        /// <summary>9-slice a generated texture over a rect. edgeScale shrinks or grows the corners (zoomed canvases).</summary>
        public static void Slice(string texture, int x, int y, int w, int h, int inset, int color, float edgeScale = 1f)
        {
            int tex = Gui.Renderer.GetTexture(texture);
            if (tex < 0 || w <= 0 || h <= 0) return;
            var size = Gui.Renderer.GetTextureSize(tex);

            int edge = Math.Min(Math.Max(1, (int)MathF.Round(inset * edgeScale)), Math.Min(w, h) / 2);
            Span<int> dx = [x, x + edge, x + w - edge], dw = [edge, w - edge * 2, edge];
            Span<int> dy = [y, y + edge, y + h - edge], dh = [edge, h - edge * 2, edge];
            Span<int> sx = [0, inset, size.x - inset], sw = [inset, size.x - inset * 2, inset];
            Span<int> sy = [0, inset, size.y - inset], sh = [inset, size.y - inset * 2, inset];

            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    if (dw[col] <= 0 || dh[row] <= 0) continue;
                    Gui.Renderer.DrawTexture(tex, dx[col], dy[row], dw[col], dh[row],
                        new Squid.Rectangle(sx[col], sy[row], sw[col], sh[row]), color);
                }
            }
        }

        /// <summary>Fill a rect with a texture, cropping it to the rect's aspect (CSS "cover").</summary>
        public static void Cover(string texture, int x, int y, int w, int h, int color = -1)
        {
            int tex = Gui.Renderer.GetTexture(texture);
            if (tex < 0 || w <= 0 || h <= 0) return;
            var size = Gui.Renderer.GetTextureSize(tex);
            if (size.x <= 0 || size.y <= 0) return;

            float target = (float)w / h;
            Squid.Rectangle source;
            if ((float)size.x / size.y > target)
            {
                int sw = (int)(size.y * target);
                source = new Squid.Rectangle((size.x - sw) / 2, 0, sw, size.y);
            }
            else
            {
                int sh = (int)(size.x / target);
                source = new Squid.Rectangle(0, (size.y - sh) / 2, size.x, sh);
            }

            Gui.Renderer.DrawTexture(tex, x, y, w, h, source, color);
        }

        public static void Stretch(string texture, int x, int y, int w, int h, int color)
        {
            int tex = Gui.Renderer.GetTexture(texture);
            if (tex < 0) return;
            Gui.Renderer.DrawTexture(tex, x, y, w, h, new Squid.Rectangle(Squid.Point.Zero, Gui.Renderer.GetTextureSize(tex)), color);
        }

        public static void Centered(string texture, int cx, int cy, int color)
        {
            int tex = Gui.Renderer.GetTexture(texture);
            if (tex < 0) return;
            var size = Gui.Renderer.GetTextureSize(tex);
            Gui.Renderer.DrawTexture(tex, cx - size.x / 2, cy - size.y / 2, size.x, size.y, new Squid.Rectangle(Squid.Point.Zero, size), color);
        }

        public static void DrawText(string font, string text, int x, int y, int color)
        {
            if (string.IsNullOrEmpty(text)) return;
            Gui.Renderer.DrawText(text, x, y, Gui.Renderer.GetFont(font), color);
        }

        public static Squid.Point Measure(string font, string text)
        {
            return string.IsNullOrEmpty(text) ? new Squid.Point() : Gui.Renderer.GetTextSize(text, Gui.Renderer.GetFont(font));
        }

        /// <summary>Shorten text to fit a width. Paths keep their tail, everything else its head.</summary>
        public static string Fit(string font, string text, int maxWidth, bool keepEnd = false)
        {
            if (string.IsNullOrEmpty(text) || Measure(font, text).x <= maxWidth) return text ?? "";

            for (int cut = 1; cut < text.Length; cut++)
            {
                var candidate = keepEnd ? "…" + text.Substring(cut) : text.Substring(0, text.Length - cut) + "…";
                if (Measure(font, candidate).x <= maxWidth) return candidate;
            }
            return "…";
        }

        public static string TimeAgo(DateTime utc)
        {
            if (utc == default) return "";
            var span = DateTime.UtcNow - utc;
            if (span.TotalMinutes < 1) return "just now";
            if (span.TotalMinutes < 60) return Plural((int)span.TotalMinutes, "minute");
            if (span.TotalHours < 24) return Plural((int)span.TotalHours, "hour");
            if (span.TotalDays < 2) return "yesterday";
            if (span.TotalDays < 14) return Plural((int)span.TotalDays, "day");
            if (span.TotalDays < 60) return Plural((int)(span.TotalDays / 7), "week");
            return utc.ToLocalTime().ToString("MMM d, yyyy");
        }

        private static string Plural(int count, string unit) => $"{count} {unit}{(count == 1 ? "" : "s")} ago";
    }
}
