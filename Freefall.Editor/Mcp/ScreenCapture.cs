using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Freefall.Editor.Mcp
{
    public enum CaptureTarget
    {
        /// <summary>The whole editor window, UI included.</summary>
        Editor,
        /// <summary>Just the scene viewport panel.</summary>
        Viewport,
    }

    /// <param name="TargetWidth">Size of the capture target (window or viewport) before crop/scale — crop coordinates live in this space.</param>
    /// <param name="Region">Captured rectangle in window pixels.</param>
    public record CaptureResult(byte[] Data, string MimeType, int TargetWidth, int TargetHeight,
        int Width, int Height, Rectangle Region);

    /// <summary>
    /// Grabs the editor window via PrintWindow, optionally crops to the scene viewport or a
    /// sub-rectangle, downscales to fit a max edge and encodes. Must run on the main thread.
    /// </summary>
    public static class ScreenCapture
    {
        [DllImport("user32.dll")]
        private static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint nFlags);

        // CLIENTONLY: without it the bitmap starts at the title bar and every region is shifted by the frame.
        private const uint PW_CLIENTONLY = 0x00000001;
        private const uint PW_RENDERFULLCONTENT = 0x00000002;

        /// <param name="crop">Optional region in the target's own pixel space (before scaling).</param>
        /// <param name="maxSize">Longest edge of the output; 0 = no downscale.</param>
        public static CaptureResult Capture(CaptureTarget target, Rectangle? crop, int maxSize, bool png, int jpegQuality = 85)
        {
            var form = Program.Form;
            var client = form.ClientRectangle;

            using var window = new Bitmap(client.Width, client.Height, PixelFormat.Format32bppArgb);
            using (var g = System.Drawing.Graphics.FromImage(window))
            {
                IntPtr hdc = g.GetHdc();
                PrintWindow(form.Handle, hdc, PW_CLIENTONLY | PW_RENDERFULLCONTENT);
                g.ReleaseHdc(hdc);
            }

            var bounds = new Rectangle(0, 0, window.Width, window.Height);
            var region = bounds;

            if (target == CaptureTarget.Viewport)
            {
                var vp = Program.EditorUI?.SceneViewport
                    ?? throw new InvalidOperationException("No scene viewport — is a project open?");
                region = Rectangle.Intersect(bounds,
                    new Rectangle(vp.Location.x, vp.Location.y, vp.Size.x, vp.Size.y));
                if (region.Width <= 0 || region.Height <= 0)
                    throw new InvalidOperationException("Scene viewport is not visible (hidden or docked behind another tab).");
            }

            var targetArea = region;

            if (crop is { } c)
            {
                var sub = new Rectangle(region.X + c.X, region.Y + c.Y, c.Width, c.Height);
                region = Rectangle.Intersect(region, sub);
                if (region.Width <= 0 || region.Height <= 0)
                    throw new ArgumentException("Crop rectangle lies outside the captured area.");
            }

            float scale = 1f;
            if (maxSize > 0)
                scale = Math.Min(1f, maxSize / (float)Math.Max(region.Width, region.Height));

            int outW = Math.Max(1, (int)Math.Round(region.Width * scale));
            int outH = Math.Max(1, (int)Math.Round(region.Height * scale));

            using var output = new Bitmap(outW, outH, PixelFormat.Format24bppRgb);
            using (var g = System.Drawing.Graphics.FromImage(output))
            {
                g.InterpolationMode = scale < 1f ? InterpolationMode.HighQualityBicubic : InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawImage(window, new Rectangle(0, 0, outW, outH), region, GraphicsUnit.Pixel);
            }

            using var ms = new MemoryStream();
            if (png)
            {
                output.Save(ms, ImageFormat.Png);
            }
            else
            {
                var codec = ImageCodecInfo.GetImageEncoders().First(e => e.FormatID == ImageFormat.Jpeg.Guid);
                using var ep = new EncoderParameters(1);
                ep.Param[0] = new EncoderParameter(Encoder.Quality, (long)Math.Clamp(jpegQuality, 1, 100));
                output.Save(ms, codec, ep);
            }

            return new CaptureResult(ms.ToArray(), png ? "image/png" : "image/jpeg",
                targetArea.Width, targetArea.Height, outW, outH, region);
        }
    }
}
