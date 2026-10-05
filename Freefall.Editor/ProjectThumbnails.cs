using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Freefall.Editor.Mcp;
using Freefall.Graphics;

namespace Freefall.Editor
{
    /// <summary>
    /// Viewport snapshots shown on the landing page. One JPEG per project under
    /// %APPDATA%/Freefall/thumbnails, keyed by project path; refreshed when a scene is saved
    /// and when the editor closes, together with the entry's scene name and entity count.
    /// </summary>
    public static class ProjectThumbnails
    {
        private static readonly string ThumbnailDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Freefall", "thumbnails");

        // Project path → Squid texture name, or null when the project has no snapshot.
        private static readonly Dictionary<string, string> _loaded = new(StringComparer.OrdinalIgnoreCase);

        private static string FileFor(string projectPath)
        {
            var hash = SHA1.HashData(Encoding.UTF8.GetBytes(projectPath.ToLowerInvariant()));
            return Path.Combine(ThumbnailDir, Convert.ToHexString(hash, 0, 8) + ".jpg");
        }

        /// <summary>
        /// Snapshot the scene viewport of the open project. Main thread only; does nothing on the landing page.
        /// </summary>
        public static void Capture()
        {
            var entry = RecentProjects.Current;
            var editor = Program.EditorUI;
            if (entry == null || editor == null) return;

            // No scene open → the viewport is empty; keep whatever the last real session left behind.
            if (string.IsNullOrEmpty(editor.CurrentScenePath)) return;

            try
            {
                entry.EntityCount = Freefall.Base.EntityManager.Entities.Count;
                entry.Scene = Path.GetFileNameWithoutExtension(editor.CurrentScenePath);
                RecentProjects.Save();

                var shot = ScreenCapture.Capture(CaptureTarget.Viewport, null, 1280, png: false, jpegQuality: 88);
                Directory.CreateDirectory(ThumbnailDir);
                File.WriteAllBytes(FileFor(entry.Path), shot.Data);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("ProjectThumbnails", $"Capture failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Squid texture name for a project's snapshot, or null if it has none. Loads on first request.
        /// </summary>
        public static string GetTexture(string projectPath)
        {
            if (_loaded.TryGetValue(projectPath, out var name))
                return name;

            name = null;
            var file = FileFor(projectPath);
            if (File.Exists(file) && Squid.Gui.Renderer is SquidRenderer renderer)
            {
                try
                {
                    using var bitmap = new Bitmap(file);
                    name = "project_thumb:" + Path.GetFileNameWithoutExtension(file);
                    renderer.InsertTexture(name, ToTexture(bitmap));
                }
                catch (Exception ex)
                {
                    name = null;
                    Debug.LogWarning("ProjectThumbnails", $"Failed to load {file}: {ex.Message}");
                }
            }

            _loaded[projectPath] = name;
            return name;
        }

        /// <summary>Upload a GDI+ bitmap as an RGBA texture.</summary>
        internal static Texture ToTexture(Bitmap bitmap)
        {
            int width = bitmap.Width, height = bitmap.Height;
            var data = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var pixels = new byte[width * height * 4];
            for (int row = 0; row < height; row++)
                Marshal.Copy(data.Scan0 + row * data.Stride, pixels, row * width * 4, width * 4);
            bitmap.UnlockBits(data);

            // GDI+ is BGRA in memory
            for (int i = 0; i < pixels.Length; i += 4)
                (pixels[i], pixels[i + 2]) = (pixels[i + 2], pixels[i]);

            return Texture.CreateFromData(Engine.Device, width, height, pixels);
        }
    }
}
