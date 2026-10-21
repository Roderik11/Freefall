using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using Freefall.Assets.Packs;

namespace Freefall.Assets
{
    /// <summary>
    /// The engine's own files, read by their path under &lt;exe&gt;/Resources: the shaders with their includes, the
    /// default textures. A published game has no Resources folder: they come from Data/engine.ffpack, which the player
    /// mounts before the engine starts (<see cref="Mount"/>; the first shaders compile in Engine.Initialize). Without a
    /// pack, and for any path the pack does not hold (a project's own .fx in Assets, the editor's source checkout),
    /// they come from the disk. A pack entry's key is a hash of the file's path relative to Resources (<see cref="KeyOf"/>),
    /// so code keeps its paths. The shader code (Effect, ComputeShader, Shader, ShaderHotReload, SpriteBatch) and
    /// Texture's DDS reader call Exists/ReadAllText/ReadAllBytes instead of File's.
    /// </summary>
    public static class EngineFiles
    {
        /// <summary>The pack's file name in Data/.</summary>
        public const string PackFile = "engine.ffpack";
        public const string DataType = "EngineFile";

        private static PackSet _pack;
        private static long _fromPack;

        /// <summary>Where the engine's files are read from on disk, and what their pack keys are relative to.</summary>
        public static string Root => Path.Combine(AppContext.BaseDirectory, "Resources");

        /// <summary>The mounted pack, or null.</summary>
        public static PackSet Pack => _pack;

        /// <summary>Files read from the pack since it was mounted.</summary>
        public static long ReadFromPack => Interlocked.Read(ref _fromPack);

        /// <summary>Read the engine's files from <paramref name="pack"/> from now on, before the disk.</summary>
        public static void Mount(PackSet pack)
        {
            Interlocked.Exchange(ref _pack, pack)?.Dispose();
            Interlocked.Exchange(ref _fromPack, 0);
        }

        /// <summary>The pack key of a file at this path relative to Resources (case and slashes do not matter): its MD5 as a GUID.</summary>
        public static string KeyOf(string relativePath)
        {
            var normal = relativePath.Replace('\\', '/').TrimStart('/').ToLowerInvariant();
            return new Guid(MD5.HashData(Encoding.UTF8.GetBytes(normal))).ToString("N");
        }

        public static bool Exists(string path) => PackedKey(path) != null || File.Exists(path);

        public static string ReadAllText(string path)
        {
            var key = PackedKey(path);
            if (key == null) return File.ReadAllText(path);
            using var reader = new StreamReader(OpenPacked(key), Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            return reader.ReadToEnd();
        }

        public static byte[] ReadAllBytes(string path)
        {
            var key = PackedKey(path);
            if (key == null) return File.ReadAllBytes(path);
            using var stream = OpenPacked(key);
            var bytes = new byte[stream.Length];
            stream.ReadExactly(bytes);
            return bytes;
        }

        /// <summary>The key of a path under <see cref="Root"/> that the mounted pack holds, else null (the disk then).</summary>
        private static string PackedKey(string path)
        {
            var pack = _pack;
            if (pack == null || string.IsNullOrEmpty(path)) return null;

            var full = Path.GetFullPath(path);
            var root = Path.GetFullPath(Root).TrimEnd('\\', '/') + Path.DirectorySeparatorChar;
            if (!full.StartsWith(root, StringComparison.OrdinalIgnoreCase)) return null;

            var key = KeyOf(full.Substring(root.Length));
            return pack.Contains(key) ? key : null;
        }

        private static Stream OpenPacked(string key)
        {
            Interlocked.Increment(ref _fromPack);
            return _pack.Open(key);
        }
    }
}
