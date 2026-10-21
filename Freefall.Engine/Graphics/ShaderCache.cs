using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Vortice.Dxc;

namespace Freefall.Graphics
{
    /// <summary>
    /// DXC bytecode on disk, one file per compiled shader: %LOCALAPPDATA%\Freefall\ShaderCache\&lt;key&gt;.cso. The key
    /// hashes DXC's version, the arguments and the source with its includes inlined, so a file is only ever reused
    /// for exactly what produced it; nothing needs invalidating. Shared by every editor and game build of this user
    /// (the key holds no paths). Files unused for 30 days are deleted. FREEFALL_SHADER_CACHE=0 turns it off.
    /// Any thread: reads and writes are independent files, a write is a rename of a finished temp file.
    /// </summary>
    public static class ShaderCache
    {
        private const int PruneAfterDays = 30;

        private static readonly bool Enabled = Environment.GetEnvironmentVariable("FREEFALL_SHADER_CACHE") != "0";

        private static string Root = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Freefall", "ShaderCache");

        /// <summary>
        /// Keep the cache in another folder (a published game: %LOCALAPPDATA%\&lt;product&gt;\ShaderCache, not the editor's).
        /// Before the first shader compiles; later calls are ignored with a warning.
        /// </summary>
        public static void SetRoot(string folder)
        {
            if (Volatile.Read(ref _directoryReady) != 0)
            {
                Debug.LogWarning("ShaderCache", $"Shaders were cached in {Root} already; {folder} is not used");
                return;
            }
            Root = folder;
        }

        private static readonly Lazy<string> CompilerIdentity = new(FindCompilerIdentity);
        private static int _directoryReady;   // 0 = not yet, 1 = created and pruning started

        /// <summary>Cache key for a compile: hex SHA-256 of DXC's identity, the arguments and the full source.</summary>
        public static string Key(string[] args, string source)
        {
            var text = CompilerIdentity.Value + "\n" + string.Join(" ", args) + "\n" + source;
            return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
        }

        /// <summary>The bytecode cached for this key, if a complete DXIL container is on disk.</summary>
        public static bool TryRead(string key, out byte[] bytecode)
        {
            bytecode = null;
            if (!Enabled) return false;

            var path = PathOf(key);
            try
            {
                EnsureDirectory();
                if (!File.Exists(path)) return false;
                var bytes = File.ReadAllBytes(path);
                if (!IsCompleteContainer(bytes)) return false;

                // Last write = last use, for pruning
                File.SetLastWriteTimeUtc(path, DateTime.UtcNow);
                bytecode = bytes;
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }

        /// <summary>Store freshly compiled bytecode (best effort: a failure only means compiling again next time).</summary>
        public static void Write(string key, byte[] bytecode)
        {
            if (!Enabled || !IsCompleteContainer(bytecode)) return;

            string temp = null;
            try
            {
                EnsureDirectory();
                var path = PathOf(key);
                temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
                File.WriteAllBytes(temp, bytecode);
                File.Move(temp, path, overwrite: true);
                temp = null;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Another process wrote the same key at the same moment, or the disk refused: no harm
            }
            finally
            {
                if (temp != null)
                    try { File.Delete(temp); } catch { /* best effort */ }
            }
        }

        private static string PathOf(string key) => Path.Combine(Root, key + ".cso");

        /// <summary>A DXIL/DXBC container: "DXBC" and its own size field (offset 24) equal to the length.</summary>
        private static bool IsCompleteContainer(byte[] bytes)
        {
            return bytes.Length >= 32
                && bytes[0] == 'D' && bytes[1] == 'X' && bytes[2] == 'B' && bytes[3] == 'C'
                && BitConverter.ToUInt32(bytes, 24) == (uint)bytes.Length;
        }

        private static void EnsureDirectory()
        {
            if (Interlocked.CompareExchange(ref _directoryReady, 1, 0) != 0) return;

            Directory.CreateDirectory(Root);
            Task.Run(Prune);
        }

        /// <summary>Delete entries unused for PruneAfterDays (shaders edited since) and temp files left by a crash.</summary>
        private static void Prune()
        {
            try
            {
                var now = DateTime.UtcNow;
                foreach (var file in new DirectoryInfo(Root).EnumerateFiles())
                {
                    bool stale = file.Extension.Equals(".tmp", StringComparison.OrdinalIgnoreCase)
                        ? now - file.LastWriteTimeUtc > TimeSpan.FromDays(1)
                        : now - file.LastWriteTimeUtc > TimeSpan.FromDays(PruneAfterDays);
                    if (stale)
                        try { file.Delete(); } catch { /* in use or gone: next time */ }
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
            }
        }

        /// <summary>Which DXC compiles: the Vortice.Dxc version and the native dxcompiler.dll's version and size.</summary>
        private static string FindCompilerIdentity()
        {
            var identity = "vortice " + typeof(DxcCompiler).Assembly.GetName().Version;
            var rid = RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "win-arm64" : "win-x64";
            foreach (var path in new[]
            {
                Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", "dxcompiler.dll"),
                Path.Combine(AppContext.BaseDirectory, "dxcompiler.dll"),
            })
            {
                if (!File.Exists(path)) continue;
                var info = System.Diagnostics.FileVersionInfo.GetVersionInfo(path);
                return $"{identity}; dxcompiler {info.FileVersion} {new FileInfo(path).Length}";
            }
            return identity;
        }
    }
}
