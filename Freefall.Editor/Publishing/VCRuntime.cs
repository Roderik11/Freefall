using System.Diagnostics;
using System.Text;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// The Visual C++ runtime a published game carries next to its exe (app-local), so it runs on a PC that never had
    /// Visual Studio or the VC++ redistributable. What the player's native DLLs import from it (dumpbin /dependents,
    /// 2026-10-23): PhysX and PhysX.Net vcruntime140.dll, PhysX.Net msvcp140.dll, WinPixEventRuntime and assimp all three.
    /// The Universal CRT they import too (api-ms-win-crt-*) is part of Windows 10 and later.
    /// <para>App-local rather than vc_redist.x64.exe: the redistributable installs into System32 and needs an admin prompt,
    /// while the game's installer runs without one, and a game folder started without the installer needs it as well.
    /// The copies also make sure the runtime is at least as new as the toolset PhysX.Net was built with (14.51); an older
    /// runtime under newer-built code is not supported.</para>
    /// The files come from Visual Studio's redist folder (VC\Redist\MSVC\&lt;ver&gt;\x64\Microsoft.VC1xx.CRT), the newest
    /// of every installed Visual Studio, or from the folder FREEFALL_VCREDIST names.
    /// </summary>
    public static class VCRuntime
    {
        public static readonly string[] Files = { "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll" };

        /// <summary>The folder to copy <see cref="Files"/> from and their version; null when no Visual Studio has the C++ redist. Main thread is fine (vswhere: ~50 ms).</summary>
        public static (string Folder, string Version)? Find()
        {
            var candidates = new List<string>();
            if (Environment.GetEnvironmentVariable("FREEFALL_VCREDIST") is { Length: > 0 } own)
                candidates.Add(own);
            else
                foreach (var install in VisualStudios())
                {
                    var redist = Path.Combine(install, "VC", "Redist", "MSVC");
                    if (!Directory.Exists(redist)) continue;
                    foreach (var version in Directory.GetDirectories(redist))
                    {
                        var x64 = Path.Combine(version, "x64");
                        if (Directory.Exists(x64)) candidates.AddRange(Directory.GetDirectories(x64, "Microsoft.VC*.CRT"));
                    }
                }

            return candidates.Where(c => Files.All(f => File.Exists(Path.Combine(c, f))))
                             .Select(c => (Folder: c, Version: FileVersion(Path.Combine(c, Files[0]))))
                             .OrderByDescending(c => c.Version)
                             .Select(c => ((string, string)?)(c.Folder, c.Version.ToString()))
                             .FirstOrDefault();
        }

        /// <summary>Worker: <see cref="Files"/> from <paramref name="folder"/> into the game's folder, beside its exe. Returns the bytes copied.</summary>
        public static long Copy(string folder, string output)
        {
            long bytes = 0;
            foreach (var file in Files)
            {
                var target = Path.Combine(output, file);
                File.Copy(Path.Combine(folder, file), target, overwrite: true);
                bytes += new FileInfo(target).Length;
            }
            return bytes;
        }

        static Version FileVersion(string file)
        {
            var info = FileVersionInfo.GetVersionInfo(file);
            return new Version(info.FileMajorPart, info.FileMinorPart, info.FileBuildPart, info.FilePrivatePart);
        }

        // The Visual Studio installer's vswhere lists every installation, Build Tools and previews included
        static IEnumerable<string> VisualStudios()
        {
            var vswhere = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "Installer", "vswhere.exe");
            if (!File.Exists(vswhere)) return Array.Empty<string>();
            var info = new ProcessStartInfo(vswhere)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                StandardOutputEncoding = Encoding.UTF8,
            };
            foreach (var arg in new[] { "-all", "-prerelease", "-products", "*", "-property", "installationPath", "-utf8" })
                info.ArgumentList.Add(arg);
            try
            {
                using var process = Process.Start(info);
                if (process == null) return Array.Empty<string>();
                var lines = process.StandardOutput.ReadToEnd().Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                process.WaitForExit();
                return lines;
            }
            catch (System.ComponentModel.Win32Exception) { return Array.Empty<string>(); }
        }
    }
}
