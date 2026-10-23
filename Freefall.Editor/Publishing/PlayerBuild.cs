using System.Diagnostics;
using System.Text;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// The player's binaries for a publish: <c>dotnet publish</c> of Freefall.Player (Release, win-x64, self-contained:
    /// the .NET runtime ships with the game) from the engine's source checkout, into a staging folder in the project's
    /// Library. The build gets its own artifacts path there, so it never touches the editor's bin and obj folders, and
    /// the next publish rebuilds only what changed. It builds the engine from its sources as they are now, which is
    /// the editor's engine as long as the editor was built from them. Needs the .NET SDK and the checkout (a shipped
    /// editor would carry a prebuilt player instead).
    /// </summary>
    public static class PlayerBuild
    {
        public const string Runtime = "win-x64";
        /// <summary>The player's exe as the build writes it; a publish renames it after the product.</summary>
        public const string AppHost = "Freefall.Player.exe";
        const string ProjectFile = "Freefall.Player.csproj";

        /// <summary>The player project, found by walking up from the editor's folder; null outside a source checkout.</summary>
        public static string? FindProject()
        {
            for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, "Freefall.Player", ProjectFile);
                if (File.Exists(candidate)) return candidate;
            }
            return null;
        }

        /// <summary>The engine's Resources in the checkout the player is built from: its shaders match that engine.</summary>
        public static string EngineResources(string project) =>
            Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(project)!)!, "Freefall.Engine", "Resources");

        /// <summary>The icon the player's exe is built with (its ApplicationIcon, the editor's icon.ico), for the installer too.</summary>
        public static string AppIcon(string project) =>
            Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(project)!)!, "Freefall.Editor", "icon.ico");

        /// <summary>Resources folders nothing reads at run time: not packed (Terrain: old splat maps).</summary>
        public static readonly string[] UnusedResources = { "Terrain" };

        /// <summary>Where a project's player is built: Library/Player/win-x64 (build/ for obj and bin, publish/ for the result).</summary>
        public static string StagingFolder(string libraryDirectory) => Path.Combine(libraryDirectory, "Player", Runtime);

        /// <summary>
        /// Worker: build and publish the player into <paramref name="staging"/>/publish and return that folder.
        /// <paramref name="report"/> gets the build's lines and a rough share done (each project built counts).
        /// Throws with the build's errors.
        /// </summary>
        public static string Build(string project, string staging, Action<string, float> report)
        {
            var publish = Path.Combine(staging, "publish");
            // dotnet publish -o adds to the folder and never takes anything out, so a file an earlier build had would ship
            // on (the debug PhysX.Net package's PhysXGpu_64.dll, 29 MB). The build in build/ stays incremental.
            if (Directory.Exists(publish)) Directory.Delete(publish, recursive: true);
            var info = new ProcessStartInfo("dotnet")
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                WorkingDirectory = Path.GetDirectoryName(project)!,
            };
            foreach (var arg in new[]
            {
                "publish", project, "-c", "Release", "-r", Runtime, "--self-contained",
                "--artifacts-path", Path.Combine(staging, "build"), "-o", publish,
                "-nologo", "-tl:off", "-v:minimal", "-clp:NoSummary",
            })
                info.ArgumentList.Add(arg);
            info.Environment["DOTNET_CLI_UI_LANGUAGE"] = "en";

            // Squid, the engine, the player, then the publish itself: one " -> " line each
            const int Outputs = 4;
            int outputs = 0;
            var errors = new List<string>();
            void Line(string? line)
            {
                if (string.IsNullOrWhiteSpace(line) || line.Contains(": warning ")) return;
                line = line.Trim();
                if (line.Contains(": error ")) lock (errors) errors.Add(line);
                if (line.Contains(" -> ")) Interlocked.Increment(ref outputs);
                report(line, Math.Min(1f, outputs / (float)Outputs));
            }

            using var process = Process.Start(info) ?? throw new InvalidOperationException("Could not start dotnet: is the .NET SDK installed?");
            process.OutputDataReceived += (s, e) => Line(e.Data);
            process.ErrorDataReceived += (s, e) => Line(e.Data);
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                var first = errors.Distinct().Take(5).ToList();
                throw new InvalidOperationException($"Building the player failed (dotnet publish exited with {process.ExitCode})" +
                    (first.Count > 0 ? ":\n" + string.Join("\n", first) : ""));
            }
            if (!File.Exists(Path.Combine(publish, AppHost)))
                throw new InvalidOperationException($"The player build wrote no {AppHost} into {publish}");
            return publish;
        }

        /// <summary>
        /// Worker: the published player into the game's folder, its exe renamed to <paramref name="exeName"/> (the apphost
        /// finds Freefall.Player.dll by the name built into it, so only the exe changes). Returns the files and bytes copied.
        /// </summary>
        public static (int Files, long Bytes) Copy(string publish, string output, string exeName, Action<string, float> report)
        {
            var files = Directory.GetFiles(publish, "*", SearchOption.AllDirectories);
            long bytes = 0;
            for (int i = 0; i < files.Length; i++)
            {
                var relative = Path.GetRelativePath(publish, files[i]);
                var target = Path.Combine(output, relative.Equals(AppHost, StringComparison.OrdinalIgnoreCase) ? exeName : relative);
                report(relative, i / (float)files.Length);
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                File.Copy(files[i], target, overwrite: true);
                bytes += new FileInfo(target).Length;
            }
            return (files.Length, bytes);
        }
    }
}
