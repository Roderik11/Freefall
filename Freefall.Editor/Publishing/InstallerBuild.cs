using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.Win32;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// The installer for a published game (BuildProfile.MakeInstaller): an Inno Setup 6 script written from the profile
    /// and compiled with its ISCC.exe into "&lt;output&gt; Installer/&lt;Product&gt; Setup.exe", next to the game's folder.
    /// It installs the game for the user who runs it, with no admin prompt, into %LOCALAPPDATA%\Programs\&lt;Product&gt;:
    /// <list type="bullet">
    /// <item>a Start menu shortcut, a desktop shortcut if ticked, and "Launch" on the last page;</item>
    /// <item>an uninstaller in Apps &amp; features, which also removes the shader cache and the logs from
    /// %LOCALAPPDATA%\&lt;Product&gt; (what a game saves there later stays).</item>
    /// </list>
    /// The packs are stored without compression: their textures are GPU-compressed and gain little, and compressing them
    /// would take minutes. Freefall-Solid's 2.9 GB game: a 2.8 GB Setup.exe in about 50 s, installed in about 40 s. The AppId comes from the profile's GUID, so a newer installer of the same profile updates the
    /// game in place.
    /// </summary>
    public static class InstallerBuild
    {
        /// <summary>Inno Setup's command-line compiler: FREEFALL_ISCC, then where its installer registered it, then the usual folders. Null if none.</summary>
        public static string? FindCompiler()
        {
            var candidates = new List<string?> { Environment.GetEnvironmentVariable("FREEFALL_ISCC") };
            foreach (var root in new[] { Registry.CurrentUser, Registry.LocalMachine })
                foreach (var key in new[] { @"Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
                                            @"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1" })
                    if (root.OpenSubKey(key)?.GetValue("InstallLocation") is string folder)
                        candidates.Add(Path.Combine(folder, "ISCC.exe"));
            candidates.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Inno Setup 6", "ISCC.exe"));
            candidates.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Inno Setup 6", "ISCC.exe"));
            candidates.Add(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Inno Setup 6", "ISCC.exe"));
            return candidates.FirstOrDefault(c => !string.IsNullOrEmpty(c) && File.Exists(c));
        }

        /// <summary>Where the installer of a game written to <paramref name="output"/> goes: a folder beside it.</summary>
        public static string FolderFor(string output) => output.TrimEnd('\\', '/') + " Installer";

        /// <summary>The setup's file name without .exe.</summary>
        public static string SetupName(string product) => $"{product} Setup";

        /// <summary>
        /// Worker: write the script for the game in <paramref name="game"/> (its exe <paramref name="exeName"/>) and compile
        /// it. Returns the setup exe. Throws with ISCC's message when it fails.
        /// </summary>
        /// <param name="setupIcon">The setup exe's icon (.ico), the game's own; null = Inno's.</param>
        public static string Build(string iscc, BuildProfile profile, string product, string game, string exeName,
                                   string scriptFolder, string? setupIcon, Action<string, float> report)
        {
            var folder = FolderFor(game);
            Directory.CreateDirectory(folder);
            foreach (var old in Directory.GetFiles(folder, SetupName(product) + "*"))
                File.Delete(old);   // a setup that spanned more .bin files last time

            long bytes = Directory.GetFiles(game, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length);
            int files = Directory.GetFiles(game, "*", SearchOption.AllDirectories).Length;
            Directory.CreateDirectory(scriptFolder);
            var script = Path.Combine(scriptFolder, profile.Name + ".iss");
            File.WriteAllText(script, Script(profile, product, game, exeName, folder, bytes, setupIcon), new UTF8Encoding(true));

            var info = new ProcessStartInfo(iscc)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            info.ArgumentList.Add(script);

            // ISCC prints a "Compressing:" line per file it stores
            int done = 0;
            var lines = new List<string>();
            void Line(string? line)
            {
                if (string.IsNullOrWhiteSpace(line)) return;
                line = line.Trim();
                lock (lines) lines.Add(line);
                if (line.StartsWith("Compressing:", StringComparison.Ordinal))
                {
                    int n = Interlocked.Increment(ref done);
                    report(Path.GetFileName(line.Substring("Compressing:".Length).Trim().Split("   ")[0]), Math.Min(0.99f, n / (float)Math.Max(1, files)));
                }
            }

            using var process = Process.Start(info) ?? throw new InvalidOperationException($"Could not start {iscc}");
            process.OutputDataReceived += (s, e) => Line(e.Data);
            process.ErrorDataReceived += (s, e) => Line(e.Data);
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();

            var setup = Path.Combine(folder, SetupName(product) + ".exe");
            if (process.ExitCode != 0 || !File.Exists(setup))
            {
                var error = lines.Where(l => l.StartsWith("Error", StringComparison.OrdinalIgnoreCase) || l.Contains("error", StringComparison.OrdinalIgnoreCase)).Take(4).ToList();
                throw new InvalidOperationException($"Building the installer failed (ISCC exited with {process.ExitCode}; script {script})" +
                    (error.Count > 0 ? ":\n" + string.Join("\n", error) : ""));
            }
            return setup;
        }

        // One Setup.exe: Inno 6.7 wrote a 2.8 GB one without disk spanning that installed fine (2026-10-22; older versions
        // stopped at about 2.1 GB). Past 4 GB, which was not tried, it spans: Setup.exe plus Setup-1.bin, …, shipped together
        const long SpanAbove = 4_000_000_000;

        /// <summary>The Inno Setup script for the game in <paramref name="game"/>.</summary>
        static string Script(BuildProfile profile, string product, string game, string exeName, string outputFolder, long bytes, string? setupIcon)
        {
            static string Q(string s) => s.Replace("\"", "\"\"");   // inside "…" in a script line
            string version = string.IsNullOrWhiteSpace(profile.Version) ? "0.1.0" : profile.Version.Trim();
            string numeric = NumericVersion(version);
            string appId = Guid.TryParse(profile.Guid, out var guid) ? guid.ToString("D").ToUpperInvariant() : Guid.NewGuid().ToString("D").ToUpperInvariant();
            string icon = Path.Combine(game, exeName);

            var s = new StringBuilder();
            s.AppendLine("; Written by the Freefall editor's Publish from the build profile " + profile.Name + "; it is replaced on every publish.");
            s.AppendLine("[Setup]");
            s.AppendLine($"AppId={{{{{appId}}}");   // "{{" escapes Inno's own braces
            s.AppendLine($"AppName={product}");
            s.AppendLine($"AppVersion={version}");
            s.AppendLine($"AppVerName={product} {version}");
            if (!string.IsNullOrWhiteSpace(profile.Company)) s.AppendLine($"AppPublisher={profile.Company.Trim()}");
            s.AppendLine($"VersionInfoVersion={numeric}");
            s.AppendLine("PrivilegesRequired=lowest");                // for the user who runs it: no admin prompt
            s.AppendLine($"DefaultDirName={{autopf}}\\{product}");    // with lowest privileges: %LOCALAPPDATA%\Programs
            s.AppendLine($"DefaultGroupName={product}");
            s.AppendLine("DisableProgramGroupPage=yes");
            s.AppendLine($"OutputDir={outputFolder}");
            s.AppendLine($"OutputBaseFilename={SetupName(product)}");
            if (File.Exists(icon)) s.AppendLine($"UninstallDisplayIcon={{app}}\\{exeName}");
            if (setupIcon != null && File.Exists(setupIcon)) s.AppendLine($"SetupIconFile={setupIcon}");
            s.AppendLine("ArchitecturesAllowed=x64compatible");
            s.AppendLine("ArchitecturesInstallIn64BitMode=x64compatible");
            s.AppendLine("MinVersion=10.0");
            s.AppendLine("WizardStyle=modern");
            s.AppendLine("Compression=lzma2/fast");
            s.AppendLine("SolidCompression=no");
            if (bytes > SpanAbove)
            {
                s.AppendLine("DiskSpanning=yes");
                s.AppendLine("DiskSliceSize=max");
            }
            s.AppendLine();
            s.AppendLine("[Languages]");
            s.AppendLine("Name: \"english\"; MessagesFile: \"compiler:Default.isl\"");
            s.AppendLine();
            s.AppendLine("[Tasks]");
            s.AppendLine("Name: \"desktopicon\"; Description: \"{cm:CreateDesktopIcon}\"; GroupDescription: \"{cm:AdditionalIcons}\"");
            s.AppendLine();
            s.AppendLine("[Files]");
            s.AppendLine($"Source: \"{Q(game)}\\*\"; Excludes: \"\\{Publisher.MarkerFile},\\{PackBuilder.DataFolderName}\\*.ffpack\"; DestDir: \"{{app}}\"; " +
                         "Flags: ignoreversion recursesubdirs createallsubdirs");
            s.AppendLine($"Source: \"{Q(game)}\\{PackBuilder.DataFolderName}\\*.ffpack\"; DestDir: \"{{app}}\\{PackBuilder.DataFolderName}\"; Flags: ignoreversion nocompression");
            s.AppendLine();
            s.AppendLine("[Icons]");
            s.AppendLine($"Name: \"{{autoprograms}}\\{Q(product)}\"; Filename: \"{{app}}\\{Q(exeName)}\"");
            s.AppendLine($"Name: \"{{autodesktop}}\\{Q(product)}\"; Filename: \"{{app}}\\{Q(exeName)}\"; Tasks: desktopicon");
            s.AppendLine();
            s.AppendLine("[Run]");
            s.AppendLine($"Filename: \"{{app}}\\{Q(exeName)}\"; Description: \"{{cm:LaunchProgram,{Q(product)}}}\"; Flags: nowait postinstall skipifsilent");
            s.AppendLine();
            s.AppendLine("[UninstallDelete]");
            s.AppendLine($"Type: filesandordirs; Name: \"{{localappdata}}\\{Q(product)}\\ShaderCache\"");
            s.AppendLine($"Type: files; Name: \"{{localappdata}}\\{Q(product)}\\player*.log\"");
            s.AppendLine($"Type: dirifempty; Name: \"{{localappdata}}\\{Q(product)}\"");
            return s.ToString();
        }

        // VersionInfoVersion takes up to four numbers: "0.3.1-beta" gives 0.3.1.0
        static string NumericVersion(string version)
        {
            var parts = version.Split('.', '-', '+', ' ').Select(p => int.TryParse(p, NumberStyles.None, CultureInfo.InvariantCulture, out int n) ? n : -1)
                               .TakeWhile(n => n >= 0).Take(4).ToList();
            while (parts.Count < 4) parts.Add(0);
            return string.Join(".", parts);
        }
    }
}
