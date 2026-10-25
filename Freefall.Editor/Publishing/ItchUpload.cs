using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// Pushes a published game to itch.io with butler (BuildProfile.UploadToItch): <c>butler push &lt;output&gt; &lt;target&gt;:&lt;channel&gt;</c>
    /// with the profile's Version. butler diffs the folder against the channel's last build and uploads only what changed
    /// (block by block, so contents that moved inside a pack still count as unchanged), and the itch app updates the
    /// copies installed from that channel. A restricted itch project keeps it to the people holding a download key.
    /// <list type="bullet">
    /// <item>butler: FREEFALL_BUTLER, then butler on the PATH, then the itch app's own (%APPDATA%\itch\broth\butler, the
    /// version its .chosen-version names).</item>
    /// <item>Its key: whoever publishes logs butler in once (<c>butler login</c>), which writes
    /// %USERPROFILE%\.config\itch\butler_creds; or BUTLER_API_KEY. Without either the step refuses before anything
    /// starts: butler would otherwise wait for a browser login that nobody sees.</item>
    /// <item>A .itch.toml in the folder names the exe, so the itch app launches the game and not createdump.exe.</item>
    /// </list>
    /// </summary>
    public static class ItchUpload
    {
        /// <summary>The itch app's launch manifest, written into the game's folder before the push.</summary>
        public const string ManifestFile = ".itch.toml";

        /// <summary>butler.exe, or null when there is none.</summary>
        public static string? FindButler()
        {
            var candidates = new List<string?> { Environment.GetEnvironmentVariable("FREEFALL_BUTLER") };
            foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                candidates.Add(Path.Combine(dir.Trim(), "butler.exe"));

            // The itch app keeps butler under broth\butler\versions\<version>, the one in use named by .chosen-version
            var broth = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "itch", "broth", "butler");
            var chosen = Path.Combine(broth, ".chosen-version");
            if (File.Exists(chosen))
                candidates.Add(Path.Combine(broth, "versions", File.ReadAllText(chosen).Trim(), "butler.exe"));
            if (Directory.Exists(Path.Combine(broth, "versions")))
                candidates.AddRange(Directory.GetDirectories(Path.Combine(broth, "versions"))
                    .OrderByDescending(d => Version.TryParse(Path.GetFileName(d), out var v) ? v : new Version())
                    .Select(d => Path.Combine(d, "butler.exe")));
            return candidates.FirstOrDefault(c => !string.IsNullOrEmpty(c) && File.Exists(c));
        }

        /// <summary>The key file <c>butler login</c> writes.</summary>
        public static string CredentialsFile =>
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".config", "itch", "butler_creds");

        /// <summary>Whether butler has a key: the file <c>butler login</c> writes, or BUTLER_API_KEY.</summary>
        public static bool HasCredentials =>
            File.Exists(CredentialsFile) || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("BUTLER_API_KEY"));

        /// <summary>
        /// The version a push goes out as: <paramref name="version"/> with its last number counted up ("0.1.9" → "0.1.10",
        /// "0.2.0-beta" → "0.2.1-beta"). Every push to itch.io bumps the profile's Version this way first.
        /// </summary>
        public static string NextVersion(string version)
        {
            var v = string.IsNullOrWhiteSpace(version) ? "0.1.0" : version.Trim();
            var m = System.Text.RegularExpressions.Regex.Match(v, @"^(.*?)(\d+)(\D*)$");
            return m.Success && long.TryParse(m.Groups[2].Value, NumberStyles.None, CultureInfo.InvariantCulture, out long n)
                ? m.Groups[1].Value + (n + 1).ToString(CultureInfo.InvariantCulture) + m.Groups[3].Value
                : v + ".1";
        }

        /// <summary>"user/game:channel" from the profile, or null when its target is empty or malformed.</summary>
        public static string? Target(BuildProfile profile)
        {
            var project = profile.ItchTarget?.Trim() ?? "";
            if (project.Length == 0 || project.Contains(':')) return null;
            var channel = string.IsNullOrWhiteSpace(profile.ItchChannel) ? "windows" : profile.ItchChannel.Trim();
            return $"{project}:{channel}";
        }

        /// <summary>
        /// Worker: write the launch manifest into <paramref name="game"/> and push it. Returns butler's last lines (what it
        /// uploaded, the build it made). Throws with butler's message when it fails.
        /// </summary>
        public static string Push(string butler, BuildProfile profile, string game, string exeName, Action<string, float> report)
        {
            var target = Target(profile) ?? throw new InvalidOperationException(
                $"The profile's itch target \"{profile.ItchTarget}\" is not user/game");
            File.WriteAllText(Path.Combine(game, ManifestFile),
                "# Written by the Freefall editor's Publish: what the itch app starts\n" +
                "[[actions]]\nname = \"play\"\npath = \"" + exeName.Replace("\\", "/").Replace("\"", "\\\"") + "\"\n");

            var info = new ProcessStartInfo(butler)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = true,   // closed at once: butler never waits on a question
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            foreach (var arg in new[] { "--json", "push", game, target, "--ignore", Publisher.MarkerFile })
                info.ArgumentList.Add(arg);
            if (!string.IsNullOrWhiteSpace(profile.Version))
            {
                info.ArgumentList.Add("--userversion");
                info.ArgumentList.Add(profile.Version.Trim());
            }

            // JSON lines: {"type":"log","level","message"}, {"type":"progress","progress":0..1,"bps","eta"},
            // {"type":"login","uri"} when there is no key (it would wait for a browser), {"type":"error","message"}
            var lines = new List<string>();
            string? failure = null;
            float uploaded = 0f;   // the last progress fraction, kept while log lines change the detail
            using var process = Process.Start(info) ?? throw new InvalidOperationException($"Could not start {butler}");
            process.StandardInput.Close();
            void Line(string? line)
            {
                if (string.IsNullOrWhiteSpace(line)) return;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;
                    switch (root.TryGetProperty("type", out var t) ? t.GetString() : null)
                    {
                        case "progress":
                            float f = root.TryGetProperty("progress", out var p) ? (float)p.GetDouble() : 0f;
                            string speed = root.TryGetProperty("bps", out var bps) && bps.ValueKind == JsonValueKind.Number
                                ? $", {BuildScan.Size((long)bps.GetDouble())}/s" : "";
                            uploaded = Math.Clamp(f, 0f, 0.99f);
                            report($"Uploading to {target}: {f.ToString("P0", CultureInfo.InvariantCulture)}{speed}", uploaded);
                            break;
                        case "log":
                            var message = root.TryGetProperty("message", out var m) ? m.GetString() ?? "" : "";
                            lock (lines) lines.Add(message);
                            if (message.Length > 0) report(message, uploaded);
                            break;
                        case "login":
                            failure = "butler has no itch.io key and wants a browser login: run butler login once";
                            try { process.Kill(); } catch (InvalidOperationException) { }
                            break;
                        case "error":
                            failure = root.TryGetProperty("message", out var e) ? e.GetString() : line;
                            break;
                    }
                }
                catch (JsonException)
                {
                    lock (lines) lines.Add(line.Trim());
                }
            }
            process.OutputDataReceived += (s, e) => Line(e.Data);
            process.ErrorDataReceived += (s, e) => Line(e.Data);
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();

            List<string> last;
            lock (lines) last = lines.Where(l => l.Length > 0).TakeLast(4).ToList();
            if (failure != null || process.ExitCode != 0)
                throw new InvalidOperationException($"Pushing to itch.io failed ({target}, butler exited with {process.ExitCode})" +
                    (failure != null ? ": " + failure : last.Count > 0 ? ":\n" + string.Join("\n", last) : ""));
            return $"{target}: " + string.Join(" ", last);
        }
    }
}
