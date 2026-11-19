using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// ThirdPartyNotices.txt beside a published game's exe: for each third-party component the game folder holds, its
    /// files with their versions and the notice its licence asks a binary redistribution to carry. Written after the
    /// player is copied. The licence texts:
    /// <list type="bullet">
    /// <item>ThirdParty/Licenses/*.txt, committed copies (a <c>!</c> line in .gitignore, which ignores *.txt):
    /// PhysX (BSD-3, from the release build's PhysX-4.1/LICENSE.md), PhysX.Net (MIT, that build's PhysX.Net/LICENSE),
    /// assimp and Assimp.Net (BSD-3 and MIT, StirlingLabs.Assimp.Net 23.4.1's License.txt), DotRecast (zlib, its
    /// repository's LICENSE.txt), Vortice (MIT, Vortice.Win32 2.3.0's LICENSE, the same as Vortice.Windows'),
    /// NVorbis (MIT, its package's LICENSE), DXC (Vortice.Dxc.Native 1.0.4's three licences), PIX (Vortice.Pix.Native
    /// 1.0.1's licence and notices), and from the repositories, since their packages only name a licence: SharpGen
    /// (SharpGenTools LICENSE.txt), LiteYaml (License.md), RawInput.Sharp (zlib, LICENSE.txt).</item>
    /// <item>Beside the sources: meshoptimizer's LICENSE; AMD's license.md, fetched with the FidelityFX DLLs.</item>
    /// <item>The .NET runtime's, read at each publish from the runtime packs the player was built with (their versions
    /// from the player's deps.json, the files from the NuGet packages folder), since they change with the SDK.</item>
    /// <item>Written here: NVIDIA DLSS's attribution, the VC++ runtime's and StbTrueTypeSharp's credit (its repository
    /// has no licence file; its README says Public Domain).</item>
    /// </list>
    /// A component counts as shipped when any of its file patterns matches in the folder; a licence text that cannot be
    /// found is a warning and the file lacks it. A new third-party DLL in the player gets a row in <see cref="Components"/>.
    /// Squid is not here: it is the engine author's own library.
    /// </summary>
    public static class ThirdPartyNotices
    {
        public const string FileName = "ThirdPartyNotices.txt";

        private sealed record Context(string Checkout, string Output, string Product, List<string> Warnings);

        // A component: its title, the file patterns that ship it, and the text under its file list
        private sealed record Component(string Title, string[] Files, Func<Context, string> Text);

        private static readonly Component[] Components =
        {
            new(".NET runtime (Microsoft .NET and the Windows Desktop runtime)", new[] { "coreclr.dll", "PresentationCore.dll", "System.Windows.Forms.dll" },
                c => "The game is self-contained: the .NET runtime's own files lie beside the exe with these.\n\n" +
                     RuntimePack(c, "Microsoft.NETCore.App.Runtime.win-x64", "LICENSE.TXT", "THIRD-PARTY-NOTICES.TXT") + "\n\n" +
                     RuntimePack(c, "Microsoft.WindowsDesktop.App.Runtime.win-x64", "LICENSE")),
            new("Microsoft Visual C++ runtime", VCRuntime.Files, VisualCpp),
            new("NVIDIA PhysX SDK 4.1", new[] { "PhysX*_64.dll" }, Licence("ThirdParty/Licenses/PhysX.txt")),
            new("PhysX.Net", new[] { "PhysX.Net.dll" }, Licence("ThirdParty/Licenses/PhysX.Net.txt")),
            new("Vortice.Windows, Vortice.Mathematics and Vortice.Win32", new[] { "Vortice.*.dll" }, Licence("ThirdParty/Licenses/Vortice.txt")),
            new("SharpGen.Runtime", new[] { "SharpGen.Runtime*.dll" }, Licence("ThirdParty/Licenses/SharpGen.txt")),
            new("DirectX Shader Compiler", new[] { "dxcompiler.dll", "dxil.dll" }, Licence("ThirdParty/Licenses/DXC.txt")),
            new("WinPixEventRuntime", new[] { "WinPixEventRuntime.dll" }, Licence("ThirdParty/Licenses/PIX.txt")),
            new("Open Asset Import Library (assimp) and Assimp.Net", new[] { "assimp.dll", "StirlingLabs.Assimp.Net.dll" },
                Licence("ThirdParty/Licenses/assimp.txt")),
            new("DotRecast", new[] { "DotRecast.*.dll" }, Licence("ThirdParty/Licenses/DotRecast.txt")),
            new("meshoptimizer", new[] { "meshoptimizer.dll" }, Licence("ThirdParty/meshoptimizer/LICENSE")),
            new("NVorbis", new[] { "NVorbis.dll" }, Licence("ThirdParty/Licenses/NVorbis.txt")),
            new("LiteYaml", new[] { "LiteYaml.dll" }, Licence("ThirdParty/Licenses/LiteYaml.txt")),
            new("RawInput.Sharp", new[] { "RawInput.Sharp.dll" }, Licence("ThirdParty/Licenses/RawInput.Sharp.txt")),
            new("StbTrueTypeSharp", new[] { "StbTrueTypeSharp.dll" },
                c => "A C# port of stb_truetype by Sean Barrett (https://github.com/nothings/stb). Its repository " +
                     "(https://github.com/StbSharp/StbTrueTypeSharp) gives its licence as Public Domain."),
            new("AMD FidelityFX SDK: FSR upscaling", UpscalerFiles.Fsr, Licence("ThirdParty/FidelityFX/license.md")),
            new("NVIDIA DLSS", UpscalerFiles.Dlss, DlssAttribution),
        };

        /// <summary>
        /// Worker, after the player is copied into <paramref name="output"/>: writes the file for the components there and
        /// returns their titles. <paramref name="checkout"/> is the engine's source checkout the player was built from.
        /// </summary>
        public static List<string> Write(string checkout, string output, string product, List<string> warnings)
        {
            var context = new Context(checkout, output, product, warnings);
            var shipped = new List<string>();
            var text = new StringBuilder();
            text.AppendLine($"Third-party software in {product}");
            text.AppendLine();
            text.AppendLine("These components ship beside the game's exe under their makers' terms.");
            text.AppendLine();

            foreach (var c in Components)
            {
                var files = c.Files.SelectMany(pattern => Directory.GetFiles(output, pattern))
                                   .Select(Path.GetFileName).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToList();
                if (files.Count == 0) continue;
                shipped.Add(c.Title);

                text.AppendLine(new string('-', 78));
                text.AppendLine(c.Title);
                text.AppendLine(string.Join(", ", files.Select(f => Version(output, f!) is { Length: > 0 } v ? $"{f} {v}" : f)));
                text.AppendLine(new string('-', 78));
                text.AppendLine();
                text.AppendLine(c.Text(context).Trim());
                text.AppendLine();
                text.AppendLine();
            }

            // With a byte order mark: the licences' curly quotes and narrow spaces read right in any editor
            if (shipped.Count > 0)
                File.WriteAllText(Path.Combine(output, FileName), text.ToString(), new UTF8Encoding(true));
            return shipped;
        }

        // "10,0,1026,32716 @Commit: …" → "10.0.1026.32716"
        private static string Version(string output, string file)
        {
            var v = FileVersionInfo.GetVersionInfo(Path.Combine(output, file)).FileVersion ?? "";
            int space = v.IndexOf(' ');
            return (space > 0 ? v[..space] : v).Replace(',', '.');
        }

        // Licence texts from the checkout, one after the other
        private static Func<Context, string> Licence(params string[] paths) => c =>
        {
            var parts = new List<string>();
            foreach (var relative in paths)
            {
                var path = Path.Combine(c.Checkout, relative);
                if (File.Exists(path)) parts.Add(File.ReadAllText(path).Trim());
                else c.Warnings.Add($"The licence text {relative} is not in the checkout, so {FileName} lacks it");
            }
            return string.Join("\n\n", parts);
        };

        // A runtime pack's own licence files, from the version the player was built with (its deps.json names it) in the
        // NuGet packages folder the build restored it into
        private static string RuntimePack(Context c, string id, params string[] files)
        {
            string? version = null;
            try
            {
                using var deps = JsonDocument.Parse(File.ReadAllText(Path.Combine(c.Output, Path.ChangeExtension(PlayerBuild.AppHost, ".deps.json"))));
                foreach (var library in deps.RootElement.GetProperty("libraries").EnumerateObject())
                    if (library.Name.StartsWith($"runtimepack.{id}/", StringComparison.OrdinalIgnoreCase))
                        version = library.Name[(library.Name.IndexOf('/') + 1)..];
            }
            catch (Exception ex) when (ex is IOException or JsonException or KeyNotFoundException) { }
            if (version == null)
            {
                c.Warnings.Add($"The player's deps.json names no {id}, so {FileName} lacks its licence");
                return "";
            }

            var packages = Environment.GetEnvironmentVariable("NUGET_PACKAGES") is { Length: > 0 } own
                ? own : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
            var folder = Path.Combine(packages, id.ToLowerInvariant(), version);
            var text = new StringBuilder($"{id} {version}\n\n");
            foreach (var file in files)
            {
                var path = Path.Combine(folder, file);
                if (File.Exists(path)) text.Append(File.ReadAllText(path).Trim()).Append("\n\n");
                else c.Warnings.Add($"{path} is not there, so {FileName} lacks it");
            }
            return text.ToString();
        }

        private static string VisualCpp(Context c)
        {
            var copyright = FileVersionInfo.GetVersionInfo(Path.Combine(c.Output, VCRuntime.Files[0])).LegalCopyright?.Trim();
            return "The Microsoft Visual C++ runtime, copied from Visual Studio's redistributable files." +
                   (copyright is { Length: > 0 } ? $" {copyright}" : "");
        }

        // The DLSS SDK's supplement asks for the SDK's use to be attributed (7.1(b)); the splash screen and credits it
        // also asks for are the game's own
        private static string DlssAttribution(Context c)
        {
            var model = FileVersionInfo.GetVersionInfo(Path.Combine(c.Output, UpscalerFiles.Dlss[1]));
            string copyright = model.LegalCopyright is { Length: > 0 } s ? s.Trim() : "Copyright NVIDIA Corporation";
            return $"{c.Product} uses NVIDIA DLSS from the NVIDIA DLSS SDK ({copyright}). {UpscalerFiles.Dlss[0]} holds NVIDIA's NGX library.\n" +
                   "NVIDIA, GeForce RTX and DLSS are trademarks and/or registered trademarks of NVIDIA Corporation in the U.S. and other countries.";
        }
    }
}
