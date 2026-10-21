using System.ComponentModel;
using Freefall.Assets;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// What a published game holds: its scenes, in order (the game starts in the first), and where it is written.
    /// Scan walks the references from the scenes (DependencyScanner) and writes the build report; publishing itself
    /// comes later (PUBLISH_PLAN.md). Nothing references a profile, so it is never packed.
    /// </summary>
    [CreateAsset("Build Profile")]
    public class BuildProfile : Asset
    {
        [Description("The game's scenes, in order. The game starts in the first; a scene one of them references is packed too")]
        public List<SceneAsset> Scenes = new();

        [Description("The game's name, for the build's files. Empty = the project's name")]
        public string ProductName = "";

        [Description("Folder the game is written to, absolute or relative to the project folder. Empty = Builds/<profile name>. The build report goes next to it")]
        public string OutputFolder = "";

        /// <summary>The product name, falling back to the project's.</summary>
        public string ResolvedProductName =>
            !string.IsNullOrWhiteSpace(ProductName) ? ProductName.Trim() : Engine.Project?.Name ?? Name;

        /// <summary>The output folder as an absolute path: <see cref="OutputFolder"/>, or Builds/&lt;profile name&gt; in the project folder.</summary>
        public string ResolvedOutputFolder
        {
            get
            {
                var root = Engine.Project?.RootDirectory ?? "";
                var folder = string.IsNullOrWhiteSpace(OutputFolder) ? Path.Combine("Builds", Name) : OutputFolder.Trim();
                return Path.GetFullPath(Path.IsPathRooted(folder) ? folder : Path.Combine(root, folder));
            }
        }

        /// <summary>The build report's file: next to the output folder, never inside it.</summary>
        public string ReportPath => ResolvedOutputFolder.TrimEnd('\\', '/') + ".report.txt";

        [Button]
        [Description("Walk the references from the scenes and write the build report next to the output folder. Nothing is published")]
        public void Scan()
        {
            var scan = DependencyScanner.Scan(this);
            scan.WriteReport(ReportPath);

            var summary = $"[Build] {Name}: {scan.Artifacts.Count:N0} assets, {BuildScan.Size(scan.TotalBytes)}, " +
                          $"{scan.Warnings.Count} warnings, {scan.Elapsed.TotalSeconds:0.0} s. Report: {ReportPath}";
            Debug.Log(summary);
            Toast.Show($"{scan.Artifacts.Count:N0} assets, {BuildScan.Size(scan.TotalBytes)}: report written", 3f);
        }
    }
}
