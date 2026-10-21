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

        [Description("Largest pack file in MB: the artifacts go into data0.ffpack, data1.ffpack, … in the order the scan reached them, a new pack once this size would be passed")]
        public int PackSizeLimitMB = 2048;

        /// <summary>The profile with this GUID, or at this path relative to Assets; null if there is none.</summary>
        public static BuildProfile? Find(string key)
        {
            if (string.IsNullOrEmpty(key)) return null;
            var guid = AssetDatabase.GetMeta(key) != null ? key : AssetDatabase.PathToGuid(key);
            if (guid == null || AssetDatabase.GetAssetType(guid) is not { } type || !typeof(BuildProfile).IsAssignableFrom(type))
                return null;
            return Engine.Assets.LoadByGuid<BuildProfile>(guid);
        }

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

        [Button]
        [Description("Scan, then write the packs and the manifest into Data/ in the output folder and check every packed artifact against its Cache file (on a worker; a toast says when it is done). No executable yet")]
        public void WritePacks()
        {
            var task = PackBuilder.Start(this, verify: true);
            Toast.Show("Writing packs…", 2f);
            task.ContinueWith(t => Engine.RunOnMainThreadAsync(() =>
            {
                if (t.IsFaulted)
                {
                    Debug.LogError($"[Build] {Name}: writing packs failed: {t.Exception?.GetBaseException().Message}");
                    Toast.Show("Writing packs failed: see the console", 3f);
                    return;
                }
                var r = t.Result;
                Debug.Log($"[Build] {Name}: {r.Entries:N0} artifacts, {BuildScan.Size(r.Bytes)} in {r.Packs.Count} pack(s), " +
                          $"written in {r.WriteTime.TotalSeconds:0.0} s, {r.Mismatches} mismatches, {r.Problems.Count} problems: {r.DataFolder}");
                Toast.Show($"{r.Packs.Count} pack(s), {BuildScan.Size(r.Bytes)}, {r.Mismatches} mismatches", 3f);
            }));
        }
    }
}
