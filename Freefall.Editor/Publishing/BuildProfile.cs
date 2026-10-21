using System.ComponentModel;
using Freefall.Assets;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// What a published game holds: its scenes, in order (the game starts in the first), and where it is written.
    /// Scan walks the references from the scenes (DependencyScanner) and writes the build report; Publish writes the
    /// game (Publisher). Nothing references a profile, so it is never packed.
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

        [Description("Texture size guideline in pixels (the larger side). The build report lists the textures above it, what capping them at it would save, and how sharp each would show capped. 0 = no list. Nothing is capped: that is each texture's Max Size")]
        public int TextureBudget = 1024;

        [Description("Pixels per metre a texture should keep where it shows coarsest: the report flags textures that would fall below it at the budget, the candidates for kit pieces, tiling materials or trim sheets")]
        public float TexelDensityTarget = 512;

        [Category("Installer")]
        [Description("After Publish, an installer (Inno Setup 6, which has to be installed): \"<Product> Setup.exe\" in a folder beside the game's, " +
                     "installing it for the user who runs it (no admin prompt) with Start menu and desktop shortcuts and an uninstaller")]
        public bool MakeInstaller;

        [Description("The game's version, in the installer and Apps & features (\"0.3.1\"; up to four numbers, a tail like \"-beta\" is shown but not counted)")]
        public string Version = "0.1.0";

        [Description("Who makes the game, in Apps & features. Empty = none")]
        public string Company = "";

        /// <summary>The scan's textures against <see cref="TextureBudget"/>, or null when it is 0. Worker-safe.</summary>
        public TextureBudgetResult? MeasureTextures(BuildScan scan, Action<string, float>? report = null) =>
            TextureBudget > 0 ? TextureBudgetScan.Measure(scan, TextureBudget, TexelDensityTarget, report) : null;

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
        public string ReportPath => ReportPathFor(ResolvedOutputFolder);

        /// <summary>The build report of a game written to <paramref name="output"/>: next to it, never inside it.</summary>
        public static string ReportPathFor(string output) => output.TrimEnd('\\', '/') + ".report.txt";

        [Button]
        [Description("Walk the references from the scenes, measure the textures against the budget and write the build report next to the output folder (on workers, with a progress popup). Nothing is published")]
        public void Scan()
        {
            try { BuildScanJob.Start(this); }
            catch (InvalidOperationException ex) { ModalDialogs.ShowMessage(Program.EditorUI, $"Could not scan {Name}", ex.Message, ("OK", null)); }
        }

        [Button]
        [Description("Write the game into the output folder, ready to run: the packs, the scripts, the player and the engine's files. " +
                     "Everything a publish wrote there before is replaced. A popup shows the steps")]
        public void Publish() => Publisher.StartFromEditor(this, run: false);

        [Button]
        [Description("Publish, then start the game")]
        public void PublishAndRun() => Publisher.StartFromEditor(this, run: true);
    }
}
