using System;
using System.Linq;
using Freefall.Assets;
using Freefall.Editor.Publishing;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Scan a build profile: what a published game would pack, written to its build report.
    /// POST /api/publish/scan
    /// Body: {"profile": "&lt;BuildProfile GUID, or its path relative to Assets&gt;", "compareLoaded": false}
    /// </summary>
    [CommandRoute("POST", "/api/publish/scan")]
    public class PublishScanCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'profile'");

            using var doc = context.ParseBody();
            string? key = doc.RootElement.TryGetProperty("profile", out var p) ? p.GetString() : null;
            if (string.IsNullOrEmpty(key))
                return CommandResult.BadRequest("'profile' is required: the BuildProfile's GUID or its path relative to Assets");

            var guid = AssetDatabase.GetMeta(key) != null ? key : AssetDatabase.PathToGuid(key);
            if (guid == null || AssetDatabase.GetAssetType(guid) is not { } type || !typeof(BuildProfile).IsAssignableFrom(type))
                return CommandResult.NotFound($"No BuildProfile asset at {key}");

            var profile = Engine.Assets.LoadByGuid<BuildProfile>(guid);
            if (profile == null)
                return CommandResult.Error(500, $"The BuildProfile {key} did not load");

            bool compareLoaded = doc.RootElement.TryGetProperty("compareLoaded", out var cl) && cl.ValueKind == System.Text.Json.JsonValueKind.True;

            try
            {
                var scan = DependencyScanner.Scan(profile);
                var missed = scan.LoadedNotReached = compareLoaded ? DependencyScanner.LoadedNotReached(scan) : null;
                scan.WriteReport(profile.ReportPath);

                return CommandResult.Json(new
                {
                    status = "scanned",
                    profile = profile.Name,
                    product = profile.ResolvedProductName,
                    scenes = scan.Scenes.Select(s => new { name = s.Name, guid = s.Guid, inList = s.Root != null }).ToArray(),
                    assets = scan.Artifacts.Count,
                    size = BuildScan.Size(scan.TotalBytes),
                    bytes = scan.TotalBytes,
                    byRoot = scan.ByRoot().Select(r => new { root = r.Root, assets = r.Count, size = BuildScan.Size(r.Bytes) }).ToArray(),
                    byType = scan.ByType().Select(t => new { type = t.Type, assets = t.Count, size = BuildScan.Size(t.Bytes) }).ToArray(),
                    largestFolders = scan.ByFolder().Take(12).Select(f => new { folder = f.Folder, assets = f.Count, size = BuildScan.Size(f.Bytes) }).ToArray(),
                    warningCount = scan.Warnings.Count,
                    warnings = scan.Warnings.Take(15).ToArray(),
                    report = profile.ReportPath,
                    seconds = Math.Round(scan.Elapsed.TotalSeconds, 2),
                    loadedNotReachedCount = missed?.Count,
                    loadedNotReachedByFolder = missed?.GroupBy(m => BuildScan.FolderOf(m.SourcePath))
                        .Select(g => new { folder = g.Key, assets = g.Count() }).ToArray(),
                    loadedNotReached = missed?.Take(50)
                        .Select(m => new { guid = m.Guid, name = m.Name, type = m.Type, path = m.SourcePath }).ToArray(),
                });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Scan failed: {ex.Message}");
            }
        }
    }
}
