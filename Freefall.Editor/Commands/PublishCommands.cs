using System;
using System.Linq;
using Freefall.Assets;
using Freefall.Editor.Publishing;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Scan a build profile: what a published game would pack, written to its build report. Runs in the call on the
    /// main thread; MCP publish_scan runs the same as a job with the progress popup (BuildScanJob).
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

            var profile = BuildProfile.Find(key);
            if (profile == null)
                return CommandResult.NotFound($"No BuildProfile asset at {key}");

            bool compareLoaded = doc.RootElement.TryGetProperty("compareLoaded", out var cl) && cl.ValueKind == System.Text.Json.JsonValueKind.True;

            try
            {
                var scan = DependencyScanner.Scan(profile);
                if (compareLoaded) scan.LoadedNotReached = DependencyScanner.LoadedNotReached(scan);
                scan.Textures = profile.MeasureTextures(scan);
                scan.WriteReport(profile.ReportPath);
                BuildScanJob.Store(profile, scan);
                return CommandResult.Json(Summary(profile, scan));
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Scan failed: {ex.Message}");
            }
        }

        /// <summary>What a scan found, for REST and MCP: totals, warnings, the textures against the budget, what is loaded but not reached.</summary>
        internal static object Summary(BuildProfile profile, BuildScan scan)
        {
            var missed = scan.LoadedNotReached;
            var textures = scan.Textures;
            var over = textures?.Over;
            return new
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
                textures = textures == null ? null : new
                {
                    budget = textures.Budget,
                    target = textures.Target,
                    overBudget = over!.Count,
                    sizeNow = BuildScan.Size(over.Sum(t => t.Artifact.Size)),
                    sizeCapped = BuildScan.Size(over.Sum(t => t.CappedBytes)),
                    belowTarget = over.Count(textures.BelowTarget),
                    noUseFound = over.Count(t => double.IsNaN(t.Coarsest)),
                    capped = textures.Textures.Count(t => t.Capped),
                    seconds = Math.Round(textures.Elapsed.TotalSeconds, 2),
                    coarsest = over.Where(t => !double.IsNaN(t.CoarsestAtBudget)).OrderBy(t => t.CoarsestAtBudget).Take(15).Select(t => new
                    {
                        path = t.Artifact.SourcePath, size = $"{t.Width}x{t.Height} {t.Format}",
                        pxPerMetre = Math.Round(t.Coarsest), atBudget = Math.Round(t.CoarsestAtBudget), where = t.CoarsestUse,
                    }).ToArray(),
                },
                loadedNotReachedCount = missed?.Count,
                loadedNotReachedByFolder = missed?.GroupBy(m => BuildScan.FolderOf(m.SourcePath))
                    .Select(g => new { folder = g.Key, assets = g.Count() }).ToArray(),
                loadedNotReached = missed?.Take(50)
                    .Select(m => new { guid = m.Guid, name = m.Name, type = m.Type, path = m.SourcePath }).ToArray(),
            };
        }
    }
}
