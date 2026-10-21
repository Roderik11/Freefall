using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Freefall.Assets;
using Freefall.Assets.Packs;
using Freefall.Editor.Publishing;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class PublishTools
    {
        [McpServerTool(Name = "publish_scan", Title = "Scan a build profile", Idempotent = true, OpenWorld = false)]
        [Description("What a published game would pack: walks the references from a BuildProfile's scenes (and every asset under an " +
                     "AlwaysPublish folder) all the way down, and writes the build report next to the profile's output folder " +
                     "(every asset with its size and the chain that reached it). Returns the totals by root, type and folder, the textures over " +
                     "the profile's TextureBudget (what capping them would save, the ones that would show coarsest), and the " +
                     "first warnings (unknown GUIDs, assets not imported). Nothing is published. Runs on workers with the editor's progress " +
                     "popup, as the profile's Scan button does; the profile inspector's texture list shows the result.")]
        public static async Task<CallToolResult> Scan(
            [Description("The BuildProfile asset's GUID, or its path relative to Assets")] string profile,
            [Description("Also list the assets this editor has loaded that the scan did not reach. With the profile's scenes open " +
                         "(and played), that is what code loads by name or by a GUID written in it; the editor loads a few things " +
                         "for itself too (a selected asset, an open graph)")] bool compareLoaded = false,
            [Description("Max seconds to wait")] int waitSeconds = 120)
        {
            BuildProfile? found = null;
            Task<BuildScan> task;
            try
            {
                task = await EditorCommandServer.Instance.RunOnMainThread(() =>
                {
                    found = BuildProfile.Find(profile) ?? throw new InvalidOperationException($"No BuildProfile asset at {profile}");
                    return BuildScanJob.Start(found, compareLoaded);
                });
            }
            catch (Exception ex)
            {
                return McpBridge.Error(ex.GetBaseException().Message);
            }

            if (await Task.WhenAny(task, Task.Delay(TimeSpan.FromSeconds(waitSeconds))) != task)
                return McpBridge.Error($"Still scanning after {waitSeconds} s; editor_status reports it");
            if (task.IsFaulted)
                return McpBridge.Error("The scan failed: " + task.Exception!.GetBaseException().Message);
            return EditorTools.Text(System.Text.Json.JsonSerializer.SerializeToNode(Commands.PublishScanCommand.Summary(found!, task.Result))!);
        }

        [McpServerTool(Name = "game_publish", Title = "Publish a build profile's game", OpenWorld = false)]
        [Description("Write a BuildProfile's game into its output folder (or 'output'), ready to run, as the profile's Publish button does: " +
                     "scan, compile the scripts (Release), build the player (dotnet publish, win-x64, self-contained; kept in the project's " +
                     "Library between publishes), copy it with the exe named after the product, write the packs and the manifest into Data/, " +
                     "and append a summary to the build report next to the folder; with the profile's MakeInstaller (or installer=true) also " +
                     "an Inno Setup installer in '<output> Installer'. The editor shows its modal progress popup meanwhile, and a " +
                     "popup on failure. Everything a publish wrote in the folder before is replaced; a non-empty folder without .freefall-build " +
                     "is refused. Changed assets the build reaches stop it unless saveChanged. Gigabytes: this call waits up to waitSeconds; " +
                     "editor_status reports the step meanwhile.")]
        public static async Task<CallToolResult> Publish(
            [Description("The BuildProfile asset's GUID, or its path relative to Assets")] string profile,
            [Description("Write the game here (absolute) instead of the profile's output folder")] string? output = null,
            [Description("Read every packed artifact back and compare it with its Cache file (about 13 s for 4.7 GB)")] bool verify = false,
            [Description("Save the changed assets the build reaches first (as Save Changed Assets does)")] bool saveChanged = false,
            [Description("Start the game once it is written")] bool run = false,
            [Description("Make the installer (Inno Setup 6) or not, whatever the profile's MakeInstaller says; omit = the profile's setting")] bool? installer = null,
            [Description("Max seconds to wait")] int waitSeconds = 900)
        {
            Task<PublishResult> task;
            try
            {
                task = await EditorCommandServer.Instance.RunOnMainThread(() =>
                {
                    var p = BuildProfile.Find(profile) ?? throw new InvalidOperationException($"No BuildProfile asset at {profile}");
                    return Publisher.Start(p, new PublishOptions { Output = output, Verify = verify, SaveChanged = saveChanged, Run = run, Installer = installer });
                });
            }
            catch (Exception ex)
            {
                return McpBridge.Error(ex.GetBaseException().Message);
            }

            if (await Task.WhenAny(task, Task.Delay(TimeSpan.FromSeconds(waitSeconds))) != task)
            {
                var s = ProgressJob.Running?.Current;
                return McpBridge.Error($"Still publishing after {waitSeconds} s" + (s != null ? $" (step: {s.Step}, {s.Fraction:P0})" : "") +
                                       "; editor_status reports it");
            }
            if (task.IsFaulted)
                return McpBridge.Error("Publishing failed: " + task.Exception!.GetBaseException().Message);

            var r = task.Result;
            return EditorTools.Text(new JsonObject
            {
                ["status"] = "published",
                ["product"] = r.Product,
                ["output"] = r.Output,
                ["exe"] = r.Exe,
                ["report"] = r.Report,
                ["seconds"] = Math.Round(r.Elapsed.TotalSeconds, 1),
                ["steps"] = new JsonArray(r.Steps.Select(s => (JsonNode)new JsonObject
                {
                    ["step"] = s.Step, ["seconds"] = Math.Round(s.Seconds, 1),
                }).ToArray()),
                ["player"] = new JsonObject { ["files"] = r.PlayerFiles, ["size"] = BuildScan.Size(r.PlayerBytes) },
                ["engineFiles"] = new JsonObject { ["files"] = r.EngineFiles, ["size"] = BuildScan.Size(r.EngineBytes) },
                ["scripts"] = r.Scripts,
                ["artifacts"] = r.Packs.Entries,
                ["size"] = BuildScan.Size(r.Packs.Bytes),
                ["packs"] = new JsonArray(r.Packs.Packs.Select(p => (JsonNode)new JsonObject
                {
                    ["file"] = p.File, ["artifacts"] = p.Entries, ["size"] = BuildScan.Size(p.Bytes),
                }).ToArray()),
                ["verified"] = r.Packs.Verified,
                ["mismatches"] = r.Packs.Mismatches,
                ["installer"] = r.Installer.Length > 0 ? new JsonObject { ["setup"] = r.Installer, ["size"] = BuildScan.Size(r.InstallerBytes) } : null,
                ["warnings"] = new JsonArray(r.Warnings.Take(30).Select(w => (JsonNode)w).ToArray()),
            });
        }

        [McpServerTool(Name = "texture_caps", Title = "Cap or uncap a profile's textures", OpenWorld = false)]
        [Description("Set Max Size on textures from a BuildProfile's latest scan (Scan or publish_scan first), as the profile " +
                     "inspector's Cap and Uncap buttons do. cap=true takes the textures under 'paths' (folders or files relative " +
                     "to Assets) that are over the profile's TextureBudget and caps them at it; cap=false takes the capped ones " +
                     "back to their source size. They reimport side by side, then the profile is scanned again, with the editor's " +
                     "progress popup. The source files are not touched.")]
        public static async Task<CallToolResult> TextureCaps(
            [Description("The BuildProfile asset's GUID, or its path relative to Assets")] string profile,
            [Description("Folders or files relative to Assets, e.g. ['Characters\\\\Paladin']")] string[] paths,
            [Description("true = cap at the profile's TextureBudget, false = uncap")] bool cap = true,
            [Description("Max seconds to wait")] int waitSeconds = 900)
        {
            Task<int> task;
            int count = 0;
            try
            {
                task = await EditorCommandServer.Instance.RunOnMainThread(() =>
                {
                    var p = BuildProfile.Find(profile) ?? throw new InvalidOperationException($"No BuildProfile asset at {profile}");
                    var rows = BuildScanJob.Latest(p)?.Textures?.Rows
                        ?? throw new InvalidOperationException("The profile has no scan in this editor yet: publish_scan first");
                    var wanted = paths.Select(x => x.Trim().TrimEnd('\\', '/').Replace('/', '\\')).Where(x => x.Length > 0).ToList();
                    var picked = rows.Where(t => wanted.Any(w => t.Artifact.SourcePath.Equals(w, StringComparison.OrdinalIgnoreCase)
                                                             || t.Artifact.SourcePath.StartsWith(w + "\\", StringComparison.OrdinalIgnoreCase)))
                                     .Where(t => cap ? t.OverBudget(p.TextureBudget) && !t.Capped : t.Capped)
                                     .Select(t => t.Artifact.Guid).ToList();
                    count = picked.Count;
                    if (count == 0) return Task.FromResult(0);
                    return Publishing.TextureCaps.Start(p, picked, cap ? p.TextureBudget : 0);
                });
            }
            catch (Exception ex)
            {
                return McpBridge.Error(ex.GetBaseException().Message);
            }

            if (await Task.WhenAny(task, Task.Delay(TimeSpan.FromSeconds(waitSeconds))) != task)
                return McpBridge.Error($"Still working after {waitSeconds} s; editor_status reports it");
            if (task.IsFaulted)
                return McpBridge.Error("Failed: " + task.Exception!.GetBaseException().Message);
            return EditorTools.Text(new JsonObject { ["matched"] = count, ["changed"] = task.Result, ["capped"] = cap });
        }
        [McpServerTool(Name = "publish_mount", Title = "Mount a profile's packs over the Cache", OpenWorld = false)]
        [Description("Test packs in the editor: mount=true makes every artifact load from a BuildProfile's packs (Data/ in its output folder) " +
                     "before the Cache folder, mount=false goes back to the folder alone. Only what loads after mounting comes from the packs, " +
                     "so mount right after starting the editor, then load the scenes. Every call reports the mount and its counters: " +
                     "artifacts opened from the packs, and those still opened from the folder (what the packs lack).")]
        public static async Task<CallToolResult> Mount(
            [Description("The BuildProfile asset's GUID or path (for mount=true)")] string? profile = null,
            [Description("true = mount, false = unmount, omitted = report only")] bool? mount = null)
        {
            try
            {
                var info = await EditorCommandServer.Instance.RunOnMainThread(() =>
                {
                    if (mount == true)
                    {
                        var p = BuildProfile.Find(profile ?? "") ?? throw new InvalidOperationException($"No BuildProfile asset at {profile}");
                        var data = Path.Combine(p.ResolvedOutputFolder, PackBuilder.DataFolderName);
                        if (!Directory.Exists(data)) throw new InvalidOperationException($"No packs at {data}: game_publish first");
                        Artifacts.Mount(PackSet.OpenFolder(data));
                    }
                    else if (mount == false)
                        Artifacts.Unmount();

                    var packs = Artifacts.Packs;
                    return new JsonObject
                    {
                        ["mounted"] = packs != null,
                        ["packs"] = packs == null ? null : new JsonArray(packs.Packs.Select(x => (JsonNode)Path.GetFileName(x.Path)).ToArray()),
                        ["artifacts"] = packs?.Count,
                        ["openedFromPacks"] = Artifacts.OpenedFromPacks,
                        ["openedFromFolder"] = Artifacts.OpenedFromFolder,
                        ["folderReads"] = new JsonArray(Artifacts.FolderReads.Take(40).Select(f => (JsonNode)f).ToArray()),
                    };
                });
                return EditorTools.Text(info);
            }
            catch (Exception ex)
            {
                return McpBridge.Error(ex.GetBaseException().Message);
            }
        }
    }
}
