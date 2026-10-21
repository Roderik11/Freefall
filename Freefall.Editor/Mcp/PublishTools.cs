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
                     "(every asset with its size and the chain that reached it). Returns the totals by root, type and folder, and the " +
                     "first warnings (unknown GUIDs, assets not imported). Nothing is published.")]
        public static Task<CallToolResult> Scan(
            [Description("The BuildProfile asset's GUID, or its path relative to Assets")] string profile,
            [Description("Also list the assets this editor has loaded that the scan did not reach. With the profile's scenes open " +
                         "(and played), that is what code loads by name or by a GUID written in it; the editor loads a few things " +
                         "for itself too (a selected asset, an open graph)")] bool compareLoaded = false)
            => McpBridge.Post("/api/publish/scan", new { profile, compareLoaded });

        [McpServerTool(Name = "publish_pack", Title = "Write a build profile's packs", OpenWorld = false)]
        [Description("Scan a BuildProfile, then write its packs (Data/data0.ffpack, … : the Cache artifacts byte for byte, a new pack at " +
                     "the profile's PackSizeLimitMB) and Data/game.manifest into the profile's output folder, and with verify read every " +
                     "artifact back from the packs and compare it with its Cache file. Gigabytes: runs on a worker, this call waits for " +
                     "it. The output folder must be empty or one Publish made (.freefall-build). No executable is written yet.")]
        public static async Task<CallToolResult> Pack(
            [Description("The BuildProfile asset's GUID, or its path relative to Assets")] string profile,
            [Description("Read every packed artifact back and compare it with its Cache file")] bool verify = true,
            [Description("Max seconds to wait for the write")] int waitSeconds = 900)
        {
            Task<PackResult> task;
            try
            {
                task = await EditorCommandServer.Instance.RunOnMainThread(() =>
                {
                    var p = BuildProfile.Find(profile) ?? throw new InvalidOperationException($"No BuildProfile asset at {profile}");
                    return PackBuilder.Start(p, verify);
                });
            }
            catch (Exception ex)
            {
                return McpBridge.Error(ex.GetBaseException().Message);
            }

            if (await Task.WhenAny(task, Task.Delay(TimeSpan.FromSeconds(waitSeconds))) != task)
                return McpBridge.Error($"Still writing after {waitSeconds} s; call publish_mount (no arguments) later, or look in the output folder");
            if (task.IsFaulted)
                return McpBridge.Error("Writing the packs failed: " + task.Exception!.GetBaseException().Message);

            var r = task.Result;
            return EditorTools.Text(new JsonObject
            {
                ["status"] = "written",
                ["data"] = r.DataFolder,
                ["artifacts"] = r.Entries,
                ["size"] = BuildScan.Size(r.Bytes),
                ["packs"] = new JsonArray(r.Packs.Select(p => (JsonNode)new JsonObject
                {
                    ["file"] = p.File, ["artifacts"] = p.Entries, ["size"] = BuildScan.Size(p.Bytes),
                }).ToArray()),
                ["scanSeconds"] = Math.Round(r.ScanTime.TotalSeconds, 2),
                ["writeSeconds"] = Math.Round(r.WriteTime.TotalSeconds, 1),
                ["verified"] = r.Verified,
                ["verifySeconds"] = Math.Round(r.VerifyTime.TotalSeconds, 1),
                ["mismatches"] = r.Mismatches,
                ["scripts"] = r.Scripts,
                ["problems"] = new JsonArray(r.Problems.Take(30).Select(p => (JsonNode)p).ToArray()),
            });
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
                        if (!Directory.Exists(data)) throw new InvalidOperationException($"No packs at {data}: publish_pack first");
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
