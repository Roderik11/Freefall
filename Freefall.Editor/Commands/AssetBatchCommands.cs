using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Freefall.Assets;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Create (or update) many assets in one call and set their members.
    /// POST /api/assets/create-batch
    /// Body: {"assets": [{"type": "Stat", "path": "Characters/Stats/Strength.asset", "values": {...}}, ...]}
    ///
    /// Every asset that does not exist yet is written first and the database rescanned once; only then are the
    /// values set and each asset saved, so an asset of the batch can refer to any other ("@path", see
    /// <see cref="AssetPathReferences"/>), whatever their order. A file that exists already is updated in place:
    /// the same GUID, the members given replace its own and the rest are kept, so a batch can be run again after
    /// its data was changed. 'path' is relative to Assets; without an extension the type's own is used. The
    /// values are encoded as for /api/asset/setproperty ("$type" names the class of an element of a list of an
    /// abstract type). One failing asset does not stop the others; each is reported.
    /// </summary>
    [CommandRoute("POST", "/api/assets/create-batch")]
    public class CreateAssetBatchCommand : EditorCommand
    {
        sealed class Entry
        {
            public JsonElement Source;
            public Type Type;
            public string RelativePath, FullPath, Guid, Error;
            public bool Existed;
        }

        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required: {\"assets\": [{\"type\", \"path\", \"values\"}, ...]}");

            using var doc = context.ParseBody();
            if (!doc.RootElement.TryGetProperty("assets", out var list) || list.ValueKind != JsonValueKind.Array)
                return CommandResult.BadRequest("'assets' must be an array of {\"type\", \"path\", \"values\"}");

            var projectDir = Engine.Project?.AssetsDirectory;
            if (string.IsNullOrEmpty(projectDir))
                return CommandResult.BadRequest("No project open");

            var entries = new List<Entry>();

            // 1. The files of the new ones
            foreach (var item in list.EnumerateArray())
            {
                var entry = new Entry { Source = item };
                entries.Add(entry);
                try
                {
                    string typeName = item.TryGetProperty("type", out var t) ? t.GetString() : null;
                    string path = item.TryGetProperty("path", out var p) ? p.GetString() : null;
                    entry.RelativePath = path;
                    if (string.IsNullOrEmpty(typeName) || string.IsNullOrEmpty(path))
                        throw new InvalidOperationException("'type' and 'path' are required");

                    entry.Type = CreateAssetCommand.FindAssetType(typeName) ?? throw new InvalidOperationException(
                        $"Unknown asset type '{typeName}'. Available: {string.Join(", ", CreateAssetCommand.AssetTypeNames)}");

                    path = path.Replace('/', Path.DirectorySeparatorChar).TrimStart(Path.DirectorySeparatorChar);
                    if (string.IsNullOrEmpty(Path.GetExtension(path)))
                        path += AssetManager.GetFileExtension(entry.Type);
                    entry.RelativePath = path;
                    entry.FullPath = Path.Combine(projectDir, path);

                    if (File.Exists(entry.FullPath))
                    {
                        entry.Existed = true;
                        continue;
                    }

                    Directory.CreateDirectory(Path.GetDirectoryName(entry.FullPath)!);
                    var asset = (Asset)Activator.CreateInstance(entry.Type);
                    asset.Name = Path.GetFileNameWithoutExtension(path);
                    Engine.Assets.SaveAsset(asset, entry.FullPath);
                }
                catch (Exception ex)
                {
                    entry.Error = ex.Message;
                }
            }

            // 2. One rescan registers every new file; each is imported
            AssetDatabase.Refresh();
            foreach (var entry in entries)
            {
                if (entry.Error != null) continue;
                entry.Guid = AssetDatabase.PathToGuid(entry.RelativePath);
                if (entry.Guid == null)
                {
                    entry.Error = "written, but the AssetDatabase did not register it";
                    continue;
                }
                if (!entry.Existed) AssetDatabase.ImportAssetByPath(entry.RelativePath);
            }

            // 3. The values, now that every asset of the batch has its GUID
            foreach (var entry in entries)
            {
                if (entry.Error != null) continue;
                try
                {
                    var asset = CommandHelpers.FindOrLoadAsset(entry.Guid)
                        ?? throw new InvalidOperationException("could not be loaded");
                    if (!entry.Type.IsInstanceOfType(asset))
                        throw new InvalidOperationException($"the file holds a {asset.GetType().Name}, not a {entry.Type.Name}");

                    if (entry.Source.TryGetProperty("values", out var values) && values.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var kv in values.EnumerateObject())
                        {
                            try { SetAssetPropertyCommand.ApplyValue(asset, kv.Name, kv.Value); }
                            catch (Exception ex) { throw new InvalidOperationException($"{kv.Name}: {ex.Message}", ex); }
                        }

                        // As an inspector edit does: dependents hear of it
                        asset.MarkDirty();
                        Engine.Assets.SaveAsset(asset, entry.FullPath);
                    }
                }
                catch (Exception ex)
                {
                    entry.Error = ex.Message;
                }
            }

            return CommandResult.Json(new
            {
                created = entries.Count(e => e.Error == null && !e.Existed),
                updated = entries.Count(e => e.Error == null && e.Existed),
                failed = entries.Count(e => e.Error != null),
                assets = entries.Select(e => new
                {
                    path = e.RelativePath?.Replace('\\', '/'),
                    guid = e.Guid,
                    status = e.Error != null ? "failed" : e.Existed ? "updated" : "created",
                    error = e.Error,
                }),
            });
        }
    }
}
