using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graph;
using Freefall.Graphics;

namespace Freefall.Editor.Commands
{
    /// <summary>Shared helpers for the authoring routes (graphs, materials, bulk scene edits).</summary>
    internal static class AuthoringHelpers
    {
        /// <summary>Save an asset to its source file and reimport it so the Library cache matches (same as asset setproperty).</summary>
        public static string SaveAndReimport(Asset asset, string guid)
        {
            var relativePath = AssetDatabase.GuidToPath(guid);
            if (relativePath == null || Engine.Project == null)
            {
                asset.MarkDirty();
                return null;
            }
            var fullPath = Path.Combine(Engine.Project.AssetsDirectory, relativePath);
            Engine.Assets.SaveAsset(asset, fullPath);
            AssetDatabase.ImportAssetByPath(relativePath);
            asset.ClearDirty();
            return fullPath;
        }

        private static Dictionary<string, Type> _nodeTypes;

        /// <summary>Forget the cached types after a script reload (it would pin the old script assembly).</summary>
        internal static void ResetTypeCache() => _nodeTypes = null;

        /// <summary>Every concrete graph node type, by short class name.</summary>
        public static Dictionary<string, Type> NodeTypes
        {
            get
            {
                if (_nodeTypes != null) return _nodeTypes;
                _nodeTypes = new Dictionary<string, Type>(StringComparer.OrdinalIgnoreCase);
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    if (ScriptCompiler.IsStale(asm)) continue;
                    Type[] types;
                    try { types = asm.GetTypes(); } catch { continue; }
                    foreach (var t in types)
                        if (!t.IsAbstract && typeof(Node).IsAssignableFrom(t) && t.GetConstructor(Type.EmptyTypes) != null)
                            _nodeTypes[t.Name] = t;
                }
                return _nodeTypes;
            }
        }

        /// <summary>Editable settings of a node: public fields that are not ports, not hidden, not layout.</summary>
        public static IEnumerable<FieldInfo> SettingFields(Type nodeType) =>
            nodeType.GetFields(BindingFlags.Public | BindingFlags.Instance)
                .Where(f => !f.IsInitOnly
                            && f.GetCustomAttribute<InputAttribute>() == null
                            && f.GetCustomAttribute<OutputAttribute>() == null
                            && f.GetCustomAttribute<BrowsableAttribute>()?.Browsable != false
                            && f.Name is not ("ID" or "Position" or "Expanded"));

        public static object DescribeNode(Node node) => new
        {
            id = node.ID,
            type = node.GetType().Name,
            inputs = node.Inputs.Select(p => p.Field.Name).ToArray(),
            outputs = node.Outputs.Select(p => p.Field.Name).ToArray(),
            fields = SettingFields(node.GetType()).ToDictionary(f => f.Name, f => CommandHelpers.SerializeValue(f.GetValue(node))),
        };

        /// <summary>Connections as producer → consumer, regardless of how they are stored.</summary>
        public static IEnumerable<object> DescribeConnections(NodeGraph graph)
        {
            foreach (var c in graph.Connections)
            {
                if (c.PortA == null || c.PortB == null) continue;
                var output = c.PortA.Type == Port.InOut.Output ? c.PortA : c.PortB;
                var input = output == c.PortA ? c.PortB : c.PortA;
                yield return new
                {
                    from = new { node = output.Node.ID, port = output.Field.Name },
                    to = new { node = input.Node.ID, port = input.Field.Name },
                };
            }
        }

        public static void ApplyNodeValues(Node node, JsonElement values)
        {
            if (values.ValueKind != JsonValueKind.Object) return;
            var type = node.GetType();
            foreach (var kv in values.EnumerateObject())
            {
                var field = SettingFields(type).FirstOrDefault(f => f.Name == kv.Name)
                    ?? throw new InvalidOperationException(
                        $"'{type.Name}' has no setting '{kv.Name}'. Settings: {string.Join(", ", SettingFields(type).Select(f => f.Name))}");
                field.SetValue(node, SetPropertyCommand.ConvertJsonValue(kv.Value, field.FieldType));
            }
        }

        /// <summary>Re-run every PCGComponent using this graph (via GraphChanged); returns what they spawned.</summary>
        public static List<object> NotifyGraphChanged(NodeGraph graph)
        {
            MessageDispatcher.Send(EngineMsg.GraphChanged, graph);
            var runs = new List<object>();
            foreach (var entity in EntityManager.Entities)
                foreach (var pcg in entity.Components.OfType<PCGComponent>())
                    if (pcg.Graph == graph)
                        runs.Add(new { id = entity.Id, uid = entity.UID.ToString(), name = entity.Name, spawned = CountSpawned(entity) });
            return runs;
        }

        public static int CountSpawned(Entity pcgEntity)
        {
            var t = pcgEntity.Transform;
            for (int i = 0; i < t.Count; i++)
            {
                var child = t.GetChild(i);
                if (child.Entity.Name == "PCG_Output") return child.Count;
            }
            return 0;
        }

        // ── Entity filtering (scene query / bulk delete) ──

        public static bool IsGenerated(Entity e)
        {
            for (var t = e.Transform; t != null; t = t.Parent)
                if ((t.Entity.Flags & EntityFlags.DontSave) != 0) return true;
            return false;
        }

        /// <summary>
        /// Entities matching the body's filters: name (wildcards), component, prefab (guid or name), area
        /// ({center:{x,z}, radius} or {min:{x,z}, max:{x,z}}), topLevelOnly (default true),
        /// includeGenerated (PCG output etc., default false).
        /// </summary>
        public static List<Entity> Filter(JsonElement root)
        {
            Regex nameRegex = null;
            if (root.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String && n.GetString() is { Length: > 0 } pattern)
                nameRegex = new Regex("^" + Regex.Escape(pattern).Replace("\\*", ".*") + "$", RegexOptions.IgnoreCase);

            Type compType = null;
            if (root.TryGetProperty("component", out var c) && c.ValueKind == JsonValueKind.String && c.GetString() is { Length: > 0 } cname)
                compType = CommandHelpers.FindComponentType(cname) ?? throw new InvalidOperationException($"Component type '{cname}' not found");

            string prefab = root.TryGetProperty("prefab", out var p) && p.ValueKind == JsonValueKind.String ? p.GetString() : null;

            bool hasCircle = false, hasRect = false;
            Vector2 center = default, min = default, max = default;
            float radius = 0;
            if (root.TryGetProperty("area", out var a) && a.ValueKind == JsonValueKind.Object)
            {
                if (a.TryGetProperty("center", out var ce))
                {
                    hasCircle = true;
                    center = new Vector2(ce.GetProperty("x").GetSingle(), ce.GetProperty("z").GetSingle());
                    radius = a.TryGetProperty("radius", out var r) ? r.GetSingle() : 100f;
                }
                else if (a.TryGetProperty("min", out var mi) && a.TryGetProperty("max", out var ma))
                {
                    hasRect = true;
                    min = new Vector2(mi.GetProperty("x").GetSingle(), mi.GetProperty("z").GetSingle());
                    max = new Vector2(ma.GetProperty("x").GetSingle(), ma.GetProperty("z").GetSingle());
                }
            }

            bool topLevelOnly = !root.TryGetProperty("topLevelOnly", out var tl) || tl.GetBoolean();
            bool includeGenerated = root.TryGetProperty("includeGenerated", out var ig) && ig.GetBoolean();

            var result = new List<Entity>();
            foreach (var e in EntityManager.Entities)
            {
                if (topLevelOnly && e.Transform.Parent != null) continue;
                if (!includeGenerated && IsGenerated(e)) continue;
                if (nameRegex != null && !nameRegex.IsMatch(e.Name)) continue;
                if (compType != null && !e.Components.Any(x => compType.IsAssignableFrom(x.GetType()))) continue;
                if (prefab != null && !(e.Prefab != null &&
                    (string.Equals(e.Prefab.Guid, prefab, StringComparison.OrdinalIgnoreCase) ||
                     string.Equals(e.Prefab.Name, prefab, StringComparison.OrdinalIgnoreCase)))) continue;
                if (hasCircle || hasRect)
                {
                    var wp = e.Transform.WorldPosition;
                    var xz = new Vector2(wp.X, wp.Z);
                    if (hasCircle && Vector2.Distance(xz, center) > radius) continue;
                    if (hasRect && (xz.X < min.X || xz.X > max.X || xz.Y < min.Y || xz.Y > max.Y)) continue;
                }
                result.Add(e);
            }
            return result;
        }
    }

    // ═══════════════════════════ Graphs ═══════════════════════════

    /// <summary>GET /api/graph/nodetypes — every node type with its category, ports and settings (with defaults).</summary>
    [CommandRoute("GET", "/api/graph/nodetypes")]
    public class GraphNodeTypesCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var types = AuthoringHelpers.NodeTypes.Values.OrderBy(t => t.Namespace).ThenBy(t => t.Name).Select(t =>
            {
                Node sample = null;
                try { sample = (Node)Activator.CreateInstance(t); } catch { }
                return new
                {
                    type = t.Name,
                    ns = t.Namespace,
                    category = t.GetCustomAttribute<CategoryAttribute>()?.Category,
                    inputs = t.GetFields().Where(f => f.GetCustomAttribute<InputAttribute>() != null).Select(f => f.Name).ToArray(),
                    outputs = t.GetFields().Where(f => f.GetCustomAttribute<OutputAttribute>() != null).Select(f => f.Name).ToArray(),
                    settings = AuthoringHelpers.SettingFields(t).ToDictionary(
                        f => f.Name,
                        f => (object)new { type = f.FieldType.Name, @default = sample != null ? CommandHelpers.SerializeValue(f.GetValue(sample)) : null }),
                };
            }).ToArray();
            return CommandResult.Json(new { count = types.Length, types });
        }
    }

    /// <summary>GET /api/graph/get?guid= — nodes (settings, ports) and connections (producer → consumer).</summary>
    [CommandRoute("GET", "/api/graph/get")]
    public class GraphGetCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var qs = CommandHelpers.ParseQueryString(context.Path);
            if (!qs.TryGetValue("guid", out var guid)) return CommandResult.BadRequest("'guid' required");
            if (CommandHelpers.FindOrLoadAsset(guid) is not NodeGraph graph) return CommandResult.NotFound($"Graph not found: {guid}");

            return CommandResult.Json(new
            {
                guid,
                name = graph.Name,
                type = graph.GetType().Name,
                seed = graph.Seed,
                nodes = graph.Nodes.Select(AuthoringHelpers.DescribeNode).ToArray(),
                connections = AuthoringHelpers.DescribeConnections(graph).ToArray(),
                usedBy = EntityManager.Entities.Where(e => e.Components.OfType<PCGComponent>().Any(p => p.Graph == graph))
                    .Select(e => new { id = e.Id, uid = e.UID.ToString(), name = e.Name }).ToArray(),
            });
        }
    }

    /// <summary>
    /// POST /api/graph/edit — one or more graph edits in order, then save + reimport + re-run users.
    /// Body: {"guid":"...", "ops":[
    ///   {"op":"add", "type":"DensityNoise", "values":{...}, "ref":"noise"},        // ref names the new node for later ops
    ///   {"op":"set", "node":5, "values":{...}},
    ///   {"op":"remove", "node":5},
    ///   {"op":"connect", "from":"noise", "fromPort":"Output", "to":7, "toPort":"Input"},
    ///   {"op":"disconnect", "from":3, "to":7}], "run": true}
    /// Node references are ids or refs from earlier "add" ops in the same call.
    /// </summary>
    [CommandRoute("POST", "/api/graph/edit")]
    public class GraphEditCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body)) return CommandResult.BadRequest("Body required");
            using var doc = context.ParseBody();
            var root = doc.RootElement;

            string guid = root.TryGetProperty("guid", out var g) ? g.GetString() : null;
            if (CommandHelpers.FindOrLoadAsset(guid ?? "") is not NodeGraph graph) return CommandResult.NotFound($"Graph not found: {guid}");
            if (!root.TryGetProperty("ops", out var ops) || ops.ValueKind != JsonValueKind.Array)
                return CommandResult.BadRequest("'ops' array required");

            var refs = new Dictionary<string, Node>(StringComparer.OrdinalIgnoreCase);
            var results = new List<object>();

            Node Resolve(JsonElement el)
            {
                if (el.ValueKind == JsonValueKind.Number)
                    return graph.Nodes.FirstOrDefault(x => x.ID == el.GetInt32()) ?? throw new InvalidOperationException($"No node {el.GetInt32()}");
                var s = el.GetString();
                if (refs.TryGetValue(s, out var r)) return r;
                if (int.TryParse(s, out var id)) return graph.Nodes.FirstOrDefault(x => x.ID == id) ?? throw new InvalidOperationException($"No node {id}");
                throw new InvalidOperationException($"Unknown node ref '{s}'");
            }

            int i = 0;
            try
            {
                foreach (var op in ops.EnumerateArray())
                {
                    string kind = op.GetProperty("op").GetString()?.ToLowerInvariant();
                    switch (kind)
                    {
                        case "add":
                        {
                            var typeName = op.GetProperty("type").GetString();
                            if (!AuthoringHelpers.NodeTypes.TryGetValue(typeName, out var t))
                                throw new InvalidOperationException($"Unknown node type '{typeName}' (see graph_node_types)");
                            var node = (Node)Activator.CreateInstance(t);
                            graph.AddNode(node);
                            // Lay new nodes out in a column to the right of the existing ones
                            float maxX = graph.Nodes.Where(x => x != node).Select(x => x.Position.X).DefaultIfEmpty(500000).Max();
                            node.Position = new Vector2(maxX + 220, 500000 + (node.ID % 8) * 120);
                            if (op.TryGetProperty("values", out var v)) AuthoringHelpers.ApplyNodeValues(node, v);
                            if (op.TryGetProperty("ref", out var rf)) refs[rf.GetString()] = node;
                            results.Add(new { op = "add", id = node.ID, type = t.Name });
                            break;
                        }
                        case "set":
                        {
                            var node = Resolve(op.GetProperty("node"));
                            AuthoringHelpers.ApplyNodeValues(node, op.GetProperty("values"));
                            results.Add(new { op = "set", id = node.ID });
                            break;
                        }
                        case "remove":
                        {
                            var node = Resolve(op.GetProperty("node"));
                            graph.RemoveNode(node);
                            results.Add(new { op = "remove", id = node.ID });
                            break;
                        }
                        case "connect":
                        case "disconnect":
                        {
                            var from = Resolve(op.GetProperty("from"));
                            var to = Resolve(op.GetProperty("to"));
                            string fromPortName = op.TryGetProperty("fromPort", out var fp) ? fp.GetString() : "Output";
                            string toPortName = op.TryGetProperty("toPort", out var tp) ? tp.GetString() : "Input";
                            var outPort = from.GetPort(fromPortName) ?? throw new InvalidOperationException($"Node {from.ID} ({from.GetType().Name}) has no port '{fromPortName}'");
                            var inPort = to.GetPort(toPortName) ?? throw new InvalidOperationException($"Node {to.ID} ({to.GetType().Name}) has no port '{toPortName}'");
                            if (outPort.Type != Port.InOut.Output) throw new InvalidOperationException($"'{fromPortName}' on node {from.ID} is not an output");
                            if (inPort.Type != Port.InOut.Input) throw new InvalidOperationException($"'{toPortName}' on node {to.ID} is not an input");
                            if (kind == "connect")
                            {
                                if (!graph.CanConnect(inPort, outPort))
                                    throw new InvalidOperationException($"Cannot connect {from.ID}.{fromPortName} → {to.ID}.{toPortName} (type mismatch or already connected)");
                                // Stored as (input, output), matching graphs authored in the editor
                                graph.AddConnection(inPort, outPort);
                            }
                            else graph.RemoveConnection(inPort, outPort);
                            results.Add(new { op = kind, from = from.ID, to = to.ID });
                            break;
                        }
                        default:
                            throw new InvalidOperationException($"Unknown op '{kind}' (add, set, remove, connect, disconnect)");
                    }
                    i++;
                }
            }
            catch (Exception ex)
            {
                // Earlier ops stay applied in memory; nothing is saved
                return CommandResult.Error(400, $"Op #{i} failed: {ex.Message}. Earlier ops are applied in memory but not saved; fix and resend, or reload.");
            }

            string saved = AuthoringHelpers.SaveAndReimport(graph, guid);
            bool run = !root.TryGetProperty("run", out var rn) || rn.GetBoolean();
            var runs = run ? AuthoringHelpers.NotifyGraphChanged(graph) : new List<object>();

            return CommandResult.Json(new { status = "edited", guid, results, saved, runs });
        }
    }

    // ═══════════════════════════ PCG ═══════════════════════════

    /// <summary>POST /api/pcg/execute {"id":123} (or no id = every PCG component) — run and report spawn counts.</summary>
    [CommandRoute("POST", "/api/pcg/execute")]
    public class PcgExecuteCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            int? id = null;
            if (!string.IsNullOrEmpty(context.Body))
            {
                using var doc = context.ParseBody();
                if (doc.RootElement.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.Number) id = idEl.GetInt32();
            }

            var entities = id.HasValue
                ? new[] { CommandHelpers.FindEntityById(id.Value) ?? throw new InvalidOperationException($"Entity {id} not found") }
                : EntityManager.Entities.Where(e => e.Components.OfType<PCGComponent>().Any()).ToArray();

            var runs = new List<object>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            foreach (var e in entities)
            {
                foreach (var pcg in e.Components.OfType<PCGComponent>())
                {
                    var t0 = sw.Elapsed.TotalMilliseconds;
                    pcg.Execute();
                    runs.Add(new { id = e.Id, uid = e.UID.ToString(), name = e.Name, graph = pcg.Graph?.Name, spawned = AuthoringHelpers.CountSpawned(e), ms = Math.Round(sw.Elapsed.TotalMilliseconds - t0, 1) });
                }
            }
            if (runs.Count == 0) return CommandResult.NotFound("No PCGComponent found");
            return CommandResult.Json(new { count = runs.Count, runs });
        }
    }

    // ═══════════════════════════ Materials ═══════════════════════════

    /// <summary>
    /// POST /api/material/settextures {"guid":"...", "textures":{"Albedo":"texGuid", "Normal":"texGuid"}, "effect":"effectGuid"?}
    /// Assigns texture slots by name, then saves + reimports. Returns the material's slot names.
    /// </summary>
    [CommandRoute("POST", "/api/material/settextures")]
    public class MaterialSetTexturesCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body)) return CommandResult.BadRequest("Body required");
            using var doc = context.ParseBody();
            var root = doc.RootElement;
            string guid = root.TryGetProperty("guid", out var g) ? g.GetString() : null;
            if (CommandHelpers.FindOrLoadAsset(guid ?? "") is not Material material) return CommandResult.NotFound($"Material not found: {guid}");

            if (root.TryGetProperty("effect", out var ef) && ef.GetString() is { Length: > 0 } effectGuid)
            {
                if (CommandHelpers.FindOrLoadAsset(effectGuid) is not Effect effect) return CommandResult.NotFound($"Effect not found: {effectGuid}");
                material.SetEffect(effect);
            }

            var applied = new Dictionary<string, string>();
            if (root.TryGetProperty("textures", out var texs) && texs.ValueKind == JsonValueKind.Object)
            {
                var slots = material.TextureParameters.Select(tp => tp.Name).ToHashSet(StringComparer.Ordinal);
                foreach (var kv in texs.EnumerateObject())
                {
                    if (slots.Count > 0 && !slots.Contains(kv.Name))
                        return CommandResult.BadRequest($"Material has no texture slot '{kv.Name}'. Slots: {string.Join(", ", slots)}");
                    // null or "" clears the slot (the shader then uses its default for that map)
                    if (kv.Value.ValueKind == JsonValueKind.Null || string.IsNullOrEmpty(kv.Value.GetString()))
                    {
                        material.ClearTexture(kv.Name);
                        applied[kv.Name] = null;
                        continue;
                    }
                    if (CommandHelpers.FindOrLoadAsset(kv.Value.GetString()) is not Texture tex)
                        return CommandResult.NotFound($"Texture not found for '{kv.Name}': {kv.Value}");
                    material.SetTexture(kv.Name, tex);
                    applied[kv.Name] = tex.Name;
                }
            }

            string saved = AuthoringHelpers.SaveAndReimport(material, guid);
            return CommandResult.Json(new
            {
                status = "set",
                guid,
                name = material.Name,
                effect = material.Effect?.Name,
                applied,
                slots = material.TextureParameters.Select(tp => new { tp.Name, value = (tp.Value as Asset)?.Name }).ToArray(),
                saved,
            });
        }
    }

    // ═══════════════════════════ Scene query / bulk edit ═══════════════════════════

    /// <summary>POST /api/scene/query — entities matching filters with position/rotation/scale/prefab, plus a per-name histogram.</summary>
    [CommandRoute("POST", "/api/scene/query")]
    public class SceneQueryCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            using var doc = JsonDocument.Parse(string.IsNullOrEmpty(context.Body) ? "{}" : context.Body);
            var root = doc.RootElement;
            List<Entity> matches;
            try { matches = AuthoringHelpers.Filter(root); }
            catch (Exception ex) { return CommandResult.BadRequest(ex.Message); }

            int limit = root.ValueKind == JsonValueKind.Object && root.TryGetProperty("limit", out var l) ? l.GetInt32() : 200;
            bool summaryOnly = root.ValueKind == JsonValueKind.Object && root.TryGetProperty("summaryOnly", out var so) && so.GetBoolean();

            var histogram = matches.GroupBy(e => e.Name).OrderByDescending(x => x.Count()).ToDictionary(x => x.Key, x => x.Count());
            var entities = summaryOnly ? Array.Empty<object>() : matches.Take(limit).Select(e => (object)new
            {
                id = e.Id, uid = e.UID.ToString(),
                name = e.Name,
                prefab = e.Prefab?.Guid,
                position = CommandHelpers.Vec3(e.Transform.WorldPosition),
                rotation = CommandHelpers.Quat(e.Transform.Rotation),
                scale = CommandHelpers.Vec3(e.Transform.Scale),
            }).ToArray();

            return CommandResult.Json(new { count = matches.Count, returned = entities.Length, histogram, entities });
        }
    }

    /// <summary>
    /// POST /api/scene/delete — delete by "ids" and/or the scene/query filters. "dryRun": true only reports.
    /// Refuses a filter-less call that would delete everything.
    /// </summary>
    [CommandRoute("POST", "/api/scene/delete")]
    public class SceneDeleteCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body)) return CommandResult.BadRequest("Body required");
            using var doc = context.ParseBody();
            var root = doc.RootElement;

            var targets = new List<Entity>();
            bool hasIds = root.TryGetProperty("ids", out var ids) && ids.ValueKind == JsonValueKind.Array;
            if (hasIds)
                foreach (var idEl in ids.EnumerateArray())
                    if (CommandHelpers.FindEntityById(idEl.GetInt32()) is { } e) targets.Add(e);

            bool hasFilter = new[] { "name", "component", "prefab", "area" }.Any(k => root.TryGetProperty(k, out _));
            if (hasFilter)
            {
                try { targets.AddRange(AuthoringHelpers.Filter(root)); }
                catch (Exception ex) { return CommandResult.BadRequest(ex.Message); }
            }
            if (!hasIds && !hasFilter)
                return CommandResult.BadRequest("Give 'ids' and/or a filter (name, component, prefab, area)");

            targets = targets.Distinct().ToList();
            var histogram = targets.GroupBy(e => e.Name).OrderByDescending(x => x.Count()).ToDictionary(x => x.Key, x => x.Count());
            bool dryRun = root.TryGetProperty("dryRun", out var dr) && dr.GetBoolean();
            if (!dryRun)
            {
                if (Selector.SelectedEntity != null && targets.Contains(Selector.SelectedEntity)) Selector.SelectedObject = null;
                foreach (var e in targets) e.Destroy();
                MessageDispatcher.Send(Msg.RefreshExplorer);
            }
            return CommandResult.Json(new { status = dryRun ? "dryRun" : "deleted", count = targets.Count, histogram });
        }
    }

    /// <summary>POST /api/entity/{id}/removecomponent {"type":"MeshRenderer"}</summary>
    [CommandRoute("POST", "/api/entity/{id}/removecomponent")]
    public class RemoveComponentCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null) return CommandResult.NotFound($"Entity {id} not found");
            using var doc = context.ParseBody();
            var typeName = doc.RootElement.TryGetProperty("type", out var t) ? t.GetString() : null;
            var type = CommandHelpers.FindComponentType(typeName ?? "");
            if (type == null) return CommandResult.NotFound($"Component type '{typeName}' not found");
            if (type == typeof(Transform)) return CommandResult.BadRequest("Transform cannot be removed");
            var comp = entity.Components.FirstOrDefault(c => type.IsAssignableFrom(c.GetType()));
            if (comp == null) return CommandResult.NotFound($"Entity {id} has no '{type.Name}'");
            entity.RemoveComponent(comp);
            MessageDispatcher.Send(Msg.RefreshExplorer);
            return CommandResult.Json(new { status = "removed", id, name = entity.Name, component = comp.GetType().Name });
        }
    }

    /// <summary>POST /api/prefab/measure {"guids":[...]} — local-space bounds of each prefab (instantiated off-scene and destroyed).</summary>
    [CommandRoute("POST", "/api/prefab/measure")]
    public class PrefabMeasureCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            using var doc = context.ParseBody();
            if (!doc.RootElement.TryGetProperty("guids", out var guids) || guids.ValueKind != JsonValueKind.Array)
                return CommandResult.BadRequest("'guids' array required");

            var results = new List<object>();
            foreach (var gEl in guids.EnumerateArray())
            {
                var guid = gEl.GetString();
                if (CommandHelpers.FindOrLoadAsset(guid ?? "") is not Prefab prefab)
                {
                    results.Add(new { guid, error = "not a prefab" });
                    continue;
                }
                var instance = prefab.Instantiate();
                if (instance == null)
                {
                    results.Add(new { guid, name = prefab.Name, error = "instantiate failed" });
                    continue;
                }
                instance.Flags |= EntityFlags.DontSave;
                instance.Transform.Position = Vector3.Zero;
                var bounds = CommandHelpers.WorldBounds(instance);
                instance.Destroy();
                // Broken prefabs can report a zero-size box (mesh with an empty bounding box) instead of null
                dynamic b = bounds;
                bool empty = bounds == null || (b.size.x < 1e-4f && b.size.y < 1e-4f && b.size.z < 1e-4f);
                results.Add(new { guid, name = prefab.Name, bounds, empty });
            }
            return CommandResult.Json(new { count = results.Count, results });
        }
    }
}
