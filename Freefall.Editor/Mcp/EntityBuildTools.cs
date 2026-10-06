using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [Description("One component to add: its type name and member values (same encodings as entity_set_properties; " +
                 "strings starting with '@' are asset names resolved to GUIDs, e.g. \"@Stone 1\")")]
    public record ComponentSpec(string Type, JsonObject? Values = null);

    /// <summary>
    /// entity_build: an entity with a whole component set in one call. Scene elements are compositions
    /// (Spline + MeshRenderer + RuntimeMesh + stamps + PCG); building them through entity_create / add_component /
    /// set_properties took ~7 calls each and left half-built entities behind when one step failed.
    /// </summary>
    [McpServerToolType]
    public static class EntityBuildTools
    {
        [McpServerTool(Name = "entity_build", Title = "Build entity with components", OpenWorld = false)]
        [Description("Create an entity with several components in one call (instead of entity_create + entity_add_component + " +
                     "entity_set_properties per component): transform, optional parent, optional prefab to start from, and a " +
                     "component list [{type, values}]. Spline is added first, MeshRenderer before RuntimeMesh and PCGComponent last " +
                     "automatically; the PCG runs afterwards. '@Name' strings in values resolve to asset GUIDs. " +
                     "Spline.Points are WORLD positions: the entity pivot is placed at their centre (XZ centre of their bounds, " +
                     "average Y) — or at 'position' if given — and the points are stored relative to it, so the element can be " +
                     "moved/rotated later by its transform. For flat RuntimeMesh streets give the points at ground height. " +
                     "If any step fails the entity is removed again and the error names the failing member.")]
        public static async Task<CallToolResult> BuildEntity(
            string name,
            ComponentSpec[]? components = null,
            [Description("Prefab GUID or '@Prefab Name' to instantiate instead of an empty entity")] string? prefab = null,
            [Description("Entity position (world, or local to 'parent'). With a Spline this is the pivot; omit to use the spline centre.")] Vec3? position = null,
            [Description("Euler degrees (x=pitch, y=yaw, z=roll)")] Vec3? rotationEuler = null,
            Vec3? scale = null,
            [Description("Parent entity id; transform values are local to it (spline points are then local to the parent)")] int? parent = null,
            [Description("Run the entity's PCGComponent after building")] bool runPcg = true,
            [Description("Parent by persistent UID (string) instead of id")] string? parentUid = null)
        {
            var (parentId, parentErr) = await EntityRefs.Resolve(parent, parentUid, required: false);
            if (parentErr != null) return parentErr;
            parent = parentId;

            var comps = (components ?? Array.Empty<ComponentSpec>())
                .Select(c => new ComponentSpec(c.Type, c.Values?.DeepClone().AsObject() ?? new JsonObject()))
                .ToList();

            // World-space spline points → pivot near the spline, points relative to it.
            var splineSpec = comps.FirstOrDefault(c => c.Type.Equals("Spline", StringComparison.OrdinalIgnoreCase));
            if (splineSpec?.Values!["Points"] is JsonArray worldPts && worldPts.Count > 0)
            {
                List<Vector3> pts;
                try { pts = worldPts.Select(ReadVec3).ToList(); }
                catch (Exception ex) { return McpBridge.Error($"Spline.Points: {ex.Message}"); }

                var pivot = position != null ? new Vector3(position.X, position.Y, position.Z) : Centre(pts);
                position = new Vec3(pivot.X, pivot.Y, pivot.Z);

                var inv = Quaternion.Inverse(ToQuat(FromEuler(rotationEuler)));
                var s = scale == null ? Vector3.One : new Vector3(scale.X, scale.Y, scale.Z);
                splineSpec.Values["Points"] = new JsonArray(pts
                    .Select(p => Vector3.Transform(p - pivot, inv) / s)
                    .Select(p => (JsonNode)new JsonArray(R(p.X), R(p.Y), R(p.Z)))
                    .ToArray());
            }

            // Resolve '@Asset' names up front so a typo fails before anything is created.
            var cache = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                foreach (var c in comps) await ResolveAssets(c.Values!, cache);
                if (prefab != null && prefab.StartsWith('@')) prefab = await ResolveName(prefab[1..], cache, "Prefab");
            }
            catch (Exception ex) { return McpBridge.Error(ex.Message); }

            // Order: Spline first (others read it when added), MeshRenderer before RuntimeMesh, PCG last.
            static int Rank(string t) => t.ToLowerInvariant() switch
            {
                "spline" => 0, "meshrenderer" => 1, "runtimemesh" => 3, "pcgcomponent" => 4, _ => 2,
            };
            comps = comps.Select((c, i) => (c, i)).OrderBy(x => Rank(x.c.Type)).ThenBy(x => x.i).Select(x => x.c).ToList();

            // 1. Entity
            var created = prefab != null
                ? await McpBridge.Post("/api/entity/instantiate", new { guid = prefab, name, snapToGround = false })
                : await McpBridge.Post("/api/entity/create", new { name });
            if (created.IsError == true) return created;
            var createdJson = JsonNode.Parse(EditorTools.TextOf(created))!;
            int id = createdJson["id"]!.GetValue<int>();
            var uid = createdJson["uid"]?.GetValue<string>();

            var log = new JsonArray();
            static async Task<string?> Step(string what, Task<CallToolResult> call)
            {
                var res = await call;
                return res.IsError == true ? $"{what}: {EditorTools.TextOf(res)}" : null;
            }

            // 2. Parent before transform, so the transform is local to the parent.
            if (parent != null)
            {
                var err = await Step("setparent", McpBridge.Post($"/api/entity/{id}/setparent", $"{{\"parent\":{parent}}}"));
                if (err != null) return await Fail(id, err, log);
            }
            if (position != null || rotationEuler != null || scale != null)
            {
                var err = await Step("transform", McpBridge.Post($"/api/entity/{id}/transform", new
                {
                    position, scale, rotation = FromEuler(rotationEuler),
                }));
                if (err != null) return await Fail(id, err, log);
            }

            // Every value set below invalidates the PCG; hold it so it runs once, on the finished entity.
            JsonNode? pcg = null;
            Freefall.PCG.PCGScheduler.BeginHold();
            try
            {
                // 3. Components
                foreach (var c in comps)
                {
                    var add = await McpBridge.Post($"/api/entity/{id}/addcomponent", new { type = c.Type });
                    // Prefab instances may already carry the component; then only its values are set.
                    if (add.IsError == true && !EditorTools.TextOf(add).Contains("already", StringComparison.OrdinalIgnoreCase))
                        return await Fail(id, $"add {c.Type}: {EditorTools.TextOf(add)}", log);

                    foreach (var (member, value) in c.Values!)
                    {
                        var body = new JsonObject { ["component"] = c.Type, ["property"] = member, ["value"] = value?.DeepClone() };
                        var err = await Step($"{c.Type}.{member}", McpBridge.Post($"/api/entity/{id}/setproperty", body.ToJsonString()));
                        if (err != null) return await Fail(id, err, log);
                    }
                    log.Add($"{c.Type} ({c.Values.Count} values)");
                }

                // 4. PCG
                if (runPcg && comps.Any(c => c.Type.Equals("PCGComponent", StringComparison.OrdinalIgnoreCase)))
                {
                    var run = await McpBridge.Post("/api/pcg/execute", new { id });
                    pcg = run.IsError == true ? EditorTools.TextOf(run) : JsonNode.Parse(EditorTools.TextOf(run));
                }
            }
            finally
            {
                Freefall.PCG.PCGScheduler.EndHold();
            }

            var detail = await McpBridge.Get($"/api/scene/entity/{id}");
            var bounds = detail.IsError == true ? null : JsonNode.Parse(EditorTools.TextOf(detail))?["bounds"]?.DeepClone();

            return EditorTools.Text(new JsonObject
            {
                ["status"] = "built",
                ["id"] = id,
                ["uid"] = uid,
                ["name"] = name,
                ["components"] = log,
                ["bounds"] = bounds,
                ["pcg"] = pcg,
            });
        }

        [McpServerTool(Name = "spline_recenter", Title = "Center spline pivots", Destructive = true, OpenWorld = false)]
        [Description("Spline.CenterPivot on one or more entities: move each entity's pivot to the centre of its spline points " +
                     "(XZ bounds centre, average height; a Flat RuntimeMesh keeps its height) without moving the spline or " +
                     "its hand-made children. Pass 'ids'/'uids', or " +
                     "all=true for every Spline in the scene — e.g. to fix splines authored with the entity at the origin. " +
                     "Same as the 'Center Pivot' button in the Spline inspector.")]
        public static async Task<CallToolResult> RecenterSplines(
            int[]? ids = null,
            [Description("Every Spline component in the scene")] bool all = false,
            [Description("Skip splines whose pivot is already within this distance of the centre (m)")] float tolerance = 0.5f,
            [Description("Persistent entity UIDs (strings), in addition to 'ids'")] string[]? uids = null)
        {
            if (ids == null && uids == null && !all) return McpBridge.Error("Pass 'ids', 'uids' or all=true.");
            var (resolved, idErr) = await EntityRefs.ResolveMany(ids, uids);
            if (idErr != null) return idErr;
            var targetIds = resolved!;

            var server = EditorCommandServer.Instance;
            var report = await server.RunOnMainThread(() =>
            {
                IEnumerable<Freefall.Components.Spline> splines = all
                    ? Freefall.Base.ComponentCache<Freefall.Components.Spline>.All.OfType<Freefall.Components.Spline>().ToList()
                    : targetIds.Select(Commands.CommandHelpers.FindEntityById)
                          .Select(e => e?.GetComponent<Freefall.Components.Spline>())
                          .Where(s => s != null)!;

                var moved = new JsonArray();
                int unchanged = 0;
                foreach (var s in splines)
                {
                    if (s.Entity == null || (s.Entity.Flags & Freefall.Base.EntityFlags.DontSave) != 0) continue; // generated
                    float shift = s.CenterPivot(tolerance);
                    if (shift <= 0f) { unchanged++; continue; }
                    var p = s.Transform.Position;
                    moved.Add(new JsonObject
                    {
                        ["id"] = s.Entity.Id, ["uid"] = s.Entity.UID.ToString(), ["name"] = s.Entity.Name, ["shift"] = MathF.Round(shift, 1),
                        ["pivot"] = new JsonArray(R(p.X), R(p.Y), R(p.Z)),
                    });
                }
                return new JsonObject { ["recentered"] = moved.Count, ["alreadyCentred"] = unchanged, ["entities"] = moved };
            });
            return EditorTools.Text(report);
        }

        // XZ centre of the points' bounds, average Y.
        private static Vector3 Centre(List<Vector3> pts)
        {
            float minX = pts.Min(p => p.X), maxX = pts.Max(p => p.X);
            float minZ = pts.Min(p => p.Z), maxZ = pts.Max(p => p.Z);
            return new Vector3((minX + maxX) * 0.5f, pts.Average(p => p.Y), (minZ + maxZ) * 0.5f);
        }

        private static Vector3 ReadVec3(JsonNode? n) => n switch
        {
            JsonArray a when a.Count >= 3 => new Vector3(a[0]!.GetValue<float>(), a[1]!.GetValue<float>(), a[2]!.GetValue<float>()),
            JsonObject o => new Vector3(F(o, "x"), F(o, "y"), F(o, "z")),
            _ => throw new FormatException("expected [x,y,z] or {x,y,z}"),
        };

        private static float F(JsonNode o, string key) => o[key]?.GetValue<float>() ?? o[key.ToUpperInvariant()]?.GetValue<float>() ?? 0f;

        private static float R(float v) => MathF.Round(v, 3);

        private static Quaternion ToQuat(Quat? q) => q == null ? Quaternion.Identity : new Quaternion(q.X, q.Y, q.Z, q.W);

        /// <summary>A half-built entity is worse than none: delete it and report what failed.</summary>
        private static async Task<CallToolResult> Fail(int id, string error, JsonArray log)
        {
            await McpBridge.Post($"/api/entity/{id}/delete");
            return McpBridge.Error($"Build failed and the entity was removed. {error}" +
                                   (log.Count > 0 ? $"\nCompleted before the failure: {log.ToJsonString()}" : ""));
        }

        private static async Task ResolveAssets(JsonNode node, Dictionary<string, string> cache)
        {
            switch (node)
            {
                case JsonObject obj:
                    foreach (var key in obj.Select(p => p.Key).ToList())
                    {
                        if (obj[key] is JsonValue v && v.TryGetValue<string>(out var s) && s.StartsWith('@'))
                            obj[key] = await ResolveName(s[1..], cache);
                        else if (obj[key] != null)
                            await ResolveAssets(obj[key]!, cache);
                    }
                    break;
                case JsonArray arr:
                    for (int i = 0; i < arr.Count; i++)
                    {
                        if (arr[i] is JsonValue v && v.TryGetValue<string>(out var s) && s.StartsWith('@'))
                            arr[i] = await ResolveName(s[1..], cache);
                        else if (arr[i] != null)
                            await ResolveAssets(arr[i]!, cache);
                    }
                    break;
            }
        }

        private static async Task<string> ResolveName(string assetName, Dictionary<string, string> cache, string? type = null)
        {
            var key = type + "|" + assetName;
            if (cache.TryGetValue(key, out var guid)) return guid;
            var res = type == null
                ? await McpBridge.Get("/api/assets/resolve", ("name", assetName))
                : await McpBridge.Get("/api/assets/resolve", ("name", assetName), ("type", type));
            var g = res.IsError == true ? null : JsonNode.Parse(EditorTools.TextOf(res))?["guid"]?.GetValue<string>();
            if (string.IsNullOrEmpty(g))
                throw new InvalidOperationException($"Asset '@{assetName}' not found (asset_search finds the exact name).");
            return cache[key] = g;
        }

        private static Quat? FromEuler(Vec3? degrees)
        {
            if (degrees == null) return null;
            const float d2r = MathF.PI / 180f;
            var q = Quaternion.CreateFromYawPitchRoll(degrees.Y * d2r, degrees.X * d2r, degrees.Z * d2r);
            return new Quat(q.X, q.Y, q.Z, q.W);
        }
    }
}
