using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    /// <summary>
    /// prefab_inspect: what a prefab is before placing it — size, door sides and floor, whether its lights and
    /// particle emitters actually work, and its notable parts. Replaces spawning probes at y=300 and reading
    /// children by hand (door sides, "LIT" prefabs whose light or fire components were empty).
    /// </summary>
    [McpServerToolType]
    public static class PrefabTools
    {
        private static readonly string[] NotableWords =
            ["door", "light", "lamp", "lantern", "torch", "fire", "candle", "chimney", "sign", "seat", "bench"];

        [McpServerTool(Name = "prefab_inspect", Title = "Inspect prefabs", ReadOnly = true, OpenWorld = false)]
        [Description("Inspect prefabs without placing them. Returns: local bounds (pivot at the origin), doors (side ±X/±Z of the " +
                     "footprint and whether they are at ground level — use for street-facing buildings), lights and particle " +
                     "emitters with a 'works' verdict, a component histogram, notable parts (lights, fire, chimneys, signs, ...) " +
                     "and a 'problems' list (renders nothing, meshless renderers, dead lights/emitters). " +
                     "Accepts GUIDs or '@Prefab Name'.")]
        public static async Task<CallToolResult> Inspect(
            [Description("Prefab GUIDs or '@Prefab Name' references")] string[] prefabs,
            [Description("Max notable parts listed per prefab")] int maxParts = 25)
        {
            var guids = new List<(string input, string? guid)>();
            foreach (var p in prefabs)
            {
                if (!p.StartsWith('@')) { guids.Add((p, p)); continue; }
                var res = await McpBridge.Get("/api/assets/resolve", ("name", p[1..]), ("type", "Prefab"));
                var g = res.IsError == true ? null : JsonNode.Parse(EditorTools.TextOf(res))?["guid"]?.GetValue<string>();
                guids.Add((p, string.IsNullOrEmpty(g) ? null : g));
            }

            var results = await EditorCommandServer.Instance.RunOnMainThread(() =>
            {
                var arr = new JsonArray();
                foreach (var (input, guid) in guids)
                    arr.Add(guid == null ? new JsonObject { ["input"] = input, ["error"] = "asset not found" } : InspectOne(guid, maxParts));
                return arr;
            });
            return EditorTools.Text(new JsonObject { ["count"] = results.Count, ["results"] = results });
        }

        private static JsonNode InspectOne(string guid, int maxParts)
        {
            if (Commands.CommandHelpers.FindOrLoadAsset(guid) is not Prefab prefab)
                return new JsonObject { ["guid"] = guid, ["error"] = "not a prefab" };

            var root = prefab.Instantiate();
            if (root == null) return new JsonObject { ["guid"] = guid, ["name"] = prefab.Name, ["error"] = "instantiate failed" };
            root.Flags |= EntityFlags.DontSave;
            root.Transform.Position = Vector3.Zero;

            try
            {
                var problems = new JsonArray();
                var nodes = new List<(Transform t, string path)>();
                void Walk(Transform t, string path)
                {
                    nodes.Add((t, path));
                    for (int i = 0; i < t.Count; i++)
                        if (t.GetChild(i) is { } c) Walk(c, path + "/" + c.Entity.Name);
                }
                Walk(root.Transform, root.Name);

                var (bMin, bMax, any) = MeshBounds(root.Transform);
                if (!any || Vector3.Distance(bMin, bMax) < 1e-3f) problems.Add("renders nothing (no mesh bounds)");

                var histogram = new JsonObject();
                foreach (var g in nodes.SelectMany(n => n.t.Entity.Components)
                                       .Where(c => c is not Transform)
                                       .GroupBy(c => c.GetType().Name)
                                       .OrderByDescending(g => g.Count()))
                    histogram[g.Key] = g.Count();

                var meshless = nodes.Where(n => n.t.Entity.Components.OfType<MeshRenderer>().Any(r => r.Mesh == null))
                                    .Select(n => n.path).ToList();
                if (meshless.Count > 0)
                    problems.Add($"{meshless.Count} MeshRenderer(s) without a mesh, e.g. {string.Join(", ", meshless.Take(3))}");

                // Doors: outermost nodes named *door* (skip door handles/panels below a door node).
                var doors = new JsonArray();
                foreach (var (t, path) in nodes)
                {
                    if (!IsDoor(t) || (t.Parent != null && IsDoor(t.Parent))) continue;
                    var (dMin, dMax, dAny) = MeshBounds(t);
                    var centre = dAny ? (dMin + dMax) * 0.5f : t.WorldPosition;
                    float bottom = dAny ? dMin.Y : t.WorldPosition.Y;
                    float aboveBase = bottom - (any ? bMin.Y : 0f);
                    doors.Add(new JsonObject
                    {
                        ["path"] = path,
                        ["position"] = V(centre),
                        ["side"] = any ? Side(centre, bMin, bMax) : null,
                        ["bottomY"] = R(bottom),
                        ["groundLevel"] = aboveBase < 1.5f,
                    });
                }
                if (doors.Count > 0 && !doors.Any(d => d!["groundLevel"]!.GetValue<bool>()))
                    problems.Add("no ground-level door (only upper-floor doors) — not for street fronts");

                var lights = new JsonArray();
                foreach (var (t, path) in nodes)
                    foreach (var l in t.Entity.Components.OfType<PointLight>())
                    {
                        bool works = l.Enabled && l.Intensity > 0f && l.Range > 0f;
                        lights.Add(new JsonObject
                        {
                            ["path"] = path, ["position"] = V(t.WorldPosition), ["intensity"] = R(l.Intensity),
                            ["range"] = R(l.Range), ["color"] = new JsonArray(R(l.Color.R), R(l.Color.G), R(l.Color.B)),
                            ["works"] = works,
                        });
                        if (!works) problems.Add($"dead light at {path}");
                    }

                var emitters = new JsonArray();
                foreach (var (t, path) in nodes)
                    foreach (var e in t.Entity.Components.OfType<ParticleEmitter>())
                    {
                        bool works = e.Enabled && e.ParticleTexture != null && e.EmitRate > 0f;
                        emitters.Add(new JsonObject
                        {
                            ["path"] = path, ["position"] = V(t.WorldPosition), ["texture"] = e.ParticleTexture?.Name,
                            ["emitRate"] = R(e.EmitRate), ["works"] = works,
                        });
                        if (!works) problems.Add($"dead emitter at {path} (no texture or zero rate — empty component?)");
                    }

                // Converted packs lose components: a "Point Light" / "Fire" node with nothing but a Transform.
                foreach (var (t, path) in nodes.Skip(1))
                {
                    var name = t.Entity.Name;
                    bool effectName = new[] { "light", "fire", "flame", "smoke", "spark" }
                        .Any(w => name.Contains(w, StringComparison.OrdinalIgnoreCase));
                    if (effectName && t.Entity.Components.All(c => c is Transform) && t.Count == 0)
                        problems.Add($"empty part '{path}' (no components — lost in conversion?)");
                }

                bool namedLit = prefab.Name.Contains("LIT", StringComparison.Ordinal); // "(LIT)" packs
                if (namedLit && !lights.Any(l => l!["works"]!.GetValue<bool>()))
                    problems.Add("named LIT but has no working light");

                var parts = new JsonArray();
                foreach (var (t, path) in nodes.Skip(1))
                {
                    if (parts.Count >= maxParts) break;
                    var name = t.Entity.Name;
                    if (IsDoor(t)) continue; // already under 'doors'
                    bool notable = NotableWords.Any(w => name.Contains(w, StringComparison.OrdinalIgnoreCase))
                                   || t.Entity.Components.Any(c => c is not Transform and not MeshRenderer);
                    if (!notable) continue;
                    parts.Add(new JsonObject
                    {
                        ["path"] = path, ["position"] = V(t.WorldPosition),
                        ["components"] = new JsonArray(t.Entity.Components.Where(c => c is not Transform)
                                                         .Select(c => (JsonNode)c.GetType().Name).ToArray()),
                    });
                }

                return new JsonObject
                {
                    ["guid"] = guid,
                    ["name"] = prefab.Name,
                    ["bounds"] = any ? new JsonObject { ["min"] = V(bMin), ["max"] = V(bMax), ["size"] = V(bMax - bMin) } : null,
                    ["nodeCount"] = nodes.Count,
                    ["components"] = histogram,
                    ["doors"] = doors,
                    ["lights"] = lights,
                    ["emitters"] = emitters,
                    ["notableParts"] = parts,
                    ["problems"] = problems,
                };
            }
            finally
            {
                root.Destroy();
            }
        }

        private static bool IsDoor(Transform t) =>
            t.Entity.Name.Contains("door", StringComparison.OrdinalIgnoreCase)
            && !t.Entity.Name.Contains("frame", StringComparison.OrdinalIgnoreCase);

        /// <summary>Footprint face nearest to the point: which side of the building it faces.</summary>
        private static string Side(Vector3 p, Vector3 min, Vector3 max)
        {
            var sides = new (string name, float dist)[]
            {
                ("-X", p.X - min.X), ("+X", max.X - p.X), ("-Z", p.Z - min.Z), ("+Z", max.Z - p.Z),
            };
            return sides.MinBy(s => s.dist).name;
        }

        /// <summary>Bounds of every mesh at and below <paramref name="t"/>, in world space (the prefab root sits at the origin).</summary>
        private static (Vector3 min, Vector3 max, bool any) MeshBounds(Transform t)
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            bool any = false;
            void Visit(Transform n)
            {
                foreach (var r in n.Entity.Components.OfType<MeshRenderer>())
                {
                    if (r.Mesh == null) continue;
                    var bb = r.Mesh.BoundingBox;
                    var world = n.WorldMatrix;
                    for (int i = 0; i < 8; i++)
                    {
                        var corner = new Vector3((i & 1) != 0 ? bb.Max.X : bb.Min.X,
                                                 (i & 2) != 0 ? bb.Max.Y : bb.Min.Y,
                                                 (i & 4) != 0 ? bb.Max.Z : bb.Min.Z);
                        var p = Vector3.Transform(corner, world);
                        min = Vector3.Min(min, p);
                        max = Vector3.Max(max, p);
                        any = true;
                    }
                }
                for (int i = 0; i < n.Count; i++)
                    if (n.GetChild(i) is { } c) Visit(c);
            }
            Visit(t);
            return (min, max, any);
        }

        private static float R(float v) => MathF.Round(v, 3);
        private static JsonArray V(Vector3 v) => new(R(v.X), R(v.Y), R(v.Z));
    }
}
