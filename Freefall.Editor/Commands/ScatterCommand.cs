using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Scatter multiple entities in a circular area with collision avoidance and terrain snapping.
    /// POST /api/entity/scatter
    /// Body: { meshes: [{guid, weight?}], area: {center:{x,z}, radius}, count, spacing?,
    ///   scale?, scaleYVariance?, randomYaw?, snapToGround?, seed?,
    ///   baseRotation?: {x,y,z} (euler degrees, applied before random yaw),
    ///   height?: {min,max}, slope?: {min,max} (degrees 0-90) }
    /// </summary>
    [CommandRoute("POST", "/api/entity/scatter")]
    public class ScatterCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            // ── Parse meshes (supports both StaticMesh and Prefab GUIDs) ──
            if (!root.TryGetProperty("meshes", out var meshesProp) || meshesProp.ValueKind != JsonValueKind.Array)
                return CommandResult.BadRequest("'meshes' array required with [{guid, weight?}]");

            var meshEntries = new List<(string guid, float weight, Mesh mesh, Prefab prefab)>();
            float totalWeight = 0;
            foreach (var entry in meshesProp.EnumerateArray())
            {
                if (!entry.TryGetProperty("guid", out var guidProp))
                    return CommandResult.BadRequest("Each mesh entry must have 'guid'");

                var guid = guidProp.GetString();
                float weight = 1f;
                if (entry.TryGetProperty("weight", out var wProp))
                    weight = wProp.GetSingle();

                // Try loading as StaticMesh first, then Prefab
                Mesh mesh = null;
                Prefab prefab = null;
                try { mesh = Engine.Assets.LoadByGuid<Mesh>(guid); } catch { }

                if (mesh == null)
                {
                    try { prefab = Engine.Assets.LoadByGuid<Prefab>(guid); } catch { }
                }

                if (mesh == null && prefab == null)
                    return CommandResult.NotFound($"No StaticMesh or Prefab found for GUID: '{guid}'");

                meshEntries.Add((guid, weight, mesh, prefab));
                totalWeight += weight;
            }

            if (meshEntries.Count == 0)
                return CommandResult.BadRequest("At least one mesh/prefab required");

            // ── Parse area ──
            if (!root.TryGetProperty("area", out var areaProp))
                return CommandResult.BadRequest("'area' required with {center:{x,z}, radius}");

            if (!areaProp.TryGetProperty("center", out var centerProp))
                return CommandResult.BadRequest("'area.center' required with {x, z}");

            float cx = centerProp.TryGetProperty("x", out var cxp) ? cxp.GetSingle() : 0;
            float cz = centerProp.TryGetProperty("z", out var czp) ? czp.GetSingle() : 0;
            float radius = areaProp.TryGetProperty("radius", out var rp) ? rp.GetSingle() : 100;

            // ── Parse options ──
            int count = root.TryGetProperty("count", out var countProp) ? countProp.GetInt32() : 10;
            float spacing = root.TryGetProperty("spacing", out var spacingProp) ? spacingProp.GetSingle() : 10;
            float scaleMin = 0.8f, scaleMax = 1.2f;
            if (root.TryGetProperty("scale", out var scaleProp))
            {
                if (scaleProp.TryGetProperty("min", out var sMin)) scaleMin = sMin.GetSingle();
                if (scaleProp.TryGetProperty("max", out var sMax)) scaleMax = sMax.GetSingle();
            }
            float scaleYVariance = root.TryGetProperty("scaleYVariance", out var syvProp) ? syvProp.GetSingle() : 0;
            bool randomYaw = !root.TryGetProperty("randomYaw", out var ryProp) || ryProp.GetBoolean();
            bool snapToGround = !root.TryGetProperty("snapToGround", out var stgProp) || stgProp.GetBoolean();
            int seed = root.TryGetProperty("seed", out var seedProp) ? seedProp.GetInt32() : Environment.TickCount;

            // Base rotation offset (euler degrees) — applied before random yaw
            Quaternion baseRotation = Quaternion.Identity;
            if (root.TryGetProperty("baseRotation", out var brProp))
            {
                float brx = brProp.TryGetProperty("x", out var bxp) ? bxp.GetSingle() * MathF.PI / 180f : 0;
                float bry = brProp.TryGetProperty("y", out var byp) ? byp.GetSingle() * MathF.PI / 180f : 0;
                float brz = brProp.TryGetProperty("z", out var bzp) ? bzp.GetSingle() * MathF.PI / 180f : 0;
                baseRotation = Quaternion.CreateFromYawPitchRoll(bry, brx, brz);
            }

            // Height filter (world Y units)
            float heightMin = float.MinValue, heightMax = float.MaxValue;
            if (root.TryGetProperty("height", out var heightProp))
            {
                if (heightProp.TryGetProperty("min", out var hMin)) heightMin = hMin.GetSingle();
                if (heightProp.TryGetProperty("max", out var hMax)) heightMax = hMax.GetSingle();
            }

            // Slope filter (degrees, 0=flat, 90=vertical)
            float slopeMin = 0, slopeMax = 90;
            if (root.TryGetProperty("slope", out var slopeProp))
            {
                if (slopeProp.TryGetProperty("min", out var slMin)) slopeMin = slMin.GetSingle();
                if (slopeProp.TryGetProperty("max", out var slMax)) slopeMax = slMax.GetSingle();
            }
            bool hasHeightFilter = heightMin > float.MinValue || heightMax < float.MaxValue;
            bool hasSlopeFilter = slopeMin > 0 || slopeMax < 90;

            // ── Find terrain (needed for snapping + height/slope filters) ──
            TerrainRenderer terrain = TerrainHeightCommand.FindTerrainRenderer();

            // ── Generate positions via rejection sampling ──
            var rng = new Random(seed);
            var accepted = new List<(Vector2 pos, float height)>();
            int maxAttempts = count * 30; // enough headroom for filter rejection
            int attempts = 0;

            while (accepted.Count < count && attempts < maxAttempts)
            {
                attempts++;
                // Random point in circle
                float angle = (float)(rng.NextDouble() * Math.PI * 2);
                float r = radius * MathF.Sqrt((float)rng.NextDouble());
                float px = cx + MathF.Cos(angle) * r;
                float pz = cz + MathF.Sin(angle) * r;

                // Get height at this point
                float h = terrain?.GetHeight(new Vector3(px, 0, pz)) ?? 0;

                // Height filter
                if (hasHeightFilter && (h < heightMin || h > heightMax))
                    continue;

                // Slope filter (central-difference approximation)
                if (hasSlopeFilter && terrain != null)
                {
                    const float d = 1.0f; // 1m sample distance
                    float hL = terrain.GetHeight(new Vector3(px - d, 0, pz));
                    float hR = terrain.GetHeight(new Vector3(px + d, 0, pz));
                    float hB = terrain.GetHeight(new Vector3(px, 0, pz - d));
                    float hF = terrain.GetHeight(new Vector3(px, 0, pz + d));
                    float dx = (hR - hL) / (2 * d);
                    float dz = (hF - hB) / (2 * d);
                    float slopeDeg = MathF.Atan(MathF.Sqrt(dx * dx + dz * dz)) * (180f / MathF.PI);
                    if (slopeDeg < slopeMin || slopeDeg > slopeMax)
                        continue;
                }

                // Check spacing against all accepted points
                bool tooClose = false;
                for (int i = 0; i < accepted.Count; i++)
                {
                    var diff = new Vector2(px - accepted[i].pos.X, pz - accepted[i].pos.Y);
                    if (diff.LengthSquared() < spacing * spacing)
                    {
                        tooClose = true;
                        break;
                    }
                }

                if (!tooClose)
                    accepted.Add((new Vector2(px, pz), h));
            }

            // ── Instantiate entities ──
            var placed = new List<object>();

            foreach (var (pos, height) in accepted)
            {
                // Weighted random mesh/prefab selection
                float roll = (float)rng.NextDouble() * totalWeight;
                var selected = meshEntries[0];
                float cumulative = 0;
                foreach (var entry in meshEntries)
                {
                    cumulative += entry.weight;
                    if (roll <= cumulative)
                    {
                        selected = entry;
                        break;
                    }
                }

                // Random transform
                float scale = scaleMin + (float)rng.NextDouble() * (scaleMax - scaleMin);
                float yScale = scale * (1f + ((float)rng.NextDouble() * 2 - 1) * scaleYVariance);
                float yaw = randomYaw ? (float)(rng.NextDouble() * 360) : 0;
                float yawRad = yaw * MathF.PI / 180f;
                var rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitY, yawRad) * baseRotation;

                float y = snapToGround ? height : 0;

                Entity entity;
                if (selected.prefab != null)
                {
                    // Prefab instantiation: creates full entity hierarchy with materials
                    entity = selected.prefab.Instantiate();
                    if (entity == null) continue;
                }
                else
                {
                    // Mesh instantiation: bare entity + renderer
                    entity = new Entity(selected.mesh.Name ?? "Scatter");
                    var renderer = new MeshRenderer();
                    renderer.Mesh = selected.mesh;
                    entity.AddComponent(renderer);
                }

                entity.Transform.Position = new Vector3(pos.X, y, pos.Y);
                entity.Transform.Rotation = rotation;
                entity.Transform.Scale = new Vector3(scale, yScale, scale);

                placed.Add(new
                {
                    id = entity.Id, uid = entity.UID.ToString(),
                    name = entity.Name,
                    position = CommandHelpers.Vec3(entity.Transform.Position),
                    scale = CommandHelpers.Vec3(entity.Transform.Scale)
                });
            }

            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new
            {
                status = "scattered",
                requested = count,
                placed = placed.Count,
                attempts,
                seed,
                entities = placed
            });
        }
    }
}
