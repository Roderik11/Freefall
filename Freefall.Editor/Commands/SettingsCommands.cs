using System;
using System.Collections.Generic;
using System.Reflection;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/settings")]
    public class GetSettingsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var settings = Engine.Settings;
            var result = new Dictionary<string, object>();

            foreach (var prop in typeof(EngineSettings).GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                if (!prop.CanRead) continue;
                // Skip the FrozenViewProjection matrix — not useful via API
                if (prop.PropertyType == typeof(System.Numerics.Matrix4x4)) continue;

                try
                {
                    var val = prop.GetValue(settings);
                    result[ToCamelCase(prop.Name)] = val is Enum e ? e.ToString() : val;
                }
                catch { result[ToCamelCase(prop.Name)] = "<error>"; }
            }

            return CommandResult.Json(result);
        }

        private static string ToCamelCase(string s) =>
            string.IsNullOrEmpty(s) ? s : char.ToLowerInvariant(s[0]) + s.Substring(1);
    }

    [CommandRoute("POST", "/api/settings")]
    public class SetSettingsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with setting key-value pairs");

            using var doc = context.ParseBody();
            var root = doc.RootElement;
            var settings = Engine.Settings;
            var changed = new Dictionary<string, object>();

            foreach (var prop in typeof(EngineSettings).GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                if (!prop.CanWrite) continue;

                // Try camelCase and PascalCase
                var camel = char.ToLowerInvariant(prop.Name[0]) + prop.Name.Substring(1);
                System.Text.Json.JsonElement val;
                if (!root.TryGetProperty(camel, out val) && !root.TryGetProperty(prop.Name, out val))
                    continue;

                try
                {
                    object converted;
                    if (prop.PropertyType == typeof(bool))
                        converted = val.GetBoolean();
                    else if (prop.PropertyType == typeof(int))
                        converted = val.GetInt32();
                    else if (prop.PropertyType == typeof(float))
                        converted = val.GetSingle();
                    else if (prop.PropertyType.IsEnum)
                        converted = Enum.Parse(prop.PropertyType, val.GetString(), ignoreCase: true);
                    else
                        continue;

                    prop.SetValue(settings, converted);
                    changed[camel] = converted is Enum e ? e.ToString() : converted;
                }
                catch (Exception ex)
                {
                    changed[camel] = $"<error: {ex.Message}>";
                }
            }

            // Persist immediately (also saved on exit) so the change survives a crash or forced restart
            if (changed.Count > 0) EditorPreferences.Save();

            return CommandResult.Json(new { status = "updated", changed });
        }
    }

    [CommandRoute("GET", "/api/debug/stats")]
    public class GetDebugStatsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            return CommandResult.Json(new
            {
                fps = (int)(1f / Math.Max(Time.Delta, 0.001)),
                deltaMs = Time.DeltaMilliseconds,
                entityCount = EntityManager.Entities.Count,
                frameIndex = Engine.FrameIndex,
                // Frames since start: two readings a few seconds apart give an averaged frame time
                tickCount = Engine.TickCount,
                batchCount = CommandBuffer.LastBatchCount,
                drawCallCount = CommandBuffer.LastDrawCallCount,
                visibleCount = CommandBuffer.Culler?.LastVisibleCount ?? 0,
                occludedCount = CommandBuffer.Culler?.LastHiZOccludedCount ?? 0,
                grassInstanceCount = Freefall.Components.TerrainRenderer.LastInstanceCount,
                meshInstanceCount = Freefall.Components.TerrainRenderer.LastMeshInstanceCount,
                meshDrawCount = Freefall.Components.TerrainRenderer.LastMeshDrawCount,
                meshRegistryCount = MeshRegistry.Count, // sub-batch ids: GPU culler buffers hold MaxSubBatches (4096)
                terrainRebakeRequests = Freefall.Components.TerrainRenderer.RebakeRequestCount,
                pcgExecutions = Freefall.Components.PCGComponent.ExecuteCount,
                // Shared particle pool: emitters, slots handed out to them, slots the GPU pool holds
                particleEmitters = Freefall.Base.Systems.Get<ParticleSystem>()?.EmitterCount ?? 0,
                particleSlots = Freefall.Base.Systems.Get<ParticleSystem>()?.AllocatedSlots ?? 0,
                particlePoolSlots = Freefall.Base.Systems.Get<ParticleSystem>()?.PoolSlots ?? 0,
                // System tree in update order, one "Name  ms" line per system (indent = nesting)
                systems = Freefall.Base.Systems.Describe().Split('\n', StringSplitOptions.RemoveEmptyEntries)
            });
        }
    }
}
