using System;
using System.Linq;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// One-shot entity creation from a StaticMesh GUID or Prefab GUID.
    /// - mesh: Creates entity → adds StaticMeshRenderer → loads mesh → sets transform.
    /// - prefab: Loads Prefab → calls Instantiate() → sets transform.
    /// </summary>
    [CommandRoute("POST", "/api/entity/instantiate")]
    public class InstantiateCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'mesh' or 'prefab' (GUID)");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            // ── Prefab path ──
            if (root.TryGetProperty("prefab", out var prefabProp))
            {
                var prefabGuid = prefabProp.GetString();
                if (string.IsNullOrEmpty(prefabGuid))
                    return CommandResult.BadRequest("'prefab' must be a non-empty GUID string");

                Prefab prefab;
                try
                {
                    prefab = Engine.Assets.LoadByGuid<Prefab>(prefabGuid);
                }
                catch (Exception ex)
                {
                    return CommandResult.Error(500, $"Failed to load Prefab '{prefabGuid}': {ex.Message}");
                }

                if (prefab == null)
                    return CommandResult.NotFound($"Prefab not found for GUID '{prefabGuid}'");

                var entity = prefab.Instantiate();
                if (entity == null)
                    return CommandResult.Error(500, $"Failed to instantiate prefab '{prefab.Name}'");

                // Override name
                if (root.TryGetProperty("name", out var nameProp2))
                    entity.Name = nameProp2.GetString();

                // Set transform
                if (root.TryGetProperty("position", out var posProp2))
                    entity.Transform.Position = CommandHelpers.ParseVec3(posProp2);
                if (root.TryGetProperty("rotation", out var rotProp2))
                    entity.Transform.Rotation = CommandHelpers.ParseQuat(rotProp2);
                if (root.TryGetProperty("scale", out var scaleProp2))
                    entity.Transform.Scale = CommandHelpers.ParseVec3(scaleProp2);

                // Snap to ground
                bool snap = true;
                if (root.TryGetProperty("snapToGround", out var snapProp2))
                    snap = snapProp2.GetBoolean();

                if (snap)
                {
                    var terrain = TerrainHeightCommand.FindTerrainRenderer();
                    if (terrain != null)
                    {
                        var pos = entity.Transform.Position;
                        pos.Y = terrain.GetHeight(new System.Numerics.Vector3(pos.X, 0, pos.Z));
                        entity.Transform.Position = pos;
                    }
                }

                MessageDispatcher.Send(Msg.RefreshExplorer);

                // Gather mesh info for diagnostics
                var mrComp = entity.GetComponent<MeshRenderer>();
                var meshInfo = mrComp?.Mesh != null ? new
                {
                    name = mrComp.Mesh.Name,
                    partCount = mrComp.Mesh.MeshParts?.Count ?? 0,
                    lodCount = mrComp.Mesh.LODs?.Count ?? 0,
                    parts = mrComp.Mesh.MeshParts?.Select(p => p.Name).ToArray()
                } : null;

                return CommandResult.Json(new
                {
                    status = "instantiated",
                    source = "prefab",
                    id = entity.Id,
                    name = entity.Name,
                    prefab = new { name = prefab.Name, guid = prefab.Guid },
                    position = CommandHelpers.Vec3(entity.Transform.Position),
                    components = entity.Components.Select(c => c.GetType().Name).ToArray(),
                    mesh = meshInfo
                });
            }

            // ── StaticMesh path (original) ──
            // Also handles unified 'guid' that auto-detects asset type
            string meshGuid = null;
            if (root.TryGetProperty("mesh", out var meshProp))
                meshGuid = meshProp.GetString();
            else if (root.TryGetProperty("guid", out var guidProp))
            {
                // Unified: try StaticMesh first, fall back to Prefab
                var testGuid = guidProp.GetString();
                Mesh testMesh = null;
                try { testMesh = Engine.Assets.LoadByGuid<Mesh>(testGuid); } catch { }

                if (testMesh != null)
                {
                    meshGuid = testGuid;
                }
                else
                {
                    // Redirect to prefab path
                    Prefab testPrefab = null;
                    try { testPrefab = Engine.Assets.LoadByGuid<Prefab>(testGuid); } catch { }
                    if (testPrefab != null)
                    {
                        var pEntity = testPrefab.Instantiate();
                        if (pEntity == null)
                            return CommandResult.Error(500, $"Failed to instantiate prefab '{testPrefab.Name}'");

                        if (root.TryGetProperty("name", out var np)) pEntity.Name = np.GetString();
                        if (root.TryGetProperty("position", out var pp)) pEntity.Transform.Position = CommandHelpers.ParseVec3(pp);
                        if (root.TryGetProperty("rotation", out var rp2)) pEntity.Transform.Rotation = CommandHelpers.ParseQuat(rp2);
                        if (root.TryGetProperty("scale", out var sp2)) pEntity.Transform.Scale = CommandHelpers.ParseVec3(sp2);

                        bool snapP = !root.TryGetProperty("snapToGround", out var stgP) || stgP.GetBoolean();
                        if (snapP)
                        {
                            var t = TerrainHeightCommand.FindTerrainRenderer();
                            if (t != null)
                            {
                                var p = pEntity.Transform.Position;
                                p.Y = t.GetHeight(new System.Numerics.Vector3(p.X, 0, p.Z));
                                pEntity.Transform.Position = p;
                            }
                        }

                        MessageDispatcher.Send(Msg.RefreshExplorer);
                        var mrC = pEntity.GetComponent<MeshRenderer>();
                        return CommandResult.Json(new
                        {
                            status = "instantiated", source = "prefab (auto)",
                            id = pEntity.Id, name = pEntity.Name,
                            prefab = new { name = testPrefab.Name, guid = testPrefab.Guid },
                            position = CommandHelpers.Vec3(pEntity.Transform.Position),
                            components = pEntity.Components.Select(c => c.GetType().Name).ToArray()
                        });
                    }

                    return CommandResult.NotFound($"No StaticMesh or Prefab found for GUID '{testGuid}'");
                }
            }

            if (string.IsNullOrEmpty(meshGuid))
                return CommandResult.BadRequest("Body must contain 'mesh', 'prefab', or 'guid'");

            // Load the StaticMesh asset
            Mesh mesh;
            try
            {
                mesh = Engine.Assets.LoadByGuid<Mesh>(meshGuid);
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to load Mesh '{meshGuid}': {ex.Message}");
            }

            if (mesh == null)
                return CommandResult.NotFound($"Mesh not found for GUID '{meshGuid}'");

            // Determine entity name
            var name = mesh.Name ?? "Entity";
            if (root.TryGetProperty("name", out var nameProp))
                name = nameProp.GetString();

            // Create entity
            var entity2 = new Entity(name);

            // Set transform
            if (root.TryGetProperty("position", out var posProp))
                entity2.Transform.Position = CommandHelpers.ParseVec3(posProp);
            if (root.TryGetProperty("rotation", out var rotProp))
                entity2.Transform.Rotation = CommandHelpers.ParseQuat(rotProp);
            if (root.TryGetProperty("scale", out var scaleProp))
                entity2.Transform.Scale = CommandHelpers.ParseVec3(scaleProp);

            // Snap to ground: default true unless explicitly set to false
            bool snapToGround = true;
            if (root.TryGetProperty("snapToGround", out var snapProp))
                snapToGround = snapProp.GetBoolean();

            if (snapToGround)
            {
                var terrain = TerrainHeightCommand.FindTerrainRenderer();
                if (terrain != null)
                {
                    var pos = entity2.Transform.Position;
                    pos.Y = terrain.GetHeight(new System.Numerics.Vector3(pos.X, 0, pos.Z));
                    entity2.Transform.Position = pos;
                }
            }

            // Add MeshRenderer with the mesh
            var renderer = new MeshRenderer();
            renderer.Mesh = mesh;
            entity2.AddComponent(renderer);

            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new
            {
                status = "instantiated",
                source = "staticmesh",
                id = entity2.Id,
                name = entity2.Name,
                mesh = new { name = mesh.Name, guid = mesh.Guid },
                position = CommandHelpers.Vec3(entity2.Transform.Position),
                bounds = mesh!= null ? CommandHelpers.BBox(mesh.BoundingBox) : null,
                components = entity2.Components.Select(c => c.GetType().Name).ToArray()
            });
        }
    }

    /// <summary>
    /// Update prefab instances in the scene.
    /// POST /api/prefab/update
    /// Body: { guid: "prefab-guid" } — updates all instances
    ///   OR: { id: 123 } — updates single entity
    /// </summary>
    [CommandRoute("POST", "/api/prefab/update")]
    public class PrefabUpdateCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'guid' or 'id'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            // Update all instances by prefab GUID
            if (root.TryGetProperty("guid", out var guidProp))
            {
                var guid = guidProp.GetString();
                Prefab prefab;
                try
                {
                    prefab = Engine.Assets.LoadByGuid<Prefab>(guid);
                }
                catch (Exception ex)
                {
                    return CommandResult.Error(500, $"Failed to load Prefab '{guid}': {ex.Message}");
                }

                if (prefab == null)
                    return CommandResult.BadRequest($"Prefab not found: {guid}");

                int count = prefab.UpdateAllInstances();
                return CommandResult.Json(new
                {
                    status = "updated",
                    prefab = prefab.Name,
                    updated = count
                });
            }

            // Update single entity by ID
            if (root.TryGetProperty("id", out var idProp))
            {
                int id = idProp.GetInt32();
                var entity = EntityManager.GetEntity(id);
                if (entity == null)
                    return CommandResult.BadRequest($"Entity {id} not found");
                if (!entity.IsPrefabInstance)
                    return CommandResult.BadRequest($"Entity {id} is not a prefab instance");

                var serializer = new Serialization.EntitySerializer();
                var templates = serializer.LoadFromBytes(entity.Prefab.SourceYaml);
                if (templates.Count > 0)
                {
                    entity.Prefab.UpdateInstance(entity, templates[0]);
                    foreach (var te in templates)
                        EntityManager.RemoveEntity(te);
                }

                return CommandResult.Json(new
                {
                    status = "updated",
                    prefab = entity.Prefab?.Name,
                    id = entity.Id,
                    updated = 1
                });
            }

            return CommandResult.BadRequest("Provide 'guid' or 'id'");
        }
    }
}
