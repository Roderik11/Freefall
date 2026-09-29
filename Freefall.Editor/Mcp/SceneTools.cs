using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class SceneTools
    {
        // --- Scene ---

        [McpServerTool(Name = "scene_load", Title = "Load scene", Destructive = true, OpenWorld = false)]
        [Description("Replace the current scene with a .scene file. Destroys all current entities (unsaved changes are lost) and invalidates every entity id. " +
                     "Waits until the scene has settled (PCG output spawned, frames rendering normally) before returning, so the next call works.")]
        public static async Task<CallToolResult> LoadScene(
            [Description("Scene path, relative to the project Assets folder or absolute")] string path,
            [Description("Max seconds to wait for the scene to settle")] int waitSeconds = 120)
        {
            var result = await McpBridge.Post("/api/scene/load", new { path });
            if (result.IsError == true) return result;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var settled = await WaitUntilSettled(waitSeconds);
            var json = JsonNode.Parse(EditorTools.TextOf(result))!.AsObject();
            json["settled"] = settled.ok;
            json["settleMs"] = sw.ElapsedMilliseconds;
            json["entityCountSettled"] = settled.entityCount; // incl. generated PCG output
            return EditorTools.Text(json);
        }

        /// <summary>
        /// Loading returns before the heavy first frames (component start-up, PCG regeneration, terrain bake) have run.
        /// Settled = the entity count is unchanged for ~1 s while frames keep advancing at a normal frame time.
        /// </summary>
        internal static async Task<(bool ok, int entityCount)> WaitUntilSettled(int waitSeconds)
        {
            var server = EditorCommandServer.Instance;
            var deadline = System.DateTime.UtcNow.AddSeconds(waitSeconds);
            int lastCount = -1, stablePolls = 0;
            long lastFrame = -1;
            while (System.DateTime.UtcNow < deadline)
            {
                var (frame, deltaMs, count) = await server.RunOnMainThread(() =>
                    ((long)Engine.TickCount, Freefall.Base.Time.DeltaMilliseconds, Freefall.Base.EntityManager.Entities.Count));

                bool healthy = frame > lastFrame && deltaMs < 250f;
                stablePolls = healthy && count == lastCount ? stablePolls + 1 : 0;
                if (stablePolls >= 4) return (true, count);

                lastCount = count;
                lastFrame = frame;
                await Task.Delay(250);
            }
            return (false, lastCount);
        }

        [McpServerTool(Name = "scene_save", Title = "Save scene", Destructive = true, OpenWorld = false)]
        [Description("Save all entities to a .scene file. Without 'path' this overwrites the currently open scene (same as File > Save Scene).")]
        public static Task<CallToolResult> SaveScene(
            [Description("Target path relative to Assets (or absolute); omit to save over the open scene")] string? path = null)
            => McpBridge.Post("/api/scene/save", new { path });

        [McpServerTool(Name = "scene_list_entities", Title = "List entities", ReadOnly = true, OpenWorld = false)]
        [Description("Entities in the scene: id, uid, name, component types. Ids are only valid until the next scene load or editor restart; " +
                     "uids (strings) are persistent and accepted by every entity tool.")]
        public static async Task<CallToolResult> ListEntities(
            [Description("Include editor-internal entities hidden from the hierarchy")] bool includeHidden = false,
            [Description("Only entities that have a component with this type name")] string? component = null,
            int limit = 500)
        {
            var result = await McpBridge.Get("/api/scene/entities");
            if (result.IsError == true) return result;

            var all = JsonNode.Parse(EditorTools.TextOf(result))!.AsArray();
            var filtered = all.Where(e =>
                    (includeHidden || e!["hidden"]?.GetValue<bool>() != true) &&
                    (component == null || e!["components"]!.AsArray().Any(c => string.Equals(c!.GetValue<string>(), component, System.StringComparison.OrdinalIgnoreCase))))
                .ToList();

            var output = new JsonObject
            {
                ["total"] = filtered.Count,
                ["entities"] = new JsonArray(filtered.Take(limit).Select(e => e!.DeepClone()).ToArray()),
            };
            return EditorTools.Text(output);
        }

        [McpServerTool(Name = "scene_find_entities", Title = "Find entities", ReadOnly = true, OpenWorld = false)]
        [Description("Find entities by name wildcard (e.g. 'rock*', case-insensitive) or by component type (base types match too). Includes hidden entities.")]
        public static Task<CallToolResult> FindEntities(
            [Description("Name pattern with * wildcards; takes precedence over 'component'")] string? name = null,
            [Description("Component type name, e.g. 'DirectionalLight'")] string? component = null,
            int limit = 100)
            => McpBridge.Get("/api/scene/find", ("name", name), ("component", component), ("limit", limit));

        [McpServerTool(Name = "entity_get", Title = "Get entity", ReadOnly = true, OpenWorld = false)]
        [Description("Full detail of one entity: transform and every public field/property of each component (names are exact, for entity_set_properties). " +
                     "Pass 'component' to limit the output to one component.")]
        public static async Task<CallToolResult> GetEntity(
            int? id = null,
            [Description("Only return this component type (case-insensitive)")] string? component = null,
            [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            if (err != null) return err;
            var result = await McpBridge.Get($"/api/scene/entity/{rid}");
            if (result.IsError == true || component == null) return result;

            var entity = JsonNode.Parse(EditorTools.TextOf(result))!.AsObject();
            var comps = entity["components"]!.AsArray()
                .Where(c => string.Equals(c!["type"]!.GetValue<string>(), component, System.StringComparison.OrdinalIgnoreCase))
                .Select(c => c!.DeepClone()).ToArray();
            entity["components"] = new JsonArray(comps);
            return EditorTools.Text(entity);
        }

        // --- Entity lifecycle ---

        [McpServerTool(Name = "entity_create", Title = "Create empty entity", OpenWorld = false)]
        [Description("Create an empty entity (Transform only). Add components with entity_add_component.")]
        public static Task<CallToolResult> CreateEntity(string name = "Entity")
            => McpBridge.Post("/api/entity/create", new { name });

        [McpServerTool(Name = "entity_instantiate", Title = "Instantiate asset", OpenWorld = false)]
        [Description("Place a prefab or mesh in the scene. Prefer prefab GUIDs (they carry materials, LODs and components); a mesh GUID creates a bare MeshRenderer. " +
                     "With snapToGround (default) the y of 'position' is replaced by the terrain height.")]
        public static async Task<CallToolResult> Instantiate(
            [Description("Prefab or Mesh GUID (use asset_search / asset_resolve to find one)")] string guid,
            string? name = null,
            Vec3? position = null,
            [Description("Rotation as quaternion")] Quat? rotation = null,
            [Description("Rotation as Euler degrees (x=pitch, y=yaw, z=roll); ignored if 'rotation' is given")] Vec3? rotationEuler = null,
            Vec3? scale = null,
            bool snapToGround = true)
        {
            var result = await McpBridge.Post("/api/entity/instantiate", new
            {
                guid, name, position, scale, snapToGround,
                rotation = rotation ?? FromEuler(rotationEuler),
            });
            if (result.IsError == true) return result;

            // Report the placed world bounds so modular pieces can be lined up by their real size.
            var placed = JsonNode.Parse(EditorTools.TextOf(result))!.AsObject();
            var id = placed["id"]?.GetValue<int>();
            if (id != null)
            {
                var detail = await McpBridge.Get($"/api/scene/entity/{id}");
                if (detail.IsError != true)
                    placed["bounds"] = JsonNode.Parse(EditorTools.TextOf(detail))?["bounds"]?.DeepClone();
            }
            placed.Remove("components");
            return EditorTools.Text(placed);
        }

        [McpServerTool(Name = "entity_scatter", Title = "Scatter assets", OpenWorld = false)]
        [Description("Scatter many prefab/mesh instances in a circle on the terrain with Poisson spacing and height/slope filters. " +
                     "Fewer than 'count' may be placed when constraints are tight; the result reports placed vs requested.")]
        public static Task<CallToolResult> Scatter(
            [Description("Assets to pick from, weighted")] WeightedAsset[] assets,
            ScatterArea area,
            int count = 10,
            [Description("Minimum distance between instances")] float spacing = 10,
            [Description("Uniform random scale range, default 0.8–1.2")] MinMax? scale = null,
            [Description("Extra random Y-scale variance (0.2 = ±20%)")] float scaleYVariance = 0,
            bool randomYaw = true,
            [Description("Euler degrees applied before the random yaw")] Vec3? baseRotation = null,
            bool snapToGround = true,
            [Description("Only place where terrain height (world Y) is in range")] MinMax? height = null,
            [Description("Only place where slope in degrees is in range (0 flat, 90 vertical)")] MinMax? slope = null,
            [Description("Random seed for reproducible results")] int? seed = null)
            => McpBridge.Post("/api/entity/scatter", new
            {
                meshes = assets, area, count, spacing, scale, scaleYVariance, randomYaw,
                baseRotation, snapToGround, height, slope, seed,
            });

        [McpServerTool(Name = "entity_delete", Title = "Delete entity", Destructive = true, OpenWorld = false)]
        [Description("Destroy an entity.")]
        public static async Task<CallToolResult> DeleteEntity(int? id = null, [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            return err ?? await McpBridge.Post($"/api/entity/{rid}/delete");
        }

        [McpServerTool(Name = "entity_rename", Title = "Rename entity", Idempotent = true, OpenWorld = false)]
        public static async Task<CallToolResult> RenameEntity(string name, int? id = null, [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            return err ?? await McpBridge.Post($"/api/entity/{rid}/rename", new { name });
        }

        [McpServerTool(Name = "entity_clone", Title = "Clone entity", OpenWorld = false)]
        [Description("Duplicate an entity and its components (shallow copy — asset references are shared; children are not cloned).")]
        public static async Task<CallToolResult> CloneEntity(int? id = null, string? name = null, [Description("Added to the local position")] Vec3? offset = null,
            [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            return err ?? await McpBridge.Post($"/api/entity/{rid}/clone", new { name, offset });
        }

        [McpServerTool(Name = "entity_set_parent", Title = "Set parent", Idempotent = true, OpenWorld = false)]
        [Description("Parent an entity under another (local transform values are kept), or omit parent/parentUid to unparent.")]
        public static async Task<CallToolResult> SetParent(int? id = null, int? parent = null,
            [Description(EntityRefs.UidHelp)] string? uid = null,
            [Description("Parent by persistent UID (string)")] string? parentUid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            if (err != null) return err;
            var (pid, perr) = await EntityRefs.Resolve(parent, parentUid, required: false);
            if (perr != null) return perr;
            return await McpBridge.Post($"/api/entity/{rid}/setparent", $"{{\"parent\":{(pid?.ToString() ?? "null")}}}");
        }

        [McpServerTool(Name = "entity_set_transform", Title = "Set transform", Idempotent = true, OpenWorld = false)]
        [Description("Set local position / rotation / scale. Omitted parts are unchanged.")]
        public static async Task<CallToolResult> SetTransform(
            int? id = null,
            Vec3? position = null,
            [Description("Rotation as quaternion")] Quat? rotation = null,
            [Description("Rotation as Euler degrees (x=pitch, y=yaw, z=roll); ignored if 'rotation' is given")] Vec3? rotationEuler = null,
            Vec3? scale = null,
            [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            return err ?? await McpBridge.Post($"/api/entity/{rid}/transform", new
            {
                position, scale,
                rotation = rotation ?? FromEuler(rotationEuler),
            });
        }

        [McpServerTool(Name = "entity_align_to_terrain", Title = "Align to terrain", Idempotent = true, OpenWorld = false)]
        [Description("Drop an entity onto the terrain and tilt it to the ground under its footprint (keeps yaw, replaces pitch/roll).")]
        public static async Task<CallToolResult> AlignToTerrain(int? id = null, [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            return err ?? await McpBridge.Post($"/api/entity/{rid}/alignToTerrain");
        }

        [McpServerTool(Name = "entity_add_component", Title = "Add component", OpenWorld = false)]
        [Description("Add a component by type name (e.g. 'PointLight', 'HeightStamp'). Fails if the entity already has one of that type.")]
        public static async Task<CallToolResult> AddComponent(string type, int? id = null, [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            return err ?? await McpBridge.Post($"/api/entity/{rid}/addcomponent", new { type });
        }

        [McpServerTool(Name = "entity_set_properties", Title = "Set component properties", Idempotent = true, OpenWorld = false)]
        [Description("Set public fields/properties on one component of an entity, e.g. component='PointLight', values={\"Intensity\": 4, \"Color\": [1,0.8,0.6]}. " +
                     "Member names are exact PascalCase (see entity_get). Applied in order; stops at the first failure (earlier ones stay applied). " +
                     "Also works for component='Transform'.")]
        public static async Task<CallToolResult> SetProperties(
            [Description("Component type name (case-insensitive)")] string component,
            [Description("Member name → value. Encodings: numbers/bools/strings as-is; vectors {x,y,z} or [x,y,z]; quaternions {x,y,z,w}; " +
                         "Color3/Color4 [r,g,b(,a)]; enums by name; asset references by GUID string (\"\" or null clears); " +
                         "component references {\"entity\": id | \"entityUid\": \"uid\", \"component\"?: \"Type\"}; lists as JSON arrays (replaces the whole list, e.g. Spline.Points); " +
                         "nested data objects as JSON objects of their members.")]
            Dictionary<string, JsonElement> values,
            int? id = null,
            [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            if (err != null) return err;
            int eid = rid!.Value;
            var applied = new JsonObject();
            foreach (var (property, value) in values)
            {
                var body = new JsonObject
                {
                    ["component"] = component,
                    ["property"] = property,
                    ["value"] = JsonNode.Parse(value.GetRawText()),
                };
                var result = await McpBridge.Post($"/api/entity/{eid}/setproperty", body.ToJsonString());
                if (result.IsError == true)
                {
                    var msg = $"Failed on '{property}': {EditorTools.TextOf(result)}";
                    if (applied.Count > 0) msg += $"\nAlready applied: {applied.ToJsonString()}";
                    return McpBridge.Error(msg);
                }
                applied[property] = JsonNode.Parse(EditorTools.TextOf(result))?["value"]?.DeepClone();
            }
            return EditorTools.Text(new JsonObject { ["id"] = eid, ["component"] = component, ["applied"] = applied });
        }

        [McpServerTool(Name = "prefab_update", Title = "Update prefab instances", OpenWorld = false)]
        [Description("Re-apply a prefab to its instances: pass prefabGuid to update all instances, or entityId to refresh a single instance.")]
        public static async Task<CallToolResult> UpdatePrefab(string? prefabGuid = null, int? entityId = null,
            [Description("Instance by persistent UID (string)")] string? entityUid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(entityId, entityUid, required: false);
            return err ?? await McpBridge.Post("/api/prefab/update", new { guid = prefabGuid, id = rid });
        }

        // --- Selection & camera ---

        [McpServerTool(Name = "selection_get", Title = "Get selection", ReadOnly = true, OpenWorld = false)]
        [Description("The entity currently selected in the editor (what the user is looking at in the inspector), or null.")]
        public static Task<CallToolResult> GetSelection() => McpBridge.Get("/api/selection");

        [McpServerTool(Name = "selection_set", Title = "Set selection", Idempotent = true, OpenWorld = false)]
        [Description("Select an entity by id or name (shows it in the inspector), or clear=true to deselect.")]
        public static async Task<CallToolResult> SetSelection(int? id = null, string? name = null, bool clear = false,
            [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            if (clear) return await McpBridge.Post("/api/selection", new { clear = true });
            var (rid, err) = await EntityRefs.Resolve(id, uid, required: false);
            return err ?? await McpBridge.Post("/api/selection", new { id = rid, name });
        }

        [McpServerTool(Name = "camera_get", Title = "Get editor camera", ReadOnly = true, OpenWorld = false)]
        [Description("Editor camera position, rotation, forward/up/right vectors, fov and clip planes.")]
        public static Task<CallToolResult> GetCamera() => McpBridge.Get("/api/camera");

        [McpServerTool(Name = "camera_move", Title = "Move editor camera", Idempotent = true, OpenWorld = false)]
        [Description("Move the editor camera and optionally aim it at a point. Take a screenshot afterwards to see the result.")]
        public static Task<CallToolResult> MoveCamera(Vec3? position = null, [Description("World point to look at")] Vec3? lookAt = null)
            => McpBridge.Post("/api/camera/move", new { position, lookAt });

        [McpServerTool(Name = "camera_focus", Title = "Focus camera on entity", Idempotent = true, OpenWorld = false)]
        [Description("Frame an entity in the viewport (like double-clicking it in the hierarchy). The camera animates, so wait a moment before a screenshot.")]
        public static async Task<CallToolResult> FocusCamera(int? id = null, string? name = null,
            [Description(EntityRefs.UidHelp)] string? uid = null)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid, required: false);
            return err ?? await McpBridge.Post("/api/camera/focus", new { id = rid, name });
        }

        // --- helpers ---

        private static Quat? FromEuler(Vec3? degrees)
        {
            if (degrees == null) return null;
            const float d2r = System.MathF.PI / 180f;
            var q = Quaternion.CreateFromYawPitchRoll(degrees.Y * d2r, degrees.X * d2r, degrees.Z * d2r);
            return new Quat(q.X, q.Y, q.Z, q.W);
        }
    }
}
