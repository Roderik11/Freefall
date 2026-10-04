using System.Linq;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;
using Vortice.Mathematics;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Shared helpers for command handlers: entity lookup, value serialization, vector parsing.
    /// </summary>
    public static class CommandHelpers
    {
        public static Entity FindEntityById(int id)
        {
            foreach (var entity in EntityManager.Entities)
            {
                if (entity.Id == id)
                    return entity;
            }
            return null;
        }

        /// <summary>By persistent UID (saved in the scene, unlike the runtime Id which changes on every load).</summary>
        public static Entity FindEntityByUid(ulong uid)
        {
            foreach (var entity in EntityManager.Entities)
            {
                if (entity.UID == uid)
                    return entity;
            }
            return null;
        }

        public static Entity FindEntityByName(string name)
        {
            return EntityManager.Entities.FirstOrDefault(e =>
                string.Equals(e.Name, name, System.StringComparison.OrdinalIgnoreCase));
        }

        public static object SerializeTransform(Transform t) => new
        {
            position = Vec3(t.Position),
            rotation = Quat(t.Rotation),
            scale = Vec3(t.Scale),
            worldPosition = Vec3(t.WorldPosition)
        };

        public static object Vec3(Vector3 v) => new { x = v.X, y = v.Y, z = v.Z };
        public static object Quat(Quaternion q) => new { x = q.X, y = q.Y, z = q.Z, w = q.W };

        public static object BBox(BoundingBox bb)
        {
            var size = bb.Max - bb.Min;
            return new
            {
                min = Vec3(bb.Min),
                max = Vec3(bb.Max),
                size = Vec3(size)
            };
        }

        /// <summary>Component <paramref name="i"/> of a vector given as {x,y,z,w} or [x,y,z,w].</summary>
        private static float VecComponent(System.Text.Json.JsonElement el, int i, float fallback)
        {
            if (el.ValueKind == System.Text.Json.JsonValueKind.Array)
                return i < el.GetArrayLength() ? el[i].GetSingle() : fallback;
            return el.TryGetProperty("xyzw"[i].ToString(), out var p) ? p.GetSingle() : fallback;
        }

        public static Vector3 ParseVec3(System.Text.Json.JsonElement el)
            => new(VecComponent(el, 0, 0), VecComponent(el, 1, 0), VecComponent(el, 2, 0));

        public static Vector2 ParseVec2(System.Text.Json.JsonElement el)
            => new(VecComponent(el, 0, 0), VecComponent(el, 1, 0));

        public static Vector4 ParseVec4(System.Text.Json.JsonElement el)
            => new(VecComponent(el, 0, 0), VecComponent(el, 1, 0), VecComponent(el, 2, 0), VecComponent(el, 3, 0));

        public static Quaternion ParseQuat(System.Text.Json.JsonElement el)
            => new(VecComponent(el, 0, 0), VecComponent(el, 1, 0), VecComponent(el, 2, 0), VecComponent(el, 3, 1));

        public static object SerializeValue(object value) => SerializeValue(value, 0);

        private static object SerializeValue(object value, int depth)
        {
            if (value == null) return null;

            // Engine data (lists of points/layers, nested settings classes) is shown structurally, a few levels deep.
            if (depth < 3)
            {
                if (value is System.Collections.IList list && value is not string && value.GetType() != typeof(byte[]))
                {
                    var items = new System.Collections.Generic.List<object>();
                    for (int i = 0; i < list.Count && i < 256; i++)
                        items.Add(SerializeValue(list[i], depth + 1));
                    return items;
                }
                var t = value.GetType();
                if (t.Namespace?.StartsWith("Freefall") == true && !t.IsEnum
                    && value is not Freefall.Assets.Asset && value is not Entity && value is not Component)
                {
                    var members = new System.Collections.Generic.Dictionary<string, object>();
                    foreach (var field in Reflector.GetMapping(t))
                    {
                        try { members[field.Name] = SerializeValue(field.GetValue(value), depth + 1); }
                        catch { members[field.Name] = "<error>"; }
                    }
                    return members;
                }
            }

            return value switch
            {
                Vector3 v => Vec3(v),
                Vector4 v => new { x = v.X, y = v.Y, z = v.Z, w = v.W },
                Quaternion q => Quat(q),
                Vector2 v => new { x = v.X, y = v.Y },
                Color3 c => new[] { c.R, c.G, c.B },
                Color4 c => new[] { c.R, c.G, c.B, c.A },
                Matrix4x4 => "<matrix4x4>",
                // System.Text.Json refuses NaN/Infinity and would fail the whole response
                float f when !float.IsFinite(f) => f.ToString(System.Globalization.CultureInfo.InvariantCulture),
                double d when !double.IsFinite(d) => d.ToString(System.Globalization.CultureInfo.InvariantCulture),
                float f => f,
                double d => d,
                int i => i,
                uint u => u,
                long l => l,
                ulong u => u,
                System.Collections.ICollection c => $"<{value.GetType().Name} count={c.Count}>",
                bool b => b,
                string s => s,
                System.Enum e => e.ToString(),
                Freefall.Assets.Asset a => new { type = a.GetType().Name, name = a.Name, guid = a.Guid },
                Entity e => new { id = e.Id, uid = e.UID.ToString(), name = e.Name },
                Component c => new { entity = c.Entity?.Id, entityUid = c.Entity?.UID.ToString(), component = c.GetType().Name },
                _ => value.ToString()
            };
        }

        public static object SerializeEntityBrief(Entity entity) => new
        {
            id = entity.Id, uid = entity.UID.ToString(),
            name = entity.Name,
            hidden = entity.HideInHierarchy,
            components = entity.Components.Select(c => c.GetType().Name).ToArray()
        };

        /// <summary>All reflected public members of an object as {MemberName: serialized value}.</summary>
        public static System.Collections.Generic.Dictionary<string, object> SerializeMembers(object obj)
        {
            var fields = new System.Collections.Generic.Dictionary<string, object>();
            foreach (var field in Reflector.GetMapping(obj.GetType()))
            {
                try { fields[field.Name] = SerializeValue(field.GetValue(obj)); }
                catch { fields[field.Name] = "<error>"; }
            }
            return fields;
        }

        /// <summary>
        /// Find a loaded asset by GUID, or load it using the type recorded in the asset database.
        /// (Never probe by trying every Asset type: a failed load per type can block the main thread for minutes.)
        /// </summary>
        public static Freefall.Assets.Asset FindOrLoadAsset(string guid)
        {
            var asset = Engine.Assets.FindByGuid(guid);
            if (asset != null) return asset;

            var type = Freefall.Assets.AssetDatabase.GetAssetType(guid);

            // Some importers (e.g. TerrainImporter) don't record a type name in the meta, but they declare AssetType.
            if (type == null && Freefall.Assets.AssetDatabase.GetMeta(guid) is { } meta
                && string.Equals(meta.Guid, guid, System.StringComparison.OrdinalIgnoreCase))
            {
                try { type = Freefall.Assets.AssetDatabase.GetImporter(guid)?.AssetType; } catch { }
            }

            if (type == null)
            {
                // No registered alias — match the meta's type name against Asset subclasses instead.
                var typeName = Freefall.Assets.AssetDatabase.GetAssetTypeName(guid);
                if (typeName == null) return null;
                type = System.AppDomain.CurrentDomain.GetAssemblies()
                    .Where(a => !ScriptCompiler.IsStale(a))
                    .SelectMany(a => { try { return a.GetTypes(); } catch { return System.Type.EmptyTypes; } })
                    .FirstOrDefault(t => !t.IsAbstract && typeof(Freefall.Assets.Asset).IsAssignableFrom(t)
                                         && string.Equals(t.Name, typeName, System.StringComparison.OrdinalIgnoreCase));
                if (type == null) return null;
            }

            return Engine.Assets.LoadByGuid(guid, type);
        }

        /// <summary>
        /// World-space AABB over every MeshRenderer on the entity and its descendants (prefab parts included),
        /// or null if it renders nothing. Lets agents lay out modular pieces (walls, docks) by real size.
        /// </summary>
        public static object WorldBounds(Entity entity)
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            bool any = false;

            void Visit(Transform t)
            {
                if (t == null) return;
                foreach (var comp in t.Entity.Components)
                {
                    if (comp is not MeshRenderer { Mesh: { } mesh }) continue;
                    var bb = mesh.BoundingBox;
                    var world = t.WorldMatrix;
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
                for (int i = 0; i < t.Count; i++)
                    Visit(t.GetChild(i));
            }

            Visit(entity.Transform);
            return any ? new { min = Vec3(min), max = Vec3(max), size = Vec3(max - min) } : null;
        }

        public static object SerializeEntityFull(Entity entity)
        {
            var components = new System.Collections.Generic.List<object>();
            foreach (var comp in entity.Components)
                components.Add(new { type = comp.GetType().Name, fields = SerializeMembers(comp) });

            return new
            {
                id = entity.Id, uid = entity.UID.ToString(),
                name = entity.Name,
                hidden = entity.HideInHierarchy,
                transform = SerializeTransform(entity.Transform),
                bounds = WorldBounds(entity),
                components
            };
        }

        /// <summary>
        /// Find a Component subclass by its simple name (e.g. "PointLight", "StaticMeshRenderer").
        /// Searches all assemblies registered with Reflector.
        /// </summary>
        public static System.Type FindComponentType(string typeName)
        {
            // Try simple name lookup via Reflector
            var type = Reflector.FindTypeBySimpleName(typeName);
            if (type != null && typeof(Component).IsAssignableFrom(type))
                return type;

            // Fallback: search all known Component subtypes
            var componentTypes = Reflector.GetTypes<Component>();
            foreach (var ct in componentTypes)
            {
                if (string.Equals(ct.Name, typeName, System.StringComparison.OrdinalIgnoreCase))
                    return ct;
            }

            return null;
        }

        /// <summary>
        /// Parse query string parameters from a URL path like "/api/foo?count=10&offset=5".
        /// Returns a dictionary of key-value pairs.
        /// </summary>
        public static System.Collections.Generic.Dictionary<string, string> ParseQueryString(string path)
        {
            var result = new System.Collections.Generic.Dictionary<string, string>(System.StringComparer.OrdinalIgnoreCase);
            var qIdx = path.IndexOf('?');
            if (qIdx < 0) return result;

            var query = path.Substring(qIdx + 1);
            foreach (var pair in query.Split('&'))
            {
                var eqIdx = pair.IndexOf('=');
                if (eqIdx > 0)
                    result[System.Uri.UnescapeDataString(pair.Substring(0, eqIdx))] =
                        System.Uri.UnescapeDataString(pair.Substring(eqIdx + 1));
            }
            return result;
        }
    }
}
