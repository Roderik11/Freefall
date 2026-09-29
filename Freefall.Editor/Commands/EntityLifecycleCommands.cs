using System;
using System.Linq;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Editor.Commands
{
    [CommandRoute("POST", "/api/entity/create")]
    public class CreateEntityCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            var name = "Entity";
            if (!string.IsNullOrEmpty(context.Body))
            {
                using var doc = context.ParseBody();
                if (doc.RootElement.TryGetProperty("name", out var nameProp))
                    name = nameProp.GetString();
            }

            var entity = new Entity(name);
            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new
            {
                status = "created",
                id = entity.Id, uid = entity.UID.ToString(),
                name = entity.Name
            });
        }
    }

    [CommandRoute("POST", "/api/entity/{id}/delete")]
    public class DeleteEntityCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity {id} not found");

            // Clear selection if this entity is selected
            if (Selector.SelectedEntity == entity)
                Selector.SelectedObject = null;

            var name = entity.Name;
            entity.Destroy();
            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new { status = "deleted", id, name });
        }
    }

    [CommandRoute("POST", "/api/entity/{id}/rename")]
    public class RenameEntityCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity {id} not found");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'name'");

            using var doc = context.ParseBody();
            if (!doc.RootElement.TryGetProperty("name", out var nameProp))
                return CommandResult.BadRequest("Body must contain 'name'");

            var oldName = entity.Name;
            entity.Name = nameProp.GetString();
            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new { id, oldName, name = entity.Name });
        }
    }

    [CommandRoute("POST", "/api/entity/{id}/setvisible")]
    public class SetVisibleCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity {id} not found");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'visible'");

            using var doc = context.ParseBody();
            if (!doc.RootElement.TryGetProperty("visible", out var visProp))
                return CommandResult.BadRequest("Body must contain 'visible'");

            entity.HideInHierarchy = !visProp.GetBoolean();
            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new { id, name = entity.Name, visible = !entity.HideInHierarchy });
        }
    }

    [CommandRoute("POST", "/api/entity/{id}/addcomponent")]
    public class AddComponentCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity {id} not found");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'type'");

            using var doc = context.ParseBody();
            if (!doc.RootElement.TryGetProperty("type", out var typeProp))
                return CommandResult.BadRequest("Body must contain 'type'");

            var typeName = typeProp.GetString();
            var componentType = CommandHelpers.FindComponentType(typeName);
            if (componentType == null)
                return CommandResult.NotFound($"Component type '{typeName}' not found");

            // Check if entity already has this component
            var existing = entity.Components.FirstOrDefault(c => c.GetType() == componentType);
            if (existing != null)
                return CommandResult.Error(409, $"Entity already has component '{typeName}'");

            try
            {
                var instance = (Component)Activator.CreateInstance(componentType);
                entity.AddComponent(instance);
                MessageDispatcher.Send(Msg.RefreshExplorer);

                return CommandResult.Json(new
                {
                    status = "added",
                    id,
                    name = entity.Name,
                    component = componentType.Name
                });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(500, $"Failed to create component: {ex.Message}");
            }
        }
    }

    [CommandRoute("POST", "/api/entity/{id}/setproperty")]
    public class SetPropertyCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity {id} not found");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'component', 'property', 'value'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (!root.TryGetProperty("component", out var compProp))
                return CommandResult.BadRequest("Body must contain 'component'");
            if (!root.TryGetProperty("property", out var propProp))
                return CommandResult.BadRequest("Body must contain 'property'");
            if (!root.TryGetProperty("value", out var valueProp))
                return CommandResult.BadRequest("Body must contain 'value'");

            var componentName = compProp.GetString();
            var propertyName = propProp.GetString();

            // Find the component on the entity
            var component = entity.Components.FirstOrDefault(c =>
                string.Equals(c.GetType().Name, componentName, StringComparison.OrdinalIgnoreCase));
            if (component == null)
                return CommandResult.NotFound($"Component '{componentName}' not found on entity");

            // Find the field
            var field = Reflector.GetField(component.GetType(), propertyName);
            if (field == null)
                return CommandResult.NotFound($"Property '{propertyName}' not found on '{componentName}'");

            try
            {
                object converted = ConvertJsonValue(valueProp, field.Type);
                field.SetValue(component, converted);

                // Same notification the inspector sends, so stamps rebake, splines update, etc.
                component.OnMemberChanged();

                return CommandResult.Json(new
                {
                    status = "set",
                    id,
                    component = componentName,
                    property = propertyName,
                    value = CommandHelpers.SerializeValue(field.GetValue(component))
                });
            }
            catch (Exception ex)
            {
                return CommandResult.Error(400, $"Failed to set property: {ex.Message}");
            }
        }

        /// <summary>Convert a JSON value to a field type. Shared with the asset setproperty command.</summary>
        internal static object ConvertJsonValue(System.Text.Json.JsonElement el, Type targetType)
        {
            if (targetType == typeof(float))
                return el.GetSingle();
            if (targetType == typeof(Vortice.Mathematics.Color3))
            {
                // Accept [r,g,b] or {"r":..,"g":..,"b":..}
                if (el.ValueKind == System.Text.Json.JsonValueKind.Array)
                {
                    var a = el.EnumerateArray().Select(x => x.GetSingle()).ToArray();
                    return new Vortice.Mathematics.Color3(a[0], a[1], a[2]);
                }
                return new Vortice.Mathematics.Color3(
                    el.GetProperty("r").GetSingle(), el.GetProperty("g").GetSingle(), el.GetProperty("b").GetSingle());
            }
            if (targetType == typeof(Vortice.Mathematics.Color4))
            {
                // Accept [r,g,b,a] or {"r":..,"g":..,"b":..,"a":..}; alpha defaults to 1
                if (el.ValueKind == System.Text.Json.JsonValueKind.Array)
                {
                    var a = el.EnumerateArray().Select(x => x.GetSingle()).ToArray();
                    return new Vortice.Mathematics.Color4(a[0], a[1], a[2], a.Length > 3 ? a[3] : 1f);
                }
                return new Vortice.Mathematics.Color4(
                    el.GetProperty("r").GetSingle(), el.GetProperty("g").GetSingle(), el.GetProperty("b").GetSingle(),
                    el.TryGetProperty("a", out var ap) ? ap.GetSingle() : 1f);
            }
            if (targetType == typeof(double))
                return el.GetDouble();
            if (targetType == typeof(int))
                return el.GetInt32();
            if (targetType == typeof(uint))
                return el.GetUInt32();
            if (targetType == typeof(long))
                return el.GetInt64();
            if (targetType == typeof(ulong))
                return el.GetUInt64();
            if (targetType == typeof(bool))
                return el.GetBoolean();
            if (targetType == typeof(string))
                return el.GetString();
            if (targetType == typeof(Vector2))
                return CommandHelpers.ParseVec2(el);
            if (targetType == typeof(Vector3))
                return CommandHelpers.ParseVec3(el);
            if (targetType == typeof(Vector4))
                return CommandHelpers.ParseVec4(el);
            if (targetType == typeof(Quaternion))
                return CommandHelpers.ParseQuat(el);
            if (targetType.IsEnum)
                return el.ValueKind == System.Text.Json.JsonValueKind.Number
                    ? Enum.ToObject(targetType, el.GetInt64())
                    : Enum.Parse(targetType, el.GetString(), ignoreCase: true);

            // Asset types: accept GUID string, resolve via AssetManager
            if (typeof(Asset).IsAssignableFrom(targetType))
            {
                var guid = el.GetString();
                if (string.IsNullOrEmpty(guid))
                    return null;
                var asset = Engine.Assets.LoadByGuid(guid, targetType);
                if (asset == null)
                    throw new InvalidOperationException($"Asset not found for GUID '{guid}' (type {targetType.Name})");
                return asset;
            }

            // Component references: accept {"entity": id} or {"entity": id, "component": "TypeName"}
            // Used for linking e.g. SkyboxRenderer.SunLight → Sun entity's DirectionalLight
            if (typeof(Component).IsAssignableFrom(targetType))
            {
                if (el.ValueKind == System.Text.Json.JsonValueKind.Null)
                    return null;

                if (el.ValueKind != System.Text.Json.JsonValueKind.Object)
                    throw new InvalidOperationException($"Component reference must be an object with 'entity' (id). Got {el.ValueKind}");

                Entity refEntity;
                if (el.TryGetProperty("entityUid", out var uidProp))
                {
                    // Persistent UID as a string (UIDs exceed JSON's exact integer range)
                    var uidText = uidProp.ValueKind == System.Text.Json.JsonValueKind.String ? uidProp.GetString() : uidProp.GetRawText();
                    if (!ulong.TryParse(uidText, out var refUid))
                        throw new InvalidOperationException($"'entityUid' must be a UID string, got {uidText}");
                    refEntity = CommandHelpers.FindEntityByUid(refUid)
                        ?? throw new InvalidOperationException($"Referenced entity UID {refUid} not found");
                }
                else if (el.TryGetProperty("entity", out var entityIdProp))
                {
                    var refEntityId = entityIdProp.GetInt32();
                    refEntity = CommandHelpers.FindEntityById(refEntityId)
                        ?? throw new InvalidOperationException($"Referenced entity {refEntityId} not found");
                }
                else
                    throw new InvalidOperationException("Component reference must contain 'entity' (id) or 'entityUid' (UID string)");

                // If "component" is specified, find that specific type; otherwise use the field's type
                Type compType = targetType;
                if (el.TryGetProperty("component", out var compTypeProp))
                {
                    var compTypeName = compTypeProp.GetString();
                    compType = CommandHelpers.FindComponentType(compTypeName) ?? targetType;
                }

                var comp = refEntity.Components.FirstOrDefault(c => compType.IsAssignableFrom(c.GetType()));
                if (comp == null)
                    throw new InvalidOperationException($"Component '{compType.Name}' not found on entity {refEntity.Id} ('{refEntity.Name}')");

                return comp;
            }

            // Lists and arrays: convert each element with these same rules (e.g. Spline.Points, Terrain.Layers)
            if (el.ValueKind == System.Text.Json.JsonValueKind.Array)
            {
                if (targetType.IsArray)
                {
                    var elementType = targetType.GetElementType();
                    var array = Array.CreateInstance(elementType, el.GetArrayLength());
                    int i = 0;
                    foreach (var item in el.EnumerateArray())
                        array.SetValue(ConvertJsonValue(item, elementType), i++);
                    return array;
                }
                // Any IList<T> with a parameterless ctor — includes subclasses like LayerMask : List<ulong>
                var listInterface = targetType.GetInterfaces().Append(targetType)
                    .FirstOrDefault(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IList<>));
                if (listInterface != null && !targetType.IsAbstract && !targetType.IsInterface
                    && typeof(System.Collections.IList).IsAssignableFrom(targetType))
                {
                    var elementType = listInterface.GetGenericArguments()[0];
                    var list = (System.Collections.IList)Activator.CreateInstance(targetType);
                    foreach (var item in el.EnumerateArray())
                        list.Add(ConvertJsonValue(item, elementType));
                    return list;
                }
            }

            // Plain data objects (e.g. Terrain.TextureLayer, SpectrumBand): build a fresh instance and set members
            // by exact name; members not mentioned keep their constructor defaults.
            if (el.ValueKind == System.Text.Json.JsonValueKind.Object && !targetType.IsPrimitive && !targetType.IsAbstract
                && (targetType.IsValueType || targetType.GetConstructor(Type.EmptyTypes) != null))
            {
                var instance = Activator.CreateInstance(targetType);
                foreach (var member in el.EnumerateObject())
                {
                    var memberField = Reflector.GetField(targetType, member.Name)
                        ?? throw new InvalidOperationException($"'{targetType.Name}' has no member '{member.Name}'");
                    memberField.SetValue(instance, ConvertJsonValue(member.Value, memberField.Type));
                }
                return instance;
            }

            throw new NotSupportedException($"Cannot convert JSON to {targetType.Name}");
        }
    }

    [CommandRoute("POST", "/api/entity/{id}/setparent")]
    public class SetParentCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity {id} not found");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'parent'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (!root.TryGetProperty("parent", out var parentProp))
                return CommandResult.BadRequest("Body must contain 'parent' (entity id or null)");

            if (parentProp.ValueKind == System.Text.Json.JsonValueKind.Null)
            {
                entity.Transform.Parent = null;
                MessageDispatcher.Send(Msg.RefreshExplorer);
                return CommandResult.Json(new { id, name = entity.Name, parent = (object)null });
            }

            var parentId = parentProp.GetInt32();
            var parentEntity = CommandHelpers.FindEntityById(parentId);
            if (parentEntity == null)
                return CommandResult.NotFound($"Parent entity {parentId} not found");

            // Prevent circular parenting
            if (parentEntity.Transform.IsChildOf(entity.Transform))
                return CommandResult.Error(400, "Cannot parent to a descendant (circular)");

            entity.Transform.Parent = parentEntity.Transform;
            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new
            {
                id,
                name = entity.Name,
                parent = new { id = parentEntity.Id, uid = parentEntity.UID.ToString(), name = parentEntity.Name }
            });
        }
    }

    [CommandRoute("POST", "/api/entity/{id}/clone")]
    public class CloneEntityCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var source = CommandHelpers.FindEntityById(id);
            if (source == null)
                return CommandResult.NotFound($"Entity {id} not found");

            var cloneName = source.Name + "_clone";
            var offset = System.Numerics.Vector3.Zero;

            if (!string.IsNullOrEmpty(context.Body))
            {
                using var doc = context.ParseBody();
                var root = doc.RootElement;
                if (root.TryGetProperty("name", out var nameProp))
                    cloneName = nameProp.GetString();
                if (root.TryGetProperty("offset", out var offsetProp))
                    offset = CommandHelpers.ParseVec3(offsetProp);
            }

            var clone = new Entity(cloneName);

            // Copy transform
            clone.Transform.Position = source.Transform.Position + offset;
            clone.Transform.Rotation = source.Transform.Rotation;
            clone.Transform.Scale = source.Transform.Scale;

            // Copy components (skip Transform, it's auto-created)
            foreach (var comp in source.Components)
            {
                if (comp is Transform) continue;

                var compType = comp.GetType();
                var newComp = (Component)Activator.CreateInstance(compType);

                // Copy all reflected fields
                var mapping = Reflector.GetMapping(compType);
                foreach (var field in mapping)
                {
                    if (!field.CanWrite) continue;
                    if (field.Name == "Id" || field.Name == "UID") continue;
                    if (field.Name == "Entity" || field.Name == "Transform") continue;
                    try
                    {
                        var val = field.GetValue(comp);
                        field.SetValue(newComp, val);
                    }
                    catch { /* skip non-copyable fields */ }
                }

                clone.AddComponent(newComp);
            }

            MessageDispatcher.Send(Msg.RefreshExplorer);

            return CommandResult.Json(new
            {
                status = "cloned",
                sourceId = id,
                id = clone.Id, uid = clone.UID.ToString(),
                name = clone.Name,
                position = CommandHelpers.Vec3(clone.Transform.Position),
                components = clone.Components.Select(c => c.GetType().Name).ToArray()
            });
        }
    }
}
