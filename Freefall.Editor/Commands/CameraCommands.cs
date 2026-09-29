using System;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Editor.Commands
{
    [CommandRoute("POST", "/api/camera/move")]
    public class CameraMoveCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            var editorCam = FindEditorCamera();
            if (editorCam == null)
                return CommandResult.NotFound("Editor camera not found");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Request body required");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            // Parse position (default to current)
            var position = editorCam.Entity.Transform.Position;
            if (root.TryGetProperty("position", out var pos))
                position = CommandHelpers.ParseVec3(pos);

            // Parse lookAt target (optional)
            Vector3? lookAt = null;
            if (root.TryGetProperty("lookAt", out var lookAtProp))
                lookAt = CommandHelpers.ParseVec3(lookAtProp);

            // Use SetView — this updates internal yaw/pitch so Update() doesn't overwrite
            editorCam.SetView(position, lookAt);

            return CommandResult.Json(new
            {
                position = CommandHelpers.Vec3(editorCam.Entity.Transform.Position),
                forward = CommandHelpers.Vec3(editorCam.Entity.Transform.Forward)
            });
        }

        internal static EditorCamera FindEditorCamera()
        {
            foreach (var entity in EntityManager.Entities)
            {
                var cam = entity.GetComponent<EditorCamera>();
                if (cam != null) return cam;
            }
            return null;
        }
    }

    [CommandRoute("GET", "/api/camera")]
    public class GetCameraCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            Entity camEntity = null;
            Camera camComponent = null;

            foreach (var entity in EntityManager.Entities)
            {
                if (entity.GetComponent<EditorCamera>() != null)
                {
                    camEntity = entity;
                    camComponent = entity.GetComponent<Camera>();
                    break;
                }
            }

            if (camEntity == null)
                return CommandResult.NotFound("Editor camera not found");

            var result = new
            {
                position = CommandHelpers.Vec3(camEntity.Transform.Position),
                rotation = CommandHelpers.Quat(camEntity.Transform.Rotation),
                forward = CommandHelpers.Vec3(camEntity.Transform.Forward),
                up = CommandHelpers.Vec3(camEntity.Transform.Up),
                right = CommandHelpers.Vec3(camEntity.Transform.Right),
                fov = camComponent?.FieldOfView ?? 0,
                near = camComponent?.NearPlane ?? 0,
                far = camComponent?.FarPlane ?? 0
            };

            return CommandResult.Json(result);
        }
    }

    [CommandRoute("POST", "/api/camera/focus")]
    public class CameraFocusCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Request body required");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            Entity target = null;

            if (root.TryGetProperty("id", out var idProp))
                target = CommandHelpers.FindEntityById(idProp.GetInt32());
            else if (root.TryGetProperty("name", out var nameProp))
                target = CommandHelpers.FindEntityByName(nameProp.GetString());

            if (target == null)
                return CommandResult.NotFound("Target entity not found");

            // Use the existing FocusEntity message
            MessageDispatcher.Send(Msg.FocusEntity, target);

            return CommandResult.Json(new
            {
                focused = target.Name,
                id = target.Id,
                position = CommandHelpers.Vec3(target.Transform.WorldPosition)
            });
        }
    }
}
