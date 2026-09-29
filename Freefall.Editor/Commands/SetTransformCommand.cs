namespace Freefall.Editor.Commands
{
    [CommandRoute("POST", "/api/entity/{id}/transform")]
    public class SetTransformCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity with ID {id} not found");

            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Request body required");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (root.TryGetProperty("position", out var pos))
                entity.Transform.Position = CommandHelpers.ParseVec3(pos);

            if (root.TryGetProperty("rotation", out var rot))
                entity.Transform.Rotation = CommandHelpers.ParseQuat(rot);

            if (root.TryGetProperty("scale", out var scl))
                entity.Transform.Scale = CommandHelpers.ParseVec3(scl);

            return CommandResult.Json(new
            {
                id = entity.Id, uid = entity.UID.ToString(),
                name = entity.Name,
                transform = CommandHelpers.SerializeTransform(entity.Transform)
            });
        }
    }
}
