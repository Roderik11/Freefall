using System.Linq;
using Freefall.Base;

namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/selection")]
    public class GetSelectionCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var selected = Selector.SelectedEntity;
            if (selected == null)
                return CommandResult.Json(new { selected = (object)null });

            return CommandResult.Json(new
            {
                selected = new
                {
                    id = selected.Id, uid = selected.UID.ToString(),
                    name = selected.Name,
                    transform = CommandHelpers.SerializeTransform(selected.Transform)
                }
            });
        }
    }

    [CommandRoute("POST", "/api/selection")]
    public class SetSelectionCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Request body required");

            using var doc = context.ParseBody();
            var root = doc.RootElement;

            if (root.TryGetProperty("id", out var idProp))
            {
                var entity = CommandHelpers.FindEntityById(idProp.GetInt32());
                if (entity == null)
                    return CommandResult.NotFound("Entity not found");

                Selector.SelectedEntity = entity;
                return CommandResult.Json(new { selected = entity.Id, uid = entity.UID.ToString(), name = entity.Name });
            }

            if (root.TryGetProperty("name", out var nameProp))
            {
                var entity = CommandHelpers.FindEntityByName(nameProp.GetString());
                if (entity == null)
                    return CommandResult.NotFound($"Entity named '{nameProp.GetString()}' not found");

                Selector.SelectedEntity = entity;
                return CommandResult.Json(new { selected = entity.Id, uid = entity.UID.ToString(), name = entity.Name });
            }

            if (root.TryGetProperty("clear", out _))
            {
                Selector.SelectedEntity = null;
                return CommandResult.Json(new { selected = (object)null });
            }

            return CommandResult.BadRequest("Body must contain 'id', 'name', or 'clear'");
        }
    }
}
