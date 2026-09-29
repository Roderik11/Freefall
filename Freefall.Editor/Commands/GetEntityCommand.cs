namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/scene/entity/{id}")]
    public class GetEntityCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var id = context.GetInt("id");
            var entity = CommandHelpers.FindEntityById(id);
            if (entity == null)
                return CommandResult.NotFound($"Entity with ID {id} not found");

            return CommandResult.Json(CommandHelpers.SerializeEntityFull(entity));
        }
    }
}
