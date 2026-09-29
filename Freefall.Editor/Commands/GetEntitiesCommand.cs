using System.Collections.Generic;
using Freefall.Base;

namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/scene/entities")]
    public class GetEntitiesCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var list = new List<object>();
            foreach (var entity in EntityManager.Entities)
                list.Add(CommandHelpers.SerializeEntityBrief(entity));

            return CommandResult.Json(list);
        }
    }
}
