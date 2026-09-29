using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/scene/find")]
    public class FindEntitiesCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!Program.IsProjectOpen)
                return CommandResult.Error(409, "No project is open");

            var qs = CommandHelpers.ParseQueryString(context.Path);
            var limit = qs.TryGetValue("limit", out var limitStr) && int.TryParse(limitStr, out var l) ? l : 100;

            var results = new List<object>();

            // Filter by name pattern (supports * wildcards)
            if (qs.TryGetValue("name", out var namePattern))
            {
                var regex = new Regex(
                    "^" + Regex.Escape(namePattern).Replace("\\*", ".*") + "$",
                    RegexOptions.IgnoreCase);

                foreach (var entity in EntityManager.Entities)
                {
                    if (regex.IsMatch(entity.Name))
                    {
                        results.Add(CommandHelpers.SerializeEntityBrief(entity));
                        if (results.Count >= limit) break;
                    }
                }
            }
            // Filter by component type
            else if (qs.TryGetValue("component", out var componentName))
            {
                var type = CommandHelpers.FindComponentType(componentName);
                if (type == null)
                    return CommandResult.NotFound($"Component type '{componentName}' not found");

                foreach (var entity in EntityManager.Entities)
                {
                    if (entity.Components.Any(c => type.IsAssignableFrom(c.GetType())))
                    {
                        results.Add(CommandHelpers.SerializeEntityBrief(entity));
                        if (results.Count >= limit) break;
                    }
                }
            }
            else
            {
                return CommandResult.BadRequest("Query parameter 'name' or 'component' required");
            }

            return CommandResult.Json(new
            {
                count = results.Count,
                entities = results
            });
        }
    }
}
