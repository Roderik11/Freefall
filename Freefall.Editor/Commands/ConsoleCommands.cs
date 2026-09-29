using System;
using System.Linq;
using Freefall.Base;

namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/console/log")]
    public class GetConsoleLogCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var qs = CommandHelpers.ParseQueryString(context.Path);
            int count = qs.TryGetValue("count", out var countStr) && int.TryParse(countStr, out var c) ? c : 50;
            int offset = qs.TryGetValue("offset", out var offsetStr) && int.TryParse(offsetStr, out var o) ? o : 0;

            var lines = Debug.Lines;
            var total = lines.Count;

            // Return most recent entries (end of list), or apply offset
            var start = Math.Max(0, total - count - offset);
            var end = Math.Min(total, start + count);

            var entries = new object[end - start];
            for (int i = start; i < end; i++)
            {
                entries[i - start] = new
                {
                    index = i,
                    message = lines[i].Message
                };
            }

            return CommandResult.Json(new
            {
                total,
                offset,
                count = entries.Length,
                entries
            });
        }
    }

    [CommandRoute("GET", "/api/console/clear")]
    public class ClearConsoleCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var count = Debug.Lines.Count;
            Debug.ClearLog();
            return CommandResult.Json(new { status = "cleared", entriesRemoved = count });
        }
    }
}
