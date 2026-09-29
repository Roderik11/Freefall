using Freefall.Base;

namespace Freefall.Editor.Commands
{
    [CommandRoute("GET", "/api/engine/stats")]
    public class GetStatsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            return CommandResult.Json(new
            {
                fps = Freefall.Base.Time.FPS,
                deltaMs = Freefall.Base.Time.DeltaMilliseconds,
                entityCount = EntityManager.Entities.Count,
                frameIndex = Engine.FrameIndex
            });
        }
    }
}
