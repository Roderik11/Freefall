using System.ComponentModel;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;

namespace Freefall.Editor.Mcp
{
    [McpServerToolType]
    public static class ClipTools
    {
        [McpServerTool(Name = "clip_get", Title = "Get animation clip", ReadOnly = true, OpenWorld = false)]
        [Description("An animation clip (a sub-asset of a model file; find its guid with asset_resolve or asset_search): length in seconds " +
                     "and frames, and the events marked on it (name, time as a fraction of the clip, seconds, frame, bone).")]
        public static Task<CallToolResult> Get(string guid)
            => McpBridge.Get("/api/clip/get", ("guid", guid));

        [McpServerTool(Name = "clip_set_events", Title = "Mark events on a clip", Destructive = true, Idempotent = true, OpenWorld = false)]
        [Description("Replace the events marked on an animation clip (AnimationClip.Events) and save them. An event is something that " +
                     "happens at one moment of the clip (the blow of a swing lands, a tool strikes); the Animator reports it when playback " +
                     "passes that moment (Animator.GetEvents). They are kept with the model file's import settings, not in the clip: no " +
                     "reimport, and a reimport keeps them. Feet coming down need no events: those are found from the clip. " +
                     "To find a moment, look at it: clip_preview, then screenshot.")]
        public static Task<CallToolResult> SetEvents(
            string guid,
            [Description("The whole list (empty clears): [{\"name\": \"hit\", \"time\": 0.42, \"bone\": \"RightHand\"}]. The moment is " +
                         "'time' (0..1 of the clip), or 'seconds', or 'frame'. 'bone' (a HumanoidBone name) is optional: where on the body it happens.")]
            JsonArray events)
            => McpBridge.Post("/api/clip/events", new JsonObject
            {
                ["guid"] = guid,
                ["events"] = events.DeepClone(),
            }.ToJsonString());

        [McpServerTool(Name = "clip_preview", Title = "Show a clip on a character", Idempotent = true, OpenWorld = false)]
        [Description("Show an animation clip at one moment on a character in the scene, in place of what it is playing, for 'hold' seconds " +
                     "(then it goes back by itself; hold=0 ends it now). The clip shows as it is, with its travel, on that character's own " +
                     "body and with what it holds. For finding the moment of an event: preview, screenshot, adjust. Give the moment as " +
                     "'time' (0..1 of the clip), 'seconds' or 'frame' (one of them; default the start).")]
        public static async Task<CallToolResult> Preview(
            string guid,
            int? id = null,
            [Description(EntityRefs.UidHelp)] string? uid = null,
            float? time = null,
            float? seconds = null,
            float? frame = null,
            float hold = 5f)
        {
            var (rid, err) = await EntityRefs.Resolve(id, uid);
            if (err != null) return err;

            var body = new JsonObject { ["guid"] = guid, ["id"] = rid, ["hold"] = hold };
            if (time.HasValue) body["time"] = time.Value;
            else if (seconds.HasValue) body["seconds"] = seconds.Value;
            else if (frame.HasValue) body["frame"] = frame.Value;

            return await McpBridge.Post("/api/clip/preview", body.ToJsonString());
        }
    }
}
