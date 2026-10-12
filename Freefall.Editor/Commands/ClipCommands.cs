using System;
using System.Collections.Generic;
using System.Text.Json;
using Freefall.Animation;
using Freefall.Assets.Importers;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Editor.Commands
{
    internal static class ClipCommandHelpers
    {
        public static AnimationClip Find(string guid) => CommandHelpers.FindOrLoadAsset(guid) as AnimationClip;

        public static object Describe(string guid, AnimationClip clip)
        {
            var events = new List<object>();
            if (clip.Events != null)
                foreach (var e in clip.Events)
                    if (e != null)
                        events.Add(new
                        {
                            name = e.Name,
                            time = e.Time,
                            seconds = e.Time * clip.DurationSeconds,
                            frame = e.Time * clip.Duration,
                            bone = e.Bone?.ToString(),
                        });

            return new
            {
                guid,
                name = clip.Name,
                seconds = clip.DurationSeconds,
                frames = clip.Duration,
                framesPerSecond = clip.TicksPerSecond,
                channels = clip.Channels.Count,
                rig = clip.Skeleton?.Name,
                events,
            };
        }

        /// <summary>A moment of a clip as a fraction of its length, given as that ("time"), in seconds or as a frame.</summary>
        public static bool TryReadTime(JsonElement from, AnimationClip clip, out float time)
        {
            time = 0f;
            if (from.TryGetProperty("time", out var t) && t.ValueKind == JsonValueKind.Number)
                time = t.GetSingle();
            else if (from.TryGetProperty("seconds", out var s) && s.ValueKind == JsonValueKind.Number)
                time = clip.DurationSeconds > 0f ? s.GetSingle() / clip.DurationSeconds : 0f;
            else if (from.TryGetProperty("frame", out var f) && f.ValueKind == JsonValueKind.Number)
                time = clip.Duration > 0f ? f.GetSingle() / clip.Duration : 0f;
            else
                return false;

            time = Math.Clamp(time, 0f, 1f);
            return true;
        }
    }

    /// <summary>GET /api/clip/get?guid=X — an animation clip: its length and what is marked on it.</summary>
    [CommandRoute("GET", "/api/clip/get")]
    public class GetClipCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var query = CommandHelpers.ParseQueryString(context.Path);
            if (!query.TryGetValue("guid", out var guid) || string.IsNullOrWhiteSpace(guid))
                return CommandResult.BadRequest("Query parameter 'guid' is required");

            var clip = ClipCommandHelpers.Find(guid);
            if (clip == null) return CommandResult.NotFound($"Animation clip not found: {guid}");

            return CommandResult.Json(ClipCommandHelpers.Describe(guid, clip));
        }
    }

    /// <summary>
    /// POST /api/clip/events {"guid": "...", "events": [{"name": "hit", "time": 0.42, "bone": "RightHand"}]}
    /// Replace what is marked on a clip and save it with the file the clip is imported from (no reimport).
    /// An event's moment is "time" (0..1 of the clip), "seconds" or "frame"; "bone" is optional.
    /// </summary>
    [CommandRoute("POST", "/api/clip/events")]
    public class SetClipEventsCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            using var doc = context.ParseBody();
            var root = doc.RootElement;

            string guid = root.TryGetProperty("guid", out var g) ? g.GetString() : null;
            if (string.IsNullOrWhiteSpace(guid)) return CommandResult.BadRequest("'guid' is required");
            if (!root.TryGetProperty("events", out var list) || list.ValueKind != JsonValueKind.Array)
                return CommandResult.BadRequest("'events' must be an array (empty to clear)");

            var clip = ClipCommandHelpers.Find(guid);
            if (clip == null) return CommandResult.NotFound($"Animation clip not found: {guid}");

            var events = new List<AnimationEvent>();
            foreach (var item in list.EnumerateArray())
            {
                string name = item.TryGetProperty("name", out var n) ? n.GetString() : null;
                if (string.IsNullOrWhiteSpace(name)) return CommandResult.BadRequest("Every event needs a 'name'");
                if (!ClipCommandHelpers.TryReadTime(item, clip, out float time))
                    return CommandResult.BadRequest($"Event '{name}' needs 'time' (0..1), 'seconds' or 'frame'");

                HumanoidBone? bone = null;
                if (item.TryGetProperty("bone", out var b) && b.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(b.GetString()))
                {
                    if (!Enum.TryParse<HumanoidBone>(b.GetString(), ignoreCase: true, out var parsed) || parsed == HumanoidBone.Count)
                        return CommandResult.BadRequest($"Event '{name}': unknown bone '{b.GetString()}' (a HumanoidBone name, e.g. RightHand)");
                    bone = parsed;
                }

                events.Add(new AnimationEvent { Name = name, Time = time, Bone = bone });
            }

            events.Sort(static (a, b) => a.Time.CompareTo(b.Time));

            // Another list, never the same one changed: characters playing the clip read it
            clip.Events = events;
            if (!ModelImporter.SetClipEvents(clip))
                return CommandResult.Error(500, $"'{clip.Name}' does not come from a model file: its events cannot be saved");

            MessageDispatcher.Send(Msg.AssetReloaded, clip);
            return CommandResult.Json(ClipCommandHelpers.Describe(guid, clip));
        }
    }

    /// <summary>
    /// POST /api/clip/preview {"guid": "...", "id": 123, "time": 0.42, "hold": 5}
    /// Show a clip at one moment on a character for a while (Animator.Preview), to look at it in a screenshot.
    /// </summary>
    [CommandRoute("POST", "/api/clip/preview")]
    public class PreviewClipCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            using var doc = context.ParseBody();
            var root = doc.RootElement;

            string guid = root.TryGetProperty("guid", out var g) ? g.GetString() : null;
            if (string.IsNullOrWhiteSpace(guid)) return CommandResult.BadRequest("'guid' is required");
            if (!root.TryGetProperty("id", out var idProp) || idProp.ValueKind != JsonValueKind.Number)
                return CommandResult.BadRequest("'id' (the character's entity) is required");

            var clip = ClipCommandHelpers.Find(guid);
            if (clip == null) return CommandResult.NotFound($"Animation clip not found: {guid}");

            var entity = CommandHelpers.FindEntityById(idProp.GetInt32());
            var animator = entity?.GetComponent<Animator>();
            if (animator == null) return CommandResult.BadRequest("That entity has no Animator");

            if (!ClipCommandHelpers.TryReadTime(root, clip, out float time)) time = 0f;
            float hold = root.TryGetProperty("hold", out var h) && h.ValueKind == JsonValueKind.Number ? h.GetSingle() : 5f;
            hold = Math.Clamp(hold, 0f, 600f);

            if (hold <= 0f) animator.Preview(null, 0f, 0f);
            else animator.Preview(clip, time * clip.DurationSeconds, hold);

            return CommandResult.Json(new
            {
                status = hold > 0f ? "showing" : "ended",
                clip = clip.Name,
                on = entity.Name,
                time,
                seconds = time * clip.DurationSeconds,
                frame = time * clip.Duration,
                hold,
            });
        }
    }
}
