using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text.Json;
using Freefall.UI;
using Squid;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Plays input into the running game (MCP play_input): keys, mouse buttons, mouse look, the game
    /// UI's cursor, waits. POST /api/play/input
    /// Body: {"steps": [step, ...]}, each step one of:
    ///   {"key": "W", "hold": 1.5}        press, hold that long (no hold: a tap of one frame), let go
    ///   {"down": "W"} / {"up": "W"}      a key down or up, for holding across other steps
    ///   {"mouseDown": "left"} / {"mouseUp": "left"}   left, right or middle
    ///   {"click": "left", "at": [x, y]}  the game UI's cursor to at (optional), press, let go
    ///   {"drag": {"from": [x, y], "to": [x, y], "over": 0.3, "button": "left"}}
    ///   {"cursor": [x, y]} / {"cursor": null}   the game UI's cursor there / the real mouse again
    ///   {"move": [x, y], "over": 0.3}    the game UI's cursor glides there
    ///   {"look": [dx, dy], "over": 0.5}  raw mouse movement: the camera turns (about 10 counts a degree)
    ///   {"scroll": 120}                  the wheel (120 = a notch)
    ///   {"wait": 0.5}                    seconds; 0 = one frame
    /// Coordinates are pixels of the game's picture from its top left, at the viewport's full size
    /// (screenshot target='viewport' reports it; GET /api/play/ui gives controls' rects in the same
    /// pixels). A new script stops one still playing and lets go of what that held; after one that
    /// finished, its cursor and what it holds stay (a drag can go on in the next call). Play mode only.
    /// </summary>
    [CommandRoute("POST", "/api/play/input")]
    public class PlayInputCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (Program.EditorUI?.IsPlaying != true)
                return CommandResult.BadRequest("Not playing: input goes to the game in play mode only (play_mode play=true)");
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required: {\"steps\": [...]}");

            using var doc = context.ParseBody();
            if (!doc.RootElement.TryGetProperty("steps", out var list) || list.ValueKind != JsonValueKind.Array)
                return CommandResult.BadRequest("'steps' must be an array");

            var steps = new List<InputScript.Step>();
            int index = 0;
            try
            {
                foreach (var step in list.EnumerateArray())
                {
                    Parse(step, steps);
                    index++;
                }
            }
            catch (Exception ex)
            {
                return CommandResult.BadRequest($"Step {index}: {ex.Message}");
            }

            var script = new InputScript(steps);
            Input.Play(script);
            return CommandResult.Json(new { status = "playing", steps = script.Count, seconds = script.Seconds });
        }

        static void Parse(JsonElement step, List<InputScript.Step> into)
        {
            if (step.ValueKind != JsonValueKind.Object) throw new InvalidOperationException("a step is an object");

            if (step.TryGetProperty("key", out var key))
            {
                var k = KeyOf(key.GetString());
                float hold = step.TryGetProperty("hold", out var h) ? h.GetSingle() : 0f;
                into.Add(new() { Kind = InputScript.Kind.KeyDown, Key = k });
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = hold });
                into.Add(new() { Kind = InputScript.Kind.KeyUp, Key = k });
            }
            else if (step.TryGetProperty("down", out var down))
                into.Add(new() { Kind = InputScript.Kind.KeyDown, Key = KeyOf(down.GetString()) });
            else if (step.TryGetProperty("up", out var up))
                into.Add(new() { Kind = InputScript.Kind.KeyUp, Key = KeyOf(up.GetString()) });
            else if (step.TryGetProperty("mouseDown", out var mouseDown))
                into.Add(new() { Kind = InputScript.Kind.ButtonDown, Button = ButtonOf(mouseDown.GetString()) });
            else if (step.TryGetProperty("mouseUp", out var mouseUp))
                into.Add(new() { Kind = InputScript.Kind.ButtonUp, Button = ButtonOf(mouseUp.GetString()) });
            else if (step.TryGetProperty("click", out var click))
            {
                int button = ButtonOf(click.ValueKind == JsonValueKind.String ? click.GetString() : "left");
                if (step.TryGetProperty("at", out var at))
                {
                    into.Add(new() { Kind = InputScript.Kind.Cursor, To = PointOf(at) });
                    into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = 0.05f });
                }
                into.Add(new() { Kind = InputScript.Kind.ButtonDown, Button = button });
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = 0.05f });
                into.Add(new() { Kind = InputScript.Kind.ButtonUp, Button = button });
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = 0f });
            }
            else if (step.TryGetProperty("drag", out var drag))
            {
                var from = PointOf(drag.GetProperty("from"));
                var to = PointOf(drag.GetProperty("to"));
                float over = drag.TryGetProperty("over", out var o) ? o.GetSingle() : 0.3f;
                int button = ButtonOf(drag.TryGetProperty("button", out var b) ? b.GetString() : "left");
                into.Add(new() { Kind = InputScript.Kind.Cursor, To = from });
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = 0.05f });
                into.Add(new() { Kind = InputScript.Kind.ButtonDown, Button = button });
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = 0.1f });
                into.Add(new() { Kind = InputScript.Kind.Move, To = to, Seconds = over });
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = 0.1f });
                into.Add(new() { Kind = InputScript.Kind.ButtonUp, Button = button });
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = 0.05f });
            }
            else if (step.TryGetProperty("cursor", out var cursor))
            {
                if (cursor.ValueKind == JsonValueKind.Null)
                    into.Add(new() { Kind = InputScript.Kind.ReleaseCursor });
                else
                    into.Add(new() { Kind = InputScript.Kind.Cursor, To = PointOf(cursor) });
            }
            else if (step.TryGetProperty("move", out var move))
                into.Add(new() { Kind = InputScript.Kind.Move, To = PointOf(move), Seconds = step.TryGetProperty("over", out var o) ? o.GetSingle() : 0.3f });
            else if (step.TryGetProperty("look", out var look))
                into.Add(new() { Kind = InputScript.Kind.Look, To = PointOf(look), Seconds = step.TryGetProperty("over", out var o) ? o.GetSingle() : 0.3f });
            else if (step.TryGetProperty("scroll", out var scroll))
                into.Add(new() { Kind = InputScript.Kind.Wheel, Amount = scroll.GetInt32() });
            else if (step.TryGetProperty("wait", out var wait))
                into.Add(new() { Kind = InputScript.Kind.Wait, Seconds = wait.GetSingle() });
            else
                throw new InvalidOperationException("unknown step (key, down, up, mouseDown, mouseUp, click, drag, cursor, move, look, scroll, wait)");
        }

        static Keys KeyOf(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) throw new InvalidOperationException("no key named");
            string n = name.Trim();
            if (n.Length == 1 && char.IsDigit(n[0])) n = "D" + n;
            n = n.ToLowerInvariant() switch
            {
                "shift" => "ShiftKey",
                "ctrl" or "control" => "ControlKey",
                "alt" => "Menu",
                "esc" => "Escape",
                "enter" => "Return",
                _ => n,
            };
            if (Enum.TryParse<Keys>(n, ignoreCase: true, out var key)) return key;
            throw new InvalidOperationException($"no key '{name}' (names of Freefall.Keys: W, Space, Escape, D2, ShiftKey, E...)");
        }

        static int ButtonOf(string name) => (name ?? "left").ToLowerInvariant() switch
        {
            "left" or "0" => 0,
            "right" or "1" => 1,
            "middle" or "2" => 2,
            _ => throw new InvalidOperationException($"no mouse button '{name}' (left, right, middle)"),
        };

        static Vector2 PointOf(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Array && element.GetArrayLength() >= 2)
                return new Vector2(element[0].GetSingle(), element[1].GetSingle());
            if (element.ValueKind == JsonValueKind.Object)
                return new Vector2(element.GetProperty("x").GetSingle(), element.GetProperty("y").GetSingle());
            throw new InvalidOperationException("a point is [x, y] or {\"x\", \"y\"}");
        }
    }

    /// <summary>GET /api/play/input: whether a script plays, and which step.</summary>
    [CommandRoute("GET", "/api/play/input")]
    public class PlayInputStatusCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var script = Input.Script;
            var cursor = Input.SimulatedCursor;
            return CommandResult.Json(new
            {
                playing = script != null,
                step = script?.Index,
                of = script?.Count,
                current = script?.Current,
                cursor = cursor is { } c ? new[] { c.X, c.Y } : null,
            });
        }
    }

    /// <summary>POST /api/play/input/stop: stops a script, lets go of what it held, gives the game UI the real mouse back.</summary>
    [CommandRoute("POST", "/api/play/input/stop")]
    public class StopPlayInputCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            Input.StopSimulation();
            return CommandResult.Json(new { status = "stopped" });
        }
    }

    /// <summary>
    /// GET /api/play/ui?filter=&amp;all=false: the game UI's controls that show, each with its type,
    /// name, style, text, tag (what it stands for: an item, a slot), rect [x, y, w, h] in pixels of
    /// the game's picture (as play_input takes them) and its path. filter = only those whose name,
    /// text, tag, style or type has it. all = also the hidden ones.
    /// </summary>
    [CommandRoute("GET", "/api/play/ui")]
    public class GameUITreeCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (!GameUI.Running)
                return CommandResult.BadRequest("The game's UI is not running (play mode only)");

            var query = CommandHelpers.ParseQueryString(context.Path);
            string filter = query.TryGetValue("filter", out var f) ? f : null;
            bool all = query.TryGetValue("all", out var a) && a == "true";
            var desktop = GameUI.Desktop;
            var origin = desktop.Position;
            var found = new List<object>();

            void Walk(Control control, string path, int depth)
            {
                if (control == null || depth > 24) return;
                if (!control.Visible && !all) return;

                string type = control.GetType().Name;
                string text = control is Label label ? label.Text : null;
                string tag = control.Tag?.ToString();
                string here = string.IsNullOrEmpty(path) ? Label(control, type) : path + "/" + Label(control, type);

                if (control != desktop)
                {
                    bool match = string.IsNullOrEmpty(filter)
                        || Has(control.Name, filter) || Has(text, filter) || Has(tag, filter) || Has(control.Style, filter) || Has(type, filter);
                    if (match && control.Size.x > 0 && control.Size.y > 0)
                    {
                        var at = control.Location;
                        found.Add(new
                        {
                            path = here,
                            type,
                            name = string.IsNullOrEmpty(control.Name) ? null : control.Name,
                            style = string.IsNullOrEmpty(control.Style) ? null : control.Style,
                            text = string.IsNullOrEmpty(text) ? null : text,
                            tag,
                            rect = new[] { at.x - origin.x, at.y - origin.y, control.Size.x, control.Size.y },
                            visible = control.Visible ? (bool?)null : false,
                            hot = desktop.HotControl == control ? true : (bool?)null,
                            selected = control is Button { Checked: true } || control is Label { Selected: true } ? true : (bool?)null,
                        });
                    }
                }

                // The desktop, frames, windows and tab pages all hold their children this way
                if (control is IControlContainer container)
                    foreach (var child in container.Controls)
                        Walk(child, here, depth + 1);
                foreach (var element in control.GetElements())
                    Walk(element, here, depth + 1);
            }

            Walk(desktop, string.Empty, 0);
            return CommandResult.Json(new
            {
                picture = new[] { desktop.Size.x, desktop.Size.y },
                menuOpen = GameUI.MenuOpen,
                count = found.Count,
                controls = found,
            });
        }

        static string Label(Control control, string type) => string.IsNullOrEmpty(control.Name) ? type : control.Name;

        static bool Has(string value, string filter)
            => !string.IsNullOrEmpty(value) && value.Contains(filter, StringComparison.OrdinalIgnoreCase);
    }
}
