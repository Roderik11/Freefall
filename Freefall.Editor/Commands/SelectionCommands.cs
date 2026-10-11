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

    /// <summary>
    /// Left-click a pixel of the scene viewport: selects what is under it through the GPU pick, as the mouse does.
    /// POST /api/viewport/click
    /// Body: {"x":640,"y":360} — viewport pixels (the space of a full-resolution viewport screenshot)
    /// </summary>
    [CommandRoute("POST", "/api/viewport/click")]
    public class ViewportClickCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            if (string.IsNullOrEmpty(context.Body))
                return CommandResult.BadRequest("Body required with 'x' and 'y'");

            using var doc = context.ParseBody();
            var root = doc.RootElement;
            if (!root.TryGetProperty("x", out var xp) || !root.TryGetProperty("y", out var yp))
                return CommandResult.BadRequest("'x' and 'y' are required");

            var viewport = Program.EditorUI?.SceneViewport;
            if (viewport == null || EditorTools.Instance == null)
                return CommandResult.BadRequest("No scene viewport — is a project open?");

            int x = xp.GetInt32(), y = yp.GetInt32();
            if (x < 0 || y < 0 || x >= viewport.Size.x || y >= viewport.Size.y)
                return CommandResult.BadRequest($"({x}, {y}) is outside the viewport ({viewport.Size.x}x{viewport.Size.y})");

            EditorTools.Instance.ClickPick(new Vortice.Mathematics.Int2(viewport.Location.x + x, viewport.Location.y + y));
            return CommandResult.Json(new { status = "clicked", x, y, note = "the pick resolves a frame or two later: read selection_get" });
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

            // An asset instead of an entity: the inspector shows whatever object is selected
            if (root.TryGetProperty("assetGuid", out var guidProp) && guidProp.ValueKind == System.Text.Json.JsonValueKind.String)
            {
                var asset = CommandHelpers.FindOrLoadAsset(guidProp.GetString());
                if (asset == null)
                    return CommandResult.NotFound($"Asset not found: {guidProp.GetString()}");

                Selector.SelectedObject = asset;
                return CommandResult.Json(new { selectedAsset = asset.Guid, type = asset.GetType().Name, name = asset.Name });
            }

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
