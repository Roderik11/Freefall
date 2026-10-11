using System.Linq;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// Start a navmesh bake on the scene's NavMeshSurface. The bake runs in the background;
    /// poll GET /api/navmesh/status for progress.
    /// POST /api/navmesh/bake with optional body: {"rebuildAll": true}
    /// </summary>
    [CommandRoute("POST", "/api/navmesh/bake")]
    public class NavMeshBakeCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var surface = FindSurface();
            if (surface == null)
                return CommandResult.NotFound("No NavMeshSurface found in scene");

            using var doc = context.ParseBody();
            bool rebuildAll = doc.RootElement.TryGetProperty("rebuildAll", out var value) && value.ValueKind == System.Text.Json.JsonValueKind.True;

            bool alreadyBaking = surface.IsBaking;
            if (!alreadyBaking)
                surface.Bake(rebuildAll);

            return CommandResult.Json(Status(surface, alreadyBaking));
        }

        internal static NavMeshSurface FindSurface()
        {
            var surfaces = ComponentCache<NavMeshSurface>.All;
            return surfaces.Count > 0 ? surfaces[0] : null;
        }

        internal static object Status(NavMeshSurface surface, bool alreadyBaking = false)
        {
            var bake = surface.GetBake();
            var result = bake?.Result;
            var asset = surface.NavMesh;

            return new
            {
                baking = surface.IsBaking,
                alreadyBaking,
                stage = bake?.Stage.ToString(),
                doneTiles = bake?.DoneCells,
                totalTiles = bake?.TotalCells,
                error = bake?.Error,
                lastBake = result == null ? null : new
                {
                    seconds = System.Math.Round(result.Seconds, 2),
                    rebuiltTiles = result.RebuiltCells,
                    unchangedTiles = result.ReusedCells,
                    // Where the scene differs from the previous bake: [minX, minZ, maxX, maxZ] per changed tile
                    changedTiles = result.ChangedRects.Select(r => new[] { r.min.X, r.min.Y, r.max.X, r.max.Y }).ToArray(),
                    changedTilesTruncated = result.RebuiltCells > result.ChangedRects.Length && result.ChangedRects.Length > 0,
                    profile = result.Profile,
                },
                navMesh = asset == null ? null : new
                {
                    name = asset.Name,
                    guid = asset.Guid,
                    polys = asset.PolyCount,
                    verts = asset.VertexCount,
                    saved = !asset.IsDirty,
                },
                entity = surface.Entity == null ? null : new { id = surface.Entity.Id, uid = surface.Entity.UID.ToString(), name = surface.Entity.Name },
            };
        }
    }

    /// <summary>GET /api/navmesh/status</summary>
    [CommandRoute("GET", "/api/navmesh/status")]
    public class NavMeshStatusCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var surface = NavMeshBakeCommand.FindSurface();
            if (surface == null)
                return CommandResult.NotFound("No NavMeshSurface found in scene");

            return CommandResult.Json(NavMeshBakeCommand.Status(surface));
        }
    }

    /// <summary>POST /api/navmesh/cancel</summary>
    [CommandRoute("POST", "/api/navmesh/cancel")]
    public class NavMeshCancelCommand : EditorCommand
    {
        public override CommandResult Execute(CommandContext context)
        {
            var surface = NavMeshBakeCommand.FindSurface();
            if (surface == null)
                return CommandResult.NotFound("No NavMeshSurface found in scene");

            surface.CancelBake();
            return CommandResult.Json(NavMeshBakeCommand.Status(surface));
        }
    }
}
