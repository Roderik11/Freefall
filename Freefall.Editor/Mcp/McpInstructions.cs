namespace Freefall.Editor.Mcp
{
    /// <summary>Sent to MCP clients on initialize — conventions an agent needs before its first call.</summary>
    public static class McpInstructions
    {
        public const string Text = """
            Controls the running Freefall editor (DX12 game engine). Every call executes on the editor's main thread.

            Workflow
            - Start with editor_status. If no project is open, use project_list_recent + project_open.
            - Scene elements are component compositions (e.g. a paved street = Spline + MeshRenderer + RuntimeMesh; a country
              road = Spline + SplatStamp + HeightStamp + PCG). Build each with one entity_build call. The how-to for roads,
              towns, terrain stamps, PCG, lighting and verified asset facts lives in the repo's Claude Code skill
              'freefall-scene-building' (.claude/skills/freefall-scene-building) — follow it when building scenes.
            - Verify visual changes with screenshot: target='editor' shows the whole UI (inspector, console, panels),
              target='viewport' only the 3D view; use 'crop' at maxSize=0 to inspect small details at full resolution.
            - After an action, console_log shows warnings/errors it produced (console_clear first to isolate them).

            Entities & components
            - Entity ids are ints that are only valid until the next scene load or editor restart. Every entity result also
              carries a 'uid' string (persistent, saved in the scene); every entity tool accepts uid instead of id. Keep uids for
              anything you will touch again later. Generated PCG output gets new uids on every regeneration.
            - Member names in entity_set_properties / asset_set_properties are exact PascalCase C# names; read them from
              entity_get / asset_get first. Only top-level members can be set (no 'A.B' paths): lists are replaced whole
              with a JSON array, nested data objects (e.g. Terrain.Layers entries) with a JSON object of their members.
            - Value encodings: vectors {x,y,z} or [x,y,z]; quaternions {x,y,z,w}; Color3/Color4 [r,g,b(,a)]; enums by name;
              asset references by GUID string; component references {"entity": id, "component": "Type"}.
            - Prefer prefab GUIDs over mesh GUIDs for entity_instantiate / entity_scatter.
            - Terrain is authored with stamp components on entities (TerrainStamp/HeightStamp/SplatStamp/DecoStamp), edited
              via entity_add_component + entity_set_properties.

            Assets
            - The editor loads assets from its Library cache: editing .asset files on disk does NOT reach it. Use asset_set_properties
              (saves + reimports), and asset_create for new assets. Material texture slots: material_set_textures.

            Placement & PCG (prefer PCG for anything repeated)
            - Scatter vegetation, rocks, clutter, lanterns etc. with PCG: an entity with a Spline (closed = area) and a
              PCGComponent whose Graph is a PCGGraph asset. Moving/reshaping the spline re-runs it; output is regenerated, never saved.
            - Spline entity pivots belong at the centre of their points, not at the origin, so users can move them by the transform.
              entity_build takes world points and sets the pivot. After editing Spline.Points directly, call spline_recenter.
            - Author graphs with graph_node_types → graph_get → graph_edit (add/set/remove/connect in one call; re-runs users).
              Key nodes: SplineSampler (Area / PerEdge + EdgeInset/OrientOutward), TerrainProjection, HeightFilter, SlopeFilter,
              DensityNoise(Scale)+DensityFilter, ExcludeStamps (roads/fields), ExcludeObstacles (buildings/props), TranformPoints,
              SelfPruning, SpawnPrefab. pcg_execute reports spawn counts.
            - prefab_inspect before placing buildings/lights: door sides + ground-level flag, working lights/emitters, problems.
            - prefab_measure gives prefab sizes without placing them; scene_query / entities_delete work on many entities at once.

            Limits
            - C# engine/editor changes need a rebuild and editor restart (editor_shutdown first; the DLL is locked while running).
            - Shaders/effects compile at startup — no hot reload; a restart is needed to see shader edits.
            - This server is stateless: after an editor restart, calls simply work again once it is back up.
            """;
    }
}
