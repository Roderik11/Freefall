using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Procedural;

namespace Freefall.Editor.Tools
{
    /// <summary>
    /// Tile definitions and mesh generation for WFC-driven building facades.
    /// Each tile represents a (cellWidth × storyHeight) panel on a building face.
    /// </summary>
    public static class BuildingTiles
    {
        // ═══════════════════════════
        // ── Tile IDs ──
        // ═══════════════════════════

        public const int WallSolid = 0;     // Plain wall panel
        public const int WallWindow = 1;    // Wall with window cutout
        public const int WallDoor = 2;      // Wall with door (ground floor only)
        public const int WallShop = 3;      // Wide opening (ground floor, like a shop front)
        public const int WallNarrow = 4;    // Narrow decorative panel (half-width look)
        public const int RoofFlat = 5;      // Flat roof cap
        public const int RoofSlope = 6;     // Sloped roof panel
        public const int TileCount = 7;

        /// <summary>
        /// Create the WFC ruleset for building facades.
        /// Grid convention: Y=0 is ground floor, Y=max is roof row.
        /// </summary>
        public static WFCSolver.RuleSet CreateRuleSet()
        {
            var rules = new WFCSolver.RuleSet(TileCount);

            // ── Weights (selection probability) ──
            // Balanced so WFC produces varied facades
            rules.SetWeight(WallSolid, 3.0f);     // Common filler
            rules.SetWeight(WallWindow, 2.5f);    // Frequent but not dominant
            rules.SetWeight(WallDoor, 1.5f);      // Constrained to ground floor
            rules.SetWeight(WallShop, 1.0f);      // Ground-floor accent
            rules.SetWeight(WallNarrow, 2.0f);    // Visual variety
            rules.SetWeight(RoofFlat, 1.0f);
            rules.SetWeight(RoofSlope, 1.0f);

            // ── Horizontal adjacency (Left/Right) ──
            // Most wall tiles can be next to each other horizontally
            int[] wallTiles = { WallSolid, WallWindow, WallDoor, WallShop, WallNarrow };
            foreach (int a in wallTiles)
                foreach (int b in wallTiles)
                    rules.Allow(a, WFCSolver.RuleSet.Right, b);

            // Roof tiles can be next to each other
            rules.Allow(RoofFlat, WFCSolver.RuleSet.Right, RoofFlat);
            rules.Allow(RoofSlope, WFCSolver.RuleSet.Right, RoofSlope);
            rules.Allow(RoofFlat, WFCSolver.RuleSet.Right, RoofSlope);

            // ── Vertical adjacency (Up/Down) ──
            // Wall tiles can stack vertically
            foreach (int a in wallTiles)
                foreach (int b in wallTiles)
                    rules.Allow(a, WFCSolver.RuleSet.Up, b);

            // Roof on top of any wall
            foreach (int w in wallTiles)
            {
                rules.Allow(w, WFCSolver.RuleSet.Up, RoofFlat);
                rules.Allow(w, WFCSolver.RuleSet.Up, RoofSlope);
            }

            // No wall tiles above roof (roof must be top row — enforced by pre-constraint)
            // No roof below wall (enforced by pre-constraint)

            return rules;
        }

        /// <summary>
        /// Set up pre-constraints on a WFC grid for a building face.
        /// Bans invalid tiles based on row position.
        /// </summary>
        public static void ApplyBuildingConstraints(WFCSolver solver, int stories, bool hasDoor)
        {
            int w = solver.Width;
            int h = solver.Height; // = stories (wall rows only, no roof row)

            for (int x = 0; x < w; x++)
            {
                for (int y = 0; y < h; y++)
                {
                    bool isGroundFloor = (y == 0);

                    // ALL rows: no roof tiles (ear-clip cap handles the roof)
                    solver.Ban(x, y, RoofFlat);
                    solver.Ban(x, y, RoofSlope);

                    if (!isGroundFloor)
                    {
                        // Upper floors: no doors/shops
                        solver.Ban(x, y, WallDoor);
                        solver.Ban(x, y, WallShop);
                    }
                }
            }

            // Force a door on the ground floor of this face
            if (hasDoor && w > 0)
            {
                int doorX = w / 2;
                solver.Constrain(doorX, 0, WallDoor);
            }
        }

        // ═══════════════════════════════════
        // ── Mesh Panel Generation ──
        // ═══════════════════════════════════

        /// <summary>
        /// Generate vertices/indices for a single tile panel.
        /// Panel occupies a quad from (0,0,0) to (width, height, 0) in local space.
        /// The caller transforms this to world space per edge/story.
        /// </summary>
        public static void GenerateTileMesh(
            int tileId, float cellWidth, float storyHeight,
            float windowWidth, float windowHeight, float windowInset,
            float doorWidth, float doorHeight,
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices,
            Vector3 origin, Vector3 right, Vector3 up, Vector3 outNormal)
        {
            switch (tileId)
            {
                case WallSolid:
                case WallNarrow:
                    EmitQuad(verts, norms, uvs, indices, origin, right * cellWidth, up * storyHeight, outNormal);
                    break;

                case WallWindow:
                    EmitWallWithOpening(verts, norms, uvs, indices, origin, right, up, outNormal,
                        cellWidth, storyHeight, windowWidth, windowHeight, windowInset,
                        storyHeight * 0.35f); // Window sill at 35% of story height
                    break;

                case WallDoor:
                    EmitWallWithOpening(verts, norms, uvs, indices, origin, right, up, outNormal,
                        cellWidth, storyHeight, doorWidth, doorHeight, windowInset,
                        0f); // Door starts at ground
                    break;

                case WallShop:
                    EmitWallWithOpening(verts, norms, uvs, indices, origin, right, up, outNormal,
                        cellWidth, storyHeight, cellWidth * 0.8f, storyHeight * 0.7f, windowInset,
                        0f); // Wide shop opening
                    break;

                case RoofFlat:
                    // Flat roof: horizontal quad on top
                    EmitQuad(verts, norms, uvs, indices, origin, right * cellWidth, up * 0.1f, outNormal);
                    break;

                case RoofSlope:
                    // Sloped panel (slight angle inward)
                    var slopeUp = Vector3.Normalize(up - outNormal * 0.3f);
                    EmitQuad(verts, norms, uvs, indices, origin, right * cellWidth, slopeUp * storyHeight * 0.5f,
                        Vector3.Normalize(outNormal + up * 0.3f));
                    break;
            }
        }

        /// <summary>
        /// Emit a wall quad with a rectangular opening (window or door).
        /// Creates the frame around the opening (8 quads forming an L-frame).
        /// </summary>
        private static void EmitWallWithOpening(
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices,
            Vector3 origin, Vector3 right, Vector3 up, Vector3 outNormal,
            float wallW, float wallH,
            float openW, float openH, float inset,
            float sillHeight)
        {
            // Opening bounds (centered horizontally)
            float oLeft = (wallW - openW) * 0.5f;
            float oRight = oLeft + openW;
            float oBottom = sillHeight;
            float oTop = sillHeight + openH;

            // Clamp
            if (oTop > wallH) oTop = wallH;
            if (oRight > wallW) oRight = wallW;

            // 5 quads forming the frame around the opening:
            // 1. Bottom strip (below opening)
            if (oBottom > 0.001f)
                EmitQuad(verts, norms, uvs, indices, origin, right * wallW, up * oBottom, outNormal);

            // 2. Top strip (above opening)
            if (oTop < wallH - 0.001f)
                EmitQuad(verts, norms, uvs, indices,
                    origin + up * oTop, right * wallW, up * (wallH - oTop), outNormal);

            // 3. Left strip (beside opening)
            if (oLeft > 0.001f)
                EmitQuad(verts, norms, uvs, indices,
                    origin + up * oBottom, right * oLeft, up * (oTop - oBottom), outNormal);

            // 4. Right strip (beside opening)
            if (wallW - oRight > 0.001f)
                EmitQuad(verts, norms, uvs, indices,
                    origin + right * oRight + up * oBottom,
                    right * (wallW - oRight), up * (oTop - oBottom), outNormal);

            // 5. Inset back wall (the recessed surface inside the opening)
            if (inset > 0.001f)
            {
                var insetOrigin = origin + right * oLeft + up * oBottom - outNormal * inset;
                EmitQuad(verts, norms, uvs, indices,
                    insetOrigin, right * openW, up * (oTop - oBottom), outNormal);

                // Inset side walls (reveal/jamb)
                var insetN = outNormal * inset;
                // Left jamb
                EmitQuad(verts, norms, uvs, indices,
                    origin + right * oLeft + up * oBottom,
                    -outNormal * inset, up * (oTop - oBottom),
                    -right);
                // Right jamb
                EmitQuad(verts, norms, uvs, indices,
                    origin + right * oRight + up * oBottom - outNormal * inset,
                    outNormal * inset, up * (oTop - oBottom),
                    right);
                // Top reveal (soffit)
                EmitQuad(verts, norms, uvs, indices,
                    origin + right * oLeft + up * oTop,
                    right * openW, -outNormal * inset,
                    up);
                // Bottom reveal (sill cap) — only for windows, not doors
                if (sillHeight > 0.001f)
                {
                    EmitQuad(verts, norms, uvs, indices,
                        origin + right * oLeft + up * oBottom - outNormal * inset,
                        right * openW, outNormal * inset,
                        -up);
                }
            }
        }

        /// <summary>Emit a simple quad (2 triangles) into the mesh buffers.</summary>
        private static void EmitQuad(
            List<Vector3> verts, List<Vector3> norms, List<Vector2> uvs, List<uint> indices,
            Vector3 origin, Vector3 axisU, Vector3 axisV, Vector3 normal)
        {
            uint baseIdx = (uint)verts.Count;

            verts.Add(origin);                     // BL
            verts.Add(origin + axisU);             // BR
            verts.Add(origin + axisV);             // TL
            verts.Add(origin + axisU + axisV);     // TR

            norms.Add(normal);
            norms.Add(normal);
            norms.Add(normal);
            norms.Add(normal);

            float uLen = axisU.Length();
            float vLen = axisV.Length();
            uvs.Add(new Vector2(0, 0));
            uvs.Add(new Vector2(uLen, 0));
            uvs.Add(new Vector2(0, vLen));
            uvs.Add(new Vector2(uLen, vLen));

            indices.Add(baseIdx + 0);
            indices.Add(baseIdx + 2);
            indices.Add(baseIdx + 1);
            indices.Add(baseIdx + 1);
            indices.Add(baseIdx + 2);
            indices.Add(baseIdx + 3);
        }
    }
}
