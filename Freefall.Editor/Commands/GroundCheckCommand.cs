using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// POST /api/scene/ground-check {center:{x,z}, radius, tolerance, rise, ring, minSize, limit, ignore:["*Cliffs*"]}: how every placed piece in a
    /// circle stands on the terrain, so a change to the ground or to what stands on it can be checked all round, not only
    /// where the work was (2026-11-17, the user: "while you focus on fixing one thing, you dont really pay attention to
    /// anything else"). Every enabled mesh counts: entities' MeshRenderers and the world cells' records alike (a kit's
    /// walls, towers and houses are records), each piece on its own.
    ///
    /// Under each piece the terrain is sampled over its footprint (its mesh's bounds, an ellipse inside them, so a round
    /// tower is not judged by the corners of its box), against the bottom of the piece there:
    ///   over a drop     some samples stand on the ground and some hang more than 'tolerance' above it: a tower or wall
    ///                   half over a cliff or a fallen bank
    ///   hovering        all samples hang above the ground, by more than 'tolerance' but less than 2 m (higher than that
    ///                   the piece stands on another one: a tower's upper storeys, a roof)
    ///   against a bank  the piece is on the ground, but just outside its footprint ('ring' m out) the ground rises more
    ///                   than 'rise' m above its footing: a house beside a raised ramp, a wall buried on one side
    /// The ground is the terrain only: a piece on a bridge, a deck or raised paving shows as hovering.
    /// </summary>
    [CommandRoute("POST", "/api/scene/ground-check")]
    public class GroundCheckCommand : EditorCommand
    {
        private sealed class Finding
        {
            public string Kind = "";
            public float Amount;
            public float Share;
            public string Owner = "";
            public string OwnerUid = "";
            public string Mesh = "";
            public Vector3 At;
        }

        public override CommandResult Execute(CommandContext context)
        {
            var terrain = TerrainHeightCommand.FindTerrainRenderer();
            if (terrain == null) return CommandResult.NotFound("No TerrainRenderer found in scene");

            using var doc = context.ParseBody();
            var root = doc.RootElement;
            if (!root.TryGetProperty("center", out var c)) return CommandResult.BadRequest("Body needs 'center' {x, z}");
            var center = new Vector2(c.GetProperty("x").GetSingle(), c.GetProperty("z").GetSingle());
            float Get(string name, float fallback) => root.TryGetProperty(name, out var v) && v.ValueKind == System.Text.Json.JsonValueKind.Number ? v.GetSingle() : fallback;
            float radius = Get("radius", 150f), tolerance = Get("tolerance", 0.3f), rise = Get("rise", 2.5f), ring = Get("ring", 3f);
            float minSize = Get("minSize", 1.5f);
            int limit = (int)Get("limit", 60f);
            // owners or meshes to leave out, by name with * wildcards (cliff rocks stand against banks on purpose)
            var ignore = new List<System.Text.RegularExpressions.Regex>();
            if (root.TryGetProperty("ignore", out var ig) && ig.ValueKind == System.Text.Json.JsonValueKind.Array)
                foreach (var pat in ig.EnumerateArray())
                    ignore.Add(new System.Text.RegularExpressions.Regex("^" + System.Text.RegularExpressions.Regex.Escape(pat.GetString() ?? "").Replace(@"\*", ".*") + "$",
                        System.Text.RegularExpressions.RegexOptions.IgnoreCase));
            bool Ignored(string? name) => name != null && ignore.Any(r => r.IsMatch(name));

            var findings = new List<Finding>();
            int pieces = 0;

            void Check(Component? owner, Entity? entity, Freefall.Graphics.Mesh mesh, Matrix4x4 world)
            {
                if (mesh == null || mesh.CellFade) return;
                if (Ignored(mesh.Name) || Ignored((owner?.Entity ?? entity)?.Name)) return;
                var box = mesh.BoundingBox;
                var lo = box.Min; var hi = box.Max;
                var mid = Vector3.Transform((lo + hi) * 0.5f, world);
                if (Vector2.Distance(new Vector2(mid.X, mid.Z), center) > radius) return;
                // its size on the ground, in world metres
                var ax = Vector3.TransformNormal(new Vector3(hi.X - lo.X, 0, 0), world);
                var az = Vector3.TransformNormal(new Vector3(0, 0, hi.Z - lo.Z), world);
                if (new Vector2(ax.X, ax.Z).Length() < minSize || new Vector2(az.X, az.Z).Length() < minSize) return;
                pieces++;

                // the footprint: points in an ellipse inside the bounds' bottom face, and a ring 'ring' m outside it
                float hx = (hi.X - lo.X) * 0.5f, hz = (hi.Z - lo.Z) * 0.5f;
                float cx = (hi.X + lo.X) * 0.5f, cz = (hi.Z + lo.Z) * 0.5f;
                float worldHx = new Vector2(ax.X, ax.Z).Length() * 0.5f, worldHz = new Vector2(az.X, az.Z).Length() * 0.5f;
                int under = 0, floating = 0;
                float maxGap = float.MinValue, minGap = float.MaxValue, footMax = float.MinValue;
                for (int i = -2; i <= 2; i++)
                    for (int j = -2; j <= 2; j++)
                    {
                        float u = i / 2.5f, v = j / 2.5f;
                        if (u * u + v * v > 0.81f) continue;
                        var p = Vector3.Transform(new Vector3(cx + u * hx, lo.Y, cz + v * hz), world);
                        if (terrain.IsHole(p)) continue;
                        float g = terrain.GetHeight(p);
                        float gap = p.Y - g;
                        maxGap = MathF.Max(maxGap, gap); minGap = MathF.Min(minGap, gap);
                        footMax = MathF.Max(footMax, g);
                        if (gap > tolerance) floating++; else under++;
                    }
                int n = under + floating;
                if (n == 0) return;

                Finding? f = null;
                if (floating > 0 && under > 0)
                    f = new Finding { Kind = "over a drop", Amount = maxGap, Share = floating / (float)n };
                else if (floating == n && minGap < 2f)
                    f = new Finding { Kind = "hovering", Amount = minGap, Share = 1f };
                else if (under > 0)
                {
                    // the ground just outside the footprint, against the highest ground under it
                    float footing = footMax;
                    float worst = float.MinValue; int high = 0, ringN = 0;
                    for (int k = 0; k < 16; k++)
                    {
                        float a = k * MathF.PI / 8f;
                        float u = MathF.Cos(a) * (1f + ring / MathF.Max(worldHx, 0.5f));
                        float v = MathF.Sin(a) * (1f + ring / MathF.Max(worldHz, 0.5f));
                        var p = Vector3.Transform(new Vector3(cx + u * hx, lo.Y, cz + v * hz), world);
                        if (terrain.IsHole(p)) continue;
                        float d = terrain.GetHeight(p) - footing;
                        ringN++;
                        worst = MathF.Max(worst, d);
                        if (d > rise) high++;
                    }
                    if (high > 0) f = new Finding { Kind = "against a bank", Amount = worst, Share = high / (float)ringN };
                }
                if (f == null) return;

                var top = owner?.Entity ?? entity;
                f.Owner = top?.Name ?? "?";
                f.OwnerUid = top?.UID.ToString() ?? "";
                f.Mesh = mesh.Name ?? "";
                f.At = mid;
                findings.Add(f);
            }

            // entities' renderers (not a near cell's copies: their records are checked below)
            foreach (var entity in EntityManager.Entities)
            {
                if (entity.Transform == null || Freefall.World.WorldCells.IsCellCopy(entity)) continue;
                foreach (var component in entity.Components)
                    if (component is MeshRenderer { Enabled: true, Mesh: { } mesh } && component is not TerrainRenderer)
                        Check(null, entity, mesh, entity.Transform.WorldMatrix);
            }
            // the world cells' records
            Freefall.World.WorldCells.VisitRecordsIn(center - new Vector2(radius), center + new Vector2(radius),
                (owner, mesh, world) => Check(owner, null, mesh, world));

            var byOwner = findings.GroupBy(f => (f.Owner, f.OwnerUid, f.Kind))
                .Select(g => new
                {
                    owner = g.Key.Owner,
                    ownerUid = g.Key.OwnerUid,
                    kind = g.Key.Kind,
                    pieces = g.Count(),
                    worst = MathF.Round(g.Max(f => f.Amount), 2),
                })
                .OrderByDescending(g => g.worst).ToList();
            var worstFirst = findings.OrderByDescending(f => f.Kind == "hovering" ? f.Amount * 3 : f.Amount).Take(limit).Select(f => new
            {
                kind = f.Kind,
                amount = MathF.Round(f.Amount, 2),
                share = MathF.Round(f.Share, 2),
                owner = f.Owner,
                ownerUid = f.OwnerUid,
                mesh = f.Mesh,
                at = new { x = MathF.Round(f.At.X, 1), y = MathF.Round(f.At.Y, 1), z = MathF.Round(f.At.Z, 1) },
            }).ToList();

            return CommandResult.Json(new
            {
                center = new { x = center.X, z = center.Y },
                radius,
                piecesChecked = pieces,
                findings = findings.Count,
                byOwner,
                worstFirst,
                note = "amount: over a drop = the most a sample hangs above the ground (m), share = the part that hangs; hovering = the " +
                       "least it hangs; against a bank = how far the ground just outside rises over its footing. The ground is the terrain only.",
            });
        }
    }
}
