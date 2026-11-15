using Freefall.Base;
using Freefall.Components;

namespace Freefall.Editor.Commands
{
    /// <summary>
    /// GET /api/scene/fingerprint — for every top-level entity, a hash of the static meshes under it
    /// (which mesh, where). Two loads of the same scene, or the state before and after regenerating,
    /// can be compared entity by entity to find a generator whose output is not reproducible.
    /// 'exact' covers every bit of the world matrices; 'coarse' only positions rounded to a centimetre,
    /// so a difference in 'exact' alone is float noise.
    /// </summary>
    [CommandRoute("GET", "/api/scene/fingerprint")]
    public class SceneFingerprintCommand : EditorCommand
    {
        // Per top-level entity: meshes, exact and coarse sums
        private sealed class Sum { public int Count; public ulong Exact, Coarse; }

        public override CommandResult Execute(CommandContext context)
        {
            var entities = new System.Collections.Generic.List<object>();
            int total = 0;
            ulong totalExact = 0, totalCoarse = 0;
            var sums = new System.Collections.Generic.Dictionary<Entity, Sum>();

            foreach (var entity in EntityManager.Entities)
            {
                if (entity.Transform?.Parent != null) continue;
                var sum = new Sum();
                Visit(entity.Transform, sum);
                sums[entity] = sum;
            }

            // The world cells' records count under their generator's top-level entity, each with the matrix its entity
            // would have, in place of the near cells' copies (skipped above): the scene as with every placement an entity,
            // wherever the camera stands
            Freefall.World.WorldCells.VisitRecords((owner, mesh, world) =>
            {
                var top = owner?.Entity?.Transform;
                while (top?.Parent != null) top = top.Parent;
                if (top?.Entity == null || !sums.TryGetValue(top.Entity, out var sum)) return;
                Add(sum, mesh, world);
            });

            foreach (var (entity, sum) in sums)
            {
                if (sum.Count == 0) continue;
                total += sum.Count;
                totalExact += sum.Exact;
                totalCoarse += sum.Coarse;
                entities.Add(new
                {
                    uid = entity.UID.ToString(),
                    name = entity.Name,
                    meshes = sum.Count,
                    exact = sum.Exact.ToString("x16"),
                    coarse = sum.Coarse.ToString("x16"),
                });
            }

            return CommandResult.Json(new
            {
                meshes = total,
                exact = totalExact.ToString("x16"),
                coarse = totalCoarse.ToString("x16"),
                entities,
            });
        }

        // Sums, so the order entities are enumerated in does not matter
        private static void Visit(Transform transform, Sum sum)
        {
            if (transform == null) return;

            // A near world cell's copies: their records are counted instead
            if (Freefall.World.WorldCells.IsCellCopy(transform.Entity)) return;

            foreach (var component in transform.Entity.Components)
            {
                if (component is not MeshRenderer { Enabled: true, Mesh: { } mesh }) continue;
                // A world cells' far mesh stands in for records counted on their own
                if (mesh.CellFade) continue;
                Add(sum, mesh, transform.Matrix);
            }

            for (int i = 0; i < transform.Count; i++)
                Visit(transform.GetChild(i), sum);
        }

        private static void Add(Sum sum, Freefall.Graphics.Mesh mesh, System.Numerics.Matrix4x4 m)
        {
            // Generated meshes (RuntimeMesh) get a new GUID-based name on every build: hash what they contain
            ulong meshHash = mesh.IsDynamic ? Content(mesh) : Mix(Text(mesh.Name), (ulong)(mesh.Positions?.Length ?? 0));

            ulong e = meshHash;
            e = Mix(e, m.M11); e = Mix(e, m.M12); e = Mix(e, m.M13);
            e = Mix(e, m.M21); e = Mix(e, m.M22); e = Mix(e, m.M23);
            e = Mix(e, m.M31); e = Mix(e, m.M32); e = Mix(e, m.M33);
            e = Mix(e, m.M41); e = Mix(e, m.M42); e = Mix(e, m.M43);

            ulong c = meshHash;
            c = Mix(c, (ulong)(long)System.MathF.Round(m.M41 * 100f));
            c = Mix(c, (ulong)(long)System.MathF.Round(m.M42 * 100f));
            c = Mix(c, (ulong)(long)System.MathF.Round(m.M43 * 100f));

            sum.Count++;
            sum.Exact += e;
            sum.Coarse += c;
        }

        private static ulong Content(Freefall.Graphics.Mesh mesh)
        {
            ulong h = 1;
            if (mesh.Positions != null)
                foreach (var p in mesh.Positions)
                {
                    h = Mix(h, p.X); h = Mix(h, p.Y); h = Mix(h, p.Z);
                }
            return Mix(h, (ulong)(mesh.CpuIndices?.Length ?? 0));
        }

        // Deterministic across runs (string.GetHashCode is not)
        private static ulong Text(string text)
        {
            ulong h = 14695981039346656037UL;
            foreach (char ch in text ?? "")
                h = (h ^ ch) * 1099511628211UL;
            return h;
        }

        private static ulong Mix(ulong h, ulong v)
        {
            ulong x = h * 0x9E3779B97F4A7C15UL + v;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
            return x ^ (x >> 31);
        }

        private static ulong Mix(ulong h, float v) => Mix(h, (ulong)System.BitConverter.SingleToUInt32Bits(v));
    }
}
