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
        public override CommandResult Execute(CommandContext context)
        {
            var entities = new System.Collections.Generic.List<object>();
            int total = 0;
            ulong totalExact = 0, totalCoarse = 0;

            foreach (var entity in EntityManager.Entities)
            {
                if (entity.Transform?.Parent != null) continue;

                int count = 0;
                ulong exact = 0, coarse = 0;
                Visit(entity.Transform, ref count, ref exact, ref coarse);
                if (count == 0) continue;

                total += count;
                totalExact += exact;
                totalCoarse += coarse;
                entities.Add(new
                {
                    uid = entity.UID.ToString(),
                    name = entity.Name,
                    meshes = count,
                    exact = exact.ToString("x16"),
                    coarse = coarse.ToString("x16"),
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
        private static void Visit(Transform transform, ref int count, ref ulong exact, ref ulong coarse)
        {
            if (transform == null) return;

            foreach (var component in transform.Entity.Components)
            {
                if (component is not MeshRenderer { Enabled: true, Mesh: { } mesh }) continue;

                // Generated meshes (RuntimeMesh) get a new GUID-based name on every build: hash what they contain
                ulong meshHash = mesh.IsDynamic ? Content(mesh) : Mix(Text(mesh.Name), (ulong)(mesh.Positions?.Length ?? 0));
                var m = transform.Matrix;

                ulong e = meshHash;
                e = Mix(e, m.M11); e = Mix(e, m.M12); e = Mix(e, m.M13);
                e = Mix(e, m.M21); e = Mix(e, m.M22); e = Mix(e, m.M23);
                e = Mix(e, m.M31); e = Mix(e, m.M32); e = Mix(e, m.M33);
                e = Mix(e, m.M41); e = Mix(e, m.M42); e = Mix(e, m.M43);

                ulong c = meshHash;
                c = Mix(c, (ulong)(long)System.MathF.Round(m.M41 * 100f));
                c = Mix(c, (ulong)(long)System.MathF.Round(m.M42 * 100f));
                c = Mix(c, (ulong)(long)System.MathF.Round(m.M43 * 100f));

                count++;
                exact += e;
                coarse += c;
            }

            for (int i = 0; i < transform.Count; i++)
                Visit(transform.GetChild(i), ref count, ref exact, ref coarse);
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
