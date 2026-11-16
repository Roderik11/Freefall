using System.Collections.Generic;
using System.Linq;
using System.Text;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;

namespace Freefall.World
{
    /// <summary>
    /// Materials a placement draws with in place of its prefab's own (WORLD_CELLS_PLAN.md stage 6): a house's palette, a
    /// fortification's Look. A record keeps it with its kind (<see cref="WorldCells.Place(Component, Prefab, System.Numerics.Vector3,
    /// System.Numerics.Quaternion, System.Numerics.Vector3, MaterialSwap)"/>): the copies near a viewer get the swapped
    /// materials on their renderers, the instances and the far forms draw with them.
    /// <para>Shared: <see cref="Of"/> gives the same object for the same pairs, so every placement of a prefab with one
    /// palette is one kind, however many houses pick it. Read-only once made, so workers may read it.</para>
    /// </summary>
    public sealed class MaterialSwap
    {
        private readonly Dictionary<Material, Material> _map;

        private MaterialSwap(Dictionary<Material, Material> map) => _map = map;

        private static readonly Dictionary<string, MaterialSwap> s_shared = new();
        private static bool s_listening;

        /// <summary>
        /// The shared swap for these pairs (a prefab's material → the one to draw with), or null when none changes
        /// anything. Main thread.
        /// </summary>
        public static MaterialSwap? Of(IEnumerable<(Material? From, Material? To)> pairs)
        {
            var map = new Dictionary<Material, Material>();
            foreach (var (from, to) in pairs)
                if (from != null && to != null && !ReferenceEquals(from, to)) map[from] = to;
            if (map.Count == 0) return null;

            var key = new StringBuilder();
            foreach (var (from, to) in map.OrderBy(p => p.Key.MaterialID))
                key.Append(from.MaterialID).Append('>').Append(to.MaterialID).Append(';');
            if (!s_listening)
            {
                s_listening = true;
                SceneManager.LoadStarted += _ => s_shared.Clear();   // materials reloaded with a scene are new objects
            }
            var k = key.ToString();
            if (!s_shared.TryGetValue(k, out var swap)) s_shared[k] = swap = new MaterialSwap(map);
            return swap;
        }

        /// <summary>
        /// The shared swap that gives each material <paramref name="prefab"/>'s renderers draw with the one
        /// <paramref name="variant"/> names (null keeps it): a Look or a palette as a function of the material. Callers
        /// placing many copies keep the result per prefab. Main thread.
        /// </summary>
        public static MaterialSwap? For(Assets.Prefab prefab, System.Func<Material, Material?> variant)
        {
            if (prefab == null) return null;
            var pairs = new List<(Material?, Material?)>();
            var seen = new HashSet<Material>();
            foreach (var rd in PrefabShapes.RenderersOf(prefab))
                foreach (var (_, material) in rd.Chains)
                    if (material != null && seen.Add(material)) pairs.Add((material, variant(material)));
            return Of(pairs);
        }

        /// <summary>The material drawn in place of <paramref name="material"/>: its swap, or itself.</summary>
        public Material Map(Material material) => material != null && _map.TryGetValue(material, out var to) ? to : material!;

        /// <summary>The swap on every renderer of a copy and under it (its material and its slots' overrides).</summary>
        public void ApplyTo(Entity copy)
        {
            var stack = new Stack<Transform>();
            stack.Push(copy.Transform);
            while (stack.Count > 0)
            {
                var t = stack.Pop();
                if (t.Entity?.GetComponent<MeshRenderer>() is { } mr)
                {
                    bool changed = false;
                    var list = new List<MaterialOverride>(mr.Materials.Count);
                    foreach (var o in mr.Materials)
                    {
                        var m = o.Material != null && _map.TryGetValue(o.Material, out var to) ? to : o.Material;
                        changed |= !ReferenceEquals(m, o.Material);
                        list.Add(new MaterialOverride { MaterialSlot = o.MaterialSlot, Material = m });
                    }
                    if (changed) mr.Materials = list;
                    if (mr.Material != null && _map.TryGetValue(mr.Material, out var own)) mr.Material = own;
                }
                for (int i = 0; i < t.Count; i++) stack.Push(t.GetChild(i));
            }
        }
    }
}
