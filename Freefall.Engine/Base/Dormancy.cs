using System.Collections.Generic;
using Freefall.Components;

namespace Freefall.Base
{
    /// <summary>
    /// A component told when its entity is put to sleep and woken (<see cref="Dormancy"/>): a body waiting in a
    /// <see cref="Freefall.Assets.PrefabPool"/>, an NPC far from every viewer. Drop on sleep what points elsewhere (a
    /// target, a partner, a capsule in the physics scene); set back on wake what must be found fresh. A pooled body
    /// serves many NPCs one after another.
    /// </summary>
    public interface IDormant
    {
        /// <summary>The entity goes to sleep; its components are disabled right after.</summary>
        void OnSleep();

        /// <summary>The entity wakes; its components were enabled as they were just before.</summary>
        void OnWake();
    }

    /// <summary>
    /// An entity and everything under it asleep: its <see cref="IDormant"/> components told, then every enabled
    /// component disabled. A disabled component does not update or draw; renderers leave the draws, the Animator is
    /// neither stepped nor posed, audio fades, nav agents leave the crowd. <see cref="Wake"/> enables them again and
    /// tells <see cref="IDormant"/>.
    ///
    /// What was disabled is remembered by entity and place in its component list, never as a component: a script
    /// component held by the engine would keep its assembly alive after a reload, and the reload puts each component's
    /// copy in the same place, so the copy is enabled. A sleeping entity is still a scene entity: one that is saved
    /// while asleep is saved with its components disabled, so only sleep what is not saved or is woken before a save.
    /// </summary>
    public sealed class Dormancy
    {
        private readonly List<(Entity Owner, int Index)> _disabled = new();
        private static readonly List<IDormant> s_told = new();

        /// <summary>The entity that sleeps.</summary>
        public Entity Root { get; }

        /// <summary>Woken already.</summary>
        public bool Awake { get; private set; }

        private Dormancy(Entity root) => Root = root;

        /// <summary>Put an entity and everything under it to sleep.</summary>
        public static Dormancy Sleep(Entity root)
        {
            var dormancy = new Dormancy(root);
            if (root?.Transform == null) return dormancy;

            Collect(root.Transform, s_told);
            foreach (var dormant in s_told) dormant.OnSleep();
            s_told.Clear();

            Disable(root.Transform, dormancy._disabled);
            return dormancy;
        }

        /// <summary>Enable what was enabled when it went to sleep, and tell its <see cref="IDormant"/> components.</summary>
        public void Wake()
        {
            if (Awake) return;
            Awake = true;
            if (Root?.Transform == null || Root.Transform.IsDestroyed) return;

            foreach (var (owner, index) in _disabled)
            {
                var components = owner.Components;
                if (index < components.Count && !components[index].IsDestroyed) components[index].Enabled = true;
            }
            _disabled.Clear();

            Collect(Root.Transform, s_told);
            foreach (var dormant in s_told) dormant.OnWake();
            s_told.Clear();
        }

        private static void Disable(Transform at, List<(Entity, int)> disabled)
        {
            var entity = at.Entity;
            var components = entity?.Components;
            if (components != null)
                for (int i = 0; i < components.Count; i++)
                {
                    var component = components[i];
                    if (component is Transform || !component.Enabled) continue;
                    component.Enabled = false;
                    disabled.Add((entity!, i));
                }
            foreach (Transform child in at) Disable(child, disabled);
        }

        private static void Collect(Transform at, List<IDormant> into)
        {
            var components = at.Entity?.Components;
            if (components != null)
                for (int i = 0; i < components.Count; i++)
                    if (components[i] is IDormant dormant) into.Add(dormant);
            foreach (Transform child in at) Collect(child, into);
        }
    }
}
