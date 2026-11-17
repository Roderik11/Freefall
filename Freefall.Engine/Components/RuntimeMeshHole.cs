using Freefall.Base;
using Component = Freefall.Base.Component;

namespace Freefall.Components
{
    /// <summary>
    /// Cuts this entity's closed spline out of its parent's closed <see cref="RuntimeMesh"/>: a garden in a paved
    /// island, a tree pit, a pool in a terrace. The parent's surface leaves the hole open to what is under it (the
    /// terrain), and its walls and curbs run round the hole's edge, facing into it. Put it on a direct child of the
    /// paved entity, with the child's own closed Spline (at least three points, wholly inside the parent's outline,
    /// apart from the other holes).
    ///
    /// The parent rebuilds when this spline changes and when the component comes or goes.
    /// </summary>
    [Icon("icon_runtime_mesh.png")]
    public class RuntimeMeshHole : Component
    {
        /// <summary>Set while the component is going away, so the parent's rebuild leaves it out.</summary>
        internal bool Gone;

        internal RuntimeMesh Paving => Entity?.Transform?.Parent?.Entity?.GetComponent<RuntimeMesh>();

        protected override void Early()
        {
            // At a scene load the parent usually built first and found this hole already; only rebuild when it did not
            if (Paving is { } paving && !paving.Cuts(this)) paving.Generate();
        }

        public override void Destroy()
        {
            Gone = true;
            var parent = Entity?.Transform?.Parent?.Entity;
            if (parent == null || parent.IsBeingDestroyed) return;
            if (Paving is { } paving && paving.Cuts(this)) paving.Generate();
        }
    }
}
