using System;

namespace Freefall.World
{
    /// <summary>
    /// A component that needs no copy of itself far away: a prefab whose components are all such (besides renderers,
    /// colliders, a static body and a navmesh blocker) is kept as a record in the world cells, its entity made only near
    /// a viewer (<see cref="WorldCells"/>). Anything without it is live: a prefab that has it is instantiated as an
    /// entity at once, as without cells, since it may act or show from anywhere (a light, a sound, an NPC place).
    ///
    /// <para>A component with state of its own loses it when its cell goes (its entity is made again from the prefab)
    /// until state per record exists (WORLD_CELLS_PLAN.md stage 5).</para>
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, Inherited = true)]
    public sealed class CellRecordAttribute : Attribute { }
}
