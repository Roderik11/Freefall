using System.Numerics;
using Component = Freefall.Base.Component;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>What the navmesh inside a NavMeshArea's box is.</summary>
    public enum NavMeshAreaKind
    {
        /// <summary>
        /// A building's ground floor: walkable, but a metre of it costs a way NavMeshWorld.IndoorsCost metres, and no
        /// body is put there or sent there to wander (NavMeshWorld.SamplePosition outdoors, RandomReachablePoint).
        /// </summary>
        Indoors,
    }

    /// <summary>
    /// Marks the navmesh inside a box as an area of its own when it is baked. Indoors: a way that ends inside goes in at
    /// the nearest door, one that only passes goes round by the street. With the ground floors as cheap as the street,
    /// the shortest way round a corner went in at one door and out at another, and a body put on "the nearest navmesh"
    /// stood in somebody's front room (2026-10-25).
    ///
    /// Only the bake reads it: adding, removing or moving one takes a bake to show.
    /// </summary>
    [Icon("icon_nav_mesh.png")]
    public class NavMeshArea : Component
    {
        public NavMeshAreaKind Kind = NavMeshAreaKind.Indoors;

        [Description("The box's middle in the entity's own space (metres).")]
        public Vector3 Center;

        [Description("The box's size in the entity's own space (metres): the navmesh within it, up or down, is the area's.")]
        public Vector3 Size = Vector3.One;
    }
}
