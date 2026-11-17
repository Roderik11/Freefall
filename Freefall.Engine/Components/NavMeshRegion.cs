using System.Numerics;
using Freefall.Base;

using Component = Freefall.Base.Component;

namespace Freefall.Components
{
    /// <summary>
    /// Where the navmesh is baked. Without any in the scene a bake covers the whole terrain; with one or more it
    /// builds only the tiles their areas reach, and the rest of the world has no navmesh. For a large world whose
    /// people walk in a few places: a 16 km terrain is 730,000 tiles, a town a thousand.
    ///
    /// The area is the entity's closed Spline (its points on the ground plane), or without one a circle of
    /// Radius round the entity; either is grown by Margin. The tile grid is the terrain's whatever the regions
    /// are, so regions added later bake their own tiles and keep the others'.
    ///
    /// Only the bake reads it: adding, removing or moving one takes a bake to show.
    /// </summary>
    [Icon("icon_nav_mesh.png")]
    public class NavMeshRegion : Component
    {
        [System.ComponentModel.Description("Metres round the entity that are baked, when the entity has no closed Spline. " +
                                           "With a closed Spline the spline's outline is the area and this is not used.")]
        [ValueRange(1f, 4000f)]
        public float Radius = 100f;

        [System.ComponentModel.Description("Metres the area is grown by on every side: the ground just outside a wall or " +
                                           "a gate that people still walk on.")]
        [ValueRange(0f, 500f)]
        public float Margin = 20f;

        /// <summary>
        /// The area on the ground plane in world space: the closed spline's points, or null for the circle
        /// (<paramref name="center"/>, Radius). Main thread.
        /// </summary>
        internal Vector2[]? Outline(out Vector2 center)
        {
            var world = Transform?.WorldMatrix ?? Matrix4x4.Identity;
            center = new Vector2(world.M41, world.M43);
            var spline = Entity?.GetComponent<Spline>();
            if (spline is not { Closed: true } || spline.Points.Count < 3) return null;

            var outline = new Vector2[spline.Points.Count];
            for (int i = 0; i < outline.Length; i++)
            {
                var p = Vector3.Transform(spline.Points[i], world);
                outline[i] = new Vector2(p.X, p.Z);
            }
            return outline;
        }
    }
}
