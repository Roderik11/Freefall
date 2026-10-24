using Component = Freefall.Base.Component;

namespace Freefall.Components
{
    /// <summary>
    /// Keeps the baked navmesh off this entity's meshes (its children's too) and off the ground
    /// round them, as far as the agent's radius: agents path round the thing, never over it.
    /// For small things lower than the navmesh's climb that are not to be walked on: benches,
    /// chairs, stools. Each mesh clears its bounds, from the climb below them to their top, so
    /// it is not for anything with a floor inside.
    ///
    /// Only the bake reads it: adding, removing or moving one takes a bake to show.
    /// </summary>
    [Icon("icon_nav_mesh.png")]
    public class NavMeshBlocker : Component
    {
    }
}
