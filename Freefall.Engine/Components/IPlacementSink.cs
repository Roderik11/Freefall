using System.Numerics;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Components
{
    /// <summary>
    /// A component on a <see cref="PCGComponent"/>'s entity that takes the prefabs its graph places instead of having
    /// them instantiated (<c>SpawnPrefab</c>): a cell spawner keeps them as records and makes entities only in the cells
    /// near a viewer. The run's output container is made as before, and each placement is in its frame, as the instance
    /// would have been placed under it. Main thread, during the run.
    /// </summary>
    public interface IPlacementSink
    {
        /// <summary>A run starts: the last run's placements are void (its container is gone). <paramref name="container"/> is this run's.</summary>
        void PlacementsBegin(Entity container);

        /// <summary>One prefab placed: position, rotation and scale in the container's frame.</summary>
        void Place(Prefab prefab, Vector3 position, Quaternion rotation, Vector3 scale);

        /// <summary>The run is done.</summary>
        void PlacementsEnd();
    }
}
