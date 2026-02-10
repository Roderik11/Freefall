using System.Numerics;

namespace Freefall.Components
{
    /// <summary>
    /// Interface for terrain height queries — allows switching between
    /// Terrain (CPU quadtree) and Landscape (GPU clipmap) implementations.
    /// </summary>
    public interface IHeightProvider
    {
        float GetHeight(Vector3 position);
    }
}
