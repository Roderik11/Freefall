using System.Numerics;

namespace Freefall.Components
{
    /// <summary>
    /// Interface for terrain height queries — allows switching between
    /// Terrain (CPU quadtree) and GPUTerrain (GPU quadtree) implementations.
    /// </summary>
    public interface IHeightProvider
    {
        float GetHeight(Vector3 position);
    }
}
