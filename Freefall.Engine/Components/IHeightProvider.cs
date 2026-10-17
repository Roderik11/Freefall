using System.Numerics;

namespace Freefall.Components
{
    /// <summary>
    /// Interface for terrain height queries — allows switching between
    /// TerrainRenderer (GPU quadtree) implementations.
    /// </summary>
    public interface IHeightProvider
    {
        float GetHeight(Vector3 position);

        /// <summary>
        /// True where the ground is open (a HoleStamp's cave mouth): GetHeight still answers there, but nothing stands
        /// on that surface. Whatever keeps things above the ground (a camera, an arrow) should let them through.
        /// </summary>
        bool IsHole(Vector3 position) => false;
    }
}
