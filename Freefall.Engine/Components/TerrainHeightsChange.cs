using System.Numerics;
using Freefall.Assets;

namespace Freefall.Components
{
    /// <summary>
    /// Payload of EngineMsg.TerrainHeightsChanged: which terrain, and which world-space XZ region of it has new
    /// heights. Listeners far away from the region skip their rebuild.
    /// </summary>
    public sealed class TerrainHeightsChange
    {
        public readonly Terrain Terrain;

        /// <summary>The whole terrain may have changed (first bake, painting, layer edits): Min/Max are not valid.</summary>
        public readonly bool All;

        /// <summary>Changed region, world XZ (X, Z).</summary>
        public readonly Vector2 Min, Max;

        public TerrainHeightsChange(Terrain terrain, bool all, Vector2 min, Vector2 max)
        {
            Terrain = terrain;
            All = all;
            Min = min;
            Max = max;
        }

        /// <summary>True if the world XZ rectangle touches the changed region.</summary>
        public bool Overlaps(Vector2 min, Vector2 max)
            => All || (min.X <= Max.X && max.X >= Min.X && min.Y <= Max.Y && max.Y >= Min.Y);
    }
}
