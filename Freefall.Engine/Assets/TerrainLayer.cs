using System.Numerics;
using Freefall.Graphics;

namespace Freefall.Assets
{
    /// <summary>
    /// A ground material for terrain: what gets painted. Where it goes is decided by the
    /// <see cref="Freefall.Components.SplatStamp"/>s that reference it, never by the layer itself,
    /// so one layer can serve a meadow, a forest floor and a village green with different rules.
    ///
    /// A terrain renders the layers its stamps reference (its palette, derived at bake time);
    /// layers that are only named in a stamp filter and never painted cost nothing.
    /// </summary>
    [CreateAsset("Terrain Layer")]
    public class TerrainLayer : Asset
    {
        public Texture Diffuse;

        public Texture Normals;

        /// <summary>Height map: shapes the transition to neighbouring layers and drives displacement.</summary>
        public Texture Height;

        /// <summary>Per-layer displacement height scale (multiplied with the global SSDM scale).</summary>
        [ValueRange(0f, 5f)]
        public float HeightScale = 1.0f;

        /// <summary>Size of one texture repeat in world units.</summary>
        public Vector2 Tiling = Vector2.One;
    }
}
