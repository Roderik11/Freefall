using System.Numerics;
using Freefall.Graphics;
using Vortice.Mathematics;

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

        /// <summary>
        /// How far, in degrees either way, the terrain may turn each patch of this texture to hide its repeats
        /// (hex-tiling): 180 for patterns without a direction (leaves, moss, gravel); a few degrees for patterns that
        /// run one way, like sand ripples or rock strata, which become swirls when every patch faces somewhere else.
        /// </summary>
        [ValueRange(0f, 180f)]
        public float TileRotation = 180f;

        /// <summary>
        /// Multiplies the diffuse map: fits a scan to the scene without editing the file, e.g. a lawn full of dead
        /// leaves that averages to straw from afar pulled towards green, or a pale rock darkened.
        /// </summary>
        public Color3 Tint = new Color3(1f, 1f, 1f);

        /// <summary>
        /// What this ground does to the colour of the decorators growing on it (grass cards, blades, flowers),
        /// weighted by how much of the layer shows at each plant's root: dry grass yellows the grass standing in it,
        /// so a dry patch runs on through the sward instead of ending where the decoration radius begins.
        /// </summary>
        public Color3 PlantTint = new Color3(1f, 1f, 1f);
    }
}
