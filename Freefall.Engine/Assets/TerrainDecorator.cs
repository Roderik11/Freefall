using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Graphics;

namespace Freefall.Assets
{
    /// <summary>
    /// How a decorator variant is drawn. The procedural modes build opaque geometry in a mesh shader instead of a
    /// mostly transparent card, coloured from the variant's texture — for near the camera:
    /// Blades = a clump of grass blades; Flowers = a few stems with paired leaves and a disc bloom (daisy, zinnia);
    /// Spikes = stems carrying a spike of small florets (loosestrife, lavender);
    /// Ferns = a crown of arching fronds, each the parts atlas' frond on a bent, cupped strip.
    /// </summary>
    public enum DecoratorMode { Mesh, Billboard, Cross, Blades, Flowers, Spikes, Ferns }

    /// <summary>
    /// Ground cover for terrain: a mix of things that are allowed in the same places.
    /// "Meadow mix" = a few grass cards, a flower and a pebble mesh.
    ///
    /// The variants share coverage and nothing else. Coverage comes from the
    /// <see cref="Freefall.Components.DecoStamp"/>s that reference the decorator; density, clumping,
    /// size and look are per variant, so grass can be near-uniform while flowers form sparse drifts
    /// inside the same mix. What a mix cannot do is give one variant its own coverage: a stamp adds or
    /// suppresses the whole decorator. Anything that needs separate coverage is a second decorator.
    ///
    /// On the GPU a decorator is one slot of the decoration control texture (32 per terrain, 8 per
    /// texel), however many variants it has.
    /// </summary>
    [CreateAsset("Terrain Decorator")]
    public class TerrainDecorator : Asset
    {
        /// <summary>Variants per decorator the spawn kernel will walk.</summary>
        public const int MaxVariants = 8;

        /// <summary>Distance bands per variant the spawn kernel reads (<see cref="DecoratorVariant.Bands"/>).</summary>
        public const int MaxBands = 4;

        public List<DecoratorVariant> Variants = [];
    }

    /// <summary>One thing a <see cref="TerrainDecorator"/> scatters.</summary>
    [Serializable]
    public class DecoratorVariant
    {
        public DecoratorMode Mode = DecoratorMode.Cross;

        /// <summary>Mesh mode: geometry + LODs.</summary>
        public Mesh Mesh;

        /// <summary>Mesh mode: material to render the mesh with (meshes don't carry one; prefabs do).</summary>
        public Material Material;

        /// <summary>Billboard/Cross mode: alpha-tested texture. Procedural modes: where their colours come from.</summary>
        public Texture Texture;

        /// <summary>
        /// Flowers / Spikes / Ferns bands: atlas of the plant's parts — a bloom seen from above (Flowers) or the flower
        /// spike from the side (Spikes), and a leaf; or one frond filling the atlas (Ferns, HeightRange = frond
        /// length) — mapped onto the procedural plant. Rendered by
        /// Tools/BlenderKits/build_flowers.py, which documents the layout. Empty = flat colours from the card.
        /// The card's own texture cannot serve: it shows the blooms side-on.
        /// </summary>
        public Texture PlantParts;

        /// <summary>
        /// How the variant is drawn at each distance, near to far (up to <see cref="TerrainDecorator.MaxBands"/>), e.g. grass as
        /// Blades to 15 m, then Cross. Empty = Mode out to the terrain's decoration radius. Beyond the last band
        /// the variant is not drawn (it shrinks away over the last 15%), so flowers can stop sooner than grass.
        /// Card variants (Texture) take every mode but Mesh; a mesh variant stays a mesh and only gets its range
        /// from the bands.
        /// </summary>
        public List<DecoratorBand> Bands = [];

        /// <summary>
        /// Random seed (0..255) for this variant's scatter positions, clumps and height patches.
        /// Variants with the same seed and settings land on the same spots; changing it reshuffles
        /// this variant without touching the others.
        /// </summary>
        [ValueRange(0, 255)]
        public int Seed = System.Security.Cryptography.RandomNumberGenerator.GetInt32(256);

        /// <summary>Instances per square meter at full coverage.</summary>
        [ValueRange(.01f, 4)]
        public float Density = 1.0f;

        /// <summary>World size (m) of the density clumps this variant grows in. 0 = uniform coverage.
        /// Real meadows are patchy: clover in drifts, grass tufts in clumps, flowers in scattered pockets.</summary>
        [ValueRange(0f, 100f)]
        public float ClusterScale = 0f;

        /// <summary>How strongly the clumps modulate density: 0 = uniform, 1 = dense clumps with bare gaps between.</summary>
        [ValueRange(0f, 1f)]
        public float ClusterAmount = 0.7f;

        /// <summary>World size (m) of short vs. lush patches, independent of the clumps. 0 = off.
        /// Density already shortens plants at clump fringes; this varies height across whole areas.</summary>
        [ValueRange(0f, 200f)]
        public float HeightNoiseScale = 0f;

        /// <summary>Height variation strength: 0 = none, 1 = 0.4x (short patches) .. 1.4x (lush patches).</summary>
        [ValueRange(0f, 1f)]
        public float HeightNoiseAmount = 0.5f;

        public Vector2 HeightRange = new(0.3f, 0.6f);

        public Vector2 WidthRange = new(0.2f, 0.4f);

        /// <summary>Root rotation applied to mesh vertices (euler degrees).</summary>
        public Vector3 RootRotation = new(-90, 0, 0);

        /// <summary>Blend factor for aligning to terrain slope (0=upright, 1=fully aligned).</summary>
        [ValueRange(-1, 1)]
        public float SlopeBias = 0.0f;

        /// <summary>Tint color for "healthy" instances (multiplicative).</summary>
        public Vector4 HealthyColor = new(1, 1, 1, 1);

        /// <summary>Tint color for "dry" instances (multiplicative).</summary>
        public Vector4 DryColor = new(1, 1, 1, 1);

        /// <summary>World-space noise frequency for healthy/dry blend. 0 = uniform healthy color.</summary>
        [ValueRange(0f, 1f)]
        public float NoiseSpread = 1.0f;

        /// <summary>True when the variant has what its mode needs to render.</summary>
        [Freefall.Reflection.DontSerialize]
        [System.Text.Json.Serialization.JsonIgnore]
        public bool IsRenderable => Mode == DecoratorMode.Mesh ? Mesh != null : Texture != null;
    }

    /// <summary>One distance band of a <see cref="DecoratorVariant"/>: how it is drawn up to a distance.</summary>
    [Serializable]
    public class DecoratorBand
    {
        public DecoratorMode Mode = DecoratorMode.Cross;

        /// <summary>
        /// Far end of the band (m). The next band takes over here, each instance at its own point within ±15%
        /// so the hand-over has no visible ring.
        /// </summary>
        [ValueRange(1f, 1000f)]
        public float MaxDistance = 30f;
    }
}
