using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Graphics;

namespace Freefall.Assets
{
    public enum DecoratorMode { Mesh, Billboard, Cross }

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

        /// <summary>Billboard/Cross mode: alpha-tested texture.</summary>
        public Texture Texture;

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
}
