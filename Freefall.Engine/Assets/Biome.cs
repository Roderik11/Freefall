using System;
using System.Collections.Generic;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Assets
{
    /// <summary>
    /// A climate's ground as one asset: the layers it paints and the ground cover it grows, each by a rule of its own
    /// ("dry grass on the flat, scree on slopes over 35°, rock over 45°"). Where it applies is decided by the
    /// <see cref="BiomeStamp"/>s that place it (an area, a corridor, or the whole terrain under a filter), never by the
    /// biome, so one biome can cover a homeland and an island alike.
    ///
    /// A biome does not shape height: relief stays in HeightStamps and the rules follow it through their filters. It
    /// does not spawn entities either; that stays PCG's. In the bake each rule becomes a paint or ground cover entry
    /// with its stamp's shape, falloff, edge noise and Priority, in the order of the lists: a later rule paints over an
    /// earlier one, as a later stamp does.
    /// </summary>
    [CreateAsset("Biome")]
    public class Biome : Asset
    {
        /// <summary>The ground it paints. Later rules paint over earlier ones.</summary>
        public List<BiomeLayerRule> Layers = [];

        /// <summary>The ground cover it grows.</summary>
        public List<BiomeDecoratorRule> Decorators = [];
    }

    /// <summary>One layer a <see cref="Biome"/> paints and where, as a <see cref="SplatStamp"/> would.</summary>
    [Serializable]
    public class BiomeLayerRule
    {
        /// <summary>The ground material. A rule without one paints nothing.</summary>
        public TerrainLayer Layer;

        public SplatOp Op = SplatOp.Paint;

        /// <summary>Paint: opacity at full weight. Remove: how much is taken away. Multiply: the factor.</summary>
        [ValueRange(0f, 1f)]
        public float Strength = 1f;

        /// <summary>Where inside the stamp's shape (and its own filter) the layer goes.</summary>
        public StampFilter Filter = new();
    }

    /// <summary>One ground cover a <see cref="Biome"/> grows and where, as a <see cref="DecoStamp"/> would.</summary>
    [Serializable]
    public class BiomeDecoratorRule
    {
        /// <summary>The ground cover. Without one, an Add rule grows nothing and a Multiply rule scales every decorator.</summary>
        public TerrainDecorator Decorator;

        public DecoOp Op = DecoOp.Add;

        /// <summary>Add: coverage 0..1. Multiply: the factor 0..2.</summary>
        [ValueRange(0f, 2f)]
        public float Weight = 1f;

        /// <summary>Where inside the stamp's shape (and its own filter) it grows.</summary>
        public StampFilter Filter = new();
    }
}
