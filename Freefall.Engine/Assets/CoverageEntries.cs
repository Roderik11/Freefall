using System.Collections.Generic;
using Freefall.Components;

namespace Freefall.Assets
{
    /// <summary>One entry of a splat bake: a layer painted in a stamp's shape through a filter.</summary>
    /// <param name="Shape">Where: shape, falloff, edge noise, fades and Priority (a SplatStamp, or the BiomeStamp of a rule).</param>
    public readonly record struct SplatEntry(TerrainStamp Shape, StampFilter Filter, TerrainLayer Layer, SplatOp Op, float Strength);

    /// <summary>One entry of a decoration bake: a decorator added or scaled in a stamp's shape through a filter.</summary>
    /// <param name="Rows">The DecoStamp whose rows the plants are moved onto; null for a biome's rule.</param>
    public readonly record struct DecoEntry(TerrainStamp Shape, StampFilter Filter, TerrainDecorator Decorator, DecoOp Op,
                                            float Weight, DecoStamp? Rows);

    /// <summary>
    /// What the splat and decoration bakes are given: the SplatStamps' and DecoStamps' own entries and every rule of
    /// the BiomeStamps, in one bake order. Stamps go by <see cref="TerrainStamp.CompareBakeOrder"/>, a biome's rules
    /// one after another at its stamp's place in it, in the order the biome lists them. After this the bake sees no
    /// biomes, only entries.
    /// </summary>
    public static class CoverageEntries
    {
        /// <summary>Both lists in bake order (as <see cref="TerrainBaker.CollectStamps{T}"/> gives them). Main thread.</summary>
        public static List<SplatEntry> Splat(IReadOnlyList<SplatStamp> splats, IReadOnlyList<BiomeStamp> biomes)
        {
            var entries = new List<SplatEntry>(splats.Count + biomes.Count * 4);
            Merge(splats, biomes,
                splat => entries.Add(new SplatEntry(splat, StampFilter.Of(splat), splat.Layer, splat.Op, splat.Strength)),
                biome =>
                {
                    var stampFilter = StampFilter.Of(biome);
                    foreach (var rule in biome.Biome.Layers)
                        if (rule != null)
                            entries.Add(new SplatEntry(biome, StampFilter.Both(stampFilter, rule.Filter ?? new()),
                                                       rule.Layer, rule.Op, rule.Strength));
                });
            return entries;
        }

        /// <summary>Both lists in bake order. Main thread.</summary>
        public static List<DecoEntry> Deco(IReadOnlyList<DecoStamp> decos, IReadOnlyList<BiomeStamp> biomes)
        {
            var entries = new List<DecoEntry>(decos.Count + biomes.Count * 2);
            Merge(decos, biomes,
                deco => entries.Add(new DecoEntry(deco, StampFilter.Of(deco), deco.Decorator, deco.Op, deco.Weight, deco)),
                biome =>
                {
                    var stampFilter = StampFilter.Of(biome);
                    foreach (var rule in biome.Biome.Decorators)
                        if (rule != null)
                            entries.Add(new DecoEntry(biome, StampFilter.Both(stampFilter, rule.Filter ?? new()),
                                                      rule.Decorator, rule.Op, rule.Weight, null));
                });
            return entries;
        }

        // Two lists already in bake order into one; biome stamps without a biome add nothing
        private static void Merge<T>(IReadOnlyList<T> stamps, IReadOnlyList<BiomeStamp> biomes,
                                     System.Action<T> addStamp, System.Action<BiomeStamp> addBiome) where T : TerrainStamp
        {
            int s = 0, b = 0;
            while (s < stamps.Count || b < biomes.Count)
            {
                bool takeStamp = b >= biomes.Count ||
                                 (s < stamps.Count && TerrainStamp.CompareBakeOrder(stamps[s], biomes[b]) <= 0);
                if (takeStamp) addStamp(stamps[s++]);
                else
                {
                    var biome = biomes[b++];
                    if (biome.Biome != null) addBiome(biome);
                }
            }
        }
    }
}
