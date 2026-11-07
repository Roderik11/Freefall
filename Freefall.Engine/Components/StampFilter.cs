using System;
using System.Collections.Generic;
using System.Numerics;
using System.Security.Cryptography;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Components
{
    /// <summary>
    /// Where inside its shape a paint or ground cover entry applies: a <see cref="CoverageStamp"/>'s filter as data,
    /// for the rules of a <see cref="Biome"/>, which have no stamp of their own. The members mean what the stamp's do.
    /// The defaults pass everything.
    /// </summary>
    [Serializable]
    public class StampFilter
    {
        /// <summary>Terrain height the rule applies in, normalized 0..1 of the terrain's MaxHeight.</summary>
        public Vector2 HeightRange = new(0, 1);

        /// <summary>Width of the fade at both ends of HeightRange (normalized 0..1).</summary>
        [ValueRange(0f, 0.5f)]
        public float HeightBlend = 0.05f;

        /// <summary>Slope the rule applies on, in degrees (0 = flat, 90 = cliff).</summary>
        public Vector2 SlopeRange = new(0, 90);

        /// <summary>Width of the fade at both ends of SlopeRange (degrees).</summary>
        [ValueRange(0f, 30f)]
        public float SlopeBlend = 5.0f;

        /// <summary>Apply only where these layers show, summed (see <see cref="CoverageStamp.RequireLayers"/>).</summary>
        public List<TerrainLayer> RequireLayers = [];

        /// <summary>Stay away from these layers (see <see cref="CoverageStamp.ExcludeLayers"/>).</summary>
        public List<TerrainLayer> ExcludeLayers = [];

        /// <summary>Apply only in irregular patches about this many metres across. 0 = everywhere.</summary>
        [ValueRange(0f, 200f)]
        public float PatchSize = 0f;

        /// <summary>About the share of the area the patches cover.</summary>
        [ValueRange(0f, 1f)]
        public float PatchCoverage = 0.3f;

        /// <summary>How gradually a patch fades out at its edge (0 = hard edge).</summary>
        [ValueRange(0f, 1f)]
        public float PatchSoftness = 0.3f;

        /// <summary>Moves the patches; the same size and seed put them in the same places, whichever stamp places the biome.</summary>
        public int PatchSeed = RandomNumberGenerator.GetInt32(99999);

        /// <summary>True if any term can reject a texel, i.e. the GPU has to evaluate it.</summary>
        public bool HasFilter =>
            HeightRange.X > 0f || HeightRange.Y < 1f || SlopeRange.X > 0f || SlopeRange.Y < 90f ||
            RequireLayers is { Count: > 0 } || ExcludeLayers is { Count: > 0 } || PatchSize > 0f;

        /// <summary>A stamp's filter as data.</summary>
        public static StampFilter Of(CoverageStamp stamp) => new()
        {
            HeightRange = stamp.HeightRange,
            HeightBlend = stamp.HeightBlend,
            SlopeRange = stamp.SlopeRange,
            SlopeBlend = stamp.SlopeBlend,
            RequireLayers = stamp.RequireLayers ?? [],
            ExcludeLayers = stamp.ExcludeLayers ?? [],
            PatchSize = stamp.PatchSize,
            PatchCoverage = stamp.PatchCoverage,
            PatchSoftness = stamp.PatchSoftness,
            PatchSeed = stamp.PatchSeed,
        };

        /// <summary>
        /// A biome stamp's filter narrowed by one of its rules': what the bake evaluates for that rule, in one filter.
        /// The height and slope ranges are the overlap of both, each with the blend of the filter that narrows it (the
        /// wider when both do: a range left open keeps its default blend, 50 m of height on a 1000 m terrain, which must
        /// not soften the other's edge); the layers kept clear of are both lists. Required layers and patches are the
        /// rule's when it has them, else the stamp's: the GPU filter holds one of each, so a stamp that requires layers
        /// and a rule that requires others take the rule's alone.
        /// </summary>
        public static StampFilter Both(StampFilter stamp, StampFilter rule)
        {
            if (!stamp.HasFilter) return rule;
            if (!rule.HasFilter) return stamp;

            bool stampHeight = stamp.HeightRange.X > 0f || stamp.HeightRange.Y < 1f;
            bool ruleHeight = rule.HeightRange.X > 0f || rule.HeightRange.Y < 1f;
            bool stampSlope = stamp.SlopeRange.X > 0f || stamp.SlopeRange.Y < 90f;
            bool ruleSlope = rule.SlopeRange.X > 0f || rule.SlopeRange.Y < 90f;

            var exclude = new List<TerrainLayer>(stamp.ExcludeLayers.Count + rule.ExcludeLayers.Count);
            exclude.AddRange(stamp.ExcludeLayers);
            foreach (var layer in rule.ExcludeLayers)
                if (!exclude.Contains(layer)) exclude.Add(layer);

            bool rulePatches = rule.PatchSize > 0f;
            return new StampFilter
            {
                HeightRange = new Vector2(MathF.Max(stamp.HeightRange.X, rule.HeightRange.X),
                                          MathF.Min(stamp.HeightRange.Y, rule.HeightRange.Y)),
                HeightBlend = stampHeight && ruleHeight ? MathF.Max(stamp.HeightBlend, rule.HeightBlend)
                            : stampHeight ? stamp.HeightBlend : rule.HeightBlend,
                SlopeRange = new Vector2(MathF.Max(stamp.SlopeRange.X, rule.SlopeRange.X),
                                         MathF.Min(stamp.SlopeRange.Y, rule.SlopeRange.Y)),
                SlopeBlend = stampSlope && ruleSlope ? MathF.Max(stamp.SlopeBlend, rule.SlopeBlend)
                           : stampSlope ? stamp.SlopeBlend : rule.SlopeBlend,
                RequireLayers = rule.RequireLayers is { Count: > 0 } ? rule.RequireLayers : stamp.RequireLayers,
                ExcludeLayers = exclude,
                PatchSize = rulePatches ? rule.PatchSize : stamp.PatchSize,
                PatchCoverage = rulePatches ? rule.PatchCoverage : stamp.PatchCoverage,
                PatchSoftness = rulePatches ? rule.PatchSoftness : stamp.PatchSoftness,
                PatchSeed = rulePatches ? rule.PatchSeed : stamp.PatchSeed,
            };
        }
    }
}
