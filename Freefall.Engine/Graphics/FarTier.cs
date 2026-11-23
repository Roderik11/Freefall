using System;
using System.Collections.Generic;
using System.Numerics;

namespace Freefall.Graphics
{
    /// <summary>
    /// The far tier: placements nearer than <see cref="From"/> metres to the camera are drawn as instances of their own
    /// (CommandBuffer.AddPersistent with farTier), further away by merged far meshes (<see cref="Mesh.CellFade"/>). Over
    /// <see cref="Band"/> metres the two dither into each other cell by cell, on the blue noise of the LOD crossfade, so
    /// together they cover each pixel once (common.fx FarTierShows); shadows switch at half way. Set into every effect's
    /// scene constants by the camera. <see cref="Band"/> 0 = no far tier: instances draw everywhere, far meshes nowhere.
    ///
    /// <para><b>Profiles</b> (WORLD_CELLS_PLAN.md stage 4): a generator may hand over to its far form at another distance
    /// (<c>PCGComponent.Ring</c>). Each distance is a profile, at most <see cref="MaxProfiles"/>; profile 0 is
    /// <see cref="From"/> itself. Its band is <see cref="Band"/> scaled with its distance. A draw names its profile: an
    /// instance in bits 8–13 of <c>InstanceDescriptor.MeshPartIdx</c>, a far mesh's part in bits 8–13 of its
    /// <c>MeshPartEntry.Flags</c> (<see cref="MeshPart.FarProfile"/>), the far forest per kind. The shaders read the
    /// table (from, band) through <c>SceneConstants.FarTierProfilesIdx</c>.</para>
    /// </summary>
    public static class FarTier
    {
        /// <summary>Metres from the camera (level) where the far meshes start to take over: profile 0.</summary>
        public static float From = 800f;

        /// <summary>Metres over which instances and far meshes dither into each other at <see cref="From"/>; 0 = off. The
        /// world cells (<see cref="Freefall.World.WorldCellSystem"/>) set it to <see cref="DefaultBand"/> while any generator
        /// keeps its placements in cells, and back to 0 when none does.</summary>
        public static float Band;

        /// <summary>The band while the far tier is on.</summary>
        public const float DefaultBand = 100f;

        /// <summary>
        /// Metres round the camera (level) within which every kind of the world cells keeps its instances and nothing takes
        /// a far form; 0 = off, the far tier as usual. While above 0 the world cells hold <see cref="Band"/> at 0 (instances
        /// draw at any distance; far meshes and the far forest nowhere) and every kind's ring reaches at least this far. For
        /// the world map's bake, which draws a chunk up to 1 km wide from straight above: a tree kind hands over where its
        /// coarsest LOD starts, as near as 128 m, so the chunk's corners showed far forms.
        /// </summary>
        public static float HeldOffWithin;

        /// <summary>Metres a side of a cell, the unit they fade by: a cell's centre is at (floor(x / CellSize) + 0.5) × CellSize.</summary>
        public static float CellSize = 64f;

        /// <summary>How many distances the shaders can tell apart (6 bits in a draw).</summary>
        public const int MaxProfiles = 64;

        // Profile → its distance; [0] stands for From whatever it is
        private static readonly List<float> s_froms = new() { 0f };
        private static StreamingBuffer<Vector2>? s_table;
        private static float s_writtenFrom = -1f, s_writtenBand = -1f;
        private static int s_writtenCount;

        /// <summary>
        /// The profile whose far side starts <paramref name="from"/> metres from the camera (to a metre): 0 for
        /// <see cref="From"/> (or 0 and below). When all <see cref="MaxProfiles"/> are taken, the nearest one.
        /// </summary>
        public static int ProfileFor(float from)
        {
            if (from <= 0f || MathF.Abs(from - From) < 0.5f) return 0;
            from = MathF.Round(from);
            int nearest = 0;
            float best = MathF.Abs(From - from);
            for (int i = 1; i < s_froms.Count; i++)
            {
                float d = MathF.Abs(s_froms[i] - from);
                if (d < 0.5f) return i;
                if (d < best) { best = d; nearest = i; }
            }
            if (s_froms.Count >= MaxProfiles)
            {
                Debug.LogWarning("FarTier", $"all {MaxProfiles} far tier profiles taken; {from} m draws as {FromOf(nearest)} m");
                return nearest;
            }
            s_froms.Add(from);
            return s_froms.Count - 1;
        }

        /// <summary>Where a profile's far side starts, in metres.</summary>
        public static float FromOf(int profile) => profile <= 0 || profile >= s_froms.Count ? From : s_froms[profile];

        /// <summary>A profile's band: <see cref="Band"/> scaled with its distance (0 while the far tier is off).</summary>
        public static float BandOf(int profile) => From > 0f ? Band * FromOf(profile) / From : Band;

        /// <summary>How many profiles there are (profile 0 included).</summary>
        public static int ProfileCount => s_froms.Count;

        /// <summary>
        /// The bindless index of this frame's profile table (StructuredBuffer of (from, band), common.fx FarTierProfile),
        /// written first where it changed. 0 until the device exists. Main thread, once a frame or more.
        /// </summary>
        internal static uint ProfilesSrv(GraphicsDevice? device)
        {
            if (device == null) return 0;
            s_table ??= new StreamingBuffer<Vector2>(device, MaxProfiles);
            if (From != s_writtenFrom || Band != s_writtenBand || s_froms.Count != s_writtenCount)
            {
                for (int i = 0; i < MaxProfiles; i++)
                    s_table.Set(i, i < s_froms.Count ? new Vector2(FromOf(i), BandOf(i)) : new Vector2(From, Band));
                s_writtenFrom = From;
                s_writtenBand = Band;
                s_writtenCount = s_froms.Count;
            }
            s_table.Upload();
            return s_table.SrvIndex;
        }
    }
}
