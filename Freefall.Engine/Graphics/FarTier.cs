namespace Freefall.Graphics
{
    /// <summary>
    /// The far tier: placements nearer than <see cref="From"/> metres to the camera are drawn as instances of their own
    /// (CommandBuffer.AddPersistent with farTier), further away by merged far meshes (<see cref="Mesh.CellFade"/>). Over
    /// <see cref="Band"/> metres the two dither into each other cell by cell, on the blue noise of the LOD crossfade, so
    /// together they cover each pixel once (common.fx FarTierShows); shadows switch at half way. Set into every effect's
    /// scene constants by the camera. <see cref="Band"/> 0 = no far tier: instances draw everywhere, far meshes nowhere.
    /// </summary>
    public static class FarTier
    {
        /// <summary>Metres from the camera (level) where the far meshes start to take over.</summary>
        public static float From = 800f;

        /// <summary>Metres over which instances and far meshes dither into each other; 0 = off.</summary>
        public static float Band;

        /// <summary>Metres a side of a cell, the unit they fade by: a cell's centre is at (floor(x / CellSize) + 0.5) × CellSize.</summary>
        public static float CellSize = 64f;
    }
}
