namespace Freefall.Assets.Packers
{
    /// <summary>
    /// Raw pixel data for a PaintHeightLayer DeltaMap (R32_Float).
    /// Stored as a hidden subasset of the .terrain file.
    /// </summary>
    public class DeltaMapData
    {
        public int Width;
        public int Height;

        /// <summary>R32_Float raw pixel bytes (Width × Height × 4 bytes).</summary>
        public byte[] Pixels;
    }
}
