using Vortice.DXGI;

namespace Freefall.Assets
{
    /// <summary>
    /// CPU-side parsed texture data. No GPU resources.
    /// Created on background threads, consumed on main thread to create Texture.
    /// </summary>
    public class CpuTextureData
    {
        public string Path;

        /// <summary>
        /// Upload buffer layout: one contiguous byte[] containing all mip levels.
        /// For WIC textures: box-filtered mip chain in R8G8B8A8.
        /// For DDS textures: raw mip data from the file.
        /// </summary>
        public byte[] PixelData;

        public int Width;
        public int Height;
        public int MipLevels;
        public int ArraySize;
        public Format Format;
        public bool IsCompressed;

        /// <summary>
        /// Per-mip layout within PixelData.
        /// </summary>
        public MipLayout[] Mips;
    }

    public struct MipLayout
    {
        /// <summary>Byte offset into CpuTextureData.PixelData where this mip starts.</summary>
        public int Offset;
        /// <summary>Row pitch in bytes (source data, not GPU-aligned).</summary>
        public int RowPitch;
        /// <summary>Number of rows (may differ from height for compressed formats).</summary>
        public int NumRows;
        /// <summary>Width of this mip level in pixels.</summary>
        public int Width;
        /// <summary>Height of this mip level in pixels.</summary>
        public int Height;
    }
}
