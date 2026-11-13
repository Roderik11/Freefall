using System;
using System.Runtime.InteropServices;
using System.Threading;
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

        /// <summary>
        /// The pixels in native memory instead of <see cref="PixelData"/>: a DDS cache file read by
        /// TextureLoader.ReadCpuData, so a texture read puts nothing on the managed heap. <see cref="Mips"/>' offsets count
        /// from its <see cref="NativePixels.Data"/>. The upload frees it once the pixels are in the upload ring
        /// (StreamingManager.RecordTextureUpload).
        /// </summary>
        public NativePixels? Native;

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

    /// <summary>
    /// A texture's pixels in native memory (<see cref="CpuTextureData.Native"/>). Texture files read into managed arrays
    /// sent ~730 MB through the large object heap on one TestWorld load (two arrays per texture: the file and a copy of its
    /// pixels), all garbage once uploaded, and the GC never compacts that heap (ram.md). Freed by <see cref="Dispose"/>
    /// (the upload, once it has copied them) or by the finalizer when a read is dropped unused (a texture let go while its
    /// read was on its way). Any thread.
    /// </summary>
    public sealed unsafe class NativePixels : IDisposable
    {
        // No GC.AddMemoryPressure: a burst of reads (hundreds of textures while a scene loads) would set off full
        // collections, and only a dropped read waits for the finalizer, which a young object reaches soon
        private IntPtr _block;

        /// <summary>The block's size: the whole DDS file.</summary>
        public long BlockBytes { get; }

        /// <summary>The first pixel, inside the block (after the file's header).</summary>
        public byte* Data { get; }

        /// <summary>Bytes from <see cref="Data"/> to the end of the block.</summary>
        public long Length { get; }

        /// <summary>Takes <paramref name="block"/> (from NativeMemory.Alloc) over; <paramref name="data"/> lies inside it.</summary>
        public NativePixels(IntPtr block, long blockBytes, byte* data, long length)
        {
            _block = block;
            BlockBytes = blockBytes;
            Data = data;
            Length = length;
        }

        public void Dispose()
        {
            Free();
            GC.SuppressFinalize(this);
        }

        ~NativePixels() => Free();

        private void Free()
        {
            var block = Interlocked.Exchange(ref _block, IntPtr.Zero);
            if (block == IntPtr.Zero) return;
            NativeMemory.Free((void*)block);
        }
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
