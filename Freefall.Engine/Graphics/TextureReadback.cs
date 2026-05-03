using System;
using System.IO;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.WIC;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU→CPU readback utilities for render targets.
    /// Used by thumbnail generation to read back rendered pixels.
    /// </summary>
    public static class TextureReadback
    {
        /// <summary>
        /// Read RGBA8 pixels from a GPU texture.
        /// Performs a synchronous GPU→CPU copy (creates a staging buffer, copies, fence-waits, maps).
        /// </summary>
        public static byte[] ReadPixels(ID3D12Resource source, int width, int height)
        {
            var device = Engine.Device;
            var nativeDevice = device.NativeDevice;

            // D3D12 requires row pitch aligned to 256 bytes
            uint rowPitch = (uint)((width * 4 + 255) & ~255);
            ulong bufferSize = (ulong)rowPitch * (uint)height;

            // Create readback buffer
            var readbackDesc = ResourceDescription.Buffer(bufferSize);
            using var readbackBuffer = nativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Readback),
                HeapFlags.None,
                readbackDesc,
                ResourceStates.CopyDest);

            // Copy texture → readback buffer
            using var allocator = nativeDevice.CreateCommandAllocator(CommandListType.Direct);
            using var cmd = nativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, allocator);

            // Transition source to CopySource
            cmd.ResourceBarrierTransition(source, ResourceStates.Common, ResourceStates.CopySource);

            var srcLoc = new TextureCopyLocation(source, 0);
            var dstLoc = new TextureCopyLocation(readbackBuffer,
                new PlacedSubresourceFootPrint
                {
                    Offset = 0,
                    Footprint = new SubresourceFootPrint(Format.R8G8B8A8_UNorm, (uint)width, (uint)height, 1, rowPitch)
                });

            cmd.CopyTextureRegion(dstLoc, 0, 0, 0, srcLoc, null);

            // Transition back to Common
            cmd.ResourceBarrierTransition(source, ResourceStates.CopySource, ResourceStates.Common);

            cmd.Close();
            device.SubmitAndWait(cmd);

            // Map and read pixels
            byte[] pixels = new byte[width * height * 4];
            unsafe
            {
                void* mappedPtr;
                readbackBuffer.Map(0, &mappedPtr);
                var mapped = (byte*)mappedPtr;

                for (int y = 0; y < height; y++)
                {
                    var srcRow = mapped + y * rowPitch;
                    Marshal.Copy((IntPtr)srcRow, pixels, y * width * 4, width * 4);
                }

                readbackBuffer.Unmap(0);
            }
            return pixels;
        }

        /// <summary>
        /// Save RGBA8 pixels to a PNG file using WIC.
        /// </summary>
        public static void SavePng(string path, byte[] pixels, int width, int height)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));

            using var factory = new IWICImagingFactory();
            using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var wicStream = factory.CreateStream(stream);
            using var encoder = factory.CreateEncoder(ContainerFormat.Png);
            encoder.Initialize(wicStream);

            using var frame = encoder.CreateNewFrame(out _);
            frame.Initialize();
            frame.SetSize((uint)width, (uint)height);
            var pixelFormat = PixelFormat.Format32bppRGBA;
            frame.SetPixelFormat(ref pixelFormat);

            // WIC may change the requested format to BGRA.
            // If so, swap R↔B before writing.
            if (pixelFormat == PixelFormat.Format32bppBGRA)
            {
                for (int i = 0; i < pixels.Length; i += 4)
                {
                    (pixels[i], pixels[i + 2]) = (pixels[i + 2], pixels[i]);
                }
            }

            // Write all scanlines at once
            uint stride = (uint)(width * 4);
            frame.WritePixels((uint)height, stride, pixels);
            frame.Commit();
            encoder.Commit();
        }
    }
}
