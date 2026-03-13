using System;
using System.Collections.Generic;
using System.IO;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.WIC;
using System.Numerics;
using Freefall;
using Freefall.Assets;

namespace Freefall.Graphics
{
    [AssetTypeAlias("DdsTextureData")]
    public class Texture : Asset, IDisposable
    {
        protected ID3D12Resource _resource;
        public ID3D12Resource Native => _resource;
        public uint BindlessIndex { get; protected set; }
        public GpuDescriptorHandle SrvHandle { get; protected set; }
        public CpuDescriptorHandle SrvCpuHandle { get; protected set; }


        public Texture(GraphicsDevice device, string path)
        {
             var cpuData = ParseFromFile(device, path);
             // Synchronous-like behavior for legacy constructor
             // We reuse the internal resource creation of CreateAsync but we must set this instance's fields
             var tex = CreateAsync(device, cpuData);
             _resource = tex._resource;
             BindlessIndex = tex.BindlessIndex;
             SrvHandle = tex.SrvHandle;
             SrvCpuHandle = tex.SrvCpuHandle;
             AssetPath = tex.AssetPath;
             Name = tex.Name;
             // Legacy code assumes ready?
             MarkReady(); 
             // Ideally we wait, but StreamingManager handles it.
        }

        public static Texture CreateFromData(GraphicsDevice device, int width, int height, byte[] data, Format format = Format.R8G8B8A8_UNorm)
        {
             var cpuData = new CpuTextureData
             {
                 Width = width,
                 Height = height,
                 Format = format,
                 PixelData = data,
                 MipLevels = 1,
                 ArraySize = 1,
                 Mips = new MipLayout[] { new MipLayout { Width = width, Height = height, RowPitch = width * FormatSize(format), NumRows = height } }
             };
             return CreateAsync(device, cpuData);
        }

        private static int FormatSize(Format fmt)
        {
            // Simple lookup for common cases
            if (fmt == Format.R8G8B8A8_UNorm) return 4;
            if (fmt == Format.R16G16B16A16_Float) return 8;
            if (fmt == Format.R32_Float) return 4;
            return 4; // Fallback
        }

        public static Texture CreateTexture2DArray(GraphicsDevice device, IList<Texture> textures, bool stripSrgb = false, bool forceDecompress = false)
        {
             if (textures.Count == 0) return null;
             
             // Ensure all source textures have been uploaded before GPU-to-GPU copy
             StreamingManager.Instance?.Flush();
             
             var first = textures[0];
             var refDesc = first.Native.Description;

             // Check if all textures share the same format AND dimensions
             bool formatsMatch = true;
             bool dimsMatch = true;
             for (int i = 1; i < textures.Count; i++)
             {
                 var d = textures[i].Native.Description;
                 if (d.Format != refDesc.Format)
                     formatsMatch = false;
                 if (d.Width != refDesc.Width || d.Height != refDesc.Height)
                     dimsMatch = false;
             }

             Debug.Log($"[Texture] CreateTexture2DArray: {textures.Count} textures, formatsMatch={formatsMatch}, dimsMatch={dimsMatch}, refFormat={refDesc.Format}, refSize={refDesc.Width}x{refDesc.Height}, stripSrgb={stripSrgb}, forceDecompress={forceDecompress}");

             // forceDecompress: always use compute path to decompress BC formats to R8G8B8A8_UNorm
             // (BC1 only has ~32 grayscale levels — far too lossy for density/control data)
             if (formatsMatch && dimsMatch && !forceDecompress)
                 return CreateTexture2DArrayCopy(device, textures, refDesc, stripSrgb);
             else
                 return CreateTexture2DArrayCompute(device, textures, refDesc);
        }

        /// <summary>Fast path: all textures share the same format. Direct GPU copy.</summary>
        private static Texture CreateTexture2DArrayCopy(GraphicsDevice device, IList<Texture> textures, ResourceDescription refDesc, bool stripSrgb = false)
        {
             var desc = refDesc;
             desc.DepthOrArraySize = (ushort)textures.Count;

             // For normal maps: create resource as typeless so we can bind a non-sRGB SRV
             var srvFormat = desc.Format;
             if (stripSrgb)
             {
                 desc.Format = ToTypeless(desc.Format);
                 srvFormat = ToLinear(refDesc.Format);
             }
             
             var arr = new Texture();
             arr._resource = device.NativeDevice.CreateCommittedResource(
                 new HeapProperties(HeapType.Default), HeapFlags.None, desc, ResourceStates.CopyDest, null);
                 
             arr.BindlessIndex = device.AllocateBindlessIndex();
             arr.SrvCpuHandle = device.GetCpuHandle(arr.BindlessIndex);
             arr.SrvHandle = device.GetGpuHandle(arr.BindlessIndex);
             
             var srvDesc = new ShaderResourceViewDescription
             {
                 Format = srvFormat,
                 ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                 Shader4ComponentMapping = ShaderComponentMapping.Default,
                 Texture2DArray = new Texture2DArrayShaderResourceView { MipLevels = refDesc.MipLevels, ArraySize = (uint)textures.Count, FirstArraySlice = 0 }
             };
             device.NativeDevice.CreateShaderResourceView(arr._resource, srvDesc, arr.SrvCpuHandle);
             
             using (var cmd = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct))
             using (var list = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, cmd))
             {
                 for(int i=0; i<textures.Count; ++i)
                 {
                     var src = textures[i];
                     list.ResourceBarrierTransition(src.Native, ResourceStates.Common, ResourceStates.CopySource);
                     
                     for(int mip=0; mip<desc.MipLevels; ++mip)
                     {
                         var dstLoc = new TextureCopyLocation(arr.Native, (uint)(i * desc.MipLevels + mip));
                         var srcLoc = new TextureCopyLocation(src.Native, (uint)mip);
                         list.CopyTextureRegion(dstLoc, 0, 0, 0, srcLoc, null);
                     }
                     
                     list.ResourceBarrierTransition(src.Native, ResourceStates.CopySource, ResourceStates.Common);
                 }
                 
                 list.ResourceBarrierTransition(arr.Native, ResourceStates.CopyDest, ResourceStates.AllShaderResource);
                 list.Close();
                 device.SubmitAndWait(list);
             }
             arr.MarkReady();
             return arr;
        }

        /// <summary>
        /// Mixed-format path: source textures have different compression formats.
        /// Creates an R8G8B8A8 array and uses a compute shader to sample each source
        /// (GPU auto-decompresses any block format) and write to the array slice.
        /// </summary>
        private static Texture CreateTexture2DArrayCompute(GraphicsDevice device, IList<Texture> textures, ResourceDescription refDesc)
        {
             Debug.Log($"[Texture] CreateTexture2DArray: mixed formats detected, using compute copy for {textures.Count} textures");

             int width = (int)refDesc.Width;
             int height = (int)refDesc.Height;
             int mipCount = refDesc.MipLevels;
             int arraySize = textures.Count;

             // Determine output format: use non-sRGB RGBA for UAV compatibility.
             // The SRV will be created with the sRGB variant if any source is sRGB.
             bool anySrgb = false;
             for (int i = 0; i < textures.Count; i++)
             {
                 var fmt = textures[i].Native.Description.Format;
                 if (fmt == Format.BC1_UNorm_SRgb || fmt == Format.BC7_UNorm_SRgb || 
                     fmt == Format.R8G8B8A8_UNorm_SRgb || fmt == Format.B8G8R8A8_UNorm_SRgb)
                     anySrgb = true;
             }
             var uavFormat = Format.R8G8B8A8_UNorm;
             var srvFormat = anySrgb ? Format.R8G8B8A8_UNorm_SRgb : Format.R8G8B8A8_UNorm;

             // Create the output texture array
             var arrayDesc = ResourceDescription.Texture2D(uavFormat, (uint)width, (uint)height,
                 (ushort)arraySize, (ushort)mipCount, 1, 0, ResourceFlags.AllowUnorderedAccess);

             var arr = new Texture();
             arr._resource = device.NativeDevice.CreateCommittedResource(
                 new HeapProperties(HeapType.Default), HeapFlags.None, arrayDesc, ResourceStates.UnorderedAccess, null);

             arr.BindlessIndex = device.AllocateBindlessIndex();
             arr.SrvCpuHandle = device.GetCpuHandle(arr.BindlessIndex);
             arr.SrvHandle = device.GetGpuHandle(arr.BindlessIndex);

             var srvDesc = new ShaderResourceViewDescription
             {
                 Format = srvFormat,
                 ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                 Shader4ComponentMapping = ShaderComponentMapping.Default,
                 Texture2DArray = new Texture2DArrayShaderResourceView { MipLevels = (uint)mipCount, ArraySize = (uint)arraySize, FirstArraySlice = 0 }
             };
             device.NativeDevice.CreateShaderResourceView(arr._resource, srvDesc, arr.SrvCpuHandle);

             // Compile compute shader inline — SM6.6 bindless, push constants at b3
             var shaderSource = @"
cbuffer PushConstants : register(b3) {
    uint SliceIndex;
    uint MipWidth;
    uint MipHeight;
    uint InputSrvIdx;
    uint OutputUavIdx;
};

[numthreads(8, 8, 1)]
void CSCopySlice(uint3 id : SV_DispatchThreadID) {
    if (id.x >= MipWidth || id.y >= MipHeight) return;
    Texture2D<float4> input = ResourceDescriptorHeap[InputSrvIdx];
    RWTexture2DArray<float4> output = ResourceDescriptorHeap[OutputUavIdx];
    output[uint3(id.xy, SliceIndex)] = input[id.xy];
}
";
             var shader = new Shader(shaderSource, "CSCopySlice", "cs_6_6");
             var pso = device.CreateComputePipelineState(shader.Bytecode);
             shader.Dispose();

             using (var cmd = device.NativeDevice.CreateCommandAllocator(CommandListType.Direct))
             using (var list = device.NativeDevice.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Direct, cmd))
             {
                 list.SetComputeRootSignature(device.GlobalRootSignature);
                 list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
                 list.SetPipelineState(pso);

                 for (int i = 0; i < textures.Count; i++)
                 {
                     var src = textures[i];

                     // Transition source to SRV
                     list.ResourceBarrierTransition(src.Native, ResourceStates.Common, ResourceStates.NonPixelShaderResource);

                     int mw = width;
                     int mh = height;
                     for (int mip = 0; mip < mipCount; mip++)
                     {
                         // Allocate per-mip UAV for the target slice+mip
                         var uavIdx = device.AllocateBindlessIndex();
                         var uavDesc = new UnorderedAccessViewDescription
                         {
                             Format = uavFormat,
                             ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                             Texture2DArray = new Texture2DArrayUnorderedAccessView
                             {
                                 MipSlice = (uint)mip,
                                 FirstArraySlice = (uint)i,
                                 ArraySize = 1
                             }
                         };
                         device.NativeDevice.CreateUnorderedAccessView(arr._resource, null, uavDesc, device.GetCpuHandle(uavIdx));

                         // Per-mip SRV for source texture
                         var srcSrvIdx = device.AllocateBindlessIndex();
                         var srcSrvDesc = new ShaderResourceViewDescription
                         {
                             Format = src.Native.Description.Format,
                             ViewDimension = ShaderResourceViewDimension.Texture2D,
                             Shader4ComponentMapping = ShaderComponentMapping.Default,
                             Texture2D = new Texture2DShaderResourceView { MostDetailedMip = (uint)mip, MipLevels = 1 }
                         };
                         device.NativeDevice.CreateShaderResourceView(src.Native, srcSrvDesc, device.GetCpuHandle(srcSrvIdx));

                         // Push constants: use root constants
                         list.SetComputeRoot32BitConstant(0, 0u, 0);            // SliceIndex = 0 (UAV view already targets slice i)
                         list.SetComputeRoot32BitConstant(0, (uint)mw, 1);      // MipWidth
                         list.SetComputeRoot32BitConstant(0, (uint)mh, 2);      // MipHeight
                         list.SetComputeRoot32BitConstant(0, srcSrvIdx, 3);     // InputSrvIdx
                         list.SetComputeRoot32BitConstant(0, uavIdx, 4);        // OutputUavIdx

                         uint gx = (uint)((mw + 7) / 8);
                         uint gy = (uint)((mh + 7) / 8);
                         list.Dispatch(gx, gy, 1);

                         list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

                         mw = Math.Max(1, mw / 2);
                         mh = Math.Max(1, mh / 2);
                     }

                     list.ResourceBarrierTransition(src.Native, ResourceStates.NonPixelShaderResource, ResourceStates.Common);
                 }

                 list.ResourceBarrierTransition(arr._resource, ResourceStates.UnorderedAccess, ResourceStates.AllShaderResource);
                 list.Close();
                 device.SubmitAndWait(list);
             }

             pso.Dispose();
             arr.MarkReady();
             return arr;
        }

        public static float[,] ReadHeightField(string path)
        {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length < 128) return new float[1, 1];
            
            // DDS header: magic(4) + size(4) + flags(4) + height(4) + width(4)
            int height = BitConverter.ToInt32(bytes, 12);
            int width = BitConverter.ToInt32(bytes, 16);
            
            // Detect DX10 extended header: if fourCC == 'DX10' (0x30315844), add 20 bytes
            int fourCC = BitConverter.ToInt32(bytes, 84);
            int dataOffset = 128;
            if (fourCC == 0x30315844) // 'DX10'
                dataOffset += 20;
            
            var field = new float[width, height];
            int remaining = bytes.Length - dataOffset;
            int bytesPerPixel = remaining / (width * height);
            
            // Detect format by bytes per pixel
            if (bytesPerPixel >= 4)
            {
                // R32_Float or R8G8B8A8 — treat as float
                for (int y = 0; y < height && dataOffset + 3 < bytes.Length; y++)
                {
                    for (int x = 0; x < width && dataOffset + 3 < bytes.Length; x++)
                    {
                        field[x, y] = BitConverter.ToSingle(bytes, dataOffset);
                        dataOffset += 4;
                    }
                }
            }
            else if (bytesPerPixel >= 2)
            {
                // R16_Float (half-float) — convert via Half, matching Apex To16BitField
                for (int y = 0; y < height && dataOffset + 1 < bytes.Length; y++)
                {
                    for (int x = 0; x < width && dataOffset + 1 < bytes.Length; x++)
                    {
                        Half halfValue = BitConverter.ToHalf(bytes, dataOffset);
                        field[x, y] = (float)halfValue;
                        dataOffset += 2;
                    }
                }
            }
            else
            {
                // R8_UNorm fallback
                for (int y = 0; y < height && dataOffset < bytes.Length; y++)
                {
                    for (int x = 0; x < width && dataOffset < bytes.Length; x++)
                    {
                        field[x, y] = bytes[dataOffset] / 255.0f;
                        dataOffset += 1;
                    }
                }
            }
            
            Debug.Log($"[Texture] ReadHeightField: {width}x{height}, {bytesPerPixel} bytes/pixel, dataOffset={128 + (fourCC == 0x30315844 ? 20 : 0)}");
            return field;
        }

        public Texture() { }

        /// <summary>
        /// Wrap an existing D3D12 resource + SRV into a Texture. Used by GPU bake pipelines
        /// that create their own resources and need a Texture wrapper for the rendering pipeline.
        /// </summary>
        public static Texture WrapNative(ID3D12Resource resource, uint srvIndex)
        {
            var tex = new Texture();
            tex._resource = resource;
            tex.BindlessIndex = srvIndex;
            tex.SrvCpuHandle = Engine.Device.GetCpuHandle(srvIndex);
            tex.SrvHandle = Engine.Device.GetGpuHandle(srvIndex);
            tex.MarkReady();
            return tex;
        }

        public static Texture CreateAsync(GraphicsDevice device, CpuTextureData cpuData)
        {
            var texture = new Texture();
            // 1. Create committed resource (fast on main thread)
            var desc = ResourceDescription.Texture2D(cpuData.Format, (uint)cpuData.Width, (uint)cpuData.Height, (ushort)cpuData.ArraySize, (ushort)cpuData.MipLevels);
            
            // Textures on copy queue should be in Common or CopyDest
            texture._resource = device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Default),
                HeapFlags.None,
                desc,
                ResourceStates.Common); // Implicit promotion to CopyDest will happen on CopyQueue

            // 2. Create SRV immediately (so we can bind it even if black/empty)
            texture.BindlessIndex = device.AllocateBindlessIndex();
            texture.SrvCpuHandle = device.GetCpuHandle(texture.BindlessIndex);
            texture.SrvHandle = device.GetGpuHandle(texture.BindlessIndex);
            
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = cpuData.Format,
                ViewDimension = cpuData.ArraySize > 1 ? ShaderResourceViewDimension.Texture2DArray : ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MipLevels = (uint)cpuData.MipLevels }
            };
            device.NativeDevice.CreateShaderResourceView(texture._resource, srvDesc, texture.SrvCpuHandle);

            // 3. Queue Upload
            StreamingManager.Instance.EnqueueTextureUpload(texture, cpuData);
            
            texture.AssetPath = cpuData.Path;
            texture.Name = Path.GetFileNameWithoutExtension(cpuData.Path);
            
            return texture;
        }

        public static Texture CreateFromCpuData(GraphicsDevice device, CpuTextureData cpuData)
        {
             // Legacy sync path reused for now, or just delegate to async and Wait?
             // For safety, let's keep the sync path synchronous or use the new async path and mark ready immediately?
             // The old CreateFromCpuData used GraphicsDevice.SubmitAndWait.
             // We can just use CreateAsync and force a wait if we really want sync.
             var tex = CreateAsync(device, cpuData);
             // TODO: If we need strictly sync return, we should wait on the fence.
             // But for now, let's assume callers can handle the 'pop in'.
             // Or we can manually flush.
             return tex;
        }

        // ... [Rest of static helpers like LoadFromFile, ParseFromFile preserved/forwarded] ...

        // Re-implementing LoadFromFile for back-compat (mostly used by tests or single asset loads)
        public static Texture LoadFromFile(GraphicsDevice device, string filePath)
        {
             var cpuData = ParseFromFile(device, filePath);
             var tex = CreateAsync(device, cpuData);
             // Verify: legacy code expects it to be ready.
             // We could block here: 
             // StreamingManager.Instance.Flush(); // if implemented
             return tex;
        }
        
         public static Freefall.Assets.CpuTextureData ParseFromFile(GraphicsDevice device, string filePath)
        {
            var ext = Path.GetExtension(filePath).ToLower();
            // ... [Same implementation as previous Turn Step 8 ] ...
             if (ext == ".dds") return ParseDDS(filePath);

            // WIC path (PNG, TGA, JPG, etc.)
            var factory = device.WicFactory;

            using var decoder = factory.CreateDecoderFromFileName(filePath);
            using var frame = decoder.GetFrame(0);
            using var converter = factory.CreateFormatConverter();

            converter.Initialize(frame, PixelFormat.Format32bppRGBA, BitmapDitherType.None, null, 0, BitmapPaletteType.Custom);

            int width = converter.Size.Width;
            int height = converter.Size.Height;
            int mipLevels = 1 + (int)Math.Floor(Math.Log2(Math.Max(width, height)));

            // Read mip 0 from WIC
            byte[] mip0Data = new byte[width * height * 4];
            converter.CopyPixels((uint)(width * 4), mip0Data);

            // Calculate total size and build mip chain
            var mips = new Freefall.Assets.MipLayout[mipLevels];
            int totalSize = 0;
            int mipW = width, mipH = height;
            for (int mip = 0; mip < mipLevels; mip++)
            {
                mips[mip] = new Freefall.Assets.MipLayout
                {
                    Offset = totalSize,
                    RowPitch = mipW * 4,
                    NumRows = mipH,
                    Width = mipW,
                    Height = mipH
                };
                totalSize += mipW * mipH * 4;
                mipW = Math.Max(1, mipW / 2);
                mipH = Math.Max(1, mipH / 2);
            }

            // Build contiguous pixel data with all mip levels
            byte[] pixelData = new byte[totalSize];
            Array.Copy(mip0Data, 0, pixelData, 0, mip0Data.Length);

            // Generate mip chain via box filter
            byte[] currentMip = mip0Data;
            mipW = width;
            mipH = height;
            for (int mip = 0; mip < mipLevels - 1; mip++)
            {
                int nextW = Math.Max(1, mipW / 2);
                int nextH = Math.Max(1, mipH / 2);
                byte[] nextMip = new byte[nextW * nextH * 4];

                for (int y = 0; y < nextH; y++)
                {
                    for (int x = 0; x < nextW; x++)
                    {
                        int srcX = x * 2;
                        int srcY = y * 2;
                        for (int c = 0; c < 4; c++)
                        {
                            int sum = 0;
                            int count = 0;
                            for (int dy = 0; dy < 2 && srcY + dy < mipH; dy++)
                            {
                                for (int dx = 0; dx < 2 && srcX + dx < mipW; dx++)
                                {
                                    sum += currentMip[((srcY + dy) * mipW + (srcX + dx)) * 4 + c];
                                    count++;
                                }
                            }
                            nextMip[(y * nextW + x) * 4 + c] = (byte)(sum / count);
                        }
                    }
                }

                Array.Copy(nextMip, 0, pixelData, mips[mip + 1].Offset, nextMip.Length);
                currentMip = nextMip;
                mipW = nextW;
                mipH = nextH;
            }

            return new Freefall.Assets.CpuTextureData
            {
                Path = filePath,
                PixelData = pixelData,
                Width = width,
                Height = height,
                MipLevels = mipLevels,
                ArraySize = 1,
                Format = Format.R8G8B8A8_UNorm,
                IsCompressed = false,
                Mips = mips
            };
        }
        
         private static Freefall.Assets.CpuTextureData ParseDDS(string filePath)
        {
            var bytes = File.ReadAllBytes(filePath);
            var result = ParseDDSFromBytes(bytes);
            result.Path = filePath;
            return result;
        }

        public static Freefall.Assets.CpuTextureData ParseDDSFromBytes(byte[] bytes)
        {
            unsafe
            {
                fixed (byte* pBytes = bytes)
                {
                    DDSTextureLoader.Parse((IntPtr)pBytes, bytes.Length, out var format, out int width, out int height, out int depth, out int mipLevels, out int arraySize, out bool isCubeMap, out IntPtr dataPtr, out int dataSize);

                    // Build mip layouts
                    var mips = new Freefall.Assets.MipLayout[mipLevels];
                    bool compressed = IsCompressedStatic(format);
                    int offset = 0;

                    for (int mip = 0; mip < mipLevels; mip++)
                    {
                        int mipW = Math.Max(1, width >> mip);
                        int mipH = Math.Max(1, height >> mip);
                        int srcPitch = ComputePitchStatic(format, mipW);
                        int numRows = compressed ? Math.Max(1, (mipH + 3) / 4) : mipH;

                        mips[mip] = new Freefall.Assets.MipLayout
                        {
                            Offset = offset,
                            RowPitch = srcPitch,
                            NumRows = numRows,
                            Width = mipW,
                            Height = mipH
                        };

                        offset += srcPitch * numRows;
                    }

                    // Copy pixel data to managed array
                    byte[] pixelData = new byte[offset];
                    System.Runtime.InteropServices.Marshal.Copy(dataPtr, pixelData, 0, Math.Min(offset, dataSize));

                    return new Freefall.Assets.CpuTextureData
                    {
                        PixelData = pixelData,
                        Width = width,
                        Height = height,
                        MipLevels = mipLevels,
                        ArraySize = arraySize,
                        Format = format,
                        IsCompressed = compressed,
                        Mips = mips
                    };
                }
            }
        }
        
        private static int ComputePitchStatic(Format fmt, int width)
        {
            if (IsCompressedStatic(fmt))
            {
                int blockSize = 16;
                if (fmt == Format.BC1_UNorm || fmt == Format.BC1_UNorm_SRgb || fmt == Format.BC4_UNorm || fmt == Format.BC4_SNorm) blockSize = 8;
                int numBlocksWide = Math.Max(1, (width + 3) / 4);
                return numBlocksWide * blockSize;
            }

            int bpp = fmt switch
            {
                Format.R8G8B8A8_UNorm => 32,
                Format.R32_Float => 32,
                Format.R16_Float => 16,
                Format.R16_UNorm => 16,
                Format.R8_UNorm => 8,
                _ => 32
            };
            return (width * bpp + 7) / 8;
        }

        private static bool IsCompressedStatic(Format fmt)
        {
            return fmt switch
            {
                Format.BC1_Typeless or Format.BC1_UNorm or Format.BC1_UNorm_SRgb or
                Format.BC2_Typeless or Format.BC2_UNorm or Format.BC2_UNorm_SRgb or
                Format.BC3_Typeless or Format.BC3_UNorm or Format.BC3_UNorm_SRgb or
                Format.BC4_Typeless or Format.BC4_UNorm or Format.BC4_SNorm or
                Format.BC5_Typeless or Format.BC5_UNorm or Format.BC5_SNorm or
                Format.BC6H_Typeless or Format.BC6H_Uf16 or Format.BC6H_Sf16 or
                Format.BC7_Typeless or Format.BC7_UNorm or Format.BC7_UNorm_SRgb => true,
                _ => false,
            };
        }

        /// <summary>Convert an sRGB or typed format to its typeless equivalent for flexible SRV creation.</summary>
        private static Format ToTypeless(Format fmt) => fmt switch
        {
            Format.BC1_UNorm or Format.BC1_UNorm_SRgb => Format.BC1_Typeless,
            Format.BC2_UNorm or Format.BC2_UNorm_SRgb => Format.BC2_Typeless,
            Format.BC3_UNorm or Format.BC3_UNorm_SRgb => Format.BC3_Typeless,
            Format.BC7_UNorm or Format.BC7_UNorm_SRgb => Format.BC7_Typeless,
            Format.R8G8B8A8_UNorm or Format.R8G8B8A8_UNorm_SRgb => Format.R8G8B8A8_Typeless,
            Format.B8G8R8A8_UNorm or Format.B8G8R8A8_UNorm_SRgb => Format.B8G8R8A8_Typeless,
            _ => fmt
        };

        /// <summary>Convert an sRGB format to its linear (non-sRGB) equivalent.</summary>
        private static Format ToLinear(Format fmt) => fmt switch
        {
            Format.BC1_UNorm_SRgb => Format.BC1_UNorm,
            Format.BC2_UNorm_SRgb => Format.BC2_UNorm,
            Format.BC3_UNorm_SRgb => Format.BC3_UNorm,
            Format.BC7_UNorm_SRgb => Format.BC7_UNorm,
            Format.R8G8B8A8_UNorm_SRgb => Format.R8G8B8A8_UNorm,
            Format.B8G8R8A8_UNorm_SRgb => Format.B8G8R8A8_UNorm,
            _ => fmt
        };

        public void Dispose()
        {
            _resource?.Dispose();
             // Release bindless index?
        }
    }
}
