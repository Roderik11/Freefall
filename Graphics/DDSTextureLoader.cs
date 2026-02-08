// DDSTextureLoader Ported to C# by Justin Stenning, March 2017
//--------------------------------------------------------------------------------------
// File: DDSTextureLoader.cpp
//
// Functions for loading a DDS texture and creating a Direct3D runtime resource for it
//
// Note these functions are useful as a light-weight runtime loader for DDS files. For
// a full-featured DDS file reader, writer, and texture processing pipeline see
// the 'Texconv' sample and the 'DirectXTex' library.
//
// THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
// PARTICULAR PURPOSE.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//
// http://go.microsoft.com/fwlink/?LinkId=248926
// http://go.microsoft.com/fwlink/?LinkId=248929
//--------------------------------------------------------------------------------------

using System;
using Vortice.Mathematics;
using Vortice.DXGI;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using System.Runtime.InteropServices;
using Device = Vortice.Direct3D12.ID3D12Device; // Modified for DX12 (Note: usage logic needs checking)
using Resource = Vortice.Direct3D12.ID3D12Resource; // Modified for DX12 
using System.IO;
using Vortice;

// Note: The original code uses D3D11 types. We are in DX12 context.
// However, the logic for parsing DDS headers is generic.
// IMPORTANT: The resource creation part is D3D11 specific.
// We only need this loader to PARSE the DDS header and get the data pointer + description.
// Then we can use our existing texture creation logic (upload heap etc)
// OR we port CreateTextureFromDDS to DX12.
// Given time constraints, I will refactor this to just return the Data/Description 
// and handle creation in Texture.cs or Refactor it to support DX12.

// Wait, this file relies HEAVILY on D3D11 Device/Context.
// Porting to DX12 is non-trivial if I keep the structure.
// BETTER APPROACH: Extract the header parsing logic to get Width/Height/Format/MipMaps
// And the data pointer.

namespace Freefall.Graphics
{
    public static class DDSTextureLoader
    {
        public enum DDS_ALPHA_MODE
        {
            DDS_ALPHA_MODE_UNKNOWN = 0,
            DDS_ALPHA_MODE_STRAIGHT = 1,
            DDS_ALPHA_MODE_PREMULTIPLIED = 2,
            DDS_ALPHA_MODE_OPAQUE = 3,
            DDS_ALPHA_MODE_CUSTOM = 4,
        };

        const int DDS_MAGIC = 0x20534444;// "DDS "

        [StructLayout(LayoutKind.Sequential)]
        public struct DDS_PIXELFORMAT
        {
            public int size;
            public int flags;
            public int fourCC;
            public int RGBBitCount;
            public uint RBitMask;
            public uint GBitMask;
            public uint BBitMask;
            public uint ABitMask;
        };

        const int DDS_FOURCC = 0x00000004;// DDPF_FOURCC
        const int DDS_RGB = 0x00000040;// DDPF_RGB
        const int DDS_RGBA = 0x00000041;// DDPF_RGB | DDPF_ALPHAPIXELS
        const int DDS_LUMINANCE = 0x00020000;// DDPF_LUMINANCE
        const int DDS_LUMINANCEA = 0x00020001;// DDPF_LUMINANCE | DDPF_ALPHAPIXELS
        const int DDS_ALPHA = 0x00000002;// DDPF_ALPHA
        const int DDS_PAL8 = 0x00000020;// DDPF_PALETTEINDEXED8

        const int DDS_HEADER_FLAGS_TEXTURE = 0x00001007;// DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
        const int DDS_HEADER_FLAGS_MIPMAP = 0x00020000;// DDSD_MIPMAPCOUNT
        const int DDS_HEADER_FLAGS_VOLUME = 0x00800000;// DDSD_DEPTH
        const int DDS_HEADER_FLAGS_PITCH = 0x00000008;// DDSD_PITCH
        const int DDS_HEADER_FLAGS_LINEARSIZE = 0x00080000;// DDSD_LINEARSIZE

        const int DDS_HEIGHT = 0x00000002;// DDSD_HEIGHT
        const int DDS_WIDTH = 0x00000004;// DDSD_WIDTH

        const int DDS_SURFACE_FLAGS_TEXTURE = 0x00001000;// DDSCAPS_TEXTURE
        const int DDS_SURFACE_FLAGS_MIPMAP = 0x00400008;// DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
        const int DDS_SURFACE_FLAGS_CUBEMAP = 0x00000008;// DDSCAPS_COMPLEX

        const int DDS_CUBEMAP_POSITIVEX = 0x00000600;// DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEX
        const int DDS_CUBEMAP_NEGATIVEX = 0x00000a00;// DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEX
        const int DDS_CUBEMAP_POSITIVEY = 0x00001200;// DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEY
        const int DDS_CUBEMAP_NEGATIVEY = 0x00002200;// DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEY
        const int DDS_CUBEMAP_POSITIVEZ = 0x00004200;// DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEZ
        const int DDS_CUBEMAP_NEGATIVEZ = 0x00008200;// DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEZ

        const int DDS_CUBEMAP_ALLFACES = (DDS_CUBEMAP_POSITIVEX | DDS_CUBEMAP_NEGATIVEX | DDS_CUBEMAP_POSITIVEY | DDS_CUBEMAP_NEGATIVEY | DDS_CUBEMAP_POSITIVEZ | DDS_CUBEMAP_NEGATIVEZ);

        const int DDS_CUBEMAP = 0x00000200;// DDSCAPS2_CUBEMAP

        const int DDS_FLAGS_VOLUME = 0x00200000;// DDSCAPS2_VOLUME

        [StructLayout(LayoutKind.Sequential)]
        public struct DDS_HEADER
        {
            public int size;
            public int flags;
            public int height;
            public int width;
            public int pitchOrLinearSize;
            public int depth; // only if DDS_HEADER_FLAGS_VOLUME is set in flags
            public int mipMapCount;
            //===11
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 11)]
            public int[] reserved1;

            public DDS_PIXELFORMAT ddspf;
            public int caps;
            public int caps2;
            public int caps3;
            public int caps4;
            public int reserved2;


        }

        enum DDS_MISC_FLAGS2
        {
            DDS_MISC_FLAGS2_ALPHA_MODE_MASK = 0x7,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct DDS_HEADER_DXT10
        {
            public Format dxgiFormat;
            public ResourceDimension resourceDimension;
            public ResourceOptionFlags miscFlag; // see D3D11_RESOURCE_MISC_FLAG
            public int arraySize;
            public int miscFlags2;
        }


        static int BitsPerPixel(Format fmt)
        {
            switch (fmt)
            {
                case Format.R32G32B32A32_Typeless:
                case Format.R32G32B32A32_Float:
                case Format.R32G32B32A32_UInt:
                case Format.R32G32B32A32_SInt:
                    return 128;

                case Format.R32G32B32_Typeless:
                case Format.R32G32B32_Float:
                case Format.R32G32B32_UInt:
                case Format.R32G32B32_SInt:
                    return 96;

                case Format.R16G16B16A16_Typeless:
                case Format.R16G16B16A16_Float:
                case Format.R16G16B16A16_UNorm:
                case Format.R16G16B16A16_UInt:
                case Format.R16G16B16A16_SNorm:
                case Format.R16G16B16A16_SInt:
                case Format.R32G32_Typeless:
                case Format.R32G32_Float:
                case Format.R32G32_UInt:
                case Format.R32G32_SInt:
                case Format.R32G8X24_Typeless:
                case Format.D32_Float_S8X24_UInt:
                case Format.R32_Float_X8X24_Typeless:
                case Format.X32_Typeless_G8X24_UInt:
                    return 64;

                case Format.R10G10B10A2_Typeless:
                case Format.R10G10B10A2_UNorm:
                case Format.R10G10B10A2_UInt:
                case Format.R11G11B10_Float:
                case Format.R8G8B8A8_Typeless:
                case Format.R8G8B8A8_UNorm:
                case Format.R8G8B8A8_UNorm_SRgb:
                case Format.R8G8B8A8_UInt:
                case Format.R8G8B8A8_SNorm:
                case Format.R8G8B8A8_SInt:
                case Format.R16G16_Typeless:
                case Format.R16G16_Float:
                case Format.R16G16_UNorm:
                case Format.R16G16_UInt:
                case Format.R16G16_SNorm:
                case Format.R16G16_SInt:
                case Format.R32_Typeless:
                case Format.D32_Float:
                case Format.R32_Float:
                case Format.R32_UInt:
                case Format.R32_SInt:
                case Format.R24G8_Typeless:
                case Format.D24_UNorm_S8_UInt:
                case Format.R24_UNorm_X8_Typeless:
                case Format.X24_Typeless_G8_UInt:
                case Format.R9G9B9E5_SharedExp:
                case Format.R8G8_B8G8_UNorm:
                case Format.G8R8_G8B8_UNorm:
                case Format.B8G8R8A8_UNorm:
                case Format.B8G8R8X8_UNorm:
                case Format.R10G10B10_Xr_Bias_A2_UNorm:
                case Format.B8G8R8A8_Typeless:
                case Format.B8G8R8A8_UNorm_SRgb:
                case Format.B8G8R8X8_Typeless:
                case Format.B8G8R8X8_UNorm_SRgb:
                    return 32;

                case Format.R8G8_Typeless:
                case Format.R8G8_UNorm:
                case Format.R8G8_UInt:
                case Format.R8G8_SNorm:
                case Format.R8G8_SInt:
                case Format.R16_Typeless:
                case Format.R16_Float:
                case Format.D16_UNorm:
                case Format.R16_UNorm:
                case Format.R16_UInt:
                case Format.R16_SNorm:
                case Format.R16_SInt:
                case Format.B5G6R5_UNorm:
                case Format.B5G5R5A1_UNorm:
                case Format.B4G4R4A4_UNorm:
                    return 16;

                case Format.R8_Typeless:
                case Format.R8_UNorm:
                case Format.R8_UInt:
                case Format.R8_SNorm:
                case Format.R8_SInt:
                case Format.A8_UNorm:
                    return 8;

                case Format.R1_UNorm:
                    return 1;

                case Format.BC1_Typeless:
                case Format.BC1_UNorm:
                case Format.BC1_UNorm_SRgb:
                case Format.BC4_Typeless:
                case Format.BC4_UNorm:
                case Format.BC4_SNorm:
                    return 4;

                case Format.BC2_Typeless:
                case Format.BC2_UNorm:
                case Format.BC2_UNorm_SRgb:
                case Format.BC3_Typeless:
                case Format.BC3_UNorm:
                case Format.BC3_UNorm_SRgb:
                case Format.BC5_Typeless:
                case Format.BC5_UNorm:
                case Format.BC5_SNorm:
                case Format.BC6H_Typeless:
                case Format.BC6H_Uf16:
                case Format.BC6H_Sf16:
                case Format.BC7_Typeless:
                case Format.BC7_UNorm:
                case Format.BC7_UNorm_SRgb:
                    return 8;

                default:
                    return 0;
            }
        }


        static bool ISBITMASK(DDS_PIXELFORMAT ddpf, uint r, uint g, uint b, uint a)
        {
            return (ddpf.RBitMask == r && ddpf.GBitMask == g && ddpf.BBitMask == b && ddpf.ABitMask == a);
        }

        static int MAKEFOURCC(int ch0, int ch1, int ch2, int ch3)
        {
            return ((int)(byte)(ch0) | ((int)(byte)(ch1) << 8) | ((int)(byte)(ch2) << 16) | ((int)(byte)(ch3) << 24));
        }


        static Format GetDXGIFormat(DDS_PIXELFORMAT ddpf)
        {

            if ((ddpf.flags & DDS_RGB) > 0)
            {
                // Note that sRGB formats are written using the "DX10" extended header

                switch (ddpf.RGBBitCount)
                {
                    case 32:
                        if (ISBITMASK(ddpf, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
                        {
                            return Format.R8G8B8A8_UNorm;
                        }

                        if (ISBITMASK(ddpf, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000))
                        {
                            return Format.B8G8R8A8_UNorm;
                        }

                        if (ISBITMASK(ddpf, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000))
                        {
                            return Format.B8G8R8X8_UNorm;
                        }

                        // No DXGI format maps to ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000) aka D3DFMT_X8B8G8R8

                        // Note that many common DDS reader/writers (including D3DX) swap the
                        // the RED/BLUE masks for 10:10:10:2 formats. We assumme
                        // below that the 'backwards' header mask is being used since it is most
                        // likely written by D3DX. The more robust solution is to use the 'DX10'
                        // header extension and specify the DXGI_FORMAT_R10G10B10A2_UNORM format directly

                        // For 'correct' writers, this should be 0x000003ff, 0x000ffc00, 0x3ff00000 for RGB data
                        if (ISBITMASK(ddpf, 0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000))
                        {
                            return Format.R10G10B10A2_UNorm;
                        }

                        // No DXGI format maps to ISBITMASK(0x000003ff, 0x000ffc00, 0x3ff00000, 0xc0000000) aka D3DFMT_A2R10G10B10

                        if (ISBITMASK(ddpf, 0x0000ffff, 0xffff0000, 0x00000000, 0x00000000))
                        {
                            return Format.R16G16_UNorm;
                        }

                        if (ISBITMASK(ddpf, 0xffffffff, 0x00000000, 0x00000000, 0x00000000))
                        {
                            // Only 32-bit color channel format in D3D9 was R32F
                            return Format.R32_Float; // D3DX writes this out as a FourCC of 114
                        }
                        break;

                    case 24:
                        // No 24bpp DXGI formats aka D3DFMT_R8G8B8
                        break;

                    case 16:
                        if (ISBITMASK(ddpf, 0x7c00, 0x03e0, 0x001f, 0x8000))
                        {
                            return Format.B5G5R5A1_UNorm;
                        }
                        if (ISBITMASK(ddpf, 0xf800, 0x07e0, 0x001f, 0x0000))
                        {
                            return Format.B5G6R5_UNorm;
                        }

                        // No DXGI format maps to ISBITMASK(0x7c00, 0x03e0, 0x001f, 0x0000) aka D3DFMT_X1R5G5B5
                        if (ISBITMASK(ddpf, 0x0f00, 0x00f0, 0x000f, 0xf000))
                        {
                            return Format.B4G4R4A4_UNorm;
                        }

                        // No DXGI format maps to ISBITMASK(0x0f00, 0x00f0, 0x000f, 0x0000) aka D3DFMT_X4R4G4B4

                        // No 3:3:2, 3:3:2:8, or paletted DXGI formats aka D3DFMT_A8R3G3B2, D3DFMT_R3G3B2, D3DFMT_P8, D3DFMT_A8P8, etc.
                        break;
                }
            }
            else if ((ddpf.flags & DDS_LUMINANCE) > 0)
            {
                if (8 == ddpf.RGBBitCount)
                {
                    if (ISBITMASK(ddpf, 0x000000ff, 0x00000000, 0x00000000, 0x00000000))
                    {
                        return Format.R8_UNorm; // D3DX10/11 writes this out as DX10 extension
                    }

                    // No DXGI format maps to ISBITMASK(0x0f, 0x00, 0x00, 0xf0) aka D3DFMT_A4L4
                }

                if (16 == ddpf.RGBBitCount)
                {
                    if (ISBITMASK(ddpf, 0x0000ffff, 0x00000000, 0x00000000, 0x00000000))
                    {
                        return Format.R16_UNorm; // D3DX10/11 writes this out as DX10 extension
                    }
                    if (ISBITMASK(ddpf, 0x000000ff, 0x00000000, 0x00000000, 0x0000ff00))
                    {
                        return Format.R8G8_UNorm; // D3DX10/11 writes this out as DX10 extension
                    }
                }
            }
            else if ((ddpf.flags & DDS_ALPHA) > 0)
            {
                if (8 == ddpf.RGBBitCount)
                {
                    return Format.A8_UNorm;
                }
            }
            else if ((ddpf.flags & DDS_FOURCC) > 0)
            {
                if (MAKEFOURCC('D', 'X', 'T', '1') == ddpf.fourCC)
                {
                    return Format.BC1_UNorm;
                }
                if (MAKEFOURCC('D', 'X', 'T', '3') == ddpf.fourCC)
                {
                    return Format.BC2_UNorm;
                }
                if (MAKEFOURCC('D', 'X', 'T', '5') == ddpf.fourCC)
                {
                    return Format.BC3_UNorm;
                }

                // While pre-mulitplied alpha isn't directly supported by the DXGI formats,
                // they are basically the same as these BC formats so they can be mapped
                if (MAKEFOURCC('D', 'X', 'T', '2') == ddpf.fourCC)
                {
                    return Format.BC2_UNorm;
                }
                if (MAKEFOURCC('D', 'X', 'T', '4') == ddpf.fourCC)
                {
                    return Format.BC3_UNorm;
                }

                if (MAKEFOURCC('A', 'T', 'I', '1') == ddpf.fourCC)
                {
                    return Format.BC4_UNorm;
                }
                if (MAKEFOURCC('B', 'C', '4', 'U') == ddpf.fourCC)
                {
                    return Format.BC4_UNorm;
                }
                if (MAKEFOURCC('B', 'C', '4', 'S') == ddpf.fourCC)
                {
                    return Format.BC4_SNorm;
                }

                if (MAKEFOURCC('A', 'T', 'I', '2') == ddpf.fourCC)
                {
                    return Format.BC5_UNorm;
                }
                if (MAKEFOURCC('B', 'C', '5', 'U') == ddpf.fourCC)
                {
                    return Format.BC5_UNorm;
                }
                if (MAKEFOURCC('B', 'C', '5', 'S') == ddpf.fourCC)
                {
                    return Format.BC5_SNorm;
                }

                // BC6H and BC7 are written using the "DX10" extended header

                if (MAKEFOURCC('R', 'G', 'B', 'G') == ddpf.fourCC)
                {
                    return Format.R8G8_B8G8_UNorm;
                }
                if (MAKEFOURCC('G', 'R', 'G', 'B') == ddpf.fourCC)
                {
                    return Format.G8R8_G8B8_UNorm;
                }

                // Check for D3DFORMAT enums being set here
                switch (ddpf.fourCC)
                {
                    case 36: // D3DFMT_A16B16G16R16
                        return Format.R16G16B16A16_UNorm;

                    case 110: // D3DFMT_Q16W16V16U16
                        return Format.R16G16B16A16_SNorm;

                    case 111: // D3DFMT_R16F
                        return Format.R16_Float;

                    case 112: // D3DFMT_G16R16F
                        return Format.R16G16_Float;

                    case 113: // D3DFMT_A16B16G16R16F
                        return Format.R16G16B16A16_Float;

                    case 114: // D3DFMT_R32F
                        return Format.R32_Float;

                    case 115: // D3DFMT_G32R32F
                        return Format.R32G32_Float;

                    case 116: // D3DFMT_A32B32G32R32F
                        return Format.R32G32B32A32_Float;
                }
            }

            return Format.Unknown;
        }

        // Exposed helper to parse DDS info
        public static void Parse(IntPtr ptr, int size, out Format format, out int width, out int height, out int depth, out int mipLevels, out int arraySize, out bool isCubeMap, out IntPtr bitData, out int bitSize)
        {
             var sizeofDDS_HEADER = Marshal.SizeOf(typeof(DDS_HEADER));
             var sizeofDDS_MAGIC = sizeof(int);
             var sizeofDDS_HEADER_DXT10 = Marshal.SizeOf(typeof(DDS_HEADER_DXT10));

             int dwMagicNumber = Marshal.ReadInt32(ptr);
             if (dwMagicNumber != DDS_MAGIC) throw new Exception("Invalid DDS Magic");

             var header = (DDS_HEADER)Marshal.PtrToStructure(ptr + sizeofDDS_MAGIC, typeof(DDS_HEADER));

             // Check for DX10 extension
             bool bDXT10Header = false;
             if ((header.ddspf.flags & DDS_FOURCC) > 0 &&
                 (MAKEFOURCC('D', 'X', '1', '0') == header.ddspf.fourCC))
             {
                 bDXT10Header = true;
             }

             width = header.width;
             height = header.height;
             depth = header.depth;
             mipLevels = header.mipMapCount == 0 ? 1 : header.mipMapCount;

             isCubeMap = false;
             arraySize = 1;

             if (bDXT10Header)
             {
                 int offset = sizeofDDS_MAGIC + sizeofDDS_HEADER;
                 var d3d10ext = (DDS_HEADER_DXT10)Marshal.PtrToStructure(ptr + offset, typeof(DDS_HEADER_DXT10));
                 
                 format = d3d10ext.dxgiFormat;
                 arraySize = d3d10ext.arraySize;
                 
                 if ((d3d10ext.miscFlag & ResourceOptionFlags.TextureCube) > 0)
                 {
                     isCubeMap = true;
                     arraySize *= 6; 
                 }
                 
                 bitData = ptr + offset + sizeofDDS_HEADER_DXT10;
                 bitSize = size - (offset + sizeofDDS_HEADER_DXT10);
             }
             else
             {
                 format = GetDXGIFormat(header.ddspf);
                 if ((header.caps2 & DDS_CUBEMAP) > 0)
                 {
                     if ((header.caps2 & DDS_CUBEMAP_ALLFACES) == DDS_CUBEMAP_ALLFACES)
                     {
                         isCubeMap = true;
                         arraySize = 6;
                     }
                 }
                 
                 bitData = ptr + sizeofDDS_MAGIC + sizeofDDS_HEADER;
                 bitSize = size - (sizeofDDS_MAGIC + sizeofDDS_HEADER);
             }
        }
    }
}
