using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using StbTrueTypeSharp;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-accelerated font using the Slug algorithm.
    /// Renders text directly from quadratic Bézier curve data on the GPU.
    /// No bitmap atlas, no distance fields — resolution independent at any size.
    /// </summary>
    public class SlugFont : IDisposable
    {
        private const int TextureWidth = 4096;

        // GPU resources
        private ID3D12Resource _curveTexture = null!;
        private ID3D12Resource _bandTexture = null!;
        private uint _curveBindlessIndex;
        private uint _bandBindlessIndex;

        // Font metrics (in font design units)
        public int UnitsPerEm { get; private set; }
        public int Ascent { get; private set; }
        public int Descent { get; private set; }
        public int LineGap { get; private set; }

        // Per-glyph data
        private Dictionary<int, SlugGlyph> _glyphs = new();

        // Kerning pairs: key = (leftCodepoint << 16 | rightCodepoint), value = kern advance in font units
        private Dictionary<int, int> _kernPairs = new();

        public uint CurveTextureIndex => _curveBindlessIndex;
        public uint BandTextureIndex => _bandBindlessIndex;

        /// <summary>
        /// Get kerning adjustment between two codepoints, in font design units.
        /// </summary>
        public int GetKerning(int leftCodepoint, int rightCodepoint)
        {
            _kernPairs.TryGetValue((leftCodepoint << 16) | rightCodepoint, out int kern);
            return kern;
        }

        public bool TryGetGlyph(int codePoint, out SlugGlyph glyph)
            => _glyphs.TryGetValue(codePoint, out glyph);

        /// <summary>
        /// Per-glyph metadata needed for rendering and layout.
        /// </summary>
        public struct SlugGlyph
        {
            // Bounding box in font design units
            public int BBoxX1, BBoxY1, BBoxX2, BBoxY2;

            // Advance width and left-side bearing in font design units
            public int AdvanceWidth;
            public int LeftSideBearing;

            // Band info for the Slug shader
            public int BandCount;       // Number of bands per axis
            public int BandDimX;        // Width of each band in font units
            public int BandDimY;        // Height of each band in font units
            public int BandTexCoordX;   // Origin in band texture (texel X)
            public int BandTexCoordY;   // Origin in band texture (texel Y)

            public int Width => BBoxX2 - BBoxX1;
            public int Height => BBoxY2 - BBoxY1;
        }

        // ─── Internal types for font processing ───

        private struct Curve
        {
            public Vector2 Start, Control, End;
            public int TexelIndex;
        }

        private struct BandTexelCoord
        {
            public ushort X, Y;
            public BandTexelCoord(ushort x, ushort y) { X = x; Y = y; }
        }

        private struct GlyphBandRange
        {
            public int HeaderStart, CurveStart, HeaderCount;
        }

        // ─── Loading ───

        /// <summary>
        /// Load a SlugFont from a TTF file and upload curve/band textures to the GPU.
        /// </summary>
        public static SlugFont LoadFromFile(GraphicsDevice device, string path)
        {
            byte[] ttfData = File.ReadAllBytes(path);
            return LoadFromMemory(device, ttfData);
        }

        /// <summary>
        /// Load a SlugFont from TTF byte data and upload curve/band textures to the GPU.
        /// </summary>
        public static unsafe SlugFont LoadFromMemory(GraphicsDevice device, byte[] ttfData)
        {
            var font = new SlugFont();

            // ── Parse TTF ──
            StbTrueType.stbtt_fontinfo fontInfo;
            fixed (byte* ptr = ttfData)
            {
                fontInfo = new StbTrueType.stbtt_fontinfo();
                if (StbTrueType.stbtt_InitFont(fontInfo, ptr, 0) == 0)
                    throw new InvalidOperationException("Failed to initialize TTF font.");

                int ascent, descent, lineGap;
                StbTrueType.stbtt_GetFontVMetrics(fontInfo, &ascent, &descent, &lineGap);

                float scale = StbTrueType.stbtt_ScaleForMappingEmToPixels(fontInfo, 1.0f);
                int unitsPerEm = scale > 0f ? (int)MathF.Round(1.0f / scale) : 1000;

                font.UnitsPerEm = unitsPerEm;
                font.Ascent = ascent;
                font.Descent = descent;
                font.LineGap = lineGap;

                // ── Process glyphs ──
                var curveTexData = new List<float>(TextureWidth * 4);
                var bandHeaderCurveCount = new List<ushort>();
                var bandHeaderOffset = new List<int>();
                var bandCurveLocs = new List<BandTexelCoord>();
                var glyphBandRanges = new List<GlyphBandRange>();
                var glyphs = new List<(int codePoint, SlugGlyph glyph)>();
                var scratchCurves = new List<Curve>();

                // Process ASCII + common Latin characters
                for (int cp = 32; cp < 127; cp++)
                    ProcessCodePoint(fontInfo, cp, curveTexData, bandHeaderCurveCount,
                        bandHeaderOffset, bandCurveLocs, glyphBandRanges, glyphs, scratchCurves);

                // Extended Latin for European languages
                for (int cp = 160; cp < 384; cp++)
                    ProcessCodePoint(fontInfo, cp, curveTexData, bandHeaderCurveCount,
                        bandHeaderOffset, bandCurveLocs, glyphBandRanges, glyphs, scratchCurves);

                // ── Build textures ──
                var (curveData, curveW, curveH) = FinalizeCurveTexture(curveTexData);
                var (bandData, bandW, bandH) = FinalizeBandTexture(
                    bandHeaderCurveCount, bandHeaderOffset, bandCurveLocs, glyphBandRanges);

                // ── Upload to GPU ──
                font.UploadTextures(device, curveData, curveW, curveH, bandData, bandW, bandH);

                // ── Store glyph data ──
                var codepoints = new List<int>();
                foreach (var (cp, g) in glyphs)
                {
                    font._glyphs[cp] = g;
                    codepoints.Add(cp);
                }

                // ── Build kerning pairs ──
                // stbtt only reads classic 'kern' table; Roboto and most modern fonts use GPOS.
                // First try stbtt (classic kern):
                int classicKernCount = 0;
                foreach (int left in codepoints)
                {
                    foreach (int right in codepoints)
                    {
                        int kern = StbTrueType.stbtt_GetCodepointKernAdvance(fontInfo, left, right);
                        if (kern != 0)
                        {
                            font._kernPairs[(left << 16) | right] = kern;
                            classicKernCount++;
                        }
                    }
                }

                // If classic kern is empty, parse GPOS table
                if (classicKernCount == 0)
                {
                    int gposKernCount = ParseGposKerning(ttfData, fontInfo, font._kernPairs, codepoints);
                    Debug.Log("SlugFont", $"Kerning: classic=0, GPOS={gposKernCount} pairs");
                }
                else
                {
                    Debug.Log("SlugFont", $"Kerning: classic={classicKernCount} pairs");
                }

                // Log sample kern pairs for diagnostics
                string[] samplePairs = { "To", "AV", "Pr", "oj", "ec", "Se", "le" };
                foreach (var pair in samplePairs)
                {
                    if (pair.Length == 2)
                    {
                        int k = font.GetKerning(pair[0], pair[1]);
                        if (k != 0) Debug.Log("SlugFont", $"Kern '{pair[0]}'+'{pair[1]}' = {k} units ({k * 16f / font.UnitsPerEm:F2}px @16px)");
                    }
                }

                // Diagnostic: log stats for the font
                Debug.Log("SlugFont", $"Loaded: unitsPerEm={font.UnitsPerEm}, ascent={font.Ascent}, descent={font.Descent}, glyphs={glyphs.Count}");
                Debug.Log("SlugFont", $"CurveTexture: {curveW}x{curveH} ({curveTexData.Count / 4} texels, {curveTexData.Count / 8} curves)");
                Debug.Log("SlugFont", $"BandTexture: {bandW}x{bandH} ({bandHeaderCurveCount.Count} headers, {bandCurveLocs.Count} curveLocs)");

                // Log a few sample glyphs
                foreach (var (cp, g) in glyphs)
                {
                    if (cp == 'A' || cp == 'e' || cp == 'l')
                    {
                        Debug.Log("SlugFont", $"Glyph '{(char)cp}': bbox=({g.BBoxX1},{g.BBoxY1})-({g.BBoxX2},{g.BBoxY2}) bands={g.BandCount} dim=({g.BandDimX},{g.BandDimY}) bandTex=({g.BandTexCoordX},{g.BandTexCoordY}) adv={g.AdvanceWidth}");
                    }
                }

                // Detailed dump for glyph 'l' to trace shader
                foreach (var (cp, g) in glyphs)
                {
                    if (cp != 'l') continue;
                    int totalH = bandHeaderCurveCount.Count;
                    int hdrStart = g.BandTexCoordX + g.BandTexCoordY * TextureWidth;

                    Debug.Log("SlugFont", $"=== Glyph 'l' band dump: totalHeaders={totalH}, headerStart={hdrStart} ===");

                    // Dump first 3 horizontal bands and first 3 vertical bands
                    for (int b = 0; b < Math.Min(3, g.BandCount); b++)
                    {
                        int hIdx = hdrStart + b;
                        int count = bandHeaderCurveCount[hIdx];
                        int offset = bandHeaderOffset[hIdx]; // already relative
                        Debug.Log("SlugFont", $"  HBand[{b}]: count={count}, relOffset={offset}");

                        // Show curve locations
                        int absTexel = hdrStart + offset;
                        for (int c = 0; c < Math.Min(count, 5); c++)
                        {
                            int lIdx = absTexel - totalH + c; // index into bandCurveLocs (after subtracting headers)
                            if (lIdx >= 0 && lIdx < bandCurveLocs.Count)
                            {
                                var loc = bandCurveLocs[lIdx];
                                // Read curve data from curveTexData at this location
                                int curTexIdx = loc.Y * TextureWidth + loc.X;
                                int fIdx = curTexIdx * 4;
                                if (fIdx + 7 < curveTexData.Count)
                                {
                                    float sx = curveTexData[fIdx], sy = curveTexData[fIdx+1];
                                    float cx = curveTexData[fIdx+2], cy = curveTexData[fIdx+3];
                                    float ex = curveTexData[fIdx+4], ey = curveTexData[fIdx+5];
                                    Debug.Log("SlugFont", $"    Curve[{c}]: loc=({loc.X},{loc.Y}) start=({sx},{sy}) ctrl=({cx},{cy}) end=({ex},{ey})");
                                }
                            }
                        }
                    }

                    // Vertical bands
                    for (int b = 0; b < Math.Min(3, g.BandCount); b++)
                    {
                        int vIdx = hdrStart + g.BandCount + b;
                        int count = bandHeaderCurveCount[vIdx];
                        int offset = bandHeaderOffset[vIdx];
                        Debug.Log("SlugFont", $"  VBand[{b}]: count={count}, relOffset={offset}");
                    }
                    break;
                }
            }

            return font;
        }

        // ─── Glyph Processing (ported from Forme's FontProcessor) ───

        private static unsafe void ProcessCodePoint(
            StbTrueType.stbtt_fontinfo fontInfo, int codePoint,
            List<float> curveTexData,
            List<ushort> bandHeaderCurveCount, List<int> bandHeaderOffset,
            List<BandTexelCoord> bandCurveLocs, List<GlyphBandRange> glyphBandRanges,
            List<(int, SlugGlyph)> glyphs, List<Curve> scratchCurves)
        {
            int glyphIdx = StbTrueType.stbtt_FindGlyphIndex(fontInfo, codePoint);
            if (glyphIdx == 0) return;

            int advanceWidth, lsb;
            StbTrueType.stbtt_GetGlyphHMetrics(fontInfo, glyphIdx, &advanceWidth, &lsb);

            StbTrueType.stbtt_vertex* verts;
            int vertCount = StbTrueType.stbtt_GetGlyphShape(fontInfo, glyphIdx, &verts);

            if (vertCount == 0)
            {
                // No outline (e.g. space) — still record advance width
                glyphs.Add((codePoint, new SlugGlyph
                {
                    AdvanceWidth = advanceWidth,
                    LeftSideBearing = lsb,
                    BandCount = 0, BandDimX = 1, BandDimY = 1
                }));
                return;
            }

            // Skip glyphs with cubic curves (not supported by Slug)
            for (int v = 0; v < vertCount; v++)
            {
                if (verts[v].type == StbTrueType.STBTT_vcubic)
                {
                    StbTrueType.stbtt_FreeShape(fontInfo, verts);
                    return;
                }
            }

            int bx1, by1, bx2, by2;
            StbTrueType.stbtt_GetGlyphBox(fontInfo, glyphIdx, &bx1, &by1, &bx2, &by2);

            // Extract quadratic Bézier curves from glyph outline
            scratchCurves.Clear();
            float curX = 0f, curY = 0f;

            for (int v = 0; v < vertCount; v++)
            {
                ref StbTrueType.stbtt_vertex vert = ref verts[v];

                switch ((int)vert.type)
                {
                    case StbTrueType.STBTT_vmove:
                        curX = vert.x;
                        curY = vert.y;
                        break;

                    case StbTrueType.STBTT_vline:
                    {
                        float nx = vert.x, ny = vert.y;
                        scratchCurves.Add(new Curve
                        {
                            Start = new Vector2(curX, curY),
                            Control = new Vector2((curX + nx) * 0.5f, (curY + ny) * 0.5f),
                            End = new Vector2(nx, ny)
                        });
                        curX = nx; curY = ny;
                        break;
                    }

                    case StbTrueType.STBTT_vcurve:
                    {
                        float nx = vert.x, ny = vert.y;
                        scratchCurves.Add(new Curve
                        {
                            Start = new Vector2(curX, curY),
                            Control = new Vector2(vert.cx, vert.cy),
                            End = new Vector2(nx, ny)
                        });
                        curX = nx; curY = ny;
                        break;
                    }
                }
            }

            StbTrueType.stbtt_FreeShape(fontInfo, verts);

            if (scratchCurves.Count == 0) return;

            // Fix degenerate control points
            for (int i = 0; i < scratchCurves.Count; i++)
            {
                var c = scratchCurves[i];
                if (c.Control == c.Start || c.Control == c.End)
                {
                    c.Control = (c.Start + c.End) * 0.5f;
                    scratchCurves[i] = c;
                }
            }

            // Append curves to curve texture
            int bandHeaderStart = bandHeaderCurveCount.Count;
            int bandCurveStart = bandCurveLocs.Count;
            int bandsTexelIndex = bandHeaderStart;

            AppendCurveTexture(scratchCurves, curveTexData);

            // Calculate band count
            int sizeX = bx2 - bx1 + 1;
            int sizeY = by2 - by1 + 1;
            int bandCount = Math.Max(1, Math.Min(16, Math.Min(sizeX, sizeY) / 2));

            AppendBandData(scratchCurves, bandCount, sizeX, sizeY, bx1, by1,
                bandHeaderCurveCount, bandHeaderOffset, bandCurveLocs);

            int bandHeaderCount = bandHeaderCurveCount.Count - bandHeaderStart;
            glyphBandRanges.Add(new GlyphBandRange
            {
                HeaderStart = bandHeaderStart,
                CurveStart = bandCurveStart,
                HeaderCount = bandHeaderCount
            });

            glyphs.Add((codePoint, new SlugGlyph
            {
                BBoxX1 = bx1, BBoxY1 = by1, BBoxX2 = bx2, BBoxY2 = by2,
                AdvanceWidth = advanceWidth,
                LeftSideBearing = lsb,
                BandCount = bandCount,
                BandDimX = (sizeX + bandCount - 1) / bandCount,
                BandDimY = (sizeY + bandCount - 1) / bandCount,
                BandTexCoordX = bandsTexelIndex % TextureWidth,
                BandTexCoordY = bandsTexelIndex / TextureWidth
            }));
        }

        private static void AppendCurveTexture(List<Curve> curves, List<float> curveTexData)
        {
            for (int i = 0; i < curves.Count; i++)
            {
                var c = curves[i];
                int nextTexel = curveTexData.Count / 4;

                // Ensure both texels of a curve fit in the same row
                if (nextTexel % TextureWidth == TextureWidth - 1)
                {
                    curveTexData.Add(0f); curveTexData.Add(0f);
                    curveTexData.Add(0f); curveTexData.Add(0f);
                    nextTexel++;
                }

                c.TexelIndex = nextTexel;
                curves[i] = c;

                // Texel 0: (P1.x, P1.y, P2.x, P2.y)
                curveTexData.Add(c.Start.X);    curveTexData.Add(c.Start.Y);
                curveTexData.Add(c.Control.X);  curveTexData.Add(c.Control.Y);

                // Texel 1: (P3.x, P3.y, 0, 0)
                curveTexData.Add(c.End.X);      curveTexData.Add(c.End.Y);
                curveTexData.Add(0f);           curveTexData.Add(0f);
            }
        }

        private static void AppendBandData(
            List<Curve> curves, int bandCount, int sizeX, int sizeY, int originX, int originY,
            List<ushort> bandHeaderCurveCount, List<int> bandHeaderOffset,
            List<BandTexelCoord> bandCurveLocs)
        {
            var localLocs = new List<BandTexelCoord>(curves.Count * 2);
            int bandDimY = (sizeY + bandCount - 1) / bandCount;
            int bandDimX = (sizeX + bandCount - 1) / bandCount;

            // ── Horizontal bands ──
            // Sort by max X descending for early shader exit
            curves.Sort((a, b) =>
            {
                float maxA = Math.Max(Math.Max(a.Start.X, a.Control.X), a.End.X);
                float maxB = Math.Max(Math.Max(b.Start.X, b.Control.X), b.End.X);
                return maxB.CompareTo(maxA);
            });

            for (int band = 0; band < bandCount; band++)
            {
                float minY = originY + band * bandDimY;
                float maxY = minY + bandDimY;

                int localOffset = localLocs.Count;
                ushort count = 0;

                foreach (var c in curves)
                {
                    // Skip horizontal lines
                    if (c.Start.Y == c.Control.Y && c.Control.Y == c.End.Y) continue;

                    float cMinY = Math.Min(Math.Min(c.Start.Y, c.Control.Y), c.End.Y);
                    float cMaxY = Math.Max(Math.Max(c.Start.Y, c.Control.Y), c.End.Y);
                    if (cMinY > maxY || cMaxY < minY) continue;

                    localLocs.Add(new BandTexelCoord(
                        (ushort)(c.TexelIndex % TextureWidth),
                        (ushort)(c.TexelIndex / TextureWidth)));
                    count++;
                }

                bandHeaderCurveCount.Add(count);
                bandHeaderOffset.Add(localOffset);
            }

            // ── Vertical bands ──
            // Sort by max Y descending for early shader exit
            curves.Sort((a, b) =>
            {
                float maxA = Math.Max(Math.Max(a.Start.Y, a.Control.Y), a.End.Y);
                float maxB = Math.Max(Math.Max(b.Start.Y, b.Control.Y), b.End.Y);
                return maxB.CompareTo(maxA);
            });

            for (int band = 0; band < bandCount; band++)
            {
                float minX = originX + band * bandDimX;
                float maxX = minX + bandDimX;

                int localOffset = localLocs.Count;
                ushort count = 0;

                foreach (var c in curves)
                {
                    // Skip vertical lines
                    if (c.Start.X == c.Control.X && c.Control.X == c.End.X) continue;

                    float cMinX = Math.Min(Math.Min(c.Start.X, c.Control.X), c.End.X);
                    float cMaxX = Math.Max(Math.Max(c.Start.X, c.Control.X), c.End.X);
                    if (cMinX > maxX || cMaxX < minX) continue;

                    localLocs.Add(new BandTexelCoord(
                        (ushort)(c.TexelIndex % TextureWidth),
                        (ushort)(c.TexelIndex / TextureWidth)));
                    count++;
                }

                bandHeaderCurveCount.Add(count);
                bandHeaderOffset.Add(localOffset);
            }

            foreach (var loc in localLocs)
                bandCurveLocs.Add(loc);
        }

        private static (float[] data, int width, int height) FinalizeCurveTexture(List<float> curveTexData)
        {
            int totalTexels = (curveTexData.Count + 3) / 4;
            int width = TextureWidth;
            int height = Math.Max(1, (totalTexels + TextureWidth - 1) / TextureWidth);

            var data = new float[width * height * 4];
            curveTexData.CopyTo(data);
            return (data, width, height);
        }

        private static (float[] data, int width, int height) FinalizeBandTexture(
            List<ushort> headerCurveCount, List<int> headerOffset,
            List<BandTexelCoord> curveLocs, List<GlyphBandRange> glyphBandRanges)
        {
            int totalHeaders = headerCurveCount.Count;

            // Fix up offsets to be ABSOLUTE texel indices into the band texture.
            // Matching Forme's approach: offset points directly to the curve location data.
            for (int gi = 0; gi < glyphBandRanges.Count; gi++)
            {
                var range = glyphBandRanges[gi];
                for (int hi = range.HeaderStart; hi < range.HeaderStart + range.HeaderCount; hi++)
                {
                    // absolute texel = totalHeaders + glyphCurveStart + localOffset
                    headerOffset[hi] = headerOffset[hi] + totalHeaders + range.CurveStart;
                }
            }

            int totalTexels = totalHeaders + curveLocs.Count;
            int width = TextureWidth;
            int height = Math.Max(1, (totalTexels + TextureWidth - 1) / TextureWidth);

            var data = new float[width * height * 2];

            for (int i = 0; i < totalHeaders; i++)
            {
                data[i * 2 + 0] = headerCurveCount[i];
                data[i * 2 + 1] = headerOffset[i];
            }

            for (int i = 0; i < curveLocs.Count; i++)
            {
                data[(totalHeaders + i) * 2 + 0] = curveLocs[i].X;
                data[(totalHeaders + i) * 2 + 1] = curveLocs[i].Y;
            }

            return (data, width, height);
        }

        // ─── GPU Upload ───

        private unsafe void UploadTextures(GraphicsDevice device,
            float[] curveData, int curveW, int curveH,
            float[] bandData, int bandW, int bandH)
        {
            // ── Curve texture: RGBA16F (4 channels of float16 per texel) ──
            _curveTexture = device.CreateTexture2D(Format.R16G16B16A16_Float, curveW, curveH);
            _curveBindlessIndex = device.AllocateBindlessIndex();

            // Convert float32 → float16 and upload
            var curveF16 = new ushort[curveW * curveH * 4];
            for (int i = 0; i < curveData.Length && i < curveF16.Length; i++)
                curveF16[i] = HalfToUShort(curveData[i]);

            UploadTextureData(device, _curveTexture, curveW, curveH,
                Format.R16G16B16A16_Float, curveF16, 4 * sizeof(ushort));

            device.NativeDevice.CreateShaderResourceView(_curveTexture, null,
                device.GetCpuHandle(_curveBindlessIndex));

            // ── Band texture: R32G32_Float (matching Forme's approach) ──
            // Store as float pairs: each texel = (curveCount_or_curveX, offset_or_curveY)
            _bandTexture = device.CreateTexture2D(Format.R32G32_Float, bandW, bandH);
            _bandBindlessIndex = device.AllocateBindlessIndex();

            // Data is already float — upload directly
            UploadTextureData(device, _bandTexture, bandW, bandH,
                Format.R32G32_Float, bandData, 2 * sizeof(float));

            // Create SRV matching the float format
            var bandSrvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R32G32_Float,
                ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MipLevels = 1 }
            };
            device.NativeDevice.CreateShaderResourceView(_bandTexture, bandSrvDesc,
                device.GetCpuHandle(_bandBindlessIndex));
        }

        private static unsafe void UploadTextureData<T>(GraphicsDevice device,
            ID3D12Resource texture, int width, int height,
            Format format, T[] data, int bytesPerPixel) where T : unmanaged
        {
            int rowPitch = AlignTo256(width * bytesPerPixel);
            int uploadSize = rowPitch * height;
            var uploadBuffer = device.CreateUploadBuffer(uploadSize);

            void* pData;
            uploadBuffer.Map(0, null, &pData);

            int srcRowPitch = width * bytesPerPixel;
            fixed (T* srcPtr = data)
            {
                byte* src = (byte*)srcPtr;
                byte* dst = (byte*)pData;
                for (int row = 0; row < height; row++)
                {
                    Buffer.MemoryCopy(src + row * srcRowPitch, dst + row * rowPitch,
                        rowPitch, srcRowPitch);
                }
            }
            uploadBuffer.Unmap(0);

            // Copy to texture
            var copyCmd = new CommandList(device);
            copyCmd.Reset();

            copyCmd.Native.ResourceBarrierTransition(texture,
                ResourceStates.Common, ResourceStates.CopyDest);

            var footprint = new PlacedSubresourceFootPrint
            {
                Offset = 0,
                Footprint = new SubresourceFootPrint(format, (uint)width, (uint)height, 1, (uint)rowPitch)
            };
            var dstLoc = new TextureCopyLocation(texture, 0);
            var srcLoc = new TextureCopyLocation(uploadBuffer, footprint);
            copyCmd.Native.CopyTextureRegion(dstLoc, 0, 0, 0, srcLoc);

            copyCmd.Native.ResourceBarrierTransition(texture,
                ResourceStates.CopyDest, ResourceStates.AllShaderResource);

            copyCmd.Close();
            device.SubmitAndWait(copyCmd.Native);
            copyCmd.Dispose();
            uploadBuffer.Dispose();
        }

        private static int AlignTo256(int value) => (value + 255) & ~255;

        private static ushort HalfToUShort(float value)
        {
            var h = (Half)value;
            return BitConverter.ToUInt16(BitConverter.GetBytes(h));
        }

        // ─── Text Layout and Measurement ───


        /// <summary>
        /// Measure text size in pixels at the given font size.
        /// </summary>
        public Point MeasureText(string text, float sizePixels)
        {
            if (string.IsNullOrEmpty(text)) return Point.Empty;

            float scale = sizePixels / Math.Max(1, UnitsPerEm);
            float totalWidth = 0;
            int lineCount = 1;

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                if (c == '\n')
                {
                    lineCount++;
                    continue;
                }

                // Apply kerning with previous character
                if (i > 0 && text[i - 1] != '\n')
                    totalWidth += GetKerning(text[i - 1], c) * scale;

                if (_glyphs.TryGetValue(c, out var glyph))
                    totalWidth += MathF.Ceiling(glyph.AdvanceWidth * scale);
            }

            int lineHeight = (int)MathF.Ceiling((Ascent - Descent + LineGap) * scale);
            return new Point((int)MathF.Ceiling(totalWidth), lineHeight * lineCount);
        }

        // ─── GPOS Table Parsing for Modern OpenType Kerning ───

        /// <summary>
        /// Parse GPOS table for pair kerning (modern OpenType fonts like Roboto).
        /// Returns number of kern pairs found.
        /// </summary>
        private static unsafe int ParseGposKerning(byte[] ttfData, StbTrueType.stbtt_fontinfo fontInfo,
            Dictionary<int, int> kernPairs, List<int> codepoints)
        {
            // Build glyph-to-codepoint mapping
            var glyphToCodepoints = new Dictionary<int, List<int>>();
            fixed (byte* ptr = ttfData)
            {
                foreach (int cp in codepoints)
                {
                    int glyphId = StbTrueType.stbtt_FindGlyphIndex(fontInfo, cp);
                    if (glyphId > 0)
                    {
                        if (!glyphToCodepoints.TryGetValue(glyphId, out var list))
                        {
                            list = new List<int>();
                            glyphToCodepoints[glyphId] = list;
                        }
                        list.Add(cp);
                    }
                }
            }

            // Find GPOS table
            int gposOffset = FindTable(ttfData, "GPOS");
            if (gposOffset < 0) return 0;

            int count = 0;
            var r = new BinaryReader2(ttfData, gposOffset);

            ushort majorVersion = r.U16(0);
            ushort minorVersion = r.U16(2);
            ushort scriptListOffset = r.U16(4);
            ushort featureListOffset = r.U16(6);
            ushort lookupListOffset = r.U16(8);

            // Read lookup list
            int llBase = gposOffset + lookupListOffset;
            var ll = new BinaryReader2(ttfData, llBase);
            ushort lookupCount = ll.U16(0);

            for (int li = 0; li < lookupCount; li++)
            {
                ushort lookupOffset = ll.U16(2 + li * 2);
                int lBase = llBase + lookupOffset;
                var lk = new BinaryReader2(ttfData, lBase);

                ushort lookupType = lk.U16(0);
                if (lookupType != 2) continue; // Only PairPos (type 2)

                ushort subTableCount = lk.U16(4);
                for (int si = 0; si < subTableCount; si++)
                {
                    ushort subOffset = lk.U16(6 + si * 2);
                    int sBase = lBase + subOffset;
                    var st = new BinaryReader2(ttfData, sBase);

                    ushort posFormat = st.U16(0);
                    ushort coverageOffset = st.U16(2);
                    ushort valueFormat1 = st.U16(4);
                    ushort valueFormat2 = st.U16(6);

                    // We only care about XAdvance in valueFormat1 (bit 2 = 0x0004)
                    int vSize1 = ValueRecordSize(valueFormat1);
                    int vSize2 = ValueRecordSize(valueFormat2);

                    if (posFormat == 1)
                    {
                        // Format 1: individual pairs
                        ushort pairSetCount = st.U16(8);
                        var coverage = ReadCoverage(ttfData, sBase + coverageOffset);

                        for (int pi = 0; pi < pairSetCount; pi++)
                        {
                            ushort pairSetOffset = st.U16(10 + pi * 2);
                            int psBase = sBase + pairSetOffset;
                            var ps = new BinaryReader2(ttfData, psBase);
                            ushort pvCount = ps.U16(0);

                            int covGlyph = coverage.Count > pi ? coverage[pi] : -1;
                            if (covGlyph < 0 || !glyphToCodepoints.ContainsKey(covGlyph)) continue;

                            int pairRecordSize = 2 + vSize1 + vSize2;
                            for (int pvi = 0; pvi < pvCount; pvi++)
                            {
                                int recOff = 2 + pvi * pairRecordSize;
                                ushort secondGlyph = ps.U16(recOff);
                                short xAdv = GetXAdvanceFromValueRecord(ttfData, psBase + recOff + 2, valueFormat1);

                                if (xAdv != 0 && glyphToCodepoints.ContainsKey(secondGlyph))
                                {
                                    foreach (int lcp in glyphToCodepoints[covGlyph])
                                        foreach (int rcp in glyphToCodepoints[secondGlyph])
                                        {
                                            kernPairs[(lcp << 16) | rcp] = xAdv;
                                            count++;
                                        }
                                }
                            }
                        }
                    }
                    else if (posFormat == 2)
                    {
                        // Format 2: class-based pairs
                        ushort classDef1Offset = st.U16(8);
                        ushort classDef2Offset = st.U16(10);
                        ushort class1Count = st.U16(12);
                        ushort class2Count = st.U16(14);

                        var classDef1 = ReadClassDef(ttfData, sBase + classDef1Offset);
                        var classDef2 = ReadClassDef(ttfData, sBase + classDef2Offset);

                        int recordSize = vSize1 + vSize2;
                        int arrayBase = 16;

                        for (int c1 = 0; c1 < class1Count; c1++)
                        {
                            for (int c2 = 0; c2 < class2Count; c2++)
                            {
                                int recOff = arrayBase + (c1 * class2Count + c2) * recordSize;
                                short xAdv = GetXAdvanceFromValueRecord(ttfData, sBase + recOff, valueFormat1);
                                if (xAdv == 0) continue;

                                // Find glyphs in these classes
                                foreach (var kvp1 in classDef1)
                                {
                                    if (kvp1.Value != c1) continue;
                                    if (!glyphToCodepoints.ContainsKey(kvp1.Key)) continue;

                                    foreach (var kvp2 in classDef2)
                                    {
                                        if (kvp2.Value != c2) continue;
                                        if (!glyphToCodepoints.ContainsKey(kvp2.Key)) continue;

                                        foreach (int lcp in glyphToCodepoints[kvp1.Key])
                                            foreach (int rcp in glyphToCodepoints[kvp2.Key])
                                            {
                                                kernPairs[(lcp << 16) | rcp] = xAdv;
                                                count++;
                                            }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return count;
        }

        private static int ValueRecordSize(ushort format)
        {
            int size = 0;
            for (int i = 0; i < 8; i++)
                if ((format & (1 << i)) != 0) size += 2;
            return size;
        }

        private static short GetXAdvanceFromValueRecord(byte[] data, int offset, ushort format)
        {
            // XAdvance is the 3rd field (bit 2 = 0x0004)
            // Skip XPlacement (bit 0) and YPlacement (bit 1) if present
            int off = offset;
            if ((format & 0x0001) != 0) off += 2; // XPlacement
            if ((format & 0x0002) != 0) off += 2; // YPlacement
            if ((format & 0x0004) != 0 && off + 1 < data.Length) // XAdvance
                return (short)((data[off] << 8) | data[off + 1]);
            return 0;
        }

        private static int FindTable(byte[] data, string tag)
        {
            if (data.Length < 12) return -1;
            ushort numTables = (ushort)((data[4] << 8) | data[5]);
            for (int i = 0; i < numTables; i++)
            {
                int rec = 12 + i * 16;
                if (rec + 16 > data.Length) break;
                if (data[rec] == tag[0] && data[rec + 1] == tag[1] &&
                    data[rec + 2] == tag[2] && data[rec + 3] == tag[3])
                {
                    return (data[rec + 8] << 24) | (data[rec + 9] << 16) |
                           (data[rec + 10] << 8) | data[rec + 11];
                }
            }
            return -1;
        }

        private static List<int> ReadCoverage(byte[] data, int offset)
        {
            var result = new List<int>();
            var r = new BinaryReader2(data, offset);
            ushort format = r.U16(0);

            if (format == 1)
            {
                ushort glyphCount = r.U16(2);
                for (int i = 0; i < glyphCount; i++)
                    result.Add(r.U16(4 + i * 2));
            }
            else if (format == 2)
            {
                ushort rangeCount = r.U16(2);
                for (int i = 0; i < rangeCount; i++)
                {
                    ushort startGlyph = r.U16(4 + i * 6);
                    ushort endGlyph = r.U16(6 + i * 6);
                    for (int g = startGlyph; g <= endGlyph; g++)
                        result.Add(g);
                }
            }
            return result;
        }

        private static Dictionary<int, int> ReadClassDef(byte[] data, int offset)
        {
            var result = new Dictionary<int, int>();
            var r = new BinaryReader2(data, offset);
            ushort format = r.U16(0);

            if (format == 1)
            {
                ushort startGlyph = r.U16(2);
                ushort glyphCount = r.U16(4);
                for (int i = 0; i < glyphCount; i++)
                    result[startGlyph + i] = r.U16(6 + i * 2);
            }
            else if (format == 2)
            {
                ushort rangeCount = r.U16(2);
                for (int i = 0; i < rangeCount; i++)
                {
                    ushort startGlyph = r.U16(4 + i * 6);
                    ushort endGlyph = r.U16(6 + i * 6);
                    ushort classValue = r.U16(8 + i * 6);
                    for (int g = startGlyph; g <= endGlyph; g++)
                        result[g] = classValue;
                }
            }
            return result;
        }

        /// <summary>
        /// Lightweight big-endian reader for TTF binary data.
        /// </summary>
        private readonly struct BinaryReader2
        {
            private readonly byte[] _data;
            private readonly int _base;

            public BinaryReader2(byte[] data, int baseOffset) { _data = data; _base = baseOffset; }

            public ushort U16(int offset)
            {
                int i = _base + offset;
                return (ushort)((_data[i] << 8) | _data[i + 1]);
            }

            public short S16(int offset)
            {
                int i = _base + offset;
                return (short)((_data[i] << 8) | _data[i + 1]);
            }
        }

        public void Dispose()
        {
            _curveTexture?.Dispose();
            _bandTexture?.Dispose();
        }
    }
}
