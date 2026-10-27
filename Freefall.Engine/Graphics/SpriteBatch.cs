using System;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// DX12 bindless sprite batch — no geometry shader, no atlas, no vertex buffer.
    /// Each sprite is written to a StructuredBuffer and carries its own texture heap index.
    /// One draw call renders ALL sprites regardless of texture count.
    /// One Begin/End per frame: End writes the frame's sprites to the start of the frame's buffer, so a
    /// second End in the same frame overwrites the first one's before the GPU draws them (draw everything
    /// over a picture in one batch, as one Squid desktop does).
    /// </summary>
    public class SpriteBatch : IDisposable
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct SpriteData
        {
            public Vector2 Position;     // pixel coords (top-left)
            public Vector2 Size;         // pixel size (width, height)
            public Color4 Color;         // tint RGBA
            public Vector4 UVs;          // minU, minV, maxU, maxV
            public uint TextureIndex;    // SRV heap index (bindless) — for Type=2: index into SlugGlyphData buffer
            public uint Type;            // 0 = Quad, 1 = Line, 2 = SlugGlyph
            public Vector2 EndPosition;  // For lines: pixel coords of end point
        }

        private const int InitialCapacity = 2048;
        private const int SpriteStride = 64;     // sizeof(SpriteData) = 64 bytes
        private const int GlyphStride = 48;     // sizeof(SlugGlyphData) = 48 bytes
        private const int InitialGlyphCapacity = 512;

        private readonly GraphicsDevice _device;
        private ID3D12PipelineState _pipelineState = null!;
        private uint _whiteTextureIndex;

        // CPU-side sprite array
        private SpriteData[] _sprites;
        private int _spriteCount;

        // Slug glyph indirection data
        private SlugGlyphData[] _glyphData;
        private int _glyphCount;

        // GPU upload buffers (mapped persistently), one pair per frame in flight: the GPU may still draw the frame
        // before from its buffers while this frame's sprites are written (one shared buffer showed later frames'
        // sprites in earlier ones). A buffer grows when its frame comes round again: RenderView.Prepare has waited
        // for the GPU to be done with that frame by then.
        private const int FramesInFlight = 3;   // RenderView.FrameCount
        private sealed class UploadBuffer
        {
            public ID3D12Resource Resource;
            public IntPtr Pointer;
            public uint BindlessIndex;
            public int Capacity;
        }
        private readonly UploadBuffer[] _spriteBuffers = new UploadBuffer[FramesInFlight];
        private readonly UploadBuffer[] _glyphBuffers = new UploadBuffer[FramesInFlight];

        // Scissor rect (pixel coords) — initialized to max so soft clipping defaults to fully open
        public RectI Scissor = new RectI(0, 0, int.MaxValue, int.MaxValue);

        // Line width in pixels
        public float LineWidth = 1;

        public SpriteBatch(GraphicsDevice device)
        {
            _device = device;
            _sprites = new SpriteData[InitialCapacity];
            _glyphData = new SlugGlyphData[InitialGlyphCapacity];

            CreatePipelineState();
            for (int i = 0; i < FramesInFlight; i++)
            {
                _spriteBuffers[i] = new UploadBuffer();
                _glyphBuffers[i] = new UploadBuffer();
                Ensure(_spriteBuffers[i], InitialCapacity, SpriteStride);
                Ensure(_glyphBuffers[i], InitialGlyphCapacity, GlyphStride);
            }
            CreateWhiteTexture();
        }

        private void CreatePipelineState()
        {
            string shaderPath = Path.Combine(Engine.RootDirectory, "Resources", "Shaders", "spritebatch.hlsl");
            string source = Freefall.Assets.EngineFiles.ReadAllText(shaderPath);

            var vs = new Shader(source, "VS", "vs_6_6");
            var ps = new Shader(source, "PS", "ps_6_6");

            var psoDesc = new GraphicsPipelineStateDescription
            {
                RootSignature = _device.GlobalRootSignature,
                VertexShader = vs.Bytecode,
                PixelShader = ps.Bytecode,
                InputLayout = default, // No vertex buffer
                RasterizerState = RasterizerDescription.CullNone,
                BlendState = CreateSpriteBlendState(),
                DepthStencilState = DepthStencilDescription.None,
                PrimitiveTopologyType = PrimitiveTopologyType.Triangle,
                RenderTargetFormats = new[] { Format.R8G8B8A8_UNorm },
                DepthStencilFormat = Format.Unknown,
                SampleDescription = new SampleDescription(1, 0),
                SampleMask = uint.MaxValue
            };

            FrameTrace.Pipeline("graphics", "SpriteBatch");
            _pipelineState = _device.NativeDevice.CreateGraphicsPipelineState(psoDesc);

            vs.Dispose();
            ps.Dispose();
        }

        private static BlendDescription CreateSpriteBlendState()
        {
            var desc = new BlendDescription
            {
                AlphaToCoverageEnable = false,
                IndependentBlendEnable = false
            };
            desc.RenderTarget[0] = new RenderTargetBlendDescription
            {
                BlendEnable = true,
                SourceBlend = Blend.SourceAlpha,
                DestinationBlend = Blend.InverseSourceAlpha,
                BlendOperation = BlendOperation.Add,
                SourceBlendAlpha = Blend.One,
                DestinationBlendAlpha = Blend.InverseSourceAlpha,
                BlendOperationAlpha = BlendOperation.Add,
                RenderTargetWriteMask = ColorWriteEnable.All
            };
            return desc;
        }

        /// <summary>A buffer for at least <paramref name="count"/> elements: made, or made anew when too small (its frame's GPU work is done).</summary>
        private void Ensure(UploadBuffer buffer, int count, int stride)
        {
            if (buffer.Resource != null && buffer.Capacity >= count) return;
            buffer.Resource?.Dispose();

            buffer.Resource = _device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)(count * stride)),
                ResourceStates.GenericRead);

            unsafe
            {
                void* ptr;
                buffer.Resource.Map(0, null, &ptr);
                buffer.Pointer = (IntPtr)ptr;
            }

            // Allocate or reuse bindless index
            if (buffer.BindlessIndex == 0)
                buffer.BindlessIndex = _device.AllocateBindlessIndex();
            _device.CreateStructuredBufferSRV(buffer.Resource, (uint)count, (uint)stride, buffer.BindlessIndex);
            buffer.Capacity = count;
        }

        private void CreateWhiteTexture()
        {
            // 1x1 white pixel texture for untextured draws
            var tex = _device.CreateTexture2D(Format.R8G8B8A8_UNorm, 1, 1);

            // Upload the white pixel
            var uploadBuffer = _device.CreateUploadBuffer(4);
            unsafe
            {
                void* ptr;
                uploadBuffer.Map(0, null, &ptr);
                var span = new Span<byte>(ptr, 4);
                span[0] = 255; span[1] = 255; span[2] = 255; span[3] = 255;
                uploadBuffer.Unmap(0);
            }

            // Copy via a temporary command list
            var copyCmd = new CommandList(_device);
            copyCmd.Reset();
            copyCmd.Native.ResourceBarrierTransition(tex, ResourceStates.Common, ResourceStates.CopyDest);
            
            var footprint = new PlacedSubresourceFootPrint
            {
                Offset = 0,
                Footprint = new SubresourceFootPrint(Format.R8G8B8A8_UNorm, 1, 1, 1, 256) // row pitch aligned to 256
            };
            var dst = new TextureCopyLocation(tex, 0);
            var src = new TextureCopyLocation(uploadBuffer, footprint);
            copyCmd.Native.CopyTextureRegion(dst, 0, 0, 0, src);
            
            copyCmd.Native.ResourceBarrierTransition(tex, ResourceStates.CopyDest, ResourceStates.AllShaderResource);
            copyCmd.Close();
            _device.SubmitAndWait(copyCmd.Native);
            copyCmd.Dispose();
            uploadBuffer.Dispose();

            _whiteTextureIndex = _device.AllocateBindlessIndex();
            _device.NativeDevice.CreateShaderResourceView(tex, null, _device.GetCpuHandle(_whiteTextureIndex));
        }

        /// <summary>
        /// White texture bindless index (for untextured draws).
        /// </summary>
        public uint WhiteTextureIndex => _whiteTextureIndex;

        public void Begin()
        {
            _spriteCount = 0;
            _glyphCount = 0;
        }

        /// <summary>
        /// Draw a textured quad. Point-sampled (pixel art drawn 1:1, the editor's UI) unless
        /// <paramref name="smooth"/>: then linear with mips, for art drawn at another size than painted.
        /// </summary>
        public void Draw(int x, int y, int width, int height, RectF rect, Color4 color, uint textureIndex, int texWidth = 0, int texHeight = 0, bool smooth = false)
        {
            // CPU-side scissor clipping (always applied, matching Apex behavior)
            {
                float fx = rect.width / width;
                float fy = rect.height / height;

                int diff;

                // X-axis clipping
                diff = Math.Max(Scissor.Left - x, 0);
                x += diff;
                width -= diff;
                rect.xMin += diff * fx;

                diff = Math.Max(x + width - Scissor.Right, 0);
                width -= diff;
                rect.xMax -= diff * fx;

                // Y-axis clipping
                diff = Math.Max(Scissor.Top - y, 0);
                y += diff;
                height -= diff;
                rect.yMin += diff * fy;

                diff = Math.Max(y + height - Scissor.Bottom, 0);
                height -= diff;
                rect.yMax -= diff * fy;

                if (width < 1 || height < 1) return;
            }

            // Normalize UVs to [0,1] using texture dimensions
            float tw = texWidth > 0 ? texWidth : 1;
            float th = texHeight > 0 ? texHeight : 1;

            EnsureCapacity();

            ref var sprite = ref _sprites[_spriteCount++];
            sprite.Position = new Vector2(x, y);
            sprite.Size = new Vector2(width, height);
            sprite.Color = color;
            sprite.UVs = new Vector4(rect.xMin / tw, rect.yMin / th, rect.xMax / tw, rect.yMax / th);
            sprite.TextureIndex = textureIndex;
            sprite.Type = smooth ? 3u : 0u;
            sprite.EndPosition = default;
        }

        /// <summary>
        /// Draw a colored line.
        /// </summary>
        public void DrawLine(int x1, int y1, int x2, int y2, Color4 color)
        {
            // CPU-side scissor clipping, like the quad path: trim the segment to the scissor rect (Liang–Barsky)
            float ax = x1, ay = y1, bx = x2, by = y2;
            {
                float dx = bx - ax, dy = by - ay;
                float t0 = 0, t1 = 1;

                if (!ClipLine(-dx, ax - Scissor.Left, ref t0, ref t1) ||
                    !ClipLine(dx, Scissor.Right - ax, ref t0, ref t1) ||
                    !ClipLine(-dy, ay - Scissor.Top, ref t0, ref t1) ||
                    !ClipLine(dy, Scissor.Bottom - ay, ref t0, ref t1))
                    return;

                bx = ax + dx * t1; by = ay + dy * t1;
                ax += dx * t0; ay += dy * t0;
            }

            EnsureCapacity();

            ref var sprite = ref _sprites[_spriteCount++];
            sprite.Position = new Vector2(ax, ay);
            sprite.Size = new Vector2(LineWidth, 0);
            sprite.Color = color;
            sprite.UVs = new Vector4(0, 0, 1, 1);
            sprite.TextureIndex = _whiteTextureIndex;
            sprite.Type = 1;
            sprite.EndPosition = new Vector2(bx, by);
        }

        // One Liang–Barsky edge test: p is the direction against the edge, q the distance to it.
        private static bool ClipLine(float p, float q, ref float t0, ref float t1)
        {
            if (p == 0) return q >= 0;

            float r = q / p;
            if (p < 0)
            {
                if (r > t1) return false;
                if (r > t0) t0 = r;
            }
            else
            {
                if (r < t0) return false;
                if (r < t1) t1 = r;
            }
            return true;
        }

        /// <summary>
        /// Draw a Slug glyph (Type=2 sprite with indirection to glyph data).
        /// </summary>
        public void DrawSlugGlyph(int x, int y, int width, int height,
            float emX0, float emY0, float emX1, float emY1,
            Color4 color, SlugGlyphData glyphData)
        {
            // CPU-side scissor clipping (same as quad path)
            {
                float fx = (emX1 - emX0) / width;
                float fy = (emY1 - emY0) / height;

                int diff;

                diff = Math.Max(Scissor.Left - x, 0);
                x += diff; width -= diff; emX0 += diff * fx;

                diff = Math.Max(x + width - Scissor.Right, 0);
                width -= diff; emX1 -= diff * fx;

                diff = Math.Max(Scissor.Top - y, 0);
                y += diff; height -= diff; emY0 += diff * fy;

                diff = Math.Max(y + height - Scissor.Bottom, 0);
                height -= diff; emY1 -= diff * fy;

                if (width < 1 || height < 1) return;
            }

            EnsureGlyphCapacity();
            EnsureCapacity();

            // Write glyph data to the indirection buffer
            int glyphIdx = _glyphCount++;
            _glyphData[glyphIdx] = glyphData;

            // Write a Type=2 sprite that references the glyph data
            ref var sprite = ref _sprites[_spriteCount++];
            sprite.Position = new Vector2(x, y);
            sprite.Size = new Vector2(width, height);
            sprite.Color = color;
            sprite.UVs = new Vector4(emX0, emY0, emX1, emY1);  // em-space bounds (interpolated by rasterizer)
            sprite.TextureIndex = (uint)glyphIdx;               // index into SlugGlyphData buffer
            sprite.Type = 2;
            sprite.EndPosition = default;
        }

        /// <summary>
        /// Flush all sprites to the GPU and draw them. Call after all Draw/DrawLine calls.
        /// </summary>
        public void End(ID3D12GraphicsCommandList commandList, int screenWidth, int screenHeight)
        {
            if (_spriteCount == 0) return;

            // This frame's buffers (the GPU may still be reading the others)
            int frame = (int)((uint)Engine.FrameIndex % FramesInFlight);
            var sprites = _spriteBuffers[frame];
            var glyphs = _glyphBuffers[frame];
            Ensure(sprites, _sprites.Length, SpriteStride);
            Ensure(glyphs, _glyphData.Length, GlyphStride);

            // Upload sprite data to GPU
            unsafe
            {
                fixed (SpriteData* src = _sprites)
                {
                    int bytes = _spriteCount * SpriteStride;
                    Buffer.MemoryCopy(src, (void*)sprites.Pointer, sprites.Capacity * SpriteStride, bytes);
                }
            }

            // Upload glyph data to GPU (if any Type=2 sprites exist)
            if (_glyphCount > 0)
            {
                unsafe
                {
                    fixed (SlugGlyphData* src = _glyphData)
                    {
                        int bytes = _glyphCount * GlyphStride;
                        Buffer.MemoryCopy(src, (void*)glyphs.Pointer, glyphs.Capacity * GlyphStride, bytes);
                    }
                }
            }

            // Set pipeline state
            commandList.SetPipelineState(_pipelineState);
            commandList.SetGraphicsRootSignature(_device.GlobalRootSignature);

            // Push constants: sprite buffer index, screen width, screen height, glyph buffer index
            commandList.SetGraphicsRoot32BitConstant(0, sprites.BindlessIndex, 0);

            unsafe
            {
                float w = screenWidth;
                float h = screenHeight;
                commandList.SetGraphicsRoot32BitConstant(0, *(uint*)&w, 1);
                commandList.SetGraphicsRoot32BitConstant(0, *(uint*)&h, 2);
            }

            commandList.SetGraphicsRoot32BitConstant(0, glyphs.BindlessIndex, 3);

            // Set topology and draw
            commandList.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);
            commandList.DrawInstanced((uint)(_spriteCount * 6), 1, 0, 0);
        }

        private void EnsureCapacity()
        {
            if (_spriteCount >= _sprites.Length)
            {
                // The GPU buffers follow in End, each when its frame comes round
                Array.Resize(ref _sprites, _sprites.Length * 2);
            }
        }

        private void EnsureGlyphCapacity()
        {
            if (_glyphCount >= _glyphData.Length)
            {
                Array.Resize(ref _glyphData, _glyphData.Length * 2);
            }
        }

        public void Dispose()
        {
            _pipelineState?.Dispose();
            foreach (var buffer in _spriteBuffers) buffer?.Resource?.Dispose();
            foreach (var buffer in _glyphBuffers) buffer?.Resource?.Dispose();
        }
    }

    /// <summary>
    /// Rectangle in float coordinates (pixel space, pre-normalization).
    /// </summary>
    public struct RectF
    {
        public float xMin, xMax, yMin, yMax;

        public float width => xMax - xMin;
        public float height => yMax - yMin;

        public RectF(float x, float y, float width, float height)
        {
            xMin = x;
            yMin = y;
            xMax = x + width;
            yMax = y + height;
        }
    }

    /// <summary>
    /// Per-glyph data for Slug text rendering.
    /// Uploaded to a StructuredBuffer and accessed via indirection from Type=2 sprites.
    /// Must match the HLSL SlugGlyphData struct layout exactly.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SlugGlyphData
    {
        public uint CurveTextureIndex;  // Bindless SRV: font's curve texture (RGBA16F)
        public uint BandTextureIndex;   // Bindless SRV: font's band texture (RG32F)
        public int  GlyphLocX;          // Band texture origin X for this glyph
        public int  GlyphLocY;          // Band texture origin Y for this glyph
        public int  BandMaxX;           // Max horizontal band index (bandCount - 1)
        public int  BandMaxY;           // Max vertical band index (bandCount - 1)
        public float BandScaleX;        // Em-to-band-index scale X
        public float BandScaleY;        // Em-to-band-index scale Y
        public float BandOffsetX;       // Em-to-band-index offset X
        public float BandOffsetY;       // Em-to-band-index offset Y
        public float PixelsPerEmX;      // CPU-computed: pixels per em-unit in X
        public float PixelsPerEmY;      // CPU-computed: pixels per em-unit in Y
    }
}
