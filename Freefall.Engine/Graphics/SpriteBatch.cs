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
            public uint TextureIndex;    // SRV heap index (bindless)
            public uint Type;            // 0 = Quad, 1 = Line
            public Vector2 EndPosition;  // For lines: pixel coords of end point
        }

        private const int InitialCapacity = 2048;
        private const int SpriteStride = 64; // sizeof(SpriteData) = 64 bytes

        private readonly GraphicsDevice _device;
        private ID3D12PipelineState _pipelineState = null!;
        private uint _whiteTextureIndex;

        // CPU-side sprite array
        private SpriteData[] _sprites;
        private int _spriteCount;

        // GPU upload buffer (mapped persistently)
        private ID3D12Resource _spriteBuffer = null!;
        private IntPtr _spriteBufferPtr;
        private uint _spriteBufferBindlessIndex;
        private int _bufferCapacity;

        // Scissor rect (pixel coords)
        public RectI Scissor;

        // Line width in pixels
        public float LineWidth = 1;

        public SpriteBatch(GraphicsDevice device)
        {
            _device = device;
            _sprites = new SpriteData[InitialCapacity];
            _bufferCapacity = InitialCapacity;

            CreatePipelineState();
            CreateSpriteBuffer(InitialCapacity);
            CreateWhiteTexture();
        }

        private void CreatePipelineState()
        {
            string shaderPath = Path.Combine(Engine.RootDirectory, "Resources", "Shaders", "spritebatch.hlsl");
            string source = File.ReadAllText(shaderPath);

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
                SourceBlendAlpha = Blend.Zero,
                DestinationBlendAlpha = Blend.Zero,
                BlendOperationAlpha = BlendOperation.Add,
                RenderTargetWriteMask = ColorWriteEnable.Red | ColorWriteEnable.Green | ColorWriteEnable.Blue
            };
            return desc;
        }

        private void CreateSpriteBuffer(int capacity)
        {
            _spriteBuffer?.Dispose();

            int bufferSize = capacity * SpriteStride;
            _spriteBuffer = _device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)bufferSize),
                ResourceStates.GenericRead);

            unsafe
            {
                void* ptr;
                _spriteBuffer.Map(0, null, &ptr);
                _spriteBufferPtr = (IntPtr)ptr;
            }

            // Allocate or reuse bindless index
            if (_spriteBufferBindlessIndex == 0)
                _spriteBufferBindlessIndex = _device.AllocateBindlessIndex();

            _device.CreateStructuredBufferSRV(
                _spriteBuffer,
                (uint)capacity,
                SpriteStride,
                _spriteBufferBindlessIndex);

            _bufferCapacity = capacity;
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
        }

        /// <summary>
        /// Draw a textured quad.
        /// </summary>
        public void Draw(int x, int y, int width, int height, RectF rect, Color4 color, uint textureIndex, int texWidth = 0, int texHeight = 0)
        {
            // CPU-side scissor clipping
            if (Scissor.Right > Scissor.Left && Scissor.Bottom > Scissor.Top)
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
            sprite.Type = 0;
            sprite.EndPosition = default;
        }

        /// <summary>
        /// Draw a colored line.
        /// </summary>
        public void DrawLine(int x1, int y1, int x2, int y2, Color4 color)
        {

            EnsureCapacity();

            ref var sprite = ref _sprites[_spriteCount++];
            sprite.Position = new Vector2(x1, y1);
            sprite.Size = new Vector2(LineWidth, 0);
            sprite.Color = color;
            sprite.UVs = new Vector4(0, 0, 1, 1);
            sprite.TextureIndex = _whiteTextureIndex;
            sprite.Type = 1;
            sprite.EndPosition = new Vector2(x2, y2);
        }

        /// <summary>
        /// Flush all sprites to the GPU and draw them. Call after all Draw/DrawLine calls.
        /// </summary>
        public void End(ID3D12GraphicsCommandList commandList, int screenWidth, int screenHeight)
        {
            if (_spriteCount == 0) return;

            // Upload sprite data to GPU
            unsafe
            {
                fixed (SpriteData* src = _sprites)
                {
                    int bytes = _spriteCount * SpriteStride;
                    Buffer.MemoryCopy(src, (void*)_spriteBufferPtr, _bufferCapacity * SpriteStride, bytes);
                }
            }

            // Set pipeline state
            commandList.SetPipelineState(_pipelineState);
            commandList.SetGraphicsRootSignature(_device.GlobalRootSignature);

            // Push constants: sprite buffer index, screen width, screen height
            commandList.SetGraphicsRoot32BitConstant(0, _spriteBufferBindlessIndex, 0);

            unsafe
            {
                float w = screenWidth;
                float h = screenHeight;
                commandList.SetGraphicsRoot32BitConstant(0, *(uint*)&w, 1);
                commandList.SetGraphicsRoot32BitConstant(0, *(uint*)&h, 2);
            }

            // Set topology and draw
            commandList.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);
            commandList.DrawInstanced((uint)(_spriteCount * 6), 1, 0, 0);
        }

        private void EnsureCapacity()
        {
            if (_spriteCount >= _sprites.Length)
            {
                int newCapacity = _sprites.Length * 2;
                Array.Resize(ref _sprites, newCapacity);

                if (newCapacity > _bufferCapacity)
                    CreateSpriteBuffer(newCapacity);
            }
        }

        public void Dispose()
        {
            _pipelineState?.Dispose();
            _spriteBuffer?.Dispose();
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
}
