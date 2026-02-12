using System;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Offscreen render target with color + depth buffers.
    /// The color buffer has a bindless SRV so it can be sampled as a texture (e.g. by Squid).
    /// Does not own a command list — uses the main RenderView's command list inline.
    /// </summary>
    public class RenderTarget : IDisposable
    {
        private readonly GraphicsDevice _device;

        private ID3D12Resource _colorBuffer = null!;
        private ID3D12Resource _depthBuffer = null!;

        private CpuDescriptorHandle _rtvHandle;
        private CpuDescriptorHandle _dsvHandle;

        private uint _bindlessIndex;
        private bool _bindlessAllocated;
        private bool _descriptorsAllocated;

        private ResourceStates _currentState = ResourceStates.Common;

        public int Width { get; private set; }
        public int Height { get; private set; }

        /// <summary>
        /// Bindless SRV index for sampling the color buffer as a texture.
        /// </summary>
        public uint BindlessIndex => _bindlessIndex;

        /// <summary>
        /// The underlying color buffer resource (for advanced usage).
        /// </summary>
        public ID3D12Resource ColorBuffer => _colorBuffer;

        public Color4 ClearColor { get; set; } = new Color4(0.39f, 0.58f, 0.93f, 1.0f);

        public RenderTarget(GraphicsDevice device, int width, int height)
        {
            _device = device;
            Width = width;
            Height = height;

            // Allocate descriptor slots (persistent, never freed — one-time)
            _rtvHandle = device.AllocateRtv();
            _dsvHandle = device.AllocateDsv();
            _bindlessIndex = device.AllocateBindlessIndex();
            _bindlessAllocated = true;
            _descriptorsAllocated = true;

            CreateBuffers();
        }

        private void CreateBuffers()
        {
            int w = Math.Max(1, Width);
            int h = Math.Max(1, Height);

            // Color buffer — R8G8B8A8_UNorm, allow render target
            _colorBuffer = _device.CreateTexture2D(
                Format.R8G8B8A8_UNorm, w, h,
                flags: ResourceFlags.AllowRenderTarget,
                initialState: ResourceStates.PixelShaderResource,
                clearValue: ClearColor);

            _currentState = ResourceStates.PixelShaderResource;

            // Create RTV
            _device.CreateRenderTargetView(_colorBuffer, null, _rtvHandle);

            // Create SRV at the bindless index
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.R8G8B8A8_UNorm,
                ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Texture2D,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2D = new Texture2DShaderResourceView { MipLevels = 1 }
            };
            _device.NativeDevice.CreateShaderResourceView(_colorBuffer, srvDesc,
                _device.GetCpuHandle(_bindlessIndex));

            // Depth buffer — D32_Float
            var depthDesc = ResourceDescription.Texture2D(
                Format.D32_Float, (uint)w, (uint)h, 1, 1, 1, 0,
                ResourceFlags.AllowDepthStencil);

            _depthBuffer = _device.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Default),
                HeapFlags.None,
                depthDesc,
                ResourceStates.DepthWrite,
                new ClearValue(Format.D32_Float, 0.0f, 0));

            var dsvDesc = new DepthStencilViewDescription
            {
                Format = Format.D32_Float,
                ViewDimension = DepthStencilViewDimension.Texture2D
            };
            _device.NativeDevice.CreateDepthStencilView(_depthBuffer, dsvDesc, _dsvHandle);
        }

        /// <summary>
        /// Resize the render target. Destroys old buffers, creates new ones.
        /// The bindless index is preserved so Squid doesn't need re-registration.
        /// Must be called when the GPU is idle (no in-flight references to old buffers).
        /// </summary>
        public void Resize(int width, int height)
        {
            if (width <= 0 || height <= 0) return;
            if (width == Width && height == Height) return;

            Width = width;
            Height = height;

            _colorBuffer?.Dispose();
            _depthBuffer?.Dispose();

            CreateBuffers();
        }

        /// <summary>
        /// Begin rendering to this target. Transitions color buffer to RenderTarget state,
        /// sets render targets, clears, and configures viewport + scissor.
        /// </summary>
        public void BeginRender(CommandList cmd)
        {
            // Transition color buffer to RenderTarget
            cmd.ResourceBarrierTransition(_colorBuffer, _currentState, ResourceStates.RenderTarget);
            _currentState = ResourceStates.RenderTarget;

            // Set render targets
            cmd.SetRenderTargets(_rtvHandle, _dsvHandle);

            // Clear
            cmd.ClearRenderTargetView(_rtvHandle, ClearColor);
            cmd.ClearDepthStencilView(_dsvHandle, ClearFlags.Depth, 0.0f, 0); // Reverse depth: far=0

            // Set viewport and scissor
            cmd.SetViewport(new Viewport(0, 0, Width, Height, 0.0f, 1.0f));
            cmd.SetScissorRect(new RectI(0, 0, Width, Height));
        }

        /// <summary>
        /// End rendering to this target. Transitions color buffer to PixelShaderResource
        /// so Squid's SpriteBatch can sample it.
        /// </summary>
        public void EndRender(CommandList cmd)
        {
            cmd.ResourceBarrierTransition(_colorBuffer, _currentState, ResourceStates.PixelShaderResource);
            _currentState = ResourceStates.PixelShaderResource;
        }

        public void Dispose()
        {
            _colorBuffer?.Dispose();
            _depthBuffer?.Dispose();

            if (_bindlessAllocated)
            {
                _device.ReleaseBindlessIndex(_bindlessIndex);
                _bindlessAllocated = false;
            }
        }
    }
}
