using System;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;
using static Vortice.Direct3D12.D3D12;
using Freefall.Components;

namespace Freefall.Graphics
{
    public class RenderView : IDisposable
    {
        private const int FrameCount = 3;
        private readonly Window _window;
        private readonly GraphicsDevice _graphicsDevice;
        
        private IDXGISwapChain3 _swapChain = null!;
        private ID3D12DescriptorHeap _rtvHeap = null!;
        private int _rtvDescriptorSize;
        private ID3D12Resource[] _renderTargets;
        private int _frameIndex;
        
        /// <summary>
        /// The current swapchain buffer index (0, 1, or 2). All frame-buffered resources should use this.
        /// </summary>
        public int FrameIndex => _frameIndex;
        
        private ID3D12DescriptorHeap _dsvHeap = null!;
        private ID3D12Resource _depthStencil = null!;

        // Synchronization logic
        private ID3D12Fence _fence = null!;
        private long _fenceValue;
        private long[] _frameFenceValues;
        private IntPtr _fenceEvent;

        // Command recording - Buffered per frame
        private CommandList[] _commandLists;
        public CommandList CommandList => _commandLists[_frameIndex];
        public RenderPipeline Pipeline { get; set; } = null!;

        private bool _tearingSupported;
        private IntPtr _frameLatencyWaitableObject;

        public int Width => _window.Width;
        public int Height => _window.Height;
        
        // Cached array for SetDescriptorHeaps call
        private ID3D12DescriptorHeap[]? _srvHeapArray;

        public RenderView(Window window, GraphicsDevice graphicsDevice)
        {
            _window = window;
            _graphicsDevice = graphicsDevice;
            _renderTargets = new ID3D12Resource[FrameCount];
            _frameFenceValues = new long[FrameCount];
            _commandLists = new CommandList[FrameCount];

            Initialize();
        }

        private void Initialize()
        {
            // Check for tearing support
            _tearingSupported = _graphicsDevice.Factory.PresentAllowTearing;

            // SwapChain - use FrameLatencyWaitableObject for proper VSync at high refresh rates
            SwapChainFlags swapFlags = SwapChainFlags.FrameLatencyWaitableObject;
            if (_tearingSupported) swapFlags |= SwapChainFlags.AllowTearing;
            
            SwapChainDescription1 swapChainDesc = new SwapChainDescription1
            {
                BufferCount = FrameCount,
                Width = (uint)_window.Width,
                Height = (uint)_window.Height,
                Format = Format.R8G8B8A8_UNorm,
                BufferUsage = Usage.RenderTargetOutput,
                SwapEffect = SwapEffect.FlipDiscard,
                SampleDescription = new SampleDescription(1, 0),
                Flags = swapFlags
            };

            using (IDXGISwapChain1 swapChain = _graphicsDevice.Factory.CreateSwapChainForHwnd(_graphicsDevice.CommandQueue, _window.Handle, swapChainDesc))
            {
                _swapChain = swapChain.QueryInterface<IDXGISwapChain3>();
                _frameIndex = (int)_swapChain.CurrentBackBufferIndex;
            }
            
            // Set frame latency to 2 for optimal pacing (1 presenting, 1 in-flight)
            // This reduces the "burst-and-stall" pattern from Investigation 316
            using (var swapChain2 = _swapChain.QueryInterface<IDXGISwapChain2>())
            {
                swapChain2.MaximumFrameLatency = 2; // Optimal for smooth triple buffering
            }
            _frameLatencyWaitableObject = _swapChain.FrameLatencyWaitableObject;

            // RTV Heap
            _rtvHeap = _graphicsDevice.NativeDevice.CreateDescriptorHeap(new DescriptorHeapDescription(DescriptorHeapType.RenderTargetView, FrameCount));
            _rtvDescriptorSize = (int)_graphicsDevice.NativeDevice.GetDescriptorHandleIncrementSize(DescriptorHeapType.RenderTargetView);

            // DSV Heap
            _dsvHeap = _graphicsDevice.NativeDevice.CreateDescriptorHeap(new DescriptorHeapDescription(DescriptorHeapType.DepthStencilView, 1));

            // Render Targets
            CreateRenderTargets();

            // Synchronization
            _fence = _graphicsDevice.NativeDevice.CreateFence(0, FenceFlags.None);
            _fenceValue = 1;
            _fenceEvent = Kernel32.CreateEvent(IntPtr.Zero, false, false, null);

            // Create per-frame CommandLists
            for (int i = 0; i < FrameCount; i++)
            {
                _commandLists[i] = new CommandList(_graphicsDevice);
            }

            _window.OnResize += HandleResize;
        }

        private void CreateDepthStencil()
        {
            var depthDesc = ResourceDescription.Texture2D(Format.D32_Float, (uint)_window.Width, (uint)_window.Height, 1, 1, 1, 0, ResourceFlags.AllowDepthStencil);
            
            var clearValue = new ClearValue(Format.D32_Float, 1.0f, 0);

            _depthStencil = _graphicsDevice.NativeDevice.CreateCommittedResource(
                new HeapProperties(HeapType.Default),
                HeapFlags.None,
                depthDesc,
                ResourceStates.DepthWrite,
                clearValue);

            var dsvDesc = new DepthStencilViewDescription
            {
                Format = Format.D32_Float,
                ViewDimension = DepthStencilViewDimension.Texture2D
            };

            _graphicsDevice.NativeDevice.CreateDepthStencilView(_depthStencil, dsvDesc, _dsvHeap.GetCPUDescriptorHandleForHeapStart());
        }

        private void CreateRenderTargets()
        {
            CpuDescriptorHandle rtvHandle = _rtvHeap.GetCPUDescriptorHandleForHeapStart();
            for (int i = 0; i < FrameCount; i++)
            {
                _renderTargets[i] = _swapChain.GetBuffer<ID3D12Resource>((uint)i);
                _graphicsDevice.NativeDevice.CreateRenderTargetView(_renderTargets[i], null, rtvHandle);
                rtvHandle += _rtvDescriptorSize;
            }

            CreateDepthStencil();
        }

        public event Action<int, int> OnResize;

        private void HandleResize(int width, int height)
        {
            if (width == 0 || height == 0) return;

            WaitForGpu(); // Wait for ALL frames to complete before destroying resources

            for (int i = 0; i < FrameCount; i++)
            {
                _renderTargets[i].Dispose();
                // We should also ensure the command lists are closed or reset. 
                // Since we waited for GPU, they are done executing.
                // We don't necessarily need to dispose the command lists, just the backbuffers.
                // But resetting them is good practice if we want to start fresh, 
                // though the next Prepare() call handles that.
            }
            _depthStencil?.Dispose();

            SwapChainFlags flags = SwapChainFlags.FrameLatencyWaitableObject;
            if (_tearingSupported) flags |= SwapChainFlags.AllowTearing;

            _swapChain.ResizeBuffers(FrameCount, (uint)width, (uint)height, Format.R8G8B8A8_UNorm, flags);
            _frameIndex = (int)_swapChain.CurrentBackBufferIndex;

            CreateRenderTargets();

            // Resize pipeline resources (G-Buffers etc)
            Pipeline?.Resize(width, height);
            
            OnResize?.Invoke(width, height);
        }

        public CpuDescriptorHandle BackBufferTarget => _rtvHeap.GetCPUDescriptorHandleForHeapStart() + (_frameIndex * _rtvDescriptorSize);
        public CpuDescriptorHandle DepthBufferTarget => _dsvHeap.GetCPUDescriptorHandleForHeapStart();
        public ID3D12Resource CurrentBackBuffer => _renderTargets[_frameIndex];

        public void Prepare()
        {
            // Wait for swap chain to signal it's ready for a new frame
            // This provides frame pacing for BOTH VSync on and off, preventing burst-and-stall
            if (_frameLatencyWaitableObject != IntPtr.Zero)
            {
                Kernel32.WaitForSingleObject(_frameLatencyWaitableObject, 1000);
            }
            
            // Wait for the GPU to be done with this specific frame buffer from the previous cycle
            SyncFrame(_frameIndex);

            // Ensure GPU has finished any pending copy queue texture uploads before we render
            _graphicsDevice.WaitForCopyQueue();
            _graphicsDevice.FlushDeferredDisposals();

            // Reset the command allocator and list for this frame
            var cmd = _commandLists[_frameIndex];
            cmd.Reset();
            
            // Set descriptor heaps ONCE per frame - critical for D3D12 performance!
            _srvHeapArray ??= new[] { _graphicsDevice.SrvHeap };
            cmd.Native.SetDescriptorHeaps(1, _srvHeapArray);

            cmd.ResourceBarrierTransition(_renderTargets[_frameIndex], ResourceStates.Present, ResourceStates.RenderTarget);

            cmd.SetRenderTargets(BackBufferTarget, DepthBufferTarget);

            cmd.SetViewport(new Viewport(0, 0, Width, Height, 0.0f, 1.0f));
            cmd.SetScissorRect(new RectI(0, 0, Width, Height));

            cmd.ClearRenderTargetView(BackBufferTarget, new Color4(0.39f, 0.58f, 0.93f, 1.0f));
            // cmd.ClearDepthStencilView(DepthBufferTarget, ClearFlags.Depth, 1.0f, 0); // Need to expose this in CommandList if not already
        }
        
        public void Render(Camera camera)
        {
             Pipeline?.Render(camera, CommandList.Native);
        }

        public void Present()
        {
            var cmd = _commandLists[_frameIndex];
            cmd.ResourceBarrierTransition(_renderTargets[_frameIndex], ResourceStates.RenderTarget, ResourceStates.Present);

            cmd.Close();
            _graphicsDevice.SubmitCommandList(cmd.Native);

            // Tearing support check
            uint syncInterval = Engine.Settings.VSync ? 1u : 0u;
            PresentFlags presentFlags = PresentFlags.None;
            if (syncInterval == 0 && _tearingSupported)
            {
                presentFlags = PresentFlags.AllowTearing;
            }

            _swapChain.Present(syncInterval, presentFlags);

            _frameFenceValues[_frameIndex] = _fenceValue;
            _graphicsDevice.CommandQueue.Signal(_fence, (ulong)_fenceValue);
            _fenceValue++;

            _frameIndex = (int)_swapChain.CurrentBackBufferIndex;
        }

        private void SyncFrame(int index)
        {
            ulong fenceValue = (ulong)_frameFenceValues[index];
            // If the fence value is 0, this frame has never been used, so no need to wait.
            if (fenceValue != 0 && _fence.CompletedValue < fenceValue)
            {
                _fence.SetEventOnCompletion(fenceValue, _fenceEvent);
                Kernel32.WaitForSingleObject(_fenceEvent, -1);
            }
        }

        private void WaitForGpu()
        {
            // Schedule a Signal command in the queue
            _graphicsDevice.CommandQueue.Signal(_fence, (ulong)_fenceValue);

            // Wait until the fence has been crossed
            _fence.SetEventOnCompletion((ulong)_fenceValue, _fenceEvent);
            Kernel32.WaitForSingleObject(_fenceEvent, -1);
            
            _fenceValue++;
        }

        public void Dispose()
        {
             WaitForGpu(); // Ensure GPU is done
             
             _window.OnResize -= OnResize;
             
             for (int i = 0; i < FrameCount; i++) _renderTargets[i]?.Dispose();
             _depthStencil?.Dispose();

             // Dispose command lists
             if (_commandLists != null)
             {
                 for(int i=0; i<_commandLists.Length; i++) _commandLists[i]?.Dispose();
             }

             _fence?.Dispose();
             _rtvHeap?.Dispose();
             _dsvHeap?.Dispose();
             _swapChain?.Dispose();
        }
    }
}
