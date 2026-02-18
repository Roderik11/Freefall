using System;
using System.Collections.Generic;
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
        private readonly IntPtr _handle;
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
        public CommandList CommandList => _commandLists?[_frameIndex];
        public RenderPipeline Pipeline { get; set; } = null!;

        private bool _tearingSupported;
        private IntPtr _frameLatencyWaitableObject;

        public int Width { get; private set; }
        public int Height { get; private set; }
        
        /// <summary>
        /// True when this RenderView owns a swapchain (main window).
        /// False for headless editor viewports that render to offscreen textures.
        /// </summary>
        public bool HasSwapChain { get; private set; }

        /// <summary>
        /// Whether this view participates in the render loop.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Optional custom render callback. When set, the render loop calls this
        /// instead of the default camera→pipeline path (used for GUI-only windows).
        /// </summary>
        public Action<RenderView> OnRender;

        public Action<RenderView> OnAfterRender;

        public event Action OnResized;

        /// <summary>
        /// All registered RenderViews. Engine.Tick iterates this list (Apex pattern).
        /// </summary>
        public static List<RenderView> All { get; } = new List<RenderView>();

        /// <summary>
        /// The primary swapchain view (Apex: RenderView.First). Used for frame synchronization.
        /// </summary>
        public static RenderView Primary { get; private set; }

        // Cached array for SetDescriptorHeaps call
        private ID3D12DescriptorHeap[]? _srvHeapArray;

        // --- Headless BackBuffer (Apex pattern: CreateBuffers non-swapchain path) ---
        private RenderTexture2D _headlessBackBuffer;
        private DepthTexture2D _headlessDepthBuffer;

        /// <summary>
        /// The BackBuffer texture for Squid rendering (headless views only).
        /// </summary>
        public RenderTexture2D BackBufferTexture => _headlessBackBuffer;

        // Deferred resize (Apex pattern: Resize sets pending, Prepare processes)
        private bool _resizePending;
        private int _pendingWidth;
        private int _pendingHeight;

        /// <summary>
        /// Create a RenderView from an external HWND (e.g. WinForms RenderForm).
        /// Call Resize() manually when the host window changes size.
        /// </summary>
        public RenderView(IntPtr handle, int width, int height, GraphicsDevice graphicsDevice)
        {
            _handle = handle;
            _graphicsDevice = graphicsDevice;
            Width = width;
            Height = height;
            HasSwapChain = true;
            _renderTargets = new ID3D12Resource[FrameCount];
            _frameFenceValues = new long[FrameCount];
            _commandLists = new CommandList[FrameCount];

            Initialize();
            All.Add(this);
            if (Primary == null) Primary = this;
        }

        /// <summary>
        /// Create a headless RenderView for editor viewports (Apex pattern).
        /// Creates own BackBuffer + DepthBuffer. Pipeline blits Composite into BackBuffer.
        /// Rendering happens on the main RenderView's command list.
        /// </summary>
        public RenderView(int width, int height, GraphicsDevice graphicsDevice)
        {
            _handle = IntPtr.Zero;
            _graphicsDevice = graphicsDevice;
            Width = Math.Max(1, width);
            Height = Math.Max(1, height);
            HasSwapChain = false;

            CreateHeadlessBuffers();
            All.Add(this);
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
                Width = (uint)Width,
                Height = (uint)Height,
                Format = Format.R8G8B8A8_UNorm,
                BufferUsage = Usage.RenderTargetOutput,
                SwapEffect = SwapEffect.FlipDiscard,
                SampleDescription = new SampleDescription(1, 0),
                Flags = swapFlags
            };

            using (IDXGISwapChain1 swapChain = _graphicsDevice.Factory.CreateSwapChainForHwnd(_graphicsDevice.CommandQueue, _handle, swapChainDesc))
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

        }

        private void CreateDepthStencil()
        {
            var depthDesc = ResourceDescription.Texture2D(Format.D32_Float, (uint)Width, (uint)Height, 1, 1, 1, 0, ResourceFlags.AllowDepthStencil);
            
            var clearValue = new ClearValue(Format.D32_Float, 0.0f, 0); // Reverse depth: far=0

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


        /// <summary>
        /// Handle resize from either internal Window.OnResize or external call.
        /// For swapchain views, resizes immediately. For headless views, this is also immediate.
        /// </summary>
        public void HandleResize(int width, int height)
        {
            if (width == 0 || height == 0) return;

            Width = width;
            Height = height;

            if (HasSwapChain)
            {
                WaitForGpu(); // Wait for ALL frames to complete before destroying resources

                for (int i = 0; i < FrameCount; i++)
                {
                    _renderTargets[i].Dispose();
                }
                _depthStencil?.Dispose();

                SwapChainFlags flags = SwapChainFlags.FrameLatencyWaitableObject;
                if (_tearingSupported) flags |= SwapChainFlags.AllowTearing;

                _swapChain.ResizeBuffers(FrameCount, (uint)width, (uint)height, Format.R8G8B8A8_UNorm, flags);
                _frameIndex = (int)_swapChain.CurrentBackBufferIndex;

                CreateRenderTargets();
            }

            // Resize pipeline resources (G-Buffers etc)
            Pipeline?.Resize(width, height);
            
            OnResized?.Invoke();
        }

        /// <summary>
        /// Deferred resize (Apex pattern). Just stores pending size.
        /// Actual resize happens in ProcessPendingResize before rendering.
        /// </summary>
        public void Resize(int width, int height)
        {
            _pendingWidth = Math.Max(1, width);
            _pendingHeight = Math.Max(1, height);
            _resizePending = true;
        }

        /// <summary>
        /// Whether a deferred resize is pending. Used by Engine.Tick to batch GPU sync.
        /// </summary>
        public bool IsResizePending => _resizePending;

        /// <summary>
        /// Process deferred resize at a GPU-safe point (Apex: inside Prepare).
        /// For headless views, call this from the render loop before camera rendering.
        /// </summary>
        public void ProcessPendingResize()
        {
            if (!_resizePending) return;
            _resizePending = false;

            if (_pendingWidth == Width && _pendingHeight == Height) return;

            Width = _pendingWidth;
            Height = _pendingHeight;

            if (HasSwapChain)
            {
                for (int i = 0; i < FrameCount; i++)
                    _renderTargets[i].Dispose();
                _depthStencil?.Dispose();

                SwapChainFlags flags = SwapChainFlags.FrameLatencyWaitableObject;
                if (_tearingSupported) flags |= SwapChainFlags.AllowTearing;

                _swapChain.ResizeBuffers(FrameCount, (uint)Width, (uint)Height, Format.R8G8B8A8_UNorm, flags);
                _frameIndex = (int)_swapChain.CurrentBackBufferIndex;

                CreateRenderTargets();
            }
            else
            {
                _headlessBackBuffer?.Dispose();
                _headlessDepthBuffer?.Dispose();
                CreateHeadlessBuffers();
            }

            // Resize pipeline resources (G-Buffers etc)
            Pipeline?.Resize(Width, Height);

            OnResized?.Invoke();
        }

        /// <summary>
        /// Create BackBuffer + DepthBuffer for headless views (Apex: CreateBuffers non-swapchain path).
        /// </summary>
        private void CreateHeadlessBuffers()
        {
            _headlessBackBuffer = new RenderTexture2D(_graphicsDevice, Width, Height, Format.R8G8B8A8_UNorm);
            _headlessDepthBuffer = new DepthTexture2D(Width, Height, Format.D32_Float, false);
        }

        public CpuDescriptorHandle BackBufferTarget =>
            HasSwapChain
                ? _rtvHeap.GetCPUDescriptorHandleForHeapStart() + (_frameIndex * _rtvDescriptorSize)
                : _headlessBackBuffer.RtvHandle;

        public CpuDescriptorHandle DepthBufferTarget =>
            HasSwapChain
                ? _dsvHeap.GetCPUDescriptorHandleForHeapStart()
                : _headlessDepthBuffer.DsvHandle;

        public ID3D12Resource CurrentBackBuffer =>
            HasSwapChain
                ? _renderTargets[_frameIndex]
                : _headlessBackBuffer.Native;

        public void Prepare()
        {
            // Wait for swap chain to signal it's ready for a new frame
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
             OnAfterRender?.Invoke(this);
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

        internal void WaitForGpu()
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
             All.Remove(this);

             if (HasSwapChain)
             {
                 WaitForGpu(); // Ensure GPU is done
                 
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
             else
             {
                 _headlessBackBuffer?.Dispose();
                 _headlessDepthBuffer?.Dispose();
             }

             Pipeline?.Dispose();
        }
    }
}
