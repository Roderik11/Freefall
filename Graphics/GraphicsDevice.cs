using System;
using System.Threading;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Direct3D;
using Vortice.WIC;
using Vortice.Direct3D12.Debug;
using Vortice.Mathematics;
using static Vortice.Direct3D12.D3D12;

namespace Freefall.Graphics
{
    public class GraphicsDevice : IDisposable
    {
        private ID3D12Device _device = null!;
        private ID3D12CommandQueue _commandQueue = null!;
        private ID3D12CommandQueue _copyQueue = null!;
        private IDXGIFactory5 _factory = null!;
        private IWICImagingFactory2 _wicFactory = null!;
        private ID3D12DescriptorHeap _srvHeap = null!;
        private ID3D12DescriptorHeap _rtvHeap = null!;
        private ID3D12DescriptorHeap _dsvHeap = null!;
        private ID3D12RootSignature _globalRootSignature = null!;
        private int _srvDescriptorSize;
        private bool _disposed;
        
        // Descriptor Management
        private const int MaxBindlessDescriptors = 1000000;
        private readonly Stack<uint> _freeBindlessIndices = new Stack<uint>();
        private uint _nextBindlessIndex = 0;
        private readonly object _bindlessLock = new object();
        private readonly object _queueLock = new object();

        // Cross-queue synchronization: copy queue → direct queue
        private ID3D12Fence _copyFence = null!;
        private long _copyFenceValue = 0;
        private long _lastWaitedCopyFenceValue = 0;
        private readonly object _copyFenceLock = new object();

        // Deferred disposal of upload buffers after async copy completes
        private readonly List<(long fenceValue, ID3D12Resource buffer)> _deferredDisposals = new();
        private readonly object _deferredDisposalLock = new object();


        private int _rtvDescriptorSize;
        private int _rtvDescriptorIndex;
        private int _dsvDescriptorSize;
        private int _dsvDescriptorIndex;

        public ID3D12Device NativeDevice => _device;
        public ID3D12CommandQueue CommandQueue => _commandQueue;
        public ID3D12CommandQueue CopyQueue => _copyQueue;
        public IDXGIFactory5 Factory => _factory;
        public IWICImagingFactory2 WicFactory => _wicFactory;
        public ID3D12DescriptorHeap SrvHeap => _srvHeap;
        public ID3D12RootSignature GlobalRootSignature => _globalRootSignature;

        public GraphicsDevice()
        {
            Initialize();
        }

        private void Initialize()
        {
            // D3D12 Debug Layer — disabled during scene loading (validation overhead causes TDR)
            // if (D3D12GetDebugInterface(out ID3D12Debug? debug).Success && debug != null)
            // {
            //     debug.EnableDebugLayer();
            //     debug.Dispose();
            // }
            
            _factory = DXGI.CreateDXGIFactory1<IDXGIFactory5>();
            _device = D3D12CreateDevice<ID3D12Device>(null, FeatureLevel.Level_11_0);

            // Check Shader Model 6.6 Support
            var featureData = new FeatureDataShaderModel { HighestShaderModel = (ShaderModel)0x66 };
            bool sm66Supported = _device.CheckFeatureSupport(Vortice.Direct3D12.Feature.ShaderModel, ref featureData) && (int)featureData.HighestShaderModel >= 0x66;
            
            // Check Root Signature Version Support
            var rootSigFeatureData = new FeatureDataRootSignature { HighestVersion = RootSignatureVersion.Version11 };
            _device.CheckFeatureSupport(Vortice.Direct3D12.Feature.RootSignature, ref rootSigFeatureData);

            if (sm66Supported)
            {
                Debug.Log("GraphicsDevice", $"Shader Model 6.6 Supported. Reported: {featureData.HighestShaderModel}");
            }
            else
            {
                Debug.LogWarning("GraphicsDevice", $"Shader Model 6.6 NOT supported. Highest: {featureData.HighestShaderModel}");
            }
            Debug.Log("GraphicsDevice", $"Highest Root Signature Version: {rootSigFeatureData.HighestVersion}");

            _commandQueue = _device.CreateCommandQueue(new CommandQueueDescription(CommandListType.Direct));
            _copyQueue = _device.CreateCommandQueue(new CommandQueueDescription(CommandListType.Copy));
            _copyFence = _device.CreateFence(0);

            // Initialize WIC
            _wicFactory = new IWICImagingFactory2();

            // Initialize SRV Descriptor Heap (Global Bindless Heap)
            _srvHeap = _device.CreateDescriptorHeap(new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = MaxBindlessDescriptors,
                Flags = DescriptorHeapFlags.ShaderVisible
            });
            _srvDescriptorSize = (int)_device.GetDescriptorHandleIncrementSize(DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);
            
            Debug.Log("GraphicsDevice", $"Bindless SRV Heap initialized with {MaxBindlessDescriptors} slots.");

            // Initialize RTV Descriptor Heap (Non-Shader Visible)
            _rtvHeap = _device.CreateDescriptorHeap(new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.RenderTargetView,
                DescriptorCount = 256,
                Flags = DescriptorHeapFlags.None
            });
            _rtvDescriptorSize = (int)_device.GetDescriptorHandleIncrementSize(DescriptorHeapType.RenderTargetView);

            // Initialize DSV Descriptor Heap (Non-Shader Visible)
            _dsvHeap = _device.CreateDescriptorHeap(new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.DepthStencilView,
                DescriptorCount = 64,
                Flags = DescriptorHeapFlags.None
            });
            _dsvDescriptorSize = (int)_device.GetDescriptorHandleIncrementSize(DescriptorHeapType.DepthStencilView);

            CreateGlobalRootSignature();
        }

        private void CreateGlobalRootSignature()
        {
            var rootParameters = new RootParameter1[]
            {
                new RootParameter1(new RootConstants(3, 0, 32), ShaderVisibility.All), // Slot 0: Push Constants (mapped to b3) (32 dwords = 8 * uint4)
                new RootParameter1(RootParameterType.ConstantBufferView, new RootDescriptor1(0, 0), ShaderVisibility.All), // Slot 1: Scene Constants (b0)
                new RootParameter1(RootParameterType.ConstantBufferView, new RootDescriptor1(1, 0), ShaderVisibility.All), // Slot 2: Object/Terrain Constants (b1)
                new RootParameter1(RootParameterType.ConstantBufferView, new RootDescriptor1(2, 0), ShaderVisibility.All), // Slot 3: Tiling Constants (b2)
                new RootParameter1(RootParameterType.ConstantBufferView, new RootDescriptor1(4, 0), ShaderVisibility.All), // Slot 4: TextureIndices (b4)
            };

            var staticSamplers = new StaticSamplerDescription[]
            {
                new StaticSamplerDescription(ShaderVisibility.All, 0, 0) { Filter = Filter.MinMagMipLinear, AddressU = TextureAddressMode.Wrap, AddressV = TextureAddressMode.Wrap, AddressW = TextureAddressMode.Wrap },
                new StaticSamplerDescription(ShaderVisibility.All, 1, 0) { Filter = Filter.MinMagMipPoint, AddressU = TextureAddressMode.Clamp, AddressV = TextureAddressMode.Clamp, AddressW = TextureAddressMode.Clamp },
                new StaticSamplerDescription(ShaderVisibility.All, 2, 0) { Filter = Filter.MinMagMipLinear, AddressU = TextureAddressMode.Clamp, AddressV = TextureAddressMode.Clamp, AddressW = TextureAddressMode.Clamp },
                new StaticSamplerDescription(ShaderVisibility.All, 3, 0) { Filter = Filter.ComparisonMinMagLinearMipPoint, AddressU = TextureAddressMode.Clamp, AddressV = TextureAddressMode.Clamp, AddressW = TextureAddressMode.Clamp, ComparisonFunction = ComparisonFunction.LessEqual }
            };

            // D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED = 0x400
            // D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED = 0x800
            var flags = RootSignatureFlags.None | (RootSignatureFlags)0x400 | (RootSignatureFlags)0x800;
            
            var desc = new VersionedRootSignatureDescription(new RootSignatureDescription1(flags, rootParameters, staticSamplers));
            _globalRootSignature = _device.CreateRootSignature(desc);
            Debug.Log("GraphicsDevice", "Global Bindless Root Signature created (Directly Indexed, Version 1.1).");

            CreateCommandSignatures();
        }

        private void CreateCommandSignatures()
        {
            Debug.Log("GraphicsDevice", $"Creating Command Signatures. RootSig: {_globalRootSignature.NativePointer:X}");
            
            // DrawInstanced Argument
            // IndirectArgumentDescription(IndirectArgumentType type, int constantBufferSlot = 0, int constantBufferVisiblity = 0, int descriptorTableSlot = 0, int descriptorTableNumDescriptors = 0) 
            // Actually usually it's just Type. Let's try explicit struct initialization if constructor fails.
            // Or typically: new IndirectArgumentDescription(IndirectArgumentType.Draw) works in some wrappers.
            // Vortice likely requires explicitly all or using struct initializer.
            // Let's assume constructor is (Type, ...).
            
            var argsDraw = new IndirectArgumentDescription[]
            {
                new IndirectArgumentDescription { Type = IndirectArgumentType.Draw }
            };

            // CreateCommandSignature(CommandSignatureDescription, ID3D12RootSignature)
            int strideDraw = System.Runtime.InteropServices.Marshal.SizeOf<DrawInstancedArguments>();
            Debug.Log("GraphicsDevice", $"DrawInstanced Stride: {strideDraw} (Expected 16)");
            
            var descDraw = new CommandSignatureDescription(strideDraw, argsDraw);
            try {
                _drawInstancedSignature = _device.CreateCommandSignature<ID3D12CommandSignature>(descDraw, null);
            } catch (Exception ex) {
                Debug.LogError("GraphicsDevice", $"Creating DrawInstancedSignature: {ex.Message}");
                // throw; // Let it continue to try the next one? No, it will crash later.
            }

            // DrawIndexedInstanced Argument
            var argsDrawIndexed = new IndirectArgumentDescription[]
            {
                new IndirectArgumentDescription { Type = IndirectArgumentType.DrawIndexed }
            };
            
            int strideIndexed = System.Runtime.InteropServices.Marshal.SizeOf<DrawIndexedInstancedArguments>();
             Debug.Log("GraphicsDevice", $"DrawIndexedInstanced Stride: {strideIndexed} (Expected 20)");

            var descDrawIndexed = new CommandSignatureDescription(strideIndexed, argsDrawIndexed);
            try {
                _drawIndexedInstancedSignature = _device.CreateCommandSignature<ID3D12CommandSignature>(descDrawIndexed, null);
            } catch (Exception ex) {
                 Debug.LogError("GraphicsDevice", $"Creating DrawIndexedInstancedSignature: {ex.Message}");
            }

            // Bindless command signature: 14 root constants + DrawInstanced
            // Used by GPU-driven indirect rendering with per-draw buffer indices
            var constantArg = new IndirectArgumentDescription();
            constantArg.Type = IndirectArgumentType.Constant;
            constantArg.Constant.RootParameterIndex = 0;           // Root parameter 0 (push constants)
            constantArg.Constant.DestOffsetIn32BitValues = 2;      // Start at slot 2
            constantArg.Constant.Num32BitValuesToSet = 14;         // 14 slots (2-15)
            
            var bindlessArgs = new IndirectArgumentDescription[]
            {
                constantArg,
                new IndirectArgumentDescription { Type = IndirectArgumentType.Draw }
            };
            
            // Stride = 14 constants (56 bytes) + DrawInstancedArguments (16 bytes) = 72 bytes
            int bindlessStride = 14 * sizeof(uint) + System.Runtime.InteropServices.Marshal.SizeOf<DrawInstancedArguments>();
            var bindlessDesc = new CommandSignatureDescription(bindlessStride, bindlessArgs);
            
            try {
                _bindlessCommandSignature = _device.CreateCommandSignature<ID3D12CommandSignature>(bindlessDesc, _globalRootSignature);
                Debug.Log("GraphicsDevice", $"Bindless command signature created ({bindlessStride} byte stride)");
            } catch (Exception ex) {
                Debug.LogError("GraphicsDevice", $"Creating BindlessCommandSignature: {ex.Message}");
            }
        }

        private ID3D12CommandSignature _drawInstancedSignature;
        private ID3D12CommandSignature _drawIndexedInstancedSignature;
        private ID3D12CommandSignature _bindlessCommandSignature = null!;

        public ID3D12CommandSignature DrawInstancedSignature => _drawInstancedSignature;
        public ID3D12CommandSignature DrawIndexedInstancedSignature => _drawIndexedInstancedSignature;
        /// <summary>Bindless command signature with 14 root constants + DrawInstanced</summary>
        public ID3D12CommandSignature BindlessCommandSignature => _bindlessCommandSignature;

        public void Dispose()
        {
            if (_disposed) return;
            
            _srvHeap?.Dispose();
            _rtvHeap?.Dispose();
            _dsvHeap?.Dispose();
            _globalRootSignature?.Dispose();
            _drawInstancedSignature?.Dispose();
            _drawIndexedInstancedSignature?.Dispose();
            _bindlessCommandSignature?.Dispose();
            _wicFactory?.Dispose();
            _copyFence?.Dispose();
            _copyQueue?.Dispose();
            _commandQueue?.Dispose();
            _device?.Dispose();
            _factory?.Dispose();
            _disposed = true;
        }

        public ID3D12DescriptorHeap RtvHeap => _rtvHeap;
        public ID3D12DescriptorHeap DsvHeap => _dsvHeap;
        public int SrvDescriptorSize => _srvDescriptorSize;

        public CpuDescriptorHandle AllocateRtv()
        {
             if (_rtvDescriptorIndex >= 256) throw new InvalidOperationException("Out of RTV descriptors");
             var handle = _rtvHeap.GetCPUDescriptorHandleForHeapStart() + (_rtvDescriptorIndex * _rtvDescriptorSize);
             _rtvDescriptorIndex++;
             return handle;
        }

        public CpuDescriptorHandle AllocateDsv()
        {
             if (_dsvDescriptorIndex >= 64) throw new InvalidOperationException("Out of DSV descriptors");
             var handle = _dsvHeap.GetCPUDescriptorHandleForHeapStart() + (_dsvDescriptorIndex * _dsvDescriptorSize);
             _dsvDescriptorIndex++;
             return handle;
        }

        public CpuDescriptorHandle AllocateSrv(out GpuDescriptorHandle gpuHandle)
        {
            return AllocateSrvRange(1, out gpuHandle, out _);
        }

        public CpuDescriptorHandle AllocateSrv(out GpuDescriptorHandle gpuHandle, out uint index)
        {
            return AllocateSrvRange(1, out gpuHandle, out index);
        }

        public CpuDescriptorHandle AllocateSrvRange(int count, out GpuDescriptorHandle gpuHandle)
        {
            return AllocateSrvRange(count, out gpuHandle, out _);
        }

        public CpuDescriptorHandle AllocateSrvRange(int count, out GpuDescriptorHandle gpuHandle, out uint index)
        {
            // For simplicity in this transition, we'll just bump the next bindless index
            // and assume it's at the start of the heap for legacy range compatibility if needed.
            // But ideally, all callers move to AllocateBindlessIndex().
            
            index = 0;
            if (count == 1)
            {
                index = AllocateBindlessIndex();
            }
            else
            {
                // Simple contiguous allocation for ranges
                lock (_bindlessLock)
                {
                    index = _nextBindlessIndex;
                    _nextBindlessIndex += (uint)count;
                    if (_nextBindlessIndex > MaxBindlessDescriptors)
                        throw new InvalidOperationException("Out of bindless descriptors.");
                }
            }

            var cpuHandleAvailable = _srvHeap.GetCPUDescriptorHandleForHeapStart() + ((int)index * _srvDescriptorSize);
            gpuHandle = _srvHeap.GetGPUDescriptorHandleForHeapStart() + ((int)index * _srvDescriptorSize);

            return cpuHandleAvailable;
        }

        public uint AllocateBindlessIndex()
        {
            lock (_bindlessLock)
            {
                if (_freeBindlessIndices.Count > 0)
                    return _freeBindlessIndices.Pop();

                if (_nextBindlessIndex >= MaxBindlessDescriptors)
                    throw new InvalidOperationException("Out of bindless descriptors.");

                return _nextBindlessIndex++;
            }
        }

        public void ReleaseBindlessIndex(uint index)
        {
            lock (_bindlessLock)
            {
                _freeBindlessIndices.Push(index);
            }
        }

        public CpuDescriptorHandle GetCpuHandle(uint index)
        {
            return _srvHeap.GetCPUDescriptorHandleForHeapStart() + ((int)index * _srvDescriptorSize);
        }

        public GpuDescriptorHandle GetGpuHandle(uint index)
        {
            return _srvHeap.GetGPUDescriptorHandleForHeapStart() + ((int)index * _srvDescriptorSize);
        }

        // Reset per-frame temporary descriptor allocations (deprecated for Bindless, but kept for compatibility)
        public void MarkPermanentDescriptors() { }
        public void ResetTemporaryDescriptors() { }

        /// <summary>
        /// Thread-safe command list submission. Use this instead of CommandQueue.ExecuteCommandList directly.
        /// </summary>
        public void SubmitCommandList(ID3D12GraphicsCommandList commandList)
        {
            lock (_queueLock)
            {
                _commandQueue.ExecuteCommandList(commandList);
            }
        }

        /// <summary>
        /// Thread-safe submit + GPU wait. For upload operations (textures, meshes) from any thread.
        /// Submits and signals under lock, then waits for completion outside the lock so the 
        /// render thread isn't blocked from submitting frames during the wait.
        /// </summary>
        public void SubmitAndWait(ID3D12GraphicsCommandList commandList)
        {
            var fence = _device.CreateFence(0);
            lock (_queueLock)
            {
                _commandQueue.ExecuteCommandList(commandList);
                _commandQueue.Signal(fence, 1);
            }
            // Wait outside the lock so the main render thread can still submit work
            while (fence.CompletedValue < 1) { Thread.SpinWait(100); }
            fence.Dispose();
        }

        /// <summary>
        /// Submit a command list to the dedicated copy queue and CPU-wait for completion.
        /// Also signals the cross-queue fence so the direct queue can sync before rendering.
        /// Safe to call from any thread (background loading, etc).
        /// </summary>
        public void CopyQueueSubmitAndWait(ID3D12GraphicsCommandList commandList)
        {
            long targetValue;
            lock (_copyFenceLock)
            {
                targetValue = ++_copyFenceValue;
                _copyQueue.ExecuteCommandList(commandList);
                _copyQueue.Signal(_copyFence, (ulong)targetValue);
            }
            // CPU-wait for the copy to complete
            while (_copyFence.CompletedValue < (ulong)targetValue) { Thread.SpinWait(100); }
        }

        /// <summary>
        /// Submit a command list to the copy queue WITHOUT waiting.
        /// Returns the fence value — the caller must keep upload buffers alive until this value completes.
        /// Throttled: after every MaxPendingCopies submissions, CPU-waits for the batch to drain
        /// so the copy queue doesn't flood and stall the render thread.
        /// </summary>
        private int _pendingCopies;
        private const int MaxPendingCopies = 16;

        public long CopyQueueSubmit(ID3D12GraphicsCommandList commandList)
        {
            long targetValue;
            bool shouldDrain;

            lock (_copyFenceLock)
            {
                targetValue = ++_copyFenceValue;
                _copyQueue.ExecuteCommandList(commandList);
                _copyQueue.Signal(_copyFence, (ulong)targetValue);
                shouldDrain = (++_pendingCopies >= MaxPendingCopies);
                if (shouldDrain) _pendingCopies = 0;
            }

            // Drain outside the lock so other threads can still submit
            if (shouldDrain)
            {
                while (_copyFence.CompletedValue < (ulong)targetValue) { Thread.SpinWait(100); }
            }

            return targetValue;
        }

        /// <summary>
        /// Schedule an upload buffer for disposal once a copy fence value completes.
        /// Thread-safe.
        /// </summary>
        public void DeferUploadBufferDisposal(long fenceValue, ID3D12Resource uploadBuffer)
        {
            lock (_deferredDisposalLock)
            {
                _deferredDisposals.Add((fenceValue, uploadBuffer));
            }
        }

        /// <summary>
        /// Dispose any upload buffers whose copy fence has completed.
        /// Call periodically (e.g. once per frame or after load).
        /// </summary>
        public void FlushDeferredDisposals()
        {
            lock (_deferredDisposalLock)
            {
                ulong completed = _copyFence.CompletedValue;
                for (int i = _deferredDisposals.Count - 1; i >= 0; i--)
                {
                    if ((ulong)_deferredDisposals[i].fenceValue <= completed)
                    {
                        _deferredDisposals[i].buffer.Dispose();
                        _deferredDisposals.RemoveAt(i);
                    }
                }
            }
        }

        /// <summary>
        /// Make the direct (render) queue GPU-wait for any NEW copy queue operations
        /// submitted since the last call. Skips the wait if no new copies have happened.
        /// This is a GPU-side wait — the CPU returns immediately.
        /// </summary>
        public void WaitForCopyQueue()
        {
            long pendingValue;
            lock (_copyFenceLock)
            {
                pendingValue = _copyFenceValue;
            }
            if (pendingValue > _lastWaitedCopyFenceValue)
            {
                _commandQueue.Wait(_copyFence, (ulong)pendingValue);
                _lastWaitedCopyFenceValue = pendingValue;
            }
        }

        public ID3D12Resource CreateTexture2D(Format format, int width, int height, int arraySize = 1, int mipLevels = 1, ResourceFlags flags = ResourceFlags.None, ResourceStates initialState = ResourceStates.Common, Color4? clearValue = null)
        {
            var desc = ResourceDescription.Texture2D(format, (uint)width, (uint)height, (ushort)arraySize, (ushort)mipLevels, 1, 0, flags);
            
            ClearValue? optimizedClearValue = null;
            if (clearValue.HasValue && (flags & ResourceFlags.AllowRenderTarget) != 0)
            {
                optimizedClearValue = new ClearValue(format, clearValue.Value);
            }
            else if ((flags & ResourceFlags.AllowDepthStencil) != 0)
            {
                // Default depth clear value
                Format clearFormat = format;
                if (format == Format.R32_Typeless) clearFormat = Format.D32_Float;
                else if (format == Format.R16_Typeless) clearFormat = Format.D16_UNorm;
                else if (format == Format.R24G8_Typeless) clearFormat = Format.D24_UNorm_S8_UInt;

                optimizedClearValue = new ClearValue(clearFormat, 1.0f, 0);
            }

            try
            {
                return _device.CreateCommittedResource(
                    new HeapProperties(HeapType.Default),
                    HeapFlags.None,
                    desc,
                    initialState,
                    optimizedClearValue);
            }
            catch (Exception ex)
            {
                var reason = _device.DeviceRemovedReason;
                Debug.LogError("GraphicsDevice", $"CreateCommittedResource FAILED: {ex.Message}");
                Debug.LogError("GraphicsDevice", $"DeviceRemovedReason: {reason}");
                Debug.LogError("GraphicsDevice", $"Format: {format}, Size: {width}x{height}, Flags: {flags}, State: {initialState}");
                throw;
            }
        }

        public void CreateRenderTargetView(ID3D12Resource resource, RenderTargetViewDescription? desc, CpuDescriptorHandle handle)
        {
            _device.CreateRenderTargetView(resource, desc, handle);
        }

        public void CreateDepthStencilView(ID3D12Resource resource, DepthStencilViewDescription? desc, CpuDescriptorHandle handle)
        {
            _device.CreateDepthStencilView(resource, desc, handle);
        }

        public void CreateShaderResourceView(ID3D12Resource resource, ShaderResourceViewDescription? desc, CpuDescriptorHandle handle)
        {
            _device.CreateShaderResourceView(resource, desc, handle);
        }


        public ID3D12Resource CreateUploadBuffer(int size)
        {
             return _device.CreateCommittedResource(
                new Vortice.Direct3D12.HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)size),
                ResourceStates.GenericRead,
                null);
        }

        public ID3D12Resource CreateUploadBuffer<T>(T[] data) where T : unmanaged
        {
             int size = data.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>();
             var buffer = _device.CreateCommittedResource(
                new Vortice.Direct3D12.HeapProperties(HeapType.Upload),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)size),
                ResourceStates.GenericRead,
                null);

            unsafe
            {
                void* pData;
                buffer.Map(0, null, &pData);
                var span = new Span<T>(pData, data.Length);
                data.CopyTo(span);
                buffer.Unmap(0);
            }
            return buffer;
        }

        public void CreateStructuredBufferSRV(ID3D12Resource resource, uint numElements, uint stride, uint bindlessIndex)
        {
            var srvDesc = new ShaderResourceViewDescription
            {
                Format = Format.Unknown,
                ViewDimension = Vortice.Direct3D12.ShaderResourceViewDimension.Buffer,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Buffer = new BufferShaderResourceView
                {
                    FirstElement = 0,
                    NumElements = numElements,
                    StructureByteStride = stride,
                    Flags = BufferShaderResourceViewFlags.None
                }
            };
            _device.CreateShaderResourceView(resource, srvDesc, GetCpuHandle(bindlessIndex));
        }

        /// <summary>
        /// Create a default (GPU-only) buffer for UAV usage.
        /// </summary>
        public ID3D12Resource CreateDefaultBuffer(int size, ResourceFlags flags = ResourceFlags.AllowUnorderedAccess)
        {
            return _device.CreateCommittedResource(
                new HeapProperties(HeapType.Default),
                HeapFlags.None,
                ResourceDescription.Buffer((ulong)size, flags),
                ResourceStates.Common,
                null);
        }

        /// <summary>
        /// Create a UAV descriptor for a structured buffer at a bindless index.
        /// </summary>
        public void CreateStructuredBufferUAV(ID3D12Resource resource, uint numElements, uint stride, uint bindlessIndex)
        {
            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Format.Unknown,
                ViewDimension = UnorderedAccessViewDimension.Buffer,
                Buffer = new BufferUnorderedAccessView
                {
                    FirstElement = 0,
                    NumElements = numElements,
                    StructureByteStride = stride,
                    CounterOffsetInBytes = 0,
                    Flags = BufferUnorderedAccessViewFlags.None
                }
            };
            _device.CreateUnorderedAccessView(resource, null, uavDesc, GetCpuHandle(bindlessIndex));
        }

        /// <summary>
        /// Create a compute pipeline state from shader bytecode.
        /// Uses the global root signature.
        /// </summary>
        public ID3D12PipelineState CreateComputePipelineState(byte[] shaderBytecode)
        {
            var desc = new ComputePipelineStateDescription
            {
                RootSignature = _globalRootSignature,
                ComputeShader = shaderBytecode
            };
            return _device.CreateComputePipelineState(desc);
        }
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct DrawInstancedArguments
    {
        public int VertexCountPerInstance;
        public int InstanceCount;
        public int StartVertexLocation;
        public int StartInstanceLocation;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct DrawIndexedInstancedArguments
    {
        public int IndexCountPerInstance;
        public int InstanceCount;
        public int StartIndexLocation;
        public int BaseVertexLocation;
        public int StartInstanceLocation;
    }
}
