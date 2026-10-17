using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// GPU-driven instance batch. All command generation happens on the GPU.
    /// Flow: unsorted draws â†’ GPU frustum culling â†’ cache-friendly grouping â†’ indirect commands
    /// </summary>
    public sealed class InstanceBatch : IDisposable
    {
        public const int FrameCount = 3;

        // Deferred disposal queue â€” old GPU buffers kept alive until GPU is done with them
        private static readonly List<(ID3D12Resource Resource, uint BindlessIndex, int FrameToDispose)> _deferredDisposals = new();

        /// <summary>
        /// Call once per frame (after Present) to release GPU resources from previous resizes.
        /// </summary>
        public static void FlushDeferredDisposals()
        {
            int currentFrame = Engine.TickCount;
            for (int i = _deferredDisposals.Count - 1; i >= 0; i--)
            {
                if (currentFrame >= _deferredDisposals[i].FrameToDispose)
                {
                    _deferredDisposals[i].Resource?.Dispose();
                    if (_deferredDisposals[i].BindlessIndex != 0)
                        Engine.Device?.ReleaseBindlessIndex(_deferredDisposals[i].BindlessIndex);
                    _deferredDisposals.RemoveAt(i);
                }
            }
        }

        private static void DeferDispose(ID3D12Resource? resource, uint bindlessIndex)
        {
            if (resource == null && bindlessIndex == 0) return;
            _deferredDisposals.Add((resource, bindlessIndex, Engine.TickCount + FrameCount + 1));
        }
        
        public BatchKey Key;
        public Material Material;

        public struct RawDraw
        {
            public Mesh Mesh;
            public int PartIndex;
            public MaterialBlock Block;
            public int TransformSlot;
            public uint MaterialId;
            public int MeshPartId;
        }

        internal int _activeFrame = -1;
        private RawDraw[] _draws = Array.Empty<RawDraw>();
        private HashSet<int> _uniqueMeshPartIds = new HashSet<int>();

        public int DrawCount => _drawCount;

        /// <summary>The batch's effect, for profiler regions (GPU) and markers (CPU, with the instance count).</summary>
        internal string ProfileName => _profileName ??= Key.Effect?.Name ?? Material?.Name ?? "Batch";
        internal ProfilerMarker ProfileMarker => _profileMarker ??= ProfilerMarker.Get(ProfileName);
        private string? _profileName;
        private ProfilerMarker? _profileMarker;
        public int SubBatchCount { get; private set; }

        private int capacity = 0;

        
        /// <summary>
        /// Per-instance GPU buffer for a single parameter (SoA pattern).
        /// Self-describing: carries its own PushConstantSlot and stride.
        /// </summary>
        private class PerInstanceBuffer
        {
            public int ParamHash;
            public int PushConstantSlot; // which push constant slot to bind SRV to
            public ID3D12Resource[] Buffers = new ID3D12Resource[FrameCount];
            public uint[] SRVIndices = new uint[FrameCount];
            public int Capacity; // in slots
            public int ElementsPerInstance; // e.g. numBones
            public int ElementStride; // bytes per element
            public bool Dirty;
            
            // CPU staging buffer â€” filled during MergeFromBucket, uploaded in one MemoryCopy
            public byte[] StagingData = Array.Empty<byte>();
            public int StagingCapacity; // in slots (matches Capacity growth)
        }
        
        // Active per-instance buffers for this batch
        private Dictionary<int, PerInstanceBuffer> _perInstanceBuffers = new();
        
        // Hardcoded compute push constant slots for per-instance buffers.
        // These are core pipeline constants â€” compute layout is fixed in cull_instances.hlsl.
        // Graphics slots (from shader resource bindings) are separate 
        private static readonly Dictionary<int, int> ComputeSlots = new()
        {
            // BoundingSpheres removed â€” culler reads bounds from MeshRegistry
            // Bones removed â€” bone data is now per-instance via InstanceDescriptor.BoneBufferIdx
            { DrawBucket.DescriptorsHash, 4 },            // Indices[1].x = DescriptorBufferIdx
            { DrawBucket.SubbatchIdsHash, 23 },           // Indices[5].w = SubbatchIdsIdx
        };
        
        // Cached array to avoid per-frame allocation
        private ID3D12DescriptorHeap[]? _cachedSrvHeapArray;

        // GPU output: visible instance indices (after culling)
        private ID3D12Resource[] visibleIndicesBuffers = new ID3D12Resource[FrameCount];
        private uint[] visibleIndicesUAVIndices = new uint[FrameCount];
        private uint[] visibleIndicesSRVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] visibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount];
        

        
        // GPU output: indirect commands (compute shader writes)
        private ID3D12Resource[] gpuCommandBuffers = new ID3D12Resource[FrameCount];
        private uint[] gpuCommandUAVIndices = new uint[FrameCount];

        // Per-subbatch counters (atomic increments during culling)
        private ID3D12Resource[] counterBuffers = new ID3D12Resource[FrameCount];
        private uint[] counterBufferUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] counterBufferCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // Visibility flags buffer (1=visible, 0=not)
        private ID3D12Resource[] visibilityFlagsBuffers = new ID3D12Resource[FrameCount];
        private uint[] visibilityFlagsUAVIndices = new uint[FrameCount];

        // World bounding sphere and LOD-resolved part per instance (InstanceBounds in cull_instances.hlsl), written by
        // the camera's CSVisibility and read by the cascade and point light shadow culling, which then gather no
        // descriptor, transform or mesh registry entry of their own. Valid once this frame's Cull has run (_boundsWritten).
        private const int InstanceBoundsStride = 20;
        private ID3D12Resource?[] instanceBoundsBuffers = new ID3D12Resource?[FrameCount];
        private uint[] instanceBoundsUAVIndices = new uint[FrameCount];
        private bool _boundsWritten;

        // The camera culling's transform cache (VisibilityCacheEntry in cull_instances.hlsl): each instance's world
        // matrix, transform slot, subbatch id and skinning, in instance order and in video memory. CSVisibility gathers
        // an instance's descriptor, subbatch id and transform (upload heaps, read across PCIe) again only when its slot
        // was uploaded this tick, its persistent record changed (_changedRecordBits) or it was enqueued this frame.
        // One buffer, not one per frame: only the GPU writes it, in queue order. _visibilityCacheTick: the tick of the
        // last camera culling (-1 = new buffer).
        private const int VisibilityCacheStride = 64;
        private ID3D12Resource? visibilityCacheBuffer;
        private uint visibilityCacheUAVIndex;
        private int _visibilityCacheTick = -1;

        // One bit per persistent record changed since the last camera culling (added, swap-removed into its index,
        // skinned or no longer), uploaded for that culling (ChangedRecordsIdx) and cleared
        private uint[] _changedRecordBits = Array.Empty<uint>();
        private bool _anyRecordChanged;
        private StreamingBuffer<uint>? _changedRecords;
        private uint[] visibilityFlagsSRVIndices = new uint[FrameCount]; // SRV for reading previous frame's flags
        
        private const int MaxSubBatches = 4096;
        private bool _cullerInitialized = false;
        private ID3D12DescriptorHeap? _cpuHeap;
        
        // Histogram buffer for GPU-driven per-MeshPartId instance counting
        private ID3D12Resource[] histogramBuffers = new ID3D12Resource[FrameCount];
        private uint[] histogramUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] histogramCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // Draw count buffer for ExecuteIndirect count parameter (atomically incremented by CSMain)
        private ID3D12Resource[] drawCountBuffers = new ID3D12Resource[FrameCount];
        private uint[] drawCountUAVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] drawCountCPUHandles = new CpuDescriptorHandle[FrameCount];
        
        // Shadow culling â€” union visibility + cascade mask (single-pass rendering)


        private ID3D12Resource[] shadowCombinedVisBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCombinedVisUAVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowCascadeMaskBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCascadeMaskUAVIndices = new uint[FrameCount];
        private uint[] shadowCascadeMaskSRVIndices = new uint[FrameCount]; // For VS cascade early-out
        private ID3D12Resource[] shadowVisibleIndicesBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowVisibleIndicesUAVIndices = new uint[FrameCount];
        private uint[] shadowVisibleIndicesSRVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowCommandBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCommandUAVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowCounterBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowCounterUAVIndices = new uint[FrameCount];
        private ID3D12Resource[] shadowHistogramBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowHistogramUAVIndices = new uint[FrameCount];
        private bool _shadowCullerInitialized = false;
        private ID3D12DescriptorHeap? _shadowCpuHeap;
        private CpuDescriptorHandle[] shadowCounterCPUHandles = new CpuDescriptorHandle[FrameCount];
        private CpuDescriptorHandle[] shadowVisibleIndicesCPUHandles = new CpuDescriptorHandle[FrameCount];
        private CpuDescriptorHandle[] shadowHistogramCPUHandles = new CpuDescriptorHandle[FrameCount];

        // Shadow cascade expansion buffer — stores (cascadeIdx<<30 | instanceIdx) per visible cascade
        private ID3D12Resource[] shadowExpansionBuffers = new ID3D12Resource[FrameCount];
        private uint[] shadowExpansionUAVIndices = new uint[FrameCount];
        private uint[] shadowExpansionSRVIndices = new uint[FrameCount];
        private CpuDescriptorHandle[] shadowExpansionCPUHandles = new CpuDescriptorHandle[FrameCount];

        // Point light shadows: per-instance view mask (uint4 = 128 views), made by the first CullLocalShadowAll
        private ID3D12Resource?[] localShadowMaskBuffers = new ID3D12Resource?[FrameCount];
        private uint[] localShadowMaskUAVIndices = new uint[FrameCount];

        // Depth sort keys buffer — uint2 per instance: (compositeKey, instanceIdx)
        private ID3D12Resource[] sortKeysBuffers = new ID3D12Resource[FrameCount];
        private uint[] sortKeysUAVIndices = new uint[FrameCount];


        private int _drawCount = 0;
        private int _maxTransformSlot = 0;
        internal bool _isGPUSourced = false;

        public PrimitiveTopology Topology { get; private set; }

        public InstanceBatch(BatchKey key, Material material)
        {
            Key = key;
            Material = material;
            Topology = material.HasTessellation
                        ? PrimitiveTopology.PatchListWith3ControlPoints
                        : PrimitiveTopology.TriangleList;

            ResizeBuffers(8192);
        }

        /// <summary>
        /// Start a new frame. Persistent instances occupy the front of every per-instance array
        /// ([0, PersistentCount)) and are kept; this frame's enqueued draws are appended after them.
        /// </summary>
        public void Clear()
        {
            _drawCount = _persistentCount;
            _uniqueMeshPartIds.Clear();
            SubBatchCount = 0;
            _maxTransformSlot = 0;
            _isGPUSourced = false;
            _boundsWritten = false;
            // The staging arrays are re-uploaded in full every frame (one memcpy per channel), so the
            // triple-buffered GPU copies never need per-slot dirty tracking.
            foreach (var buf in _perInstanceBuffers.Values)
                buf.Dirty = _persistentCount > 0;
        }

        /// <summary>
        /// Mark the batch as having nothing to draw this frame. Called for every batch that was not
        /// activated in a frame. The shadow pass walks ALL batches of the opaque pass, active or not
        /// (CommandBuffer.GetAllBatches), guarded only by SubBatchCount; without this an inactive
        /// batch keeps the instance count and GPU buffers of the last frame it was active in, and the
        /// shadow pass draws those stale records: instances of destroyed entities, pointing at
        /// released bone buffers, transform slots and mesh buffers (GPU device removed).
        /// </summary>
        public void Deactivate()
        {
            // Engine.FrameIndex is the back buffer index and repeats every few frames, so a stale
            // _activeFrame could otherwise match again and make the batch look already activated.
            _activeFrame = -1;
            _drawCount = 0;
            SubBatchCount = 0;
            _uniqueMeshPartIds.Clear();
            _isGPUSourced = false;
            _boundsWritten = false;
        }

        #region Persistent instances (GPU-resident draws)

        /// <summary>
        /// One persistent instance record in a batch. The batch patches Index when a swap-remove moves the record.
        /// </summary>
        public sealed class InstanceHandle
        {
            public InstanceBatch Batch { get; internal set; }
            public int Index { get; internal set; }

            internal InstanceHandle(InstanceBatch batch, int index)
            {
                Batch = batch;
                Index = index;
            }
        }

        private int _persistentCount;
        private InstanceHandle?[] _handles = Array.Empty<InstanceHandle?>();

        /// <summary>Number of persistent instance records (registered once, not re-enqueued per frame).</summary>
        public int PersistentCount => _persistentCount;

        /// <summary>
        /// Add a persistent instance record. Main thread, and only before this frame's bucket merge
        /// (CommandBuffer.FlushPersistent): the record goes at index PersistentCount, where the
        /// per-frame draws would otherwise start.
        /// </summary>
        internal InstanceHandle AddPersistent(in DrawCall call)
        {
            int index = _persistentCount;
            if (index + 1 > capacity)
                ResizeBuffers(Math.Max(capacity * 2, index + 1));

            int meshPartId = call.Mesh.GetMeshPartId(call.MeshPartIndex);
            if (meshPartId < 0)
            {
                call.Mesh.RegisterMeshParts();
                meshPartId = call.Mesh.GetMeshPartId(call.MeshPartIndex);
            }

            _draws[index] = new RawDraw
            {
                Mesh = call.Mesh,
                PartIndex = call.MeshPartIndex,
                Block = call.MaterialBlock,
                TransformSlot = call.TransformSlot,
                MaterialId = (uint)call.Material.MaterialID,
                MeshPartId = meshPartId,
            };

            // Optional channels this record doesn't supply read as zero, never as a previous record's data
            foreach (var pib in _perInstanceBuffers.Values)
            {
                int bytes = pib.ElementsPerInstance * pib.ElementStride;
                if (bytes == 0) continue;
                EnsureStaging(pib, index + 1);
                Array.Clear(pib.StagingData, index * bytes, bytes);
            }

            var descriptor = new InstanceDescriptor
            {
                TransformSlot = (uint)call.TransformSlot,
                MaterialId = (uint)call.Material.MaterialID,
                CustomDataIdx = (uint)(call.InstanceSlot > 0 ? call.InstanceSlot : call.Material.Effect?.DefaultInstanceSlot ?? 0),
                MeshPartIdx = (uint)call.MeshPartIndex,
                BoneBufferIdx = call.BoneBufferIdx
            };
            WriteCore(DrawBucket.DescriptorsHash, 20, index, descriptor);
            WriteCore(DrawBucket.SubbatchIdsHash, 4, index, (uint)meshPartId | (call.LodManaged ? DrawBucket.LodManagedBit : 0u));

            if (call.MaterialBlock != null)
            {
                foreach (var (hash, param) in call.MaterialBlock.Parameters)
                {
                    if (param is TextureParameterValue) continue;

                    if (param.PushConstantSlot < 0 && call.Material.Effect != null)
                        param.PushConstantSlot = call.Material.Effect.GetPushConstantSlot(hash);

                    int elemCount = param.GetElementCount();
                    int elemStride = param.GetElementStride();
                    if (elemCount == 0 || elemStride == 0) continue;

                    if (!_perInstanceBuffers.TryGetValue(hash, out var pib))
                    {
                        pib = new PerInstanceBuffer
                        {
                            ParamHash = hash,
                            PushConstantSlot = param.PushConstantSlot,
                            ElementStride = elemStride,
                            ElementsPerInstance = elemCount
                        };
                        _perInstanceBuffers[hash] = pib;
                    }

                    EnsureStaging(pib, index + 1);
                    param.CopyToStaging(pib.StagingData, index * pib.ElementsPerInstance * pib.ElementStride);
                }
            }

            var handle = new InstanceHandle(this, index);
            _handles[index] = handle;
            _persistentCount++;
            MarkRecordChanged(index);
            return handle;
        }

        /// <summary>
        /// Remove a persistent instance record (swap-remove: the last persistent record takes its place).
        /// Main thread, before this frame's bucket merge.
        /// </summary>
        internal void RemovePersistent(InstanceHandle handle)
        {
            int index = handle.Index;
            if (handle.Batch != this || index < 0 || index >= _persistentCount) return;

            int last = _persistentCount - 1;
            if (index != last)
            {
                _draws[index] = _draws[last];

                foreach (var pib in _perInstanceBuffers.Values)
                {
                    int bytes = pib.ElementsPerInstance * pib.ElementStride;
                    if (bytes == 0 || pib.StagingData.Length < (last + 1) * bytes) continue;
                    Buffer.BlockCopy(pib.StagingData, last * bytes, pib.StagingData, index * bytes, bytes);
                }

                var moved = _handles[last]!;
                moved.Index = index;
                _handles[index] = moved;
                MarkRecordChanged(index);
            }

            _handles[last] = null;
            _draws[last] = default;
            _persistentCount--;
            handle.Index = -1;
        }

        /// <summary>
        /// Overwrite the bone buffer SRV index of a persistent record in place (CommandBuffer.SetBoneBuffer).
        /// </summary>
        internal unsafe void SetPersistentBoneBuffer(InstanceHandle handle, uint boneBufferIdx)
        {
            int index = handle.Index;
            if (handle.Batch != this || index < 0 || index >= _persistentCount) return;
            if (!_perInstanceBuffers.TryGetValue(DrawBucket.DescriptorsHash, out var pib)) return;

            int offset = index * sizeof(InstanceDescriptor);
            if (pib.StagingData.Length < offset + sizeof(InstanceDescriptor)) return;

            fixed (byte* ptr = &pib.StagingData[offset])
            {
                var descriptor = (InstanceDescriptor*)ptr;
                // The camera culling's transform cache keeps whether the instance is skinned, not the index
                if ((descriptor->BoneBufferIdx != 0) != (boneBufferIdx != 0))
                    MarkRecordChanged(index);
                descriptor->BoneBufferIdx = boneBufferIdx;
            }
        }

        // A persistent record's index whose content changed: the camera culling's transform cache gathers it again.
        // Atomic: SetPersistentBoneBuffer runs in the parallel update phase. The bits are sized with the capacity.
        private void MarkRecordChanged(int index)
        {
            System.Threading.Interlocked.Or(ref _changedRecordBits[index >> 5], 1u << (index & 31));
            _anyRecordChanged = true;
        }

        private static void EnsureStaging(PerInstanceBuffer pib, int slots)
        {
            int required = slots * pib.ElementsPerInstance * pib.ElementStride;
            if (pib.StagingData.Length < required)
                Array.Resize(ref pib.StagingData, Math.Max(pib.StagingData.Length * 2, required));
        }

        private unsafe void WriteCore<T>(int hash, int stride, int index, T value) where T : unmanaged
        {
            if (!_perInstanceBuffers.TryGetValue(hash, out var pib))
            {
                pib = new PerInstanceBuffer
                {
                    ParamHash = hash,
                    PushConstantSlot = -1, // Core data — bound by InstanceBatch, not shader
                    ElementStride = stride,
                    ElementsPerInstance = 1
                };
                _perInstanceBuffers[hash] = pib;
            }

            EnsureStaging(pib, index + 1);
            fixed (byte* ptr = &pib.StagingData[index * stride])
                *(T*)ptr = value;
        }

        #endregion

        /// <summary>
        /// Descriptor for a GPU-generated per-instance buffer channel.
        /// </summary>
        public struct GPUBufferBinding
        {
            public int ParamHash;
            public int PushConstantSlot;
            public int ElementStride;
            public uint SrvIndex;
        }

        /// <summary>
        /// Attach GPU-generated per-instance buffers (bypasses CPU staging/upload).
        /// The provided SRV indices point at buffers filled by a compute shader.
        /// descriptorsSRV/subbatchIdsSRV are core culler infrastructure.
        /// customBindings contains additional per-instance data channels (e.g. TerrainPatchData).
        /// </summary>
        public void AttachGPUData(
            uint descriptorsSRV, uint subbatchIdsSRV,
            int instanceCount, int meshPartId,
            ReadOnlySpan<GPUBufferBinding> customBindings = default)
        {
            _isGPUSourced = true;
            _drawCount = instanceCount;
            
            // Ensure capacity is large enough for culler buffers
            if (instanceCount > capacity)
                ResizeBuffers(Math.Max(instanceCount, capacity * 2));
            
            // Register the mesh part so MeshRegistry has it
            _uniqueMeshPartIds.Add(meshPartId);
            SubBatchCount = _uniqueMeshPartIds.Count;
            
            int frameIndex = Engine.FrameIndex % FrameCount;
            
            // Core culler infrastructure buffers (bounding spheres now come from MeshRegistry)
            EnsureGPUPerInstanceBuffer(DrawBucket.DescriptorsHash, -1, 16, 1, frameIndex, descriptorsSRV);
            EnsureGPUPerInstanceBuffer(DrawBucket.SubbatchIdsHash, -1, 4, 1, frameIndex, subbatchIdsSRV);
            
            // Custom per-instance data channels (caller provides hash, slot, stride)
            foreach (ref readonly var binding in customBindings)
            {
                EnsureGPUPerInstanceBuffer(binding.ParamHash, binding.PushConstantSlot,
                    binding.ElementStride, 1, frameIndex, binding.SrvIndex);
            }
        }

        private void EnsureGPUPerInstanceBuffer(int hash, int pushConstantSlot, int elementStride, int elementsPerInstance, int frameIndex, uint srvIndex)
        {
            if (!_perInstanceBuffers.TryGetValue(hash, out var pib))
            {
                pib = new PerInstanceBuffer
                {
                    ParamHash = hash,
                    PushConstantSlot = pushConstantSlot,
                    ElementStride = elementStride,
                    ElementsPerInstance = elementsPerInstance,
                };
                _perInstanceBuffers[hash] = pib;
            }
            // Override the SRV index for this frame with the GPU-generated buffer
            pib.SRVIndices[frameIndex] = srvIndex;
            pib.Dirty = false; // Don't upload CPU data
        }


        public void MergeFromBucket(DrawBucket bucket)
        {
            if (bucket.Count == 0) return;
            
            int startIdx = _drawCount;
            int count = bucket.Count;
            int newTotal = startIdx + count;
            
            // Ensure capacity for all new draws
            if (newTotal > capacity)
                ResizeBuffers(Math.Max(capacity * 2, newTotal));
            
            // Block-copy RawDraws using Span (slice to actual count)
            var srcDraws = CollectionsMarshal.AsSpan(bucket.Draws).Slice(0, count);
            srcDraws.CopyTo(_draws.AsSpan(startIdx));
            
            foreach (var id in bucket.UniqueMeshPartIds)
                _uniqueMeshPartIds.Add(id);
            SubBatchCount = _uniqueMeshPartIds.Count;
            
            // Block-copy all pre-staged per-instance data from bucket (staged at Enqueue time)
            // This includes core channels (Descriptors, SubbatchIds) and
            // optional channels (TerrainPatchData, PointLightData, Bones, etc.)
            foreach (var (hash, staging) in bucket.PerInstanceData)
            {
                if (staging.Count == 0) continue;
                
                if (!_perInstanceBuffers.TryGetValue(hash, out var pib))
                {
                    pib = new PerInstanceBuffer
                    {
                        ParamHash = hash,
                        PushConstantSlot = staging.PushConstantSlot,
                        ElementStride = staging.ElementStride,
                        ElementsPerInstance = staging.ElementsPerInstance
                    };
                    _perInstanceBuffers[hash] = pib;
                }
                
                // Ensure staging capacity (dense, indexed by instance)
                int bytesPerInst = staging.BytesPerInstance;
                int requiredBytes = newTotal * bytesPerInst;
                if (pib.StagingData.Length < requiredBytes)
                {
                    int newCap = Math.Max(pib.StagingData.Length * 2, requiredBytes);
                    Array.Resize(ref pib.StagingData, newCap);
                }
                
                // Block-copy from bucket staging into batch staging
                int srcBytes = staging.Count * bytesPerInst;
                Buffer.BlockCopy(staging.Data, 0, pib.StagingData, startIdx * bytesPerInst, srcBytes);
                pib.Dirty = true;
            }
            
            // Track max transform slot for bone buffer sizing
            for (int i = startIdx; i < newTotal; i++)
            {
                if (_draws[i].TransformSlot > _maxTransformSlot)
                    _maxTransformSlot = _draws[i].TransformSlot;
            }
            
            _drawCount = newTotal;
        }

        private void ResizeBuffers(int newCapacity)
        {
            if (newCapacity <= capacity) return;
            capacity = newCapacity;

            // Defer disposal of old GPU buffers instead of immediate destroy
            DeferDisposeBuffers();

            Array.Resize(ref _draws, capacity);
            Array.Resize(ref _handles, capacity);
            Array.Resize(ref _changedRecordBits, (capacity + 31) / 32);

            int slotSize = capacity * 4;
            int commandSize = MaxSubBatches * IndirectDrawSizes.IndirectCommandSize;

            for (int i = 0; i < FrameCount; ++i)
            {
                // GPU output: visible indices (compute shader writes)
                visibleIndicesBuffers[i] = Engine.Device.CreateDefaultBuffer(slotSize);
                visibleIndicesUAVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferUAV(visibleIndicesBuffers[i], (uint)capacity, 4, visibleIndicesUAVIndices[i]);
                visibleIndicesSRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(visibleIndicesBuffers[i], (uint)capacity, 4, visibleIndicesSRVIndices[i]);

                // GPU output: indirect commands (compute shader writes)
                gpuCommandBuffers[i] = Engine.Device.CreateDefaultBuffer(commandSize);
                gpuCommandUAVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferUAV(gpuCommandBuffers[i], MaxSubBatches,
                    (uint)IndirectDrawSizes.IndirectCommandSize, gpuCommandUAVIndices[i]);
            }
        }

        public void UploadInstanceData(GraphicsDevice device)
        {
            if (_drawCount == 0 || _isGPUSourced) return;
            int frameIndex = Engine.FrameIndex % FrameCount;

            // All per-instance data (descriptors, bounding spheres, subbatch IDs,
            // terrain data, bones, etc.) is now uploaded through the unified path.
            UploadPerInstanceBuffers(frameIndex);
        }

        private void UploadPerInstanceBuffers(int frameIndex)
        {
            if (_drawCount == 0) return;
            int requiredSlots = _drawCount; // Dense: per-instance data is indexed by instance, not TransformSlot

            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (!pib.Dirty) continue;
                if (pib.ElementStride == 0 || pib.ElementsPerInstance == 0) continue;

                // Ensure GPU buffer capacity
                EnsurePerInstanceBuffer(pib, requiredSlots);

                // Single MemoryCopy from pre-staged byte array â€” no iteration, no reflection
                int bytesPerSlot = pib.ElementsPerInstance * pib.ElementStride;
                int uploadBytes = requiredSlots * bytesPerSlot;
                if (uploadBytes > pib.StagingData.Length)
                    uploadBytes = pib.StagingData.Length;

                var buf = pib.Buffers[frameIndex];
                unsafe
                {
                    byte* pData;
                    buf.Map(0, null, (void**)&pData);
                    fixed (byte* pSrc = pib.StagingData)
                        Buffer.MemoryCopy(pSrc, pData, uploadBytes, uploadBytes);
                    buf.Unmap(0);
                }
            }
        }

        private void EnsurePerInstanceBuffer(PerInstanceBuffer pib, int requiredSlots)
        {
            if (pib.Capacity >= requiredSlots) return;

            // Defer disposal of old buffers â€” GPU may still be reading them from in-flight frames
            for (int i = 0; i < FrameCount; ++i)
            {
                DeferDispose(pib.Buffers[i], pib.SRVIndices[i]);
            }

            int totalElements = requiredSlots * pib.ElementsPerInstance;
            int bufferSize = totalElements * pib.ElementStride;

            for (int i = 0; i < FrameCount; ++i)
            {
                pib.Buffers[i] = Engine.Device.CreateUploadBuffer(bufferSize);
                pib.SRVIndices[i] = Engine.Device.AllocateBindlessIndex();
                Engine.Device.CreateStructuredBufferSRV(
                    pib.Buffers[i],
                    (uint)totalElements,
                    (uint)pib.ElementStride,
                    pib.SRVIndices[i]);
            }

            pib.Capacity = requiredSlots;
        }

        public void Build(GraphicsDevice device, ID3D12GraphicsCommandList commandList, ulong frustumBufferAddress)
        {
            int count = _drawCount;
            if (count == 0) return;

            MeshRegistry.Upload(device);
            if (!_isGPUSourced)
                SubBatchCount = MeshRegistry.Count;
        }

        #region GPU Culler

        private void InitializeCullerResources()
        {
            if (_cullerInitialized) return;
            _cullerInitialized = true;

            var device = Engine.Device;
            int counterBufferSize = MaxSubBatches * sizeof(uint);
            int instanceBufferSize = capacity * sizeof(uint);

            // Zeroed on creation: every entry reads as never written, and the first culling refreshes them all
            visibilityCacheBuffer = device.CreateDefaultBuffer(capacity * VisibilityCacheStride);
            visibilityCacheUAVIndex = device.AllocateBindlessIndex();
            device.CreateStructuredBufferUAV(visibilityCacheBuffer, (uint)capacity, VisibilityCacheStride, visibilityCacheUAVIndex);
            _visibilityCacheTick = -1;

            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = FrameCount * 4, // counter, visibleIndices, histogram, drawCount
                Flags = DescriptorHeapFlags.None
            };
            _cpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            uint handleIncrementSize = device.NativeDevice.GetDescriptorHandleIncrementSize(
                DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);

            for (int i = 0; i < FrameCount; i++)
            {
                counterBuffers[i] = device.CreateDefaultBuffer(counterBufferSize);
                counterBufferUAVIndices[i] = device.AllocateBindlessIndex();

                var uavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = MaxSubBatches,
                        StructureByteStride = sizeof(uint),
                        Flags = BufferUnorderedAccessViewFlags.None
                    }
                };
                device.NativeDevice.CreateUnorderedAccessView(
                    counterBuffers[i], null, uavDesc, device.GetCpuHandle(counterBufferUAVIndices[i]));

                var cpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)(i * handleIncrementSize));
                device.NativeDevice.CreateUnorderedAccessView(counterBuffers[i], null, uavDesc, cpuHandle);
                counterBufferCPUHandles[i] = cpuHandle;

                var visibleIndicesCpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)((FrameCount + i) * handleIncrementSize));
                var visibleUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = (uint)capacity,
                        StructureByteStride = sizeof(uint),
                        Flags = BufferUnorderedAccessViewFlags.None
                    }
                };
                device.NativeDevice.CreateUnorderedAccessView(visibleIndicesBuffers[i], null, visibleUavDesc, visibleIndicesCpuHandle);
                visibleIndicesCPUHandles[i] = visibleIndicesCpuHandle;

                visibilityFlagsBuffers[i] = device.CreateDefaultBuffer(instanceBufferSize);
                visibilityFlagsUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(visibilityFlagsBuffers[i], (uint)capacity, sizeof(uint), 
                    visibilityFlagsUAVIndices[i]);
                // SRV for reading previous frame's visibility flags (feedback loop breaker)
                visibilityFlagsSRVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(visibilityFlagsBuffers[i], (uint)capacity, sizeof(uint),
                    visibilityFlagsSRVIndices[i]);

                instanceBoundsBuffers[i] = device.CreateDefaultBuffer(capacity * InstanceBoundsStride);
                instanceBoundsUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(instanceBoundsBuffers[i]!, (uint)capacity, InstanceBoundsStride,
                    instanceBoundsUAVIndices[i]);

                int histogramSize = MeshRegistry.MaxMeshParts * sizeof(uint);
                histogramBuffers[i] = device.CreateDefaultBuffer(histogramSize);
                histogramUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(histogramBuffers[i], (uint)MeshRegistry.MaxMeshParts, sizeof(uint),
                    histogramUAVIndices[i]);
                
                var histogramCpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)((2 * FrameCount + i) * handleIncrementSize));
                var histogramUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = (uint)MeshRegistry.MaxMeshParts,
                        StructureByteStride = sizeof(uint),
                        Flags = BufferUnorderedAccessViewFlags.None
                    }
                };
                device.NativeDevice.CreateUnorderedAccessView(histogramBuffers[i], null, histogramUavDesc, histogramCpuHandle);
                histogramCPUHandles[i] = histogramCpuHandle;
                
                // Draw count buffer for ExecuteIndirect count parameter
                drawCountBuffers[i] = device.CreateDefaultBuffer(sizeof(uint));
                drawCountUAVIndices[i] = device.AllocateBindlessIndex();
                var drawCountUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView
                    {
                        FirstElement = 0,
                        NumElements = 1,
                        StructureByteStride = sizeof(uint),
                        Flags = BufferUnorderedAccessViewFlags.None
                    }
                };
                device.NativeDevice.CreateUnorderedAccessView(
                    drawCountBuffers[i], null, drawCountUavDesc, device.GetCpuHandle(drawCountUAVIndices[i]));
                
                var drawCountCpuHandle = _cpuHeap.GetCPUDescriptorHandleForHeapStart() + ((int)((3 * FrameCount + i) * handleIncrementSize));
                device.NativeDevice.CreateUnorderedAccessView(drawCountBuffers[i], null, drawCountUavDesc, drawCountCpuHandle);
                drawCountCPUHandles[i] = drawCountCpuHandle;
                
                // Depth sort keys buffer: uint2 per entry (8 bytes) for GPU distance sorting
                // Bitonic sort needs power-of-2 count, so round up to avoid OOB writes
                int sortKeysCap = 1;
                while (sortKeysCap < capacity) sortKeysCap <<= 1;
                int sortKeysSize = sortKeysCap * 8; // sizeof(uint2) = 8
                sortKeysBuffers[i] = device.CreateDefaultBuffer(sortKeysSize);
                sortKeysUAVIndices[i] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(sortKeysBuffers[i], (uint)sortKeysCap, 8, sortKeysUAVIndices[i]);
            }
        }

        /// <summary>
        /// Push constants 33-37 for GPU LOD selection (cull_instances.hlsl ResolveLodPart).
        /// Always the main camera, for every pass including shadows, so all passes resolve the same LOD.
        /// </summary>
        private static void WriteLodConstants(uint[] constants)
        {
            var (position, factor) = LodCamera();
            constants[33] = MeshRegistry.LodSrvIndex;
            constants[34] = BitConverter.SingleToUInt32Bits(position.X);
            constants[35] = BitConverter.SingleToUInt32Bits(position.Y);
            constants[36] = BitConverter.SingleToUInt32Bits(position.Z);
            constants[37] = BitConverter.SingleToUInt32Bits(factor);
        }

        // The main camera's position and LOD factor (0 = no valid camera: LOD-managed instances are not drawn this frame)
        private static (Vector3 Position, float Factor) LodCamera()
        {
            var camera = Components.Camera.Main;
            if (camera == null)
                return (default, 0f);
            float scale = Engine.Settings.LODScale;
            return (camera.Position, camera.FoVFactor * scale * scale);
        }

        /// <summary>
        /// GPU depth sort of the visible indices by camera distance (pass 4b of Cull), keeping the MeshPartId grouping
        /// (composite key meshPartId &lt;&lt; 20 | distBits). The batch's cull constants must be set.
        /// </summary>
        private void RecordDepthSort(ID3D12GraphicsCommandList commandList, GPUCuller culler)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            int totalInstances = _drawCount;

            // Round up to next power of 2 for bitonic sort
            uint sortN = 1;
            while (sortN < (uint)totalInstances) sortN <<= 1;

            // Bind sort keys buffer
            commandList.SetComputeRoot32BitConstant(0, sortKeysUAVIndices[frameIndex], 32);

            // Pass 4b.1: CSComputeDepthKeys — pack (compositeKey, instanceIdx) into sort buffer
            // Dispatch for sortN entries (not totalInstances) so the entire power-of-2 range
            // is initialized. Entries >= TotalInstances get 0xFFFFFFFF padding keys.
            commandList.SetPipelineState(culler.ComputeDepthKeysPSO);
            commandList.Dispatch((sortN + 255) / 256, 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(sortKeysBuffers[frameIndex])));

            // Pass 4b.2: CSBitonicSortPairs — full bitonic sort on uint2 entries by .x key
            commandList.SetPipelineState(culler.BitonicSortPairsPSO);
            commandList.SetComputeRoot32BitConstant(0, sortN, 22); // SortCount = padded count

            for (uint blockSize = 2; blockSize <= sortN; blockSize <<= 1)
            {
                for (uint stride = blockSize >> 1; stride > 0; stride >>= 1)
                {
                    commandList.SetComputeRoot32BitConstant(0, blockSize, 20); // SortSize
                    commandList.SetComputeRoot32BitConstant(0, stride, 21);    // SortStride

                    uint threadCount = sortN / 2;
                    commandList.Dispatch((threadCount + 255) / 256, 1, 1);
                    commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(sortKeysBuffers[frameIndex])));
                }
            }

            // Pass 4b.3: CSWriteSortedIndices — unpack sorted indices back to visibleIndices
            commandList.SetPipelineState(culler.WriteSortedIndicesPSO);
            commandList.Dispatch((uint)((totalInstances + 255) / 256), 1, 1);
            commandList.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(visibleIndicesBuffers[frameIndex])));
        }

        private void InitializeShadowCullerResources()
        {
            if (_shadowCullerInitialized) return;
            _shadowCullerInitialized = true;

            var device = Engine.Device;
            int counterBufferSize = MaxSubBatches * sizeof(uint);
            int instanceBufferSize = capacity * sizeof(uint);
            int commandSize = MaxSubBatches * IndirectDrawSizes.IndirectCommandSize;
            int histogramSize = MeshRegistry.MaxMeshParts * sizeof(uint);

            // CPU descriptor heap for ClearUnorderedAccessViewUint (4 buffers per frame: counter, visible indices, histogram, expansion)
            int totalShadowDescriptors = FrameCount * 4;
            var cpuHeapDesc = new DescriptorHeapDescription
            {
                Type = DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView,
                DescriptorCount = (uint)totalShadowDescriptors,
                Flags = DescriptorHeapFlags.None
            };
            _shadowCpuHeap = device.NativeDevice.CreateDescriptorHeap(cpuHeapDesc);
            uint handleIncrement = device.NativeDevice.GetDescriptorHandleIncrementSize(
                DescriptorHeapType.ConstantBufferViewShaderResourceViewUnorderedAccessView);
            int descriptorIdx = 0;

            for (int f = 0; f < FrameCount; f++)
            {
                // Combined visibility flags (union of all cascades)
                shadowCombinedVisBuffers[f] = device.CreateDefaultBuffer(instanceBufferSize);
                shadowCombinedVisUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCombinedVisBuffers[f], (uint)capacity, sizeof(uint), shadowCombinedVisUAVIndices[f]);

                // Cascade mask (4-bit per instance, for VS early-out)
                shadowCascadeMaskBuffers[f] = device.CreateDefaultBuffer(instanceBufferSize);
                shadowCascadeMaskUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCascadeMaskBuffers[f], (uint)capacity, sizeof(uint), shadowCascadeMaskUAVIndices[f]);
                shadowCascadeMaskSRVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(shadowCascadeMaskBuffers[f], (uint)capacity, sizeof(uint), shadowCascadeMaskSRVIndices[f]);

                // Visible indices output (ONE buffer, union-compacted)
                shadowVisibleIndicesBuffers[f] = device.CreateDefaultBuffer(instanceBufferSize);
                shadowVisibleIndicesUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowVisibleIndicesBuffers[f], (uint)capacity, sizeof(uint), shadowVisibleIndicesUAVIndices[f]);
                shadowVisibleIndicesSRVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(shadowVisibleIndicesBuffers[f], (uint)capacity, sizeof(uint), shadowVisibleIndicesSRVIndices[f]);

                // CPU handle for visible indices clear
                var visibleCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var visibleUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)capacity, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowVisibleIndicesBuffers[f], null, visibleUavDesc, visibleCpuHandle);
                shadowVisibleIndicesCPUHandles[f] = visibleCpuHandle;
                descriptorIdx++;

                // Indirect commands (ONE buffer)
                shadowCommandBuffers[f] = device.CreateDefaultBuffer(commandSize);
                shadowCommandUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCommandBuffers[f], MaxSubBatches, (uint)IndirectDrawSizes.IndirectCommandSize, shadowCommandUAVIndices[f]);

                // Counters (ONE buffer)
                shadowCounterBuffers[f] = device.CreateDefaultBuffer(counterBufferSize);
                shadowCounterUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowCounterBuffers[f], MaxSubBatches, sizeof(uint), shadowCounterUAVIndices[f]);

                // CPU handle for counter clear
                var counterCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var counterUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = MaxSubBatches, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowCounterBuffers[f], null, counterUavDesc, counterCpuHandle);
                shadowCounterCPUHandles[f] = counterCpuHandle;
                descriptorIdx++;

                // Histogram (ONE buffer)
                shadowHistogramBuffers[f] = device.CreateDefaultBuffer(histogramSize);
                shadowHistogramUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowHistogramBuffers[f], (uint)MeshRegistry.MaxMeshParts, sizeof(uint), shadowHistogramUAVIndices[f]);

                // CPU handle for histogram clear
                var histogramCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var histogramUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)MeshRegistry.MaxMeshParts, StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowHistogramBuffers[f], null, histogramUavDesc, histogramCpuHandle);
                shadowHistogramCPUHandles[f] = histogramCpuHandle;
                descriptorIdx++;

                // Expansion buffer (capacity × 4 = worst case when every instance is in all 4 cascades)
                int expansionBufferSize = capacity * 4 * sizeof(uint);
                shadowExpansionBuffers[f] = device.CreateDefaultBuffer(expansionBufferSize);
                shadowExpansionUAVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferUAV(shadowExpansionBuffers[f], (uint)(capacity * 4), sizeof(uint), shadowExpansionUAVIndices[f]);
                shadowExpansionSRVIndices[f] = device.AllocateBindlessIndex();
                device.CreateStructuredBufferSRV(shadowExpansionBuffers[f], (uint)(capacity * 4), sizeof(uint), shadowExpansionSRVIndices[f]);

                // CPU handle for expansion clear
                var expansionCpuHandle = _shadowCpuHeap.GetCPUDescriptorHandleForHeapStart() + (int)(descriptorIdx * handleIncrement);
                var expansionUavDesc = new UnorderedAccessViewDescription
                {
                    Format = Vortice.DXGI.Format.Unknown,
                    ViewDimension = UnorderedAccessViewDimension.Buffer,
                    Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)(capacity * 4), StructureByteStride = sizeof(uint) }
                };
                device.NativeDevice.CreateUnorderedAccessView(shadowExpansionBuffers[f], null, expansionUavDesc, expansionCpuHandle);
                shadowExpansionCPUHandles[f] = expansionCpuHandle;
                descriptorIdx++;
            }
        }

        #region Culling all batches stage by stage

        // Root slot 0: 40 push constants (GraphicsDevice's global root signature)
        private const int CullConstantCount = 40;

        // This batch's push constants for the cull chain being recorded. The stage-major drivers below run one stage
        // for every batch before the next stage, so each dispatch sets its batch's constants again (one call).
        private readonly uint[] _cullConstants = new uint[CullConstantCount];

        // The batches of the chain being recorded (main thread only)
        private static readonly List<InstanceBatch> _stageBatches = new();
        private static ID3D12DescriptorHeap[]? _stageSrvHeaps;

        private enum StageItems { Instances, Parts, OneGroup }

        private unsafe void ApplyCullConstants(ID3D12GraphicsCommandList list)
        {
            fixed (uint* c = _cullConstants)
                list.SetComputeRoot32BitConstants(0, CullConstantCount, c, 0);
        }

        // Sets the per-instance buffers' SRVs (descriptors, subbatch ids) into the constants
        private void WritePerInstanceConstants(int frameIndex)
        {
            foreach (var (hash, pib) in _perInstanceBuffers)
                if (ComputeSlots.TryGetValue(hash, out int computeSlot))
                    _cullConstants[computeSlot] = pib.SRVIndices[frameIndex];
        }

        private void ClearUav(ID3D12GraphicsCommandList list, uint uavIndex, CpuDescriptorHandle cpuHandle, ID3D12Resource buffer, int value = 0) =>
            list.ClearUnorderedAccessViewUint(Engine.Device.GetGpuHandle(uavIndex), cpuHandle, buffer,
                new Vortice.Mathematics.Int4(value, value, value, value));

        // One barrier for all batches of a stage: every batch's next stage reads what its previous stage wrote
        private static void StageBarrier(ID3D12GraphicsCommandList list) =>
            list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

        // The stage's kernel once per batch, back to back with no barrier between them
        private static void DispatchStage(ID3D12GraphicsCommandList list, ID3D12PipelineState pso, StageItems items, uint groupSize = 1)
        {
            list.SetPipelineState(pso);
            foreach (var batch in _stageBatches)
            {
                batch.ApplyCullConstants(list);
                uint count = items == StageItems.Instances ? (uint)batch._drawCount : (uint)MeshRegistry.Count;
                list.Dispatch(items == StageItems.OneGroup ? 1u : (count + groupSize - 1) / groupSize, 1, 1);
            }
        }

        private static void BeginStageChain(ID3D12GraphicsCommandList list)
        {
            var device = Engine.Device;
            list.SetComputeRootSignature(device.GlobalRootSignature);
            _stageSrvHeaps ??= new[] { device.SrvHeap };
            list.SetDescriptorHeaps(1, _stageSrvHeaps);
        }

        /// <summary>
        /// The camera's culling for all of a pass's batches, stage by stage: visibility (which also writes the instance
        /// bounds for the shadow culling), histogram, prefix sum, scatter and commands are each dispatched for every
        /// batch before one barrier, so the batches' small dispatches overlap instead of each draining the GPU on its
        /// own (2026-10-16: all culling 0.97 → 0.72 ms on Eldermoor, measured against a chain per batch).
        /// </summary>
        public static void Cull(ID3D12GraphicsCommandList list, IReadOnlyList<InstanceBatch> batches,
            ulong frustumBufferGPUAddress, GPUCuller? culler)
        {
            if (culler?.VisibilityPSO == null || culler.MainPSO == null || culler.HistogramPSO == null) return;

            _stageBatches.Clear();
            foreach (var batch in batches)
                if (batch.SubBatchCount > 0)
                    _stageBatches.Add(batch);
            if (_stageBatches.Count == 0) return;

            BeginStageChain(list);
            list.SetComputeRootConstantBufferView(1, frustumBufferGPUAddress);
            // Root slot 2 (b1) must be bound too: the shader declares ShadowCascadePlanes
            list.SetComputeRootConstantBufferView(2, frustumBufferGPUAddress);

            foreach (var batch in _stageBatches)
                batch.PrepareCull(list);
            StageBarrier(list);

            PixMarker.Begin(list, "Visibility");
            DispatchStage(list, culler.VisibilityPSO, StageItems.Instances, 256);
            StageBarrier(list);
            PixMarker.End(list);
            foreach (var batch in _stageBatches)
                batch._boundsWritten = true;   // and the instance bounds, for the shadow culling

            PixMarker.Begin(list, "Compaction");
            DispatchStage(list, culler.HistogramPSO, StageItems.Instances, 64);
            StageBarrier(list);
            DispatchStage(list, culler.HistogramPrefixSumPSO, StageItems.OneGroup);
            StageBarrier(list);
            DispatchStage(list, culler.GlobalScatterPSO, StageItems.Instances, 256);
            StageBarrier(list);
            PixMarker.End(list);

            if (!Engine.Settings.DisableDepthSort
                && culler.ComputeDepthKeysPSO != null && culler.BitonicSortPairsPSO != null && culler.WriteSortedIndicesPSO != null)
            {
                PixMarker.Begin(list, "Depth sort");
                foreach (var batch in _stageBatches)
                {
                    batch.ApplyCullConstants(list);
                    batch.RecordDepthSort(list, culler);
                }
                PixMarker.End(list);
            }

            // Commands buffers transition to IndirectArgument in Draw(), no UAV barrier needed here
            PixMarker.Begin(list, "Commands");
            DispatchStage(list, culler.MainPSO, StageItems.Parts, 64);
            PixMarker.End(list);
        }

        // The tick, frame slot and instance count of this batch's last camera culling (PrepareCull)
        private int _lastCullTick = -1, _lastCullSlot, _lastCullCount;

        /// <summary>
        /// What this batch's camera culling left in the last tick or this one, for the sun shadow receivers
        /// (ShadowReceivers marks the forward instances the camera saw): each instance's bounds (InstanceBounds, UAV),
        /// its visibility flag (SRV, 0 = not seen) and how many instances there were. False when it was not culled then.
        /// </summary>
        internal bool RecentCameraCull(out uint boundsUav, out uint flagsSrv, out int count)
        {
            boundsUav = flagsSrv = 0;
            count = 0;
            if (_lastCullTick < Engine.TickCount - 1 || _lastCullCount == 0 || instanceBoundsBuffers[_lastCullSlot] == null)
                return false;
            boundsUav = instanceBoundsUAVIndices[_lastCullSlot];
            flagsSrv = visibilityFlagsSRVIndices[_lastCullSlot];
            count = _lastCullCount;
            return true;
        }

        // Constants and clears of this batch's camera culling
        private void PrepareCull(ID3D12GraphicsCommandList list)
        {
            InitializeCullerResources();
            int f = Engine.FrameIndex % FrameCount;
            int prev = (f + FrameCount - 1) % FrameCount;

            var c = _cullConstants;
            Array.Clear(c);
            c[0] = MeshRegistry.SrvIndex;
            c[1] = gpuCommandUAVIndices[f];
            c[2] = visibilityFlagsSRVIndices[prev];          // previous frame's visibility flags
            c[3] = 1;                                         // InstanceMultiplier
            c[5] = visibleIndicesUAVIndices[f];
            c[6] = counterBufferUAVIndices[f];
            c[7] = (uint)SubBatchCount;
            c[9] = visibleIndicesSRVIndices[f];
            c[10] = TransformBuffer.Instance?.SrvIndex ?? 0;
            c[11] = visibilityFlagsUAVIndices[f];
            c[13] = (uint)_drawCount;
            c[18] = drawCountUAVIndices[f];                   // DrawCountUAVIdx: packed commands
            c[25] = Material.MaterialsBufferIndex;
            c[26] = histogramUAVIndices[f];
            c[27] = (uint)MeshRegistry.Count;
            c[38] = instanceBoundsUAVIndices[f];              // InstanceBoundsUAVIdx: written for the shadow culling
            // The transform cache. Gathered again: this frame's enqueued draws (from PersistentCount on), and every
            // instance when the cache is new, the batch missed a tick (and that tick's uploaded slots) or a compute
            // shader wrote the records (EnqueueGPUBatch)
            bool refreshAll = _visibilityCacheTick < Engine.TickCount - 1 || _isGPUSourced;
            _visibilityCacheTick = Engine.TickCount;
            c[24] = refreshAll ? 0u : (uint)_persistentCount;                // CacheRefreshFrom
            c[28] = visibilityCacheUAVIndex;                                  // VisibilityCacheIdx
            c[29] = TransformBuffer.Instance?.ChangedSlotsSrvIndex ?? 0;     // ChangedSlotsIdx: slots uploaded this tick
            c[39] = UploadChangedRecords();                                   // ChangedRecordsIdx
            WriteLodConstants(c);
            WritePerInstanceConstants(f);
            _lastCullTick = Engine.TickCount;
            _lastCullSlot = f;
            _lastCullCount = _drawCount;

            ClearUav(list, counterBufferUAVIndices[f], counterBufferCPUHandles[f], counterBuffers[f]);
            ClearUav(list, visibleIndicesUAVIndices[f], visibleIndicesCPUHandles[f], visibleIndicesBuffers[f], unchecked((int)0xFFFFFFFF));
            ClearUav(list, histogramUAVIndices[f], histogramCPUHandles[f], histogramBuffers[f]);
            ClearUav(list, drawCountUAVIndices[f], drawCountCPUHandles[f], drawCountBuffers[f]);
        }

        // The records changed since the last camera culling, as this frame's bits (SRV index; 0 = none), then cleared
        private uint UploadChangedRecords()
        {
            if (!_anyRecordChanged)
                return 0;
            _changedRecords ??= new StreamingBuffer<uint>(Engine.Device, _changedRecordBits.Length);
            _changedRecords.BulkWrite(_changedRecordBits);
            Array.Clear(_changedRecordBits);
            _anyRecordChanged = false;
            return _changedRecords.SrvIndex;
        }

        /// <summary>
        /// Culls all shadow cascades for all batches, stage by stage (see Cull): CSVisibilityShadow4 writes the combined
        /// visibility (union) and a cascade mask per instance, one compaction runs on the union, CSExpandCascades
        /// expands it by the masks, and the non-empty commands are packed into each batch's opaque command buffer.
        /// </summary>
        /// <param name="minCasterClip">Cascades 1+: smallest caster diameter in clip units (EngineSettings.ShadowMinCasterTexels * 2 / resolution; 0 = off).</param>
        public static void CullShadowAll(ID3D12GraphicsCommandList list, IReadOnlyList<InstanceBatch> batches,
            uint cascadeBufferSrv, ulong cascadeBufferGpuAddress, uint shadowHiZSrv, GPUCuller? culler, float minCasterClip = 0f)
        {
            if (culler?.VisibilityShadow4PSO == null || culler.MainPSO == null || culler.HistogramPSO == null) return;

            _stageBatches.Clear();
            foreach (var batch in batches)
                if (batch.SubBatchCount > 0)
                    _stageBatches.Add(batch);
            if (_stageBatches.Count == 0) return;

            BeginStageChain(list);
            // Root signature requires all CBV slots bound (the kernels read the cascades from the SRV)
            list.SetComputeRootConstantBufferView(2, cascadeBufferGpuAddress);

            foreach (var batch in _stageBatches)
                batch.PrepareCullShadow(list, cascadeBufferSrv, shadowHiZSrv, minCasterClip);
            StageBarrier(list);

            // Combined flag + cascade mask per instance
            PixMarker.Begin(list, "Visibility");
            DispatchStage(list, culler.VisibilityShadow4PSO, StageItems.Instances, 256);
            StageBarrier(list);
            PixMarker.End(list);

            // One compaction of the union of the cascades
            PixMarker.Begin(list, "Compaction");
            DispatchStage(list, culler.HistogramPSO, StageItems.Instances, 64);
            StageBarrier(list);
            DispatchStage(list, culler.HistogramPrefixSumPSO, StageItems.OneGroup);
            StageBarrier(list);
            DispatchStage(list, culler.GlobalScatterPSO, StageItems.Instances, 256);
            StageBarrier(list);
            PixMarker.End(list);

            // Commands at fixed part indices (DrawCountUAVIdx = 0)
            PixMarker.Begin(list, "Commands");
            DispatchStage(list, culler.MainPSO, StageItems.Parts, 64);
            StageBarrier(list);
            PixMarker.End(list);

            // Expand the compacted instances by their cascade masks; the counters count the appends again
            PixMarker.Begin(list, "Cascade expansion");
            int f = Engine.FrameIndex % FrameCount;
            foreach (var batch in _stageBatches)
                batch.ClearUav(list, batch.shadowCounterUAVIndices[f], batch.shadowCounterCPUHandles[f], batch.shadowCounterBuffers[f]);
            StageBarrier(list);
            DispatchStage(list, culler.ExpandCascadesPSO, StageItems.Instances, 256);
            StageBarrier(list);
            DispatchStage(list, culler.PatchExpandedPSO, StageItems.Parts, 64);
            StageBarrier(list);
            PixMarker.End(list);

            // Pack the non-empty commands into the opaque command buffer (separate source and destination)
            PixMarker.Begin(list, "Compact commands");
            foreach (var batch in _stageBatches)
            {
                batch._cullConstants[18] = batch.drawCountUAVIndices[f];    // DrawCountUAVIdx
                batch._cullConstants[9] = batch.gpuCommandUAVIndices[f];    // dest = opaque command buffer
            }
            DispatchStage(list, culler.CompactCommandsPSO, StageItems.Parts, 64);
            StageBarrier(list);
            PixMarker.End(list);
        }

        // Constants and clears of this batch's cascade culling. The draw count is cleared
        // here already: nothing touches it before the compaction at the end.
        private void PrepareCullShadow(ID3D12GraphicsCommandList list, uint cascadeBufferSrv, uint shadowHiZSrv, float minCasterClip)
        {
            InitializeShadowCullerResources();
            int f = Engine.FrameIndex % FrameCount;

            var c = _cullConstants;
            Array.Clear(c);
            c[0] = MeshRegistry.SrvIndex;
            c[1] = shadowCommandUAVIndices[f];
            c[2] = shadowHiZSrv;                              // ShadowHiZSrvIdx
            c[3] = 1;                                         // InstanceMultiplier
            c[5] = shadowVisibleIndicesUAVIndices[f];
            c[6] = shadowCounterUAVIndices[f];
            c[7] = (uint)SubBatchCount;
            c[9] = shadowVisibleIndicesSRVIndices[f];
            c[10] = TransformBuffer.Instance?.SrvIndex ?? 0;
            c[11] = shadowCombinedVisUAVIndices[f];           // VisibilityFlagsIdx = combined
            c[13] = (uint)_drawCount;
            c[18] = 0;                                        // DrawCountUAVIdx = 0: fixed-index commands
            c[24] = shadowCombinedVisUAVIndices[f];           // CombinedShadowVisIdx
            c[25] = Material.MaterialsBufferIndex;
            c[26] = shadowHistogramUAVIndices[f];
            c[27] = (uint)MeshRegistry.Count;
            c[28] = shadowCascadeMaskUAVIndices[f];           // CascadeMaskUAVIdx
            c[29] = shadowExpansionUAVIndices[f];             // ExpansionUAVIdx
            c[30] = BitConverter.SingleToUInt32Bits(minCasterClip);
            c[31] = cascadeBufferSrv;                         // CascadeBufferSRVIdx
            // The camera's visibility pass left each instance's sphere and LOD part (0 = not this frame: gather them)
            c[38] = _boundsWritten ? instanceBoundsUAVIndices[f] : 0u;
            c[39] = DirectionalLight.CurrentShadowReceiversUav;   // ShadowReceiversUAVIdx: the sun's receiver tiles (0 = none)
            WriteLodConstants(c);                             // main camera: shadows use the main view's LOD
            WritePerInstanceConstants(f);

            ClearUav(list, shadowCounterUAVIndices[f], shadowCounterCPUHandles[f], shadowCounterBuffers[f]);
            ClearUav(list, shadowVisibleIndicesUAVIndices[f], shadowVisibleIndicesCPUHandles[f], shadowVisibleIndicesBuffers[f], unchecked((int)0xFFFFFFFF));
            ClearUav(list, shadowHistogramUAVIndices[f], shadowHistogramCPUHandles[f], shadowHistogramBuffers[f]);
            ClearUav(list, drawCountUAVIndices[f], drawCountCPUHandles[f], drawCountBuffers[f]);
        }

        /// <summary>
        /// Point light shadows: culls every batch with a LocalShadow pass against every cube face LocalShadows built
        /// this frame, stage by stage (see Cull), and expands the visible (instance, face) pairs for
        /// DrawShadowSinglePass (LocalShadow pass, view buffer in slot 21). Reuses the cascade buffers: it runs after
        /// the cascades have been drawn. Unlike the cascades the expansion is sized exactly (the histogram counts an
        /// instance once per face) and the scatter writes it directly (CS*LocalShadows in cull_instances.hlsl).
        /// </summary>
        /// <param name="countsUav">The selection's counts (local_shadow_select.hlsl), [0] = lights: picked on the GPU.</param>
        /// <param name="placeholderCbAddress">Any CBV address for root slot 2, which must be bound (the kernels do not read it).</param>
        public static void CullLocalShadowAll(ID3D12GraphicsCommandList list, IReadOnlyList<InstanceBatch> batches,
            uint viewBufferSrv, uint lightBufferSrv, uint countsUav, ulong placeholderCbAddress, GPUCuller culler)
        {
            if (culler.VisibilityLocalShadowsPSO == null || culler.HistogramLocalShadowsPSO == null
                || culler.ScatterLocalShadowsPSO == null || culler.MainPSO == null) return;

            _stageBatches.Clear();
            foreach (var batch in batches)
                if (batch.SubBatchCount > 0 && batch.Material.HasPass(RenderPass.LocalShadow))
                    _stageBatches.Add(batch);
            if (_stageBatches.Count == 0) return;

            BeginStageChain(list);
            list.SetComputeRootConstantBufferView(2, placeholderCbAddress);

            foreach (var batch in _stageBatches)
                batch.PrepareCullLocalShadow(list, viewBufferSrv, lightBufferSrv, countsUav);
            StageBarrier(list);   // also orders the camera's instance bounds before the visibility reads them

            // Combined flag + view mask per instance
            PixMarker.Begin(list, "Visibility");
            DispatchStage(list, culler.VisibilityLocalShadowsPSO, StageItems.Instances, 256);
            StageBarrier(list);
            PixMarker.End(list);

            // Entries per part, then their prefix sum (the parts' first entries)
            PixMarker.Begin(list, "Histogram");
            DispatchStage(list, culler.HistogramLocalShadowsPSO, StageItems.Instances, 64);
            StageBarrier(list);
            DispatchStage(list, culler.HistogramPrefixSumPSO, StageItems.OneGroup);
            StageBarrier(list);
            PixMarker.End(list);

            // Expansion entries, written in place
            PixMarker.Begin(list, "Scatter");
            DispatchStage(list, culler.ScatterLocalShadowsPSO, StageItems.Instances, 256);
            StageBarrier(list);
            PixMarker.End(list);

            // Commands at fixed part indices, then packed into the opaque command buffer as the cascades do
            PixMarker.Begin(list, "Commands");
            DispatchStage(list, culler.MainPSO, StageItems.Parts, 64);
            StageBarrier(list);
            int f = Engine.FrameIndex % FrameCount;
            foreach (var batch in _stageBatches)
            {
                batch._cullConstants[18] = batch.drawCountUAVIndices[f];    // DrawCountUAVIdx
                batch._cullConstants[9] = batch.gpuCommandUAVIndices[f];    // dest = opaque command buffer
            }
            DispatchStage(list, culler.CompactCommandsPSO, StageItems.Parts, 64);
            StageBarrier(list);
            PixMarker.End(list);
        }

        // Constants and clears of this batch's point light shadow culling
        private void PrepareCullLocalShadow(ID3D12GraphicsCommandList list, uint viewBufferSrv, uint lightBufferSrv, uint countsUav)
        {
            InitializeShadowCullerResources();
            int f = Engine.FrameIndex % FrameCount;
            var device = Engine.Device;

            if (localShadowMaskBuffers[f] == null)
            {
                const int maskStride = 4 * sizeof(uint);
                for (int i = 0; i < FrameCount; i++)
                {
                    localShadowMaskBuffers[i] = device.CreateDefaultBuffer(capacity * maskStride);
                    localShadowMaskUAVIndices[i] = device.AllocateBindlessIndex();
                    device.CreateStructuredBufferUAV(localShadowMaskBuffers[i]!, (uint)capacity, maskStride, localShadowMaskUAVIndices[i]);
                }
            }

            var c = _cullConstants;
            Array.Clear(c);
            c[0] = MeshRegistry.SrvIndex;
            c[1] = shadowCommandUAVIndices[f];
            c[2] = 0;                                         // no shadow Hi-Z
            c[3] = 1;                                         // InstanceMultiplier: counts are entries already
            c[6] = shadowCounterUAVIndices[f];
            c[7] = (uint)SubBatchCount;
            c[10] = TransformBuffer.Instance?.SrvIndex ?? 0;
            c[11] = shadowCombinedVisUAVIndices[f];           // VisibilityFlagsIdx = combined
            c[13] = (uint)_drawCount;
            c[16] = lightBufferSrv;                           // LocalShadowLightsIdx
            c[17] = countsUav;                                // LocalShadowCountsIdx
            c[18] = 0;                                        // DrawCountUAVIdx = 0: fixed-index CSMain
            c[19] = (uint)(capacity * 4);                     // LocalShadowExpansionCapacity
            c[24] = shadowCombinedVisUAVIndices[f];           // CombinedShadowVisIdx
            c[25] = Material.MaterialsBufferIndex;
            c[26] = shadowHistogramUAVIndices[f];
            c[27] = (uint)MeshRegistry.Count;
            c[28] = localShadowMaskUAVIndices[f];             // LocalShadowMaskIdx (uint4 view mask)
            c[29] = shadowExpansionUAVIndices[f];
            c[30] = 0;                                        // ShadowMinCasterClip: cascades only
            c[31] = viewBufferSrv;                            // LocalShadowViewsIdx
            // The camera's visibility pass left each instance's sphere and LOD part (0 = not this frame: gather them)
            c[38] = _boundsWritten ? instanceBoundsUAVIndices[f] : 0u;
            WriteLodConstants(c);                             // main camera: same LOD as the view
            WritePerInstanceConstants(f);

            ClearUav(list, shadowCounterUAVIndices[f], shadowCounterCPUHandles[f], shadowCounterBuffers[f]);
            ClearUav(list, shadowHistogramUAVIndices[f], shadowHistogramCPUHandles[f], shadowHistogramBuffers[f]);
            ClearUav(list, drawCountUAVIndices[f], drawCountCPUHandles[f], drawCountBuffers[f]);
        }

        #endregion

        /// <summary>
        /// Single-pass shadow draw: 1 ExecuteIndirect for all cascades (Shadow pass), or for all point light
        /// cube faces after CullLocalShadowAll (LocalShadow pass; shadowVPSrv is then LocalShadows' view buffer).
        /// Material/PSO/descriptors must already be set by the caller.
        /// Expansion buffer encodes (view, instanceIdx) — no view mask needed in VS.
        /// </summary>
        public void DrawShadowSinglePass(ID3D12GraphicsCommandList commandList, uint shadowVPSrv)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            if (!_shadowCullerInitialized || SubBatchCount == 0) return;

            // After CSCompactCommands, packed commands are in gpuCommandBuffers
            var commandBuffer = gpuCommandBuffers[frameIndex];
            if (commandBuffer == null) return;



            // Bind per-instance buffer SRV indices (may use any slots including high ones)
            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (pib.PushConstantSlot < 0) continue;
                commandList.SetGraphicsRoot32BitConstant(0, pib.SRVIndices[frameIndex], (uint)pib.PushConstantSlot);
            }

            // Per-batch graphics constants (slots 4-8)
            // These were previously per-draw via command signature, now set once before ExecuteIndirect.
            var descriptorPib = _perInstanceBuffers.GetValueOrDefault(DrawBucket.DescriptorsHash);
            if (descriptorPib != null)
                commandList.SetGraphicsRoot32BitConstant(0, descriptorPib.SRVIndices[frameIndex], 4); // DescriptorBufIdx
            commandList.SetGraphicsRoot32BitConstant(0, visibleIndicesSRVIndices[frameIndex], 5);     // SortedIndicesIdx
            commandList.SetGraphicsRoot32BitConstant(0, MeshRegistry.SrvIndex, 6);                   // MeshRegistryIdx
            commandList.SetGraphicsRoot32BitConstant(0, Material.MaterialsBufferIndex, 7);            // MaterialsIdx
            commandList.SetGraphicsRoot32BitConstant(0, TransformBuffer.Instance?.SrvIndex ?? 0, 8);  // GlobalTransformBufIdx

            // Set expansion buffer + VP buffer
            commandList.SetGraphicsRoot32BitConstant(0, shadowExpansionSRVIndices[frameIndex], 20); // ExpansionBufferIdx
            commandList.SetGraphicsRoot32BitConstant(0, shadowVPSrv, 21);    // ShadowVPBufferIdx

            // Transition buffers for reading
            var countBuffer = drawCountBuffers[frameIndex];

            if (countBuffer != null)
            {
                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(shadowExpansionBuffers[frameIndex],
                        ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)),
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)),
                    new ResourceBarrier(new ResourceTransitionBarrier(countBuffer,
                        ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));

                commandList.ExecuteIndirect(
                    Engine.Device.BindlessCommandSignature,
                    (uint)SubBatchCount,
                    commandBuffer,
                    0,
                    countBuffer,
                    0
                );

                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)),
                    new ResourceBarrier(new ResourceTransitionBarrier(countBuffer,
                        ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)),
                    new ResourceBarrier(new ResourceTransitionBarrier(shadowExpansionBuffers[frameIndex],
                        ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
            }
            else
            {
                // Fallback: no count buffer
                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(shadowExpansionBuffers[frameIndex],
                        ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)),
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));

                commandList.ExecuteIndirect(
                    Engine.Device.BindlessCommandSignature,
                    (uint)SubBatchCount,
                    commandBuffer,
                    0,
                    null,
                    0
                );

                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)),
                    new ResourceBarrier(new ResourceTransitionBarrier(shadowExpansionBuffers[frameIndex],
                        ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
            }
        }

        #endregion


        public void Draw(ID3D12GraphicsCommandList commandList, GraphicsDevice device)
        {
            int frameIndex = Engine.FrameIndex % FrameCount;
            if (SubBatchCount == 0) return;
            
            
            var commandBuffer = gpuCommandBuffers[frameIndex];
            if (commandBuffer == null) return;

            // Set topology: tessellation shaders require patch topology
            commandList.IASetPrimitiveTopology(Topology);

            // The command signature only writes slots 2-3 per draw (MeshPartId + InstanceBaseOffset).
            // Per-instance buffers (e.g. TerrainPatchData at slot 1) use custom slots
            // and must be set explicitly before ExecuteIndirect.
            foreach (var (hash, pib) in _perInstanceBuffers)
            {
                if (pib.PushConstantSlot < 0) continue;
                uint srvIdx = pib.SRVIndices[frameIndex];
                commandList.SetGraphicsRoot32BitConstant(0, srvIdx, (uint)pib.PushConstantSlot);
            }

            // Per-batch graphics constants (slots 4-8)
            var descriptorPib = _perInstanceBuffers.GetValueOrDefault(DrawBucket.DescriptorsHash);
            if (descriptorPib != null)
                commandList.SetGraphicsRoot32BitConstant(0, descriptorPib.SRVIndices[frameIndex], 4);
            commandList.SetGraphicsRoot32BitConstant(0, visibleIndicesSRVIndices[frameIndex], 5);
            commandList.SetGraphicsRoot32BitConstant(0, MeshRegistry.SrvIndex, 6);
            commandList.SetGraphicsRoot32BitConstant(0, Material.MaterialsBufferIndex, 7);
            commandList.SetGraphicsRoot32BitConstant(0, TransformBuffer.Instance?.SrvIndex ?? 0, 8);

            var countBuffer = drawCountBuffers[frameIndex];

            if (countBuffer != null)
            {
                // Batch all pre-draw transitions
                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(visibleIndicesBuffers[frameIndex],
                        ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)),
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)),
                    new ResourceBarrier(new ResourceTransitionBarrier(countBuffer,
                        ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));

                commandList.ExecuteIndirect(
                    Engine.Device.BindlessCommandSignature,
                    (uint)SubBatchCount,
                    commandBuffer,
                    0,
                    countBuffer,
                    0
                );

                // Batch all post-draw transitions
                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)),
                    new ResourceBarrier(new ResourceTransitionBarrier(countBuffer,
                        ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)),
                    new ResourceBarrier(new ResourceTransitionBarrier(visibleIndicesBuffers[frameIndex],
                        ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
            }
            else
            {
                // Fallback: no count buffer (after resize, before next Cull())
                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(visibleIndicesBuffers[frameIndex],
                        ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource)),
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.UnorderedAccess, ResourceStates.IndirectArgument)));

                commandList.ExecuteIndirect(
                    Engine.Device.BindlessCommandSignature,
                    (uint)SubBatchCount,
                    commandBuffer,
                    0,
                    null,
                    0
                );

                commandList.Barriers(
                    new ResourceBarrier(new ResourceTransitionBarrier(commandBuffer,
                        ResourceStates.IndirectArgument, ResourceStates.UnorderedAccess)),
                    new ResourceBarrier(new ResourceTransitionBarrier(visibleIndicesBuffers[frameIndex],
                        ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess)));
            }

        }

        /// <summary>
        /// Reset buffer states after draw for next frame. Currently empty.
        /// </summary>
        public void ResetBufferState(ID3D12GraphicsCommandList commandList) { }

        /// <summary>
        /// Queue old buffers for deferred disposal (GPU may still be using them).
        /// </summary>
        private void DeferDisposeBuffers()
        {
            for (int i = 0; i < FrameCount; ++i)
            {

                DeferDispose(visibleIndicesBuffers[i], visibleIndicesUAVIndices[i]);
                DeferDispose(null, visibleIndicesSRVIndices[i]);
                visibleIndicesBuffers[i] = null; visibleIndicesUAVIndices[i] = 0; visibleIndicesSRVIndices[i] = 0;



                DeferDispose(gpuCommandBuffers[i], gpuCommandUAVIndices[i]);
                gpuCommandBuffers[i] = null; gpuCommandUAVIndices[i] = 0;

                DeferDispose(counterBuffers[i], counterBufferUAVIndices[i]);
                counterBuffers[i] = null; counterBufferUAVIndices[i] = 0;

                DeferDispose(visibilityFlagsBuffers[i], visibilityFlagsUAVIndices[i]);
                DeferDispose(null, visibilityFlagsSRVIndices[i]);
                visibilityFlagsBuffers[i] = null; visibilityFlagsUAVIndices[i] = 0; visibilityFlagsSRVIndices[i] = 0;

                if (instanceBoundsBuffers[i] != null)
                    DeferDispose(instanceBoundsBuffers[i], instanceBoundsUAVIndices[i]);
                instanceBoundsBuffers[i] = null; instanceBoundsUAVIndices[i] = 0;

                DeferDispose(histogramBuffers[i], histogramUAVIndices[i]);
                histogramBuffers[i] = null; histogramUAVIndices[i] = 0;

                DeferDispose(drawCountBuffers[i], drawCountUAVIndices[i]);
                drawCountBuffers[i] = null; drawCountUAVIndices[i] = 0;

                DeferDispose(sortKeysBuffers[i], sortKeysUAVIndices[i]);
                sortKeysBuffers[i] = null; sortKeysUAVIndices[i] = 0;
            }

            if (visibilityCacheBuffer != null)
                DeferDispose(visibilityCacheBuffer, visibilityCacheUAVIndex);
            visibilityCacheBuffer = null; visibilityCacheUAVIndex = 0;

            _cpuHeap?.Dispose();
            _cpuHeap = null;
            _cullerInitialized = false;

            // Defer shadow union buffers
            if (_shadowCullerInitialized)
            {
                for (int f = 0; f < FrameCount; f++)
                {
                    DeferDispose(shadowCombinedVisBuffers[f], shadowCombinedVisUAVIndices[f]);
                    shadowCombinedVisBuffers[f] = null; shadowCombinedVisUAVIndices[f] = 0;

                    DeferDispose(shadowCascadeMaskBuffers[f], shadowCascadeMaskUAVIndices[f]);
                    DeferDispose(null, shadowCascadeMaskSRVIndices[f]);
                    shadowCascadeMaskBuffers[f] = null; shadowCascadeMaskUAVIndices[f] = 0; shadowCascadeMaskSRVIndices[f] = 0;

                    DeferDispose(shadowVisibleIndicesBuffers[f], shadowVisibleIndicesUAVIndices[f]);
                    DeferDispose(null, shadowVisibleIndicesSRVIndices[f]);
                    shadowVisibleIndicesBuffers[f] = null; shadowVisibleIndicesUAVIndices[f] = 0; shadowVisibleIndicesSRVIndices[f] = 0;

                    DeferDispose(shadowCommandBuffers[f], shadowCommandUAVIndices[f]);
                    shadowCommandBuffers[f] = null; shadowCommandUAVIndices[f] = 0;

                    DeferDispose(shadowCounterBuffers[f], shadowCounterUAVIndices[f]);
                    shadowCounterBuffers[f] = null; shadowCounterUAVIndices[f] = 0;

                    DeferDispose(shadowHistogramBuffers[f], shadowHistogramUAVIndices[f]);
                    shadowHistogramBuffers[f] = null; shadowHistogramUAVIndices[f] = 0;

                    DeferDispose(shadowExpansionBuffers[f], shadowExpansionUAVIndices[f]);
                    DeferDispose(null, shadowExpansionSRVIndices[f]);
                    shadowExpansionBuffers[f] = null; shadowExpansionUAVIndices[f] = 0; shadowExpansionSRVIndices[f] = 0;

                    if (localShadowMaskBuffers[f] != null)
                        DeferDispose(localShadowMaskBuffers[f], localShadowMaskUAVIndices[f]);
                    localShadowMaskBuffers[f] = null; localShadowMaskUAVIndices[f] = 0;
                }
                _shadowCullerInitialized = false;
                _shadowCpuHeap?.Dispose();
                _shadowCpuHeap = null;
            }
        }

        private void DisposeBuffers()
        {
            for (int i = 0; i < FrameCount; ++i)
            {
                visibleIndicesBuffers[i]?.Dispose();
                if (visibleIndicesUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(visibleIndicesUAVIndices[i]);
                if (visibleIndicesSRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(visibleIndicesSRVIndices[i]);
                
                gpuCommandBuffers[i]?.Dispose();
                if (gpuCommandUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(gpuCommandUAVIndices[i]);
                
                counterBuffers[i]?.Dispose();
                if (counterBufferUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(counterBufferUAVIndices[i]);

                drawCountBuffers[i]?.Dispose();
                if (drawCountUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(drawCountUAVIndices[i]);

                sortKeysBuffers[i]?.Dispose();
                if (sortKeysUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(sortKeysUAVIndices[i]);

                instanceBoundsBuffers[i]?.Dispose();
                if (instanceBoundsUAVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(instanceBoundsUAVIndices[i]);
                instanceBoundsBuffers[i] = null; instanceBoundsUAVIndices[i] = 0;

                visibleIndicesUAVIndices[i] = 0;
                visibleIndicesSRVIndices[i] = 0;
                gpuCommandUAVIndices[i] = 0;
                counterBufferUAVIndices[i] = 0;
                drawCountUAVIndices[i] = 0;
                sortKeysUAVIndices[i] = 0;
            }

            visibilityCacheBuffer?.Dispose();
            if (visibilityCacheUAVIndex != 0) Engine.Device.ReleaseBindlessIndex(visibilityCacheUAVIndex);
            visibilityCacheBuffer = null; visibilityCacheUAVIndex = 0;

            _cpuHeap?.Dispose();
            _cpuHeap = null;
            _cullerInitialized = false;
            
            // Dispose shadow union buffers
            if (_shadowCullerInitialized)
            {
                for (int f = 0; f < FrameCount; f++)
                {
                    shadowCombinedVisBuffers[f]?.Dispose();
                    if (shadowCombinedVisUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCombinedVisUAVIndices[f]);

                    shadowCascadeMaskBuffers[f]?.Dispose();
                    if (shadowCascadeMaskUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCascadeMaskUAVIndices[f]);
                    if (shadowCascadeMaskSRVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCascadeMaskSRVIndices[f]);

                    shadowVisibleIndicesBuffers[f]?.Dispose();
                    if (shadowVisibleIndicesUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowVisibleIndicesUAVIndices[f]);
                    if (shadowVisibleIndicesSRVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowVisibleIndicesSRVIndices[f]);

                    shadowCommandBuffers[f]?.Dispose();
                    if (shadowCommandUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCommandUAVIndices[f]);

                    shadowCounterBuffers[f]?.Dispose();
                    if (shadowCounterUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowCounterUAVIndices[f]);

                    shadowHistogramBuffers[f]?.Dispose();
                    if (shadowHistogramUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowHistogramUAVIndices[f]);

                    shadowExpansionBuffers[f]?.Dispose();
                    if (shadowExpansionUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowExpansionUAVIndices[f]);
                    if (shadowExpansionSRVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(shadowExpansionSRVIndices[f]);

                    localShadowMaskBuffers[f]?.Dispose();
                    if (localShadowMaskUAVIndices[f] != 0) Engine.Device.ReleaseBindlessIndex(localShadowMaskUAVIndices[f]);
                    localShadowMaskBuffers[f] = null; localShadowMaskUAVIndices[f] = 0;
                }
                _shadowCullerInitialized = false;
            }
        }

        public void Dispose()
        {
            DisposeBuffers();
            _changedRecords?.Dispose();
            _changedRecords = null;
            foreach (var kvp in _perInstanceBuffers)
            {
                for (int i = 0; i < FrameCount; ++i)
                {
                    kvp.Value.Buffers[i]?.Dispose();
                    if (kvp.Value.SRVIndices[i] != 0) Engine.Device.ReleaseBindlessIndex(kvp.Value.SRVIndices[i]);
                }
            }
            _perInstanceBuffers.Clear();
        }
    }
}
