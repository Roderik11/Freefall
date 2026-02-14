using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    public struct BatchKey : IEquatable<BatchKey>
    {
        public Effect Effect;  // Batch by Effect, not Material - materials share Effects

        public BatchKey(Effect effect)
        {
            Effect = effect;
        }

        public bool Equals(BatchKey other)
        {
            return Effect == other.Effect;
        }

        public override int GetHashCode()
        {
            return Effect.GetHashCode();
        }
    }

    // Draw call structure for batching and sorting
    public struct DrawCall
    {
        public BatchKey Key;
        public Mesh Mesh;
        public int MeshPartIndex;
        public Material Material;
        public MaterialBlock MaterialBlock;
        /// <summary>
        /// Transform slot in global TransformBuffer. -1 means use MaterialBlock["World"] fallback.
        /// </summary>
        public int TransformSlot;
    }

    /// <summary>
    /// Thread-local bucket for collecting GPU-ready arrays during parallel Enqueue.
    /// Stores parallel arrays that can be block-copied with Array.Copy.
    /// </summary>
    public class DrawBucket
    {
        // Well-known hash keys for core per-instance data channels
        public static readonly int DescriptorsHash = "Descriptors".GetHashCode();
        // Removed: BoundingSpheresHash — bounding spheres now live in MeshRegistry
        public static readonly int SubbatchIdsHash = "SubbatchIds".GetHashCode();
        
        public List<InstanceBatch.RawDraw> Draws = new();
        public HashSet<int> UniqueMeshPartIds = new();
        public Material? FirstMaterial;
        
        /// <summary>
        /// Pre-staged per-instance data: hash → contiguous byte array (filled at Enqueue time).
        /// </summary>
        public class PerInstanceStaging
        {
            public int PushConstantSlot;    // Graphics push constant slot (from shader)
            public int ElementStride;       // Bytes per element
            public int ElementsPerInstance; // Elements per instance (1 for scalar, N for array)
            public byte[] Data = Array.Empty<byte>();
            public int Count;               // Number of instances staged
            public int BytesPerInstance => ElementsPerInstance * ElementStride;
        }
        
        public Dictionary<int, PerInstanceStaging> PerInstanceData = new();
        
        public int Count => Draws.Count;
        
        public void Add(Mesh mesh, int partIndex, Material material, MaterialBlock block, int transformSlot)
        {
            // Get or register MeshPartId (done during parallel Enqueue!)
            int meshPartId = mesh.GetMeshPartId(partIndex);
            if (meshPartId < 0)
            {
                mesh.RegisterMeshParts();
                meshPartId = mesh.GetMeshPartId(partIndex);
            }
            
            if (FirstMaterial == null) FirstMaterial = material;
            
            // Store GPU-ready data in parallel arrays
            Draws.Add(new InstanceBatch.RawDraw
            {
                Mesh = mesh,
                PartIndex = partIndex,
                Block = block,
                TransformSlot = transformSlot,
                MaterialId = (uint)material.MaterialID,
                MeshPartId = meshPartId,
            });
            
            // Stage core per-instance data through the generic PerInstanceStaging system
            var descriptor = new InstanceDescriptor
            {
                TransformSlot = (uint)transformSlot,
                MaterialId = (uint)material.MaterialID,
                CustomDataIdx = 0
            };

            StageCore(DescriptorsHash, 12, descriptor);       // InstanceDescriptor: 12 bytes
            StageCore(SubbatchIdsHash, 4,  (uint)meshPartId);        // uint: 4 bytes
            UniqueMeshPartIds.Add(meshPartId);
            
            if(block == null) return;

            // Stage per-instance params into contiguous byte arrays (at Enqueue time, not MergeFromBucket)
            foreach (var (hash, param) in block.Parameters)
            {
                if (param is TextureParameterValue) continue;
                
                // Auto-resolve graphics push constant slot from shader resource bindings
                if (param.PushConstantSlot < 0 && material.Effect != null)
                    param.PushConstantSlot = material.Effect.GetPushConstantSlot(hash);
                
                int elemCount = param.GetElementCount();
                int elemStride = param.GetElementStride();
                if (elemCount == 0 || elemStride == 0) continue;
                    
                if (!PerInstanceData.TryGetValue(hash, out var staging))
                {
                    staging = new PerInstanceStaging
                    {
                        PushConstantSlot = param.PushConstantSlot,
                        ElementStride = elemStride,
                        ElementsPerInstance = elemCount,
                    };
                    PerInstanceData[hash] = staging;
                }
                    
                // Ensure capacity
                int bytesPerInst = staging.BytesPerInstance;
                int needed = (staging.Count + 1) * bytesPerInst;
                if (staging.Data.Length < needed)
                    Array.Resize(ref staging.Data, Math.Max(staging.Data.Length * 2, needed));
                    
                // Copy raw bytes at sequential offset (instance N at offset N * bytesPerInstance)
                param.CopyToStaging(staging.Data, staging.Count * bytesPerInst);
                staging.Count++;
            }
        }
        
        public void Clear()
        {
            Draws.Clear();
            UniqueMeshPartIds.Clear();
            FirstMaterial = null;
            foreach (var staging in PerInstanceData.Values)
                staging.Count = 0;
        }
        
        /// <summary>
        /// Stage a core per-instance value (Descriptor, BoundingSphere, SubbatchId) as raw bytes.
        /// </summary>
        private unsafe void StageCore<T>(int hash, int stride, T value) where T : unmanaged
        {
            if (!PerInstanceData.TryGetValue(hash, out var staging))
            {
                staging = new PerInstanceStaging
                {
                    PushConstantSlot = -1,  // Core data — bound by InstanceBatch, not shader
                    ElementStride = stride,
                    ElementsPerInstance = 1,
                };
                PerInstanceData[hash] = staging;
            }
            int needed = (staging.Count + 1) * stride;
            if (staging.Data.Length < needed)
                Array.Resize(ref staging.Data, Math.Max(staging.Data.Length * 2, needed));
            
            fixed (byte* ptr = &staging.Data[staging.Count * stride])
                *(T*)ptr = value;
            staging.Count++;
        }
    }

    public class CommandBuffer
    {
        private static CommandBuffer current;
        private static readonly Stack<CommandBuffer> stack = new Stack<CommandBuffer>();
        private static readonly Stack<CommandBuffer> freeBuffers = new Stack<CommandBuffer>();

        /// <summary>
        /// Optional GPU-based frustum culler. Set to enable GPU culling.
        /// Initialize with: CommandBuffer.GPUCuller = new GPUCuller(Engine.Device);
        /// </summary>
        public static GPUCuller? GPUCuller { get; set; }
        
        /// <summary>Last frame's opaque draw call count (for title bar stats).</summary>
        public static int LastDrawCallCount { get; set; }
        /// <summary>Last frame's opaque batch count (for title bar stats).</summary>
        public static int LastBatchCount { get; set; }
        
        /// <summary>
        /// Current frustum planes for GPU culling (6 planes as Vector4: xyz=normal, w=distance).
        /// Set by renderer before calling Execute.
        /// </summary>
        public static Vector4[]? CurrentFrustumPlanes { get; set; }

        // We assume 3 frames in flight for DX12 safety
        public const int FrameCount = 3;

        class Pass
        {
            // ThreadLocal buckets for per-thread batching - pre-computed GPU-ready arrays
            private readonly ThreadLocal<Dictionary<BatchKey, DrawBucket>> threadLocalBuckets = 
                new(() => new Dictionary<BatchKey, DrawBucket>(), trackAllValues: true);
            
            // Persistent batches - reused across frames
            private readonly Dictionary<BatchKey, InstanceBatch> batches = new Dictionary<BatchKey, InstanceBatch>();
            private readonly List<InstanceBatch> activeBatches = new List<InstanceBatch>();

            private readonly List<InstanceBatch> allBatches = new List<InstanceBatch>();
            
            /// <summary>
            /// Get the active batches from current frame for shadow rendering access.
            /// </summary>
            public IReadOnlyList<InstanceBatch> ActiveBatches => activeBatches;
            
            /// <summary>
            /// Get ALL registered batches (not just camera-visible). Used by shadow pass
            /// so off-screen objects can still cast shadows into the visible area.
            /// </summary>
            public IReadOnlyList<InstanceBatch> AllBatches => allBatches;
            
            // Shared frustum constant buffers (frame-level, uploaded once before batch loop)
            private static ID3D12Resource[]? _frustumConstantsBuffers;
            private static bool _frustumBuffersInitialized;
            
            // Previous frame's ViewProjection — Hi-Z depth is 1 frame behind,
            // so occlusion must project using the VP that matches the depth data
            private static Matrix4x4 _previousFrameViewProjection;
            
            private static void EnsureFrustumBuffers(GraphicsDevice device)
            {
                if (_frustumBuffersInitialized) return;
                
                _frustumConstantsBuffers = new ID3D12Resource[FrameCount];
                int bufferSize = 256; // 6 * 16 bytes = 96 bytes, aligned to 256 for CBV
                
                for (int i = 0; i < FrameCount; i++)
                {
                    _frustumConstantsBuffers[i] = device.CreateUploadBuffer(bufferSize);
                }
                _frustumBuffersInitialized = true;
            }
            
            private static ulong UploadFrustumPlanes(GraphicsDevice device, Vector4[] frustumPlanes)
            {
                EnsureFrustumBuffers(device);
                
                int frameIndex = Engine.FrameIndex % FrameCount;
                
                // Build full frustum constants including Hi-Z parameters
                var constants = new GPUCuller.FrustumConstants
                {
                    Plane0 = frustumPlanes[0],
                    Plane1 = frustumPlanes[1],
                    Plane2 = frustumPlanes[2],
                    Plane3 = frustumPlanes[3],
                    Plane4 = frustumPlanes[4],
                    Plane5 = frustumPlanes[5],
                };
                
                // Add Hi-Z occlusion data if available and pyramid has been generated
                var culler = CommandBuffer.Culler;
                if (culler != null && culler.HiZPyramidSRV != 0 && culler.HiZReady && !Engine.Settings.DisableHiZ)
                {
                    // Hi-Z depth is from the previous frame, so project spheres
                    // using the previous frame's VP to match the depth data
                    constants.OcclusionProjection = Engine.Settings.FreezeFrustum
                        ? Engine.Settings.FrozenViewProjection
                        : _previousFrameViewProjection;
                    constants.HiZSrvIdx = culler.HiZPyramidSRV;
                    constants.HiZWidth = culler.HiZWidth;
                    constants.HiZHeight = culler.HiZHeight;
                    constants.HiZMipCount = (uint)culler.HiZMipCount;
                    constants.NearPlane = Camera.Main.NearPlane;
                }
                
                // Debug visualization mode
                constants.DebugMode = (uint)Engine.Settings.DebugVisualizationMode;
                
                // Cull stats UAV (always set if culler is initialized)
                if (culler != null)
                    constants.CullStatsUAVIdx = culler.CullStatsUAV;
                
                // Store current VP for next frame's occlusion projection
                _previousFrameViewProjection = Camera.Main.ViewProjection;
                
                unsafe
                {
                    void* pData;
                    _frustumConstantsBuffers![frameIndex].Map(0, null, &pData);
                    *(GPUCuller.FrustumConstants*)pData = constants;
                    _frustumConstantsBuffers[frameIndex].Unmap(0);
                }
                
                return _frustumConstantsBuffers![frameIndex].GPUVirtualAddress;
            }

            public void Clear()
            {
                // Clear all thread-local buckets
                foreach (var threadBuckets in threadLocalBuckets.Values)
                    foreach (var bucket in threadBuckets.Values)
                        bucket.Clear();
            }

            public void Add(in DrawCall drawCall)
            {
                // Write to thread-local bucket with pre-computed GPU arrays (no contention)
                var buckets = threadLocalBuckets.Value;
                if (!buckets.TryGetValue(drawCall.Key, out var bucket))
                {
                    bucket = new DrawBucket();
                    buckets[drawCall.Key] = bucket;
                }
                bucket.Add(drawCall.Mesh, drawCall.MeshPartIndex, drawCall.Material, drawCall.MaterialBlock, drawCall.TransformSlot);
            }

            /// <summary>
            /// Create or find a batch for GPU-sourced data (bypasses CPU staging).
            /// Attaches pre-filled buffer SRVs from a compute shader.
            /// </summary>
            public void EnsureGPUBatch(
                BatchKey key, Material material, Mesh mesh, int meshPartId,
                uint descriptorsSRV, uint subbatchIdsSRV,
                int instanceCount, ReadOnlySpan<InstanceBatch.GPUBufferBinding> customBindings)
            {
                if (!batches.TryGetValue(key, out var batch))
                {
                    batch = new InstanceBatch(key, material);
                    batches.Add(key, batch);
                    allBatches.Add(batch);
                }

                // Activate for this frame
                if (batch._activeFrame != Engine.FrameIndex)
                {
                    batch._activeFrame = Engine.FrameIndex;
                    batch.Clear();
                    activeBatches.Add(batch);
                }

                // Attach GPU-generated buffers
                batch.AttachGPUData(
                    descriptorsSRV, subbatchIdsSRV,
                    instanceCount, meshPartId, customBindings);
            }

            public void Execute(ID3D12GraphicsCommandList commandList, GraphicsDevice device, RenderPass pass)
            {
                var totalSw = System.Diagnostics.Stopwatch.StartNew();
                                
                // 0. Execute custom actions (compute shader dispatches etc.) before batch processing
                if (current.customQueues.TryGetValue(pass, out var actions) && actions.Count > 0)
                {
                    foreach (var action in actions)
                        action(commandList);
                    actions.Clear();
                }

                // 1. Batch draw calls by Effect
                var batchingSw = System.Diagnostics.Stopwatch.StartNew();
                
                activeBatches.Clear();
                
                int drawCallCount = 0;
                
                // Merge all thread-local buckets into batches with block copy
                foreach (var threadBuckets in threadLocalBuckets.Values)
                {
                    foreach (var kvp in threadBuckets)
                    {
                        var key = kvp.Key;
                        var drawBucket = kvp.Value;
                        
                        if (drawBucket.Count == 0) continue;
                        drawCallCount += drawBucket.Count;
                        
                        // Get or create batch for this key
                        if (!batches.TryGetValue(key, out var batch))
                        {
                            batch = new InstanceBatch(key, drawBucket.FirstMaterial!);
                            batches.Add(key, batch);
                            allBatches.Add(batch);
                        }
                        
                        // Activate batch for this frame
                        if (batch._activeFrame != Engine.FrameIndex)
                        {
                            batch._activeFrame = Engine.FrameIndex;
                            batch.Clear();
                            activeBatches.Add(batch);
                        }
                        
                        // Block-copy pre-computed arrays from bucket
                        batch.MergeFromBucket(drawBucket);
                    }
                }
                
                // Re-add GPU-sourced batches that were activated during Draw() via EnsureGPUBatch.
                // These don't go through the bucket merge path but are already configured.
                foreach (var batch in allBatches)
                {
                    if (batch._isGPUSourced && batch._activeFrame == Engine.FrameIndex
                        && !activeBatches.Contains(batch))
                        activeBatches.Add(batch);
                }
                
                batchingSw.Stop();

                // GPU culling requires Culler to be initialized
                var sortDrawCallsSw = System.Diagnostics.Stopwatch.StartNew();
                bool usingGPUPath = Culler != null && pass == RenderPass.Opaque;

                // 2. Build commands (each batch type handles its own path)
                // Use frozen frustum if F3 was pressed, otherwise use current camera
                var vpMatrix = Engine.Settings.FreezeFrustum 
                    ? Engine.Settings.FrozenViewProjection 
                    : Camera.Main.ViewProjection;
                var frustum = new Frustum(vpMatrix);
                var frustumPlanes = frustum.GetPlanesAsVector4();
                ulong frustumBufferGPUAddress = UploadFrustumPlanes(device, frustumPlanes);

                foreach (var batch in activeBatches)
                    batch.Material.SetPass(pass);
                
                // 3. Upload transforms, materials (always needed)
                var buildSw = System.Diagnostics.Stopwatch.StartNew();
                foreach (var batch in activeBatches)
                    batch.UploadInstanceData(device);
                
                foreach (var batch in activeBatches)
                    batch.Build(device, commandList, frustumBufferGPUAddress);
                
                // 4b. Clear cull stats, dispatch GPU culling, copy stats to readback
                if (usingGPUPath)
                    CommandBuffer.Culler?.ClearCullStats(commandList);
                    
                foreach (var batch in activeBatches)
                    batch.Cull(commandList, frustumBufferGPUAddress, Culler);
                
                if (usingGPUPath)
                    CommandBuffer.Culler?.CopyCullStatsToReadback(commandList);
                
                buildSw.Stop();
                sortDrawCallsSw.Stop();

                // 7. Draw (topology set per-batch: patch for tessellation, triangle list otherwise)
                var applySw = System.Diagnostics.Stopwatch.StartNew();
                var drawSw = System.Diagnostics.Stopwatch.StartNew();
                
                foreach (var batch in activeBatches)
                {
                    // Apply Material (PSO)
                    applySw.Start();
                    batch.Material.Apply(commandList, device);
                    // Set topology per-batch: tessellation shaders require patch topology
                    var topology = batch.Material.HasTessellation
                        ? Vortice.Direct3D.PrimitiveTopology.PatchListWith3ControlPoints
                        : Vortice.Direct3D.PrimitiveTopology.TriangleList;
                    commandList.IASetPrimitiveTopology(topology);
                    // Push constant slot 16: debug mode (not touched by command signature slots 2-15)
                    commandList.SetGraphicsRoot32BitConstant(0, (uint)Engine.Settings.DebugVisualizationMode, 16);
                    applySw.Stop();
                    
                    drawSw.Start();
                    batch.Draw(commandList, device);
                    drawSw.Stop();
                }
                
                foreach (var batch in activeBatches)
                    batch.ResetBufferState(commandList);
                    
                    
                totalSw.Stop();
                
                bool isOpaque = pass == RenderPass.Opaque;
                
                // Expose stats for title bar
                if (isOpaque)
                {
                    CommandBuffer.LastDrawCallCount = drawCallCount;
                    CommandBuffer.LastBatchCount = activeBatches.Count;
                }
                
                if (Engine.FrameIndex % 60 == 0 && isOpaque)
                {
                    Debug.Log($"[Pass.Execute] DrawCalls: {drawCallCount} | Batches: {activeBatches.Count} | Total: {totalSw.Elapsed.TotalMilliseconds:F2}ms | Batching: {batchingSw.Elapsed.TotalMilliseconds:F2}ms | BuildBuffers: {buildSw.Elapsed.TotalMilliseconds:F2}ms | Material.Apply: {applySw.Elapsed.TotalMilliseconds:F2}ms | DrawFast: {drawSw.Elapsed.TotalMilliseconds:F2}ms");
                }
            }
        }

        private Pass[] passes;
        private Dictionary<RenderPass, List<Action<ID3D12GraphicsCommandList>>> customQueues = new Dictionary<RenderPass, List<Action<ID3D12GraphicsCommandList>>>();
        
        // Static GPU culler - shared by all batches (never run in parallel)
        public static GPUCuller? Culler { get; private set; }

        public static void InitializeCuller(GraphicsDevice device)
        {
            Culler = new GPUCuller(device);
            Culler.Initialize();
            Debug.Log("[CommandBuffer] GPU Culler initialized");
            
            // Create Hi-Z pyramid if renderer is already set up
            var renderer = DeferredRenderer.Current;
            if (renderer?.Depth != null)
            {
                int w = (int)renderer.Depth.Native.Description.Width;
                int h = (int)renderer.Depth.Native.Description.Height;
                Culler.CreateHiZPyramid(w, h);
            }
        }
        
        /// <summary>
        /// Get active batches from a render pass for shadow rendering access.
        /// Returns the instance batches that will be drawn in the specified pass.
        /// </summary>
        public static IReadOnlyList<InstanceBatch>? GetActiveBatches(RenderPass pass)
        {
            return current.passes[(int)pass].ActiveBatches;
        }
        
        /// <summary>
        /// Get ALL registered batches from a render pass, including off-screen objects.
        /// Used by shadow rendering so off-screen casters still cast visible shadows.
        /// </summary>
        public static IReadOnlyList<InstanceBatch>? GetAllBatches(RenderPass pass)
        {
            return current.passes[(int)pass].AllBatches;
        }

        static CommandBuffer()
        {
            current = new CommandBuffer();
        }

        private CommandBuffer()
        {
            passes = new Pass[Enum.GetValues(typeof(RenderPass)).Length];
            for (int i = 0; i < passes.Length; i++) passes[i] = new Pass();
            
            foreach (RenderPass pass in Enum.GetValues(typeof(RenderPass)))
                customQueues[pass] = new List<Action<ID3D12GraphicsCommandList>>();
        }

        public static void Enqueue(RenderPass pass, Action<ID3D12GraphicsCommandList> action)
        {
            current.customQueues[pass].Add(action);
        }

        /// <summary>
        /// Register a GPU-sourced batch where per-instance buffers were generated by compute shader.
        /// Bypasses CPU staging: the provided SRV indices point directly at GPU-filled buffers.
        /// </summary>
        public static void EnqueueGPUBatch(
            Material material,
            Mesh mesh,
            int meshPartId,
            uint descriptorsSRV,
            uint subbatchIdsSRV,
            int instanceCount,
            ReadOnlySpan<InstanceBatch.GPUBufferBinding> customBindings = default)
        {
            var key = new BatchKey(material.Effect);
            var pass = current.passes[(int)RenderPass.Opaque];

            pass.EnsureGPUBatch(key, material, mesh, meshPartId,
                descriptorsSRV, subbatchIdsSRV, instanceCount, customBindings);
        }


        /// <summary>
        /// Enqueue draw call into all applicable RenderPasses based on the Material's Effect passes.
        /// </summary>
        public static void Enqueue(Mesh mesh, int meshPartIndex, Material material, MaterialBlock materialBlock, int transformSlot = -1)
        {
            var key = new BatchKey(material.Effect);
            var drawCall = new DrawCall 
            {
                Key = key,
                Mesh = mesh,
                MeshPartIndex = meshPartIndex,
                Material = material,
                MaterialBlock = materialBlock,
                TransformSlot = transformSlot
            };
            
            // Iterate all passes defined in the Effect and enqueue to each
            foreach (var shaderPass in material.GetPasses())
            {
                if(shaderPass.RenderPass == RenderPass.Shadow)
                continue;
                current.passes[(int)shaderPass.RenderPass].Add(drawCall);
            }
        }
        
        /// <summary>
        /// Enqueue all mesh parts into all applicable RenderPasses based on the Material's Effect passes.
        /// </summary>
        public static void Enqueue(Mesh mesh, Material material, MaterialBlock materialBlock, int transformSlot = -1)
        {
            for (int i = 0; i < mesh.MeshParts.Count; i++)
            {
                if (mesh.MeshParts[i].Enabled)
                    Enqueue(mesh, i, material, materialBlock, transformSlot);
            }
        }

        public static void Execute(RenderPass pass, ID3D12GraphicsCommandList commandList, GraphicsDevice device)
        {
            // 1. Custom Actions (single-threaded submission, no concurrent modification possible)
            var queue = current.customQueues[pass];
            for (int i = 0; i < queue.Count; i++)
                queue[i](commandList);
            queue.Clear();

            // 2. Batches
            current.passes[(int)pass].Execute(commandList, device, pass);
        }

        public static void Clear()
        {
            foreach (var pass in current.passes) pass.Clear();
            foreach (var queue in current.customQueues.Values) queue.Clear();
        }
    }
}
