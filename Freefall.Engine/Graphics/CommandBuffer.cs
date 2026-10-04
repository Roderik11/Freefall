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
        /// <summary>
        /// Per-Animator bone buffer SRV index (0 = static mesh).
        /// </summary>
        public uint BoneBufferIdx;
        /// <summary>
        /// The GPU culler resolves the LOD: MeshPartIndex is a LOD chain head (Mesh.DrawPartIndices)
        /// and small-on-screen culling applies. False draws exactly the given part.
        /// </summary>
        public bool LodManaged;
    }

    /// <summary>
    /// The persistent (GPU-resident) draws of one renderer: instance records that stay in their
    /// batches across frames instead of being re-enqueued. Owned by the renderer, filled by
    /// CommandBuffer.AddPersistent, released as a whole by CommandBuffer.RemovePersistent.
    /// </summary>
    public sealed class DrawGroup
    {
        // One handle per (draw, render pass). Touched only on the main thread (CommandBuffer.FlushPersistent).
        internal readonly List<InstanceBatch.InstanceHandle> Handles = new();
    }

    /// <summary>
    /// A component whose draws are GPU-resident. It has no per-frame Draw(): when something its
    /// draws depend on changes it calls CommandBuffer.Invalidate(this), and RefreshDraws runs once
    /// on the main thread before the next frame is rendered.
    /// </summary>
    public interface IPersistentDrawSource
    {
        /// <summary>Bring the registered draws in line with the current state (main thread).</summary>
        void RefreshDraws();
    }

    /// <summary>
    /// Thread-local bucket for collecting GPU-ready arrays during parallel Enqueue.
    /// Stores parallel arrays that can be block-copied with Array.Copy.
    /// </summary>
    public class DrawBucket
    {
        /// <summary>
        /// Pre-staged per-instance data: hash → contiguous byte array (filled at Enqueue time).
        /// </summary>
        public class PerInstanceStaging
        {
            public int PushConstantSlot;    // Graphics push constant slot (from shader)
            public int ElementStride;       // Bytes per element
            public int ElementsPerInstance; // Elements per instance (1 for scalar, N for array)
            public byte[] Data = [];
            public int Count;               // Number of instances staged
            public int BytesPerInstance => ElementsPerInstance * ElementStride;
        }


        // Well-known hash keys for core per-instance data channels
        public static readonly int DescriptorsHash = "Descriptors".GetHashCode();
        // Removed: BoundingSpheresHash — bounding spheres now live in MeshRegistry
        public static readonly int SubbatchIdsHash = "SubbatchIds".GetHashCode();
        
        public Material? FirstMaterial;
        public HashSet<int> UniqueMeshPartIds = [];
        public List<InstanceBatch.RawDraw> Draws = [];
        public Dictionary<int, PerInstanceStaging> PerInstanceData = [];

        public int Count => Draws.Count;        
        
        /// <summary>High bit of a staged subbatch ID: the culler resolves the LOD for this instance.</summary>
        public const uint LodManagedBit = 0x80000000u;

        public void Add(Mesh mesh, int partIndex, Material material, MaterialBlock block, int transformSlot, uint boneBufferIdx = 0, bool lodManaged = false)
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
                CustomDataIdx = 0,
                MeshPartIdx = (uint)partIndex,
                BoneBufferIdx = boneBufferIdx
            };

            StageCore(DescriptorsHash, 20, descriptor);       // InstanceDescriptor: 20 bytes (5 uints)
            StageCore(SubbatchIdsHash, 4,  (uint)meshPartId | (lodManaged ? LodManagedBit : 0u)); // uint: 4 bytes
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
            
            // Shared frustum constant buffers for non-opaque passes (no Hi-Z)
            private static ID3D12Resource[]? _frustumConstantsBuffers;
            private static bool _frustumBuffersInitialized;
            
            private static void EnsureFrustumBuffers(GraphicsDevice device)
            {
                if (_frustumBuffersInitialized) return;
                
                _frustumConstantsBuffers = new ID3D12Resource[FrameCount];
                int bufferSize = 256;
                
                for (int i = 0; i < FrameCount; i++)
                    _frustumConstantsBuffers[i] = device.CreateUploadBuffer(bufferSize);
                _frustumBuffersInitialized = true;
            }
            
            /// <summary>
            /// Upload simple frustum planes (no Hi-Z) for non-opaque passes.
            /// </summary>
            private static ulong UploadSimpleFrustumPlanes(GraphicsDevice device, Vector4[] frustumPlanes, Vector3 cameraPosition, uint sortDirection)
            {
                EnsureFrustumBuffers(device);
                
                int frameIndex = Engine.FrameIndex % FrameCount;
                
                var constants = new GPUCuller.FrustumConstants
                {
                    Plane0 = frustumPlanes[0],
                    Plane1 = frustumPlanes[1],
                    Plane2 = frustumPlanes[2],
                    Plane3 = frustumPlanes[3],
                    Plane4 = frustumPlanes[4],
                    Plane5 = frustumPlanes[5],
                    CameraPosition = cameraPosition,
                    SortDirection = sortDirection,
                };
                
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
                bucket.Add(drawCall.Mesh, drawCall.MeshPartIndex, drawCall.Material, drawCall.MaterialBlock, drawCall.TransformSlot, drawCall.BoneBufferIdx, drawCall.LodManaged);
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

            /// <summary>Find or create the batch for a key (persistent draws; main thread).</summary>
            public InstanceBatch GetOrCreateBatch(BatchKey key, Material material)
            {
                if (!batches.TryGetValue(key, out var batch))
                {
                    batch = new InstanceBatch(key, material);
                    batches.Add(key, batch);
                    allBatches.Add(batch);
                }
                return batch;
            }

            /// <summary>
            /// Execute this pass: merge buckets, upload, build, cull, draw.
            /// </summary>
            /// <param name="frustumGpuAddr">GPU address of frustum constants. 
            /// If 0, a simple frustum (no Hi-Z) is built internally.</param>
            public void Execute(ID3D12GraphicsCommandList commandList, GraphicsDevice device, RenderPass pass, ulong frustumGpuAddr = 0)
            {
                var totalSw = System.Diagnostics.Stopwatch.StartNew();
                                
                // 0. Execute custom actions (compute shader dispatches etc.) before batch processing
                if (current.customQueues.TryGetValue(pass, out var actions) && actions.Count > 0)
                {
                    foreach (var action in actions)
                        action(commandList);
                    actions.Clear();
                }

                // Apply persistent add/remove requests queued during Draw(). Must happen before any
                // batch is cleared and merged this frame (persistent records sit in front of the frame's draws).
                FlushPersistent();

                // 1. Batch draw calls by Effect
                var batchingSw = System.Diagnostics.Stopwatch.StartNew();

                activeBatches.Clear();

                int drawCallCount = 0;

                // Batches with persistent instances are active every frame, even with no enqueued draws
                foreach (var batch in allBatches)
                {
                    if (batch.PersistentCount == 0 || batch._activeFrame == Engine.FrameIndex) continue;
                    batch._activeFrame = Engine.FrameIndex;
                    batch.Clear();
                    activeBatches.Add(batch);
                    drawCallCount += batch.PersistentCount;
                }

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
                
                // Batches with nothing to draw this frame must not keep last frame's instance count:
                // the shadow pass walks all batches, not only the active ones.
                foreach (var batch in allBatches)
                {
                    if (batch._activeFrame != Engine.FrameIndex)
                        batch.Deactivate();
                }

                batchingSw.Stop();

                // GPU culling requires Culler to be initialized
                var sortDrawCallsSw = System.Diagnostics.Stopwatch.StartNew();
                bool usingGPUPath = Culler != null && pass == RenderPass.Opaque;

                // 2. Build frustum if not provided externally
                ulong frustumBufferGPUAddress = frustumGpuAddr;
                if (frustumBufferGPUAddress == 0)
                {
                    var vpMatrix = Engine.Settings.FreezeFrustum 
                        ? Engine.Settings.FrozenViewProjection 
                        : Camera.Main.ViewProjection;
                    var frustum = new Frustum(vpMatrix);
                    var frustumPlanes = frustum.GetPlanesAsVector4();
                    // Sort direction: 0 = front-to-back (opaque), 1 = back-to-front (transparent/forward)
                    uint sortDir = (pass == RenderPass.Forward) ? 1u : 0u;
                    frustumBufferGPUAddress = UploadSimpleFrustumPlanes(device, frustumPlanes, Camera.Main.Position, sortDir);
                }

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
                    // Push constant slot 16: debug mode (not touched by command signature slots 2-15)
                    commandList.SetGraphicsRoot32BitConstant(0, (uint)Engine.Settings.DebugVisualizationMode, 16);
                    
                    // Forward pass: set slots 0-1 for transparent shaders (shadow map + composite snapshot)
                    // These are 'reserved' in opaque/sky shaders and harmless to overwrite.
                    // Custom actions (ocean) run before batches and set their own root state.
                    if (pass == RenderPass.Forward)
                    {
                        var renderer = DeferredRenderer.Current;
                        if (renderer != null)
                        {
                            commandList.SetGraphicsRoot32BitConstant(0, renderer.ShadowTextureArray?.BindlessIndex ?? 0u, 0);
                            commandList.SetGraphicsRoot32BitConstant(0, renderer.CompositeSnapshot?.BindlessIndex ?? 0u, 1);
                        }
                    }
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
                  //  Debug.Log($"[Pass.Execute] DrawCalls: {drawCallCount} | Batches: {activeBatches.Count} | Total: {totalSw.Elapsed.TotalMilliseconds:F2}ms | Batching: {batchingSw.Elapsed.TotalMilliseconds:F2}ms | BuildBuffers: {buildSw.Elapsed.TotalMilliseconds:F2}ms | Material.Apply: {applySw.Elapsed.TotalMilliseconds:F2}ms | DrawFast: {drawSw.Elapsed.TotalMilliseconds:F2}ms");
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
        /// Check if a render pass has any pending custom actions.
        /// Use to skip expensive setup (resource copies, transitions) for empty passes.
        /// </summary>
        public static bool HasPendingCommands(RenderPass pass)
        {
            return current.customQueues[pass].Count > 0;
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
        /// <param name="lodManaged">See <see cref="DrawCall.LodManaged"/>.</param>
        public static void Enqueue(Mesh mesh, int meshPartIndex, Material material, MaterialBlock materialBlock, int transformSlot = -1, uint boneBufferIdx = 0, bool lodManaged = false)
        {
            var key = new BatchKey(material.Effect);
            var drawCall = new DrawCall 
            {
                Key = key,
                Mesh = mesh,
                MeshPartIndex = meshPartIndex,
                Material = material,
                MaterialBlock = materialBlock,
                TransformSlot = transformSlot,
                BoneBufferIdx = boneBufferIdx,
                LodManaged = lodManaged
            };

            // Iterate all passes defined in the Effect and enqueue to each
            foreach (var shaderPass in material.GetPasses())
            {
                if(shaderPass.RenderPass == RenderPass.Shadow)
                continue;
                current.passes[(int)shaderPass.RenderPass].Add(drawCall);
            }
        }
        
        #region Persistent (GPU-resident) draws

        private struct PersistentOp
        {
            public DrawGroup Group;
            public bool Add;      // false = remove every record of the group
            public DrawCall Call; // Add only
        }

        private static int _lastFlushTick = -1;
        private static readonly List<PersistentOp> _persistentOps = new();
        private static readonly Lock _persistentLock = new();

        /// <summary>
        /// Register a draw that stays in its batches until the group is removed: nothing is enqueued
        /// per frame for it. Same parameters as Enqueue. Thread-safe (callable from parallel Draw());
        /// takes effect at the next pass execution. To change a registered draw, remove the group and
        /// add a new one.
        /// </summary>
        public static void AddPersistent(DrawGroup group, Mesh mesh, int meshPartIndex, Material material, MaterialBlock materialBlock, int transformSlot, uint boneBufferIdx = 0, bool lodManaged = false)
        {
            var op = new PersistentOp
            {
                Group = group,
                Add = true,
                Call = new DrawCall
                {
                    Key = new BatchKey(material.Effect),
                    Mesh = mesh,
                    MeshPartIndex = meshPartIndex,
                    Material = material,
                    MaterialBlock = materialBlock,
                    TransformSlot = transformSlot,
                    BoneBufferIdx = boneBufferIdx,
                    LodManaged = lodManaged
                }
            };

            lock (_persistentLock)
                _persistentOps.Add(op);
        }

        /// <summary>
        /// Remove every persistent draw of a group. Thread-safe; takes effect at the next pass execution.
        /// </summary>
        public static void RemovePersistent(DrawGroup group)
        {
            lock (_persistentLock)
                _persistentOps.Add(new PersistentOp { Group = group, Add = false });
        }

        /// <summary>
        /// Change the bone buffer SRV index of every record of a group, in place. For skinned meshes:
        /// the Animator's bone buffer is triple-buffered, so its SRV index changes every frame, and
        /// re-registering the draws each frame would defeat the point of persistent draws.
        ///
        /// Safe without a lock from the (parallel) update phase: it only writes this group's own
        /// records in the staging arrays, which are re-uploaded in full when the pass executes, and
        /// batches are only restructured during rendering (FlushPersistent, bucket merge).
        /// Records not applied yet (group registered this frame) get the value from their DrawCall.
        /// </summary>
        public static void SetBoneBuffer(DrawGroup group, uint boneBufferIdx)
        {
            var handles = group.Handles;
            for (int i = 0; i < handles.Count; i++)
                handles[i].Batch.SetPersistentBoneBuffer(handles[i], boneBufferIdx);
        }

        private static List<IPersistentDrawSource> _invalidSources = new();
        private static List<IPersistentDrawSource> _refreshingSources = new();
        private static readonly Lock _invalidLock = new();

        /// <summary>
        /// Ask for <paramref name="source"/>.RefreshDraws() to run before the next frame is rendered.
        /// Thread-safe. The source is responsible for not queueing itself twice.
        /// </summary>
        public static void Invalidate(IPersistentDrawSource source)
        {
            lock (_invalidLock)
                _invalidSources.Add(source);
        }

        /// <summary>
        /// Run RefreshDraws on every invalidated source. Main thread. The renderer calls this before
        /// the transform upload, so a transform slot first allocated by a refresh is uploaded the same
        /// frame; FlushPersistent calls it again for anything invalidated later.
        /// </summary>
        public static void RefreshDrawSources()
        {
            lock (_invalidLock)
            {
                if (_invalidSources.Count == 0) return;
                (_invalidSources, _refreshingSources) = (_refreshingSources, _invalidSources);
            }

            // Outside the lock: a refresh may invalidate other sources (they run next time)
            foreach (var source in _refreshingSources)
                source.RefreshDraws();
            _refreshingSources.Clear();
        }

        /// <summary>
        /// Apply queued persistent adds/removes in order. Main thread, called at the start of every
        /// pass execution but only acting on the first one per tick, so the batches of all passes are
        /// up to date before any is merged or culled and then stay fixed for the frame.
        /// </summary>
        private static void FlushPersistent()
        {
            // Once per tick, at the first pass that executes. Batches must not change after that:
            // the passes of a frame share them (the shadow pass culls the opaque pass's batches), and
            // an add can grow a batch, which disposes and re-creates its GPU buffers. Doing that after
            // the batch was culled would leave the shadow pass recording commands on null buffers.
            // Requests queued mid-frame wait for the next one.
            if (_lastFlushTick == Engine.TickCount) return;
            _lastFlushTick = Engine.TickCount;

            RefreshDrawSources();

            lock (_persistentLock)
            {
                if (_persistentOps.Count == 0) return;

                foreach (var op in _persistentOps)
                {
                    if (op.Add)
                    {
                        // One record per render pass of the material's effect, as Enqueue does
                        foreach (var shaderPass in op.Call.Material.GetPasses())
                        {
                            if (shaderPass.RenderPass == RenderPass.Shadow)
                                continue;
                            var batch = current.passes[(int)shaderPass.RenderPass].GetOrCreateBatch(op.Call.Key, op.Call.Material);
                            op.Group.Handles.Add(batch.AddPersistent(op.Call));
                        }
                    }
                    else
                    {
                        foreach (var handle in op.Group.Handles)
                            handle.Batch.RemovePersistent(handle);
                        op.Group.Handles.Clear();
                    }
                }

                _persistentOps.Clear();
            }
        }

        #endregion

        /// <summary>
        /// Enqueue all mesh parts into all applicable RenderPasses based on the Material's Effect passes.
        /// </summary>
        public static void Enqueue(Mesh mesh, Material material, MaterialBlock materialBlock, int transformSlot = -1, uint boneBufferIdx = 0)
        {
            for (int i = 0; i < mesh.MeshParts.Count; i++)
            {
                if (mesh.MeshParts[i].Enabled)
                    Enqueue(mesh, i, material, materialBlock, transformSlot, boneBufferIdx);
            }
        }

        public static void Execute(RenderPass pass, ID3D12GraphicsCommandList commandList, GraphicsDevice device, ulong frustumGpuAddr = 0)
        {
            // 1. Custom Actions (single-threaded submission, no concurrent modification possible)
            var queue = current.customQueues[pass];
            for (int i = 0; i < queue.Count; i++)
                queue[i](commandList);
            queue.Clear();

            // 2. Batches
            current.passes[(int)pass].Execute(commandList, device, pass, frustumGpuAddr);
        }

        /// <summary>
        /// Execute only the custom actions for a pass WITHOUT clearing. 
        /// Use with ClearCustomActions after a loop.
        /// </summary>
        public static void ExecuteCustomActions(RenderPass pass, ID3D12GraphicsCommandList commandList)
        {
            var queue = current.customQueues[pass];
            for (int i = 0; i < queue.Count; i++)
                queue[i](commandList);
        }

        /// <summary>
        /// Clear custom action queue for a render pass.
        /// </summary>
        public static void ClearCustomActions(RenderPass pass)
        {
            current.customQueues[pass].Clear();
        }

        public static void Clear()
        {
            foreach (var pass in current.passes) pass.Clear();
            foreach (var queue in current.customQueues.Values) queue.Clear();
        }
    }
}
