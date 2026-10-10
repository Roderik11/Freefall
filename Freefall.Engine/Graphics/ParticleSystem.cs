using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Freefall.Base;
using Freefall.Components;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    /// <summary>
    /// Simulates and draws every <see cref="ParticleEmitter"/> from one shared GPU pool.
    ///
    /// The pool is cut into chunks of <see cref="ChunkSize"/> slots (one compute thread group); an emitter
    /// owns a contiguous run of chunks sized from EmitRate x Lifetime and uses it as a ring: each frame
    /// the CPU advances a cursor and the slots under it respawn. No dead list, no per-emitter buffers.
    ///
    /// Per frame: this system (UpdateGroup, after scripts) builds one table row per emitter.
    /// <see cref="ParticleRenderSystem"/> (RenderGroup) uploads the table and enqueues, per rendered view,
    /// a cull dispatch (emitter bounds vs frustum + Hi-Z), one simulate dispatch over the whole pool
    /// that also appends live particles of visible emitters to a draw list, and one indirect draw per
    /// render mode. See .agent/knowledge/rendering_engine/features/particles.md.
    /// </summary>
    [UpdateInEditor]
    public sealed class ParticleSystem : EntitySystem
    {
        public const int ChunkSize = 256;           // = PARTICLE_CHUNK_SIZE
        private const uint NoEmitter = 0xFFFFFFFF;  // = PARTICLE_NO_EMITTER
        private const int MinPoolChunks = 64;
        private const int MaxPoolChunks = 16384;    // 4.2M particles
        private const int MaxEmitterChunks = 4096;  // 1M particles
        private const int RenderModeCount = 2;      // = PARTICLE_MODE_COUNT
        private const int CoreStride = 32;          // ParticleCore
        private const int VisualStride = 16;        // ParticleVisual
        private const uint FlagReset = 1;           // = PARTICLE_FLAG_RESET

        /// <summary>One table row. Must match struct ParticleEmitter in particle_common.hlsli.</summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct EmitterRow
        {
            public Vector3 Position;       public float ParticleRadius;
            public Vector3 EmitDirection;  public float SpreadCos;
            public Vector3 Gravity;        public float Lifetime;
            public Vector3 ShapeExtents;   public float ShapeRadius;
            public Vector3 AxisX;          public float ConeTan;
            public Vector3 AxisY;          public float LifetimeRandomness;
            public Vector3 AxisZ;          public float SizeRandomness;
            public Vector3 Wind;           public float Drag;
            public Vector2 SpeedRange;     public float RotationRange;      public float Bounciness;
            public uint Shape;             public uint DirectionMode;       public uint EmitFromShell;      public uint CollisionMode;
            public uint CollisionResponse; public float PlaneHeight;        public float CollisionThickness; public uint RandomSeed;
            public uint FirstSlot;         public uint Capacity;            public uint EmitStart;          public uint EmitCount;
            public Vector4 ColorStart;
            public Vector4 ColorEnd;
            public Vector2 SizeStartEnd;   public float Aspect;             public float StretchFactor;
            public uint TextureIdx;        public uint FlipbookCols;        public uint FlipbookRows;       public uint FlipbookFrameCount;
            public float FlipbookAnimSpeed; public uint BillboardMode;      public uint SoftEnabled;        public float SoftRange;
            public Vector3 BoundsCenter;   public float BoundsRadius;
            public uint RenderMode;        public uint Flags;               public uint Pad0;               public uint Pad1;
        }

        /// <summary>Everything the system keeps per emitter. The component itself only holds authored data.</summary>
        private sealed class Entry
        {
            public ParticleEmitter Emitter = null!;
            public int FirstChunk = -1;
            public int ChunkCount;
            public uint Cursor;          // next ring slot to respawn
            public float Accumulator;    // fractional particles carried to the next frame
            public uint Frame;
            public bool Reset;           // its chunks hold another emitter's particles
            public long ResetSerial;
            public bool PoolFull;

            // Derived values, recomputed only when their source changes (no trig per emitter per frame)
            public float SpreadAngle = float.NaN;
            public float ConeAngle = float.NaN;
            public Vector3 EmitDirection = new(float.NaN);
            public float SpreadCos;
            public float ConeTan;
            public Vector3 Direction;
        }

        private readonly List<Entry> _entries = [];
        private EmitterRow[] _rows = new EmitterRow[64];
        private int _rowCount;
        private readonly int[] _modeCount = new int[RenderModeCount];
        private bool _anyDepthCollision;

        // Chunk allocator: free runs sorted by start
        private readonly List<(int Start, int Count)> _free = [];
        private int _chunkCapacity;
        private int _highWater;   // chunks up to here are dispatched

        // Which gather the GPU has seen (Reset flags must survive frames that were never rendered)
        private long _gatherSerial;
        private long _uploadedSerial;
        private long _simulatedSerial = -1;

        // GPU
        private StreamingBuffer<EmitterRow>? _emitters;
        private StreamingBuffer<uint>? _chunkMap;
        private GraphicsBuffer? _core;
        private GraphicsBuffer? _visual;
        private GraphicsBuffer? _drawList;     // _poolSlots entries per render mode
        private GraphicsBuffer? _drawArgs;     // 4 uints per render mode
        private GraphicsBuffer? _visibility;   // one uint per emitter row
        private GraphicsBuffer?[] _frameConstants = new GraphicsBuffer?[CommandBuffer.FrameCount];
        private int _poolSlots;

        // Pool contents still to be copied into the grown buffers
        private GraphicsBuffer? _copyCore;
        private GraphicsBuffer? _copyVisual;
        private int _copySlots;

        private ComputeShader? _compute;
        private int _kCull, _kSimulate;
        private Effect? _drawEffect;
        private Material? _drawMaterial;
        private ID3D12DescriptorHeap[]? _heaps;
        private bool _failed;

        private int _preparedTick = -1;
        private bool _prepared;
        private int _simulatedTick = -1;
        private bool _drawReady;
        private int _uploadedRows;
        private int _uploadedHighWater;

        /// <summary>Emitters currently registered.</summary>
        public int EmitterCount => _entries.Count;

        /// <summary>Pool slots handed out to emitters (the simulate dispatch covers up to the highest one).</summary>
        public int AllocatedSlots { get; private set; }

        /// <summary>Slots the GPU pool can hold.</summary>
        public int PoolSlots => _poolSlots;

        // ────────────── Emitter registration ──────────────

        protected internal override void Initialize()
        {
            if (Unsafe.SizeOf<EmitterRow>() != 304)
                throw new InvalidOperationException("EmitterRow must match struct ParticleEmitter in particle_common.hlsli (304 bytes)");

            ComponentCache<ParticleEmitter>.Added += OnAdded;
            ComponentCache<ParticleEmitter>.Removed += OnRemoved;

            foreach (var emitter in ComponentCache<ParticleEmitter>.All)
                OnAdded(emitter);
        }

        protected internal override void Destroy()
        {
            ComponentCache<ParticleEmitter>.Added -= OnAdded;
            ComponentCache<ParticleEmitter>.Removed -= OnRemoved;
        }

        private void OnAdded(ParticleEmitter emitter)
        {
            if (emitter.SystemIndex >= 0) return;
            emitter.SystemIndex = _entries.Count;
            _entries.Add(new Entry { Emitter = emitter });
        }

        private void OnRemoved(ParticleEmitter emitter)
        {
            int index = emitter.SystemIndex;
            if (index >= 0 && index < _entries.Count && _entries[index].Emitter == emitter)
                RemoveEntry(index);
        }

        private void RemoveEntry(int index)
        {
            var entry = _entries[index];
            ReleaseChunks(entry);
            entry.Emitter.SystemIndex = -1;

            int last = _entries.Count - 1;
            if (index != last)
            {
                // The last entry takes the freed row: its chunks have to name the new row
                var moved = _entries[last];
                _entries[index] = moved;
                moved.Emitter.SystemIndex = index;
                MapChunks(moved, (uint)index);
                if (last < _rows.Length) _rows[index] = _rows[last];
            }
            _entries.RemoveAt(last);
        }

        // ────────────── Chunk allocation ──────────────

        private static int ChunksFor(ParticleEmitter emitter)
        {
            // Most particles alive at once, plus headroom for the frame-sized emission steps
            float life = MathF.Max(0.01f, emitter.Lifetime) * (1f + Math.Clamp(emitter.LifetimeRandomness, 0f, 1f));
            float slots = MathF.Min(MathF.Max(0f, emitter.EmitRate) * life * 1.05f + 16f, MaxEmitterChunks * (float)ChunkSize);
            return Math.Clamp((int)MathF.Ceiling(slots / ChunkSize), 1, MaxEmitterChunks);
        }

        private void Reallocate(Entry entry, int index, int chunks)
        {
            ReleaseChunks(entry);

            int start = AllocateChunks(chunks);
            if (start < 0)
            {
                if (!entry.PoolFull)
                    Debug.LogWarning("ParticleSystem", $"Particle pool is full ({MaxPoolChunks * ChunkSize} particles): '{entry.Emitter.Entity?.Name}' emits nothing");
                entry.PoolFull = true;
                return;
            }

            entry.PoolFull = false;
            entry.FirstChunk = start;
            entry.ChunkCount = chunks;
            entry.Cursor = 0;
            entry.Reset = true;
            entry.ResetSerial = _gatherSerial;
            AllocatedSlots += chunks * ChunkSize;
            _highWater = Math.Max(_highWater, start + chunks);
            MapChunks(entry, (uint)index);
        }

        private void ReleaseChunks(Entry entry)
        {
            if (entry.ChunkCount == 0) return;

            MapChunks(entry, NoEmitter);
            FreeChunks(entry.FirstChunk, entry.ChunkCount);
            AllocatedSlots -= entry.ChunkCount * ChunkSize;
            entry.FirstChunk = -1;
            entry.ChunkCount = 0;

            _highWater = 0;
            foreach (var other in _entries)
                _highWater = Math.Max(_highWater, other.FirstChunk + other.ChunkCount);
        }

        private void MapChunks(Entry entry, uint row)
        {
            for (int c = 0; c < entry.ChunkCount; c++)
                _chunkMap!.Set(entry.FirstChunk + c, row);
        }

        private int AllocateChunks(int count)
        {
            while (true)
            {
                for (int i = 0; i < _free.Count; i++)
                {
                    var (start, free) = _free[i];
                    if (free < count) continue;

                    if (free == count) _free.RemoveAt(i);
                    else _free[i] = (start + count, free - count);
                    return start;
                }

                if (_chunkCapacity >= MaxPoolChunks) return -1;

                // Grow: the new chunks join the free list, the GPU buffers follow in PrepareFrame
                int old = _chunkCapacity;
                _chunkCapacity = Math.Min(MaxPoolChunks, Math.Max(MinPoolChunks, old * 2));

                _chunkMap ??= new StreamingBuffer<uint>(Engine.Device, MinPoolChunks);
                _chunkMap.EnsureCapacity(_chunkCapacity);
                for (int c = old; c < _chunkCapacity; c++)
                    _chunkMap.Set(c, NoEmitter);

                FreeChunks(old, _chunkCapacity - old);
            }
        }

        private void FreeChunks(int start, int count)
        {
            int i = 0;
            while (i < _free.Count && _free[i].Start < start) i++;
            _free.Insert(i, (start, count));

            if (i + 1 < _free.Count && _free[i].Start + _free[i].Count == _free[i + 1].Start)
            {
                _free[i] = (_free[i].Start, _free[i].Count + _free[i + 1].Count);
                _free.RemoveAt(i + 1);
            }

            if (i > 0 && _free[i - 1].Start + _free[i - 1].Count == _free[i].Start)
            {
                _free[i - 1] = (_free[i - 1].Start, _free[i - 1].Count + _free[i].Count);
                _free.RemoveAt(i);
            }
        }

        // ────────────── Gather (UpdateGroup, after scripts) ──────────────

        public override void Update()
        {
            // An emitter destroyed before its Added notification never gets a Removed one
            for (int i = _entries.Count - 1; i >= 0; i--)
            {
                var emitter = _entries[i].Emitter;
                if (emitter.IsDestroyed && !emitter.Announced)
                    RemoveEntry(i);
            }

            int count = _entries.Count;
            if (_rows.Length < count)
                Array.Resize(ref _rows, Math.Max(count, _rows.Length * 2));

            _gatherSerial++;
            _anyDepthCollision = false;
            Array.Clear(_modeCount);

            float dt = (float)Time.Delta;
            for (int i = 0; i < count; i++)
                BuildRow(_entries[i], i, ref _rows[i], dt);

            _rowCount = count;
        }

        private void BuildRow(Entry entry, int index, ref EmitterRow row, float dt)
        {
            var e = entry.Emitter;

            // Destroyed or detached this frame (Removed arrives at the next wake-up): let its particles run out
            if (e.IsDestroyed || e.Entity == null)
            {
                row.EmitCount = 0;
                row.Flags = 0;
                return;
            }

            // Pool space follows the authored rate. Shrinking waits for a factor of two so a rate that
            // is being animated does not restart the emitter every frame.
            int wanted = ChunksFor(e);
            if (wanted > entry.ChunkCount || wanted * 2 < entry.ChunkCount)
                Reallocate(entry, index, wanted);

            // The GPU has killed the previous owner's particles once a simulate ran with the flag
            if (entry.Reset && _simulatedSerial >= entry.ResetSerial)
                entry.Reset = false;

            uint capacity = (uint)(entry.ChunkCount * ChunkSize);

            float rate = e.Enabled ? MathF.Max(0f, e.EmitRate) * MathF.Max(0f, e.EmitRateScale) : 0f;
            entry.Accumulator += rate * dt;
            uint emit = (uint)entry.Accumulator;
            entry.Accumulator -= emit;
            if (emit > capacity) emit = capacity;

            if (e.SpreadAngle != entry.SpreadAngle)
            {
                entry.SpreadAngle = e.SpreadAngle;
                entry.SpreadCos = MathF.Cos(Math.Clamp(e.SpreadAngle, 0f, 180f) * (MathF.PI / 180f));
            }
            if (e.ConeAngle != entry.ConeAngle)
            {
                entry.ConeAngle = e.ConeAngle;
                entry.ConeTan = MathF.Tan(Math.Clamp(e.ConeAngle, 0f, 89f) * (MathF.PI / 180f));
            }
            if (e.EmitDirection != entry.EmitDirection)
            {
                entry.EmitDirection = e.EmitDirection;
                entry.Direction = e.EmitDirection.LengthSquared() > 1e-8f ? Vector3.Normalize(e.EmitDirection) : Vector3.UnitY;
            }

            var transform = e.Transform!;
            var position = transform.WorldPosition;
            var wind = e.WindOverride ?? e.Wind;
            float drag = MathF.Max(0f, e.Drag);
            float lifetime = MathF.Max(0.01f, e.Lifetime);
            float lifeRandom = Math.Clamp(e.LifetimeRandomness, 0f, 1f);
            float sizeRandom = Math.Clamp(e.SizeRandomness, 0f, 1f);
            float aspect = MathF.Max(0.01f, e.Aspect);
            float stretch = MathF.Max(0f, e.StretchFactor);
            float speedMin = MathF.Min(e.SpeedRange.X, e.SpeedRange.Y);
            float speedMax = MathF.Max(e.SpeedRange.X, e.SpeedRange.Y);
            bool depthCollision = e.CollisionMode == ParticleCollisionMode.Depth;
            _anyDepthCollision |= depthCollision;

            // Bounds for culling, from |x|+|y|+|z| norms: a little loose, but no square roots.
            // Gravity moves a particle by up to g*L²/2, so the sphere sits a quarter of the way along.
            float life = lifetime * (1f + lifeRandom);
            float gravity = MathF.Abs(e.Gravity.X) + MathF.Abs(e.Gravity.Y) + MathF.Abs(e.Gravity.Z);
            float drift = drag > 0f ? (MathF.Abs(wind.X) + MathF.Abs(wind.Y) + MathF.Abs(wind.Z)) : 0f;
            float topSpeed = MathF.Max(MathF.Abs(speedMin), MathF.Abs(speedMax));
            float shape = e.Shape switch
            {
                EmissionShape.Point => 0f,
                EmissionShape.Box => MathF.Abs(e.ShapeExtents.X) + MathF.Abs(e.ShapeExtents.Y) + MathF.Abs(e.ShapeExtents.Z),
                _ => MathF.Abs(e.ShapeRadius),
            };
            float size = MathF.Max(e.StartSize, e.EndSize) * (1f + sizeRandom);
            float particleRadius = 0.5f * size * (1f + aspect);
            if (e.BillboardMode == ParticleBillboardMode.VelocityStretched)
                particleRadius += 0.5f * stretch * (topSpeed + gravity * life + drift);

            int mode = e.RenderMode == ParticleRenderMode.Transparent ? 1 : 0;
            _modeCount[mode]++;

            row = new EmitterRow
            {
                Position = position,
                ParticleRadius = particleRadius,
                EmitDirection = entry.Direction,
                SpreadCos = entry.SpreadCos,
                Gravity = e.Gravity,
                Lifetime = lifetime,
                ShapeExtents = Vector3.Abs(e.ShapeExtents),
                ShapeRadius = MathF.Abs(e.ShapeRadius),
                AxisX = transform.Right,
                ConeTan = entry.ConeTan,
                AxisY = transform.Up,
                LifetimeRandomness = lifeRandom,
                AxisZ = transform.Forward,
                SizeRandomness = sizeRandom,
                Wind = wind,
                Drag = drag,
                SpeedRange = new Vector2(speedMin, speedMax),
                RotationRange = e.RotationRange,
                Bounciness = Math.Clamp(e.Bounciness, 0f, 1f),
                Shape = (uint)e.Shape,
                DirectionMode = (uint)e.DirectionMode,
                EmitFromShell = e.EmitFromShell ? 1u : 0u,
                CollisionMode = (uint)e.CollisionMode,
                CollisionResponse = (uint)e.CollisionResponse,
                PlaneHeight = e.PlaneHeight,
                CollisionThickness = MathF.Max(0.01f, e.CollisionThickness),
                RandomSeed = Hash(++entry.Frame + (uint)e.Id * 0x9E3779B9u),
                FirstSlot = (uint)(Math.Max(entry.FirstChunk, 0) * ChunkSize),
                Capacity = capacity,
                EmitStart = entry.Cursor,
                EmitCount = emit,
                ColorStart = e.ColorStart,
                ColorEnd = e.ColorEnd,
                SizeStartEnd = new Vector2(e.StartSize, e.EndSize),
                Aspect = aspect,
                StretchFactor = stretch,
                TextureIdx = e.ParticleTexture?.BindlessIndex ?? 0u,
                FlipbookCols = (uint)Math.Max(1, e.FlipbookColumns),
                FlipbookRows = (uint)Math.Max(1, e.FlipbookRows),
                FlipbookFrameCount = (uint)Math.Max(0, e.FlipbookFrameCount),
                FlipbookAnimSpeed = e.FlipbookAnimSpeed,
                BillboardMode = (uint)e.BillboardMode,
                SoftEnabled = e.SoftParticles ? 1u : 0u,
                SoftRange = e.SoftRange,
                BoundsCenter = position + e.Gravity * (0.25f * life * life),
                BoundsRadius = shape + topSpeed * life + 0.25f * gravity * life * life + drift * life + particleRadius,
                RenderMode = (uint)mode,
                Flags = entry.Reset ? FlagReset : 0u,
            };

            if (capacity > 0)
                entry.Cursor = (entry.Cursor + emit) % capacity;
        }

        // PCG hash, same as the shader's
        private static uint Hash(uint input)
        {
            uint state = input * 747796405u + 2891336453u;
            uint word = ((state >> (int)((state >> 28) + 4)) ^ state) * 277803737u;
            return (word >> 22) ^ word;
        }

        // ────────────── Render side (called by ParticleRenderSystem) ──────────────

        /// <summary>
        /// Once per tick, before the first view enqueues: size the GPU pool and upload this frame's table.
        /// False when there is nothing to simulate or draw.
        /// </summary>
        internal bool PrepareFrame()
        {
            if (_preparedTick == Engine.TickCount) return _prepared;
            _preparedTick = Engine.TickCount;
            _prepared = false;

            if (_failed || _rowCount == 0 || _chunkCapacity == 0) return false;

            try
            {
                EnsureResources();
            }
            catch (Exception ex)
            {
                _failed = true;
                Debug.LogError("ParticleSystem", $"Init failed, particles are disabled: {ex.Message}");
                return false;
            }

            _emitters!.BulkWrite(_rows.AsSpan(0, _rowCount));
            _chunkMap!.Upload();

            _uploadedSerial = _gatherSerial;
            _uploadedRows = _rowCount;
            _uploadedHighWater = _highWater;
            return _prepared = _highWater > 0;
        }

        private void EnsureResources()
        {
            var device = Engine.Device;

            if (_compute == null)
            {
                _compute = new ComputeShader("particle_compute.hlsl");
                _kCull = _compute.FindKernel("CSCull");
                _kSimulate = _compute.FindKernel("CSSimulate");

                _drawEffect = new Effect("particle_draw");
                _drawMaterial = new Material(_drawEffect);

                _drawArgs = GraphicsBuffer.CreateRaw(RenderModeCount * 4, uav: true);
                for (int i = 0; i < _frameConstants.Length; i++)
                    _frameConstants[i] = GraphicsBuffer.CreateConstantBuffer<Matrix4x4>();

                _emitters = new StreamingBuffer<EmitterRow>(device, 64);
                _heaps = [device.SrvHeap];
            }

            int slots = _chunkCapacity * ChunkSize;
            if (_poolSlots < slots)
            {
                var core = GraphicsBuffer.CreateStructured(slots, CoreStride, srv: true, uav: true);
                var visual = GraphicsBuffer.CreateStructured(slots, VisualStride, srv: true, uav: true);

                if (_core != null)
                {
                    // Live particles move over on the command list (CopyGrownPool). If a copy is
                    // already pending, the buffers in between never held anything newer.
                    if (_copyCore == null)
                    {
                        _copyCore = _core;
                        _copyVisual = _visual;
                        _copySlots = _poolSlots;
                    }
                    else
                    {
                        device.DeferDispose(_core);
                        device.DeferDispose(_visual);
                    }
                }

                device.DeferDispose(_drawList);
                _drawList = GraphicsBuffer.CreateStructured<uint>(slots * RenderModeCount, srv: true, uav: true);

                _core = core;
                _visual = visual;
                _poolSlots = slots;
            }

            if (_visibility == null || _visibility.ElementCount < _rowCount)
            {
                device.DeferDispose(_visibility);
                _visibility = GraphicsBuffer.CreateStructured<uint>(Math.Max(64, _rowCount * 2), uav: true);
            }
        }

        /// <summary>Transparent pass: cull + simulate for this view, then its Transparent-mode particles.</summary>
        internal void RenderTransparent(ID3D12GraphicsCommandList cmd)
        {
            _drawReady = Compute(cmd);
            if (_drawReady) Draw(cmd, 1);
        }

        /// <summary>Forward pass of the same view: Forward-mode particles from the lists built above.</summary>
        internal void RenderForward(ID3D12GraphicsCommandList cmd)
        {
            if (_drawReady) Draw(cmd, 0);
            _drawReady = false;
        }

        private bool Compute(ID3D12GraphicsCommandList cmd)
        {
            var renderer = DeferredRenderer.Current;
            if (renderer == null || renderer.FrustumConstantsAddress == 0) return false;

            var device = Engine.Device;

            // Only the first view of a tick advances the simulation; later ones just build their lists
            bool simulate = _simulatedTick != Engine.TickCount;
            _simulatedTick = Engine.TickCount;

            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, _heaps!);

            CopyGrownPool(cmd);

            _core!.Transition(cmd, ResourceStates.UnorderedAccess);
            _visual!.Transition(cmd, ResourceStates.UnorderedAccess);
            _drawList!.Transition(cmd, ResourceStates.UnorderedAccess);
            _drawArgs!.Transition(cmd, ResourceStates.UnorderedAccess);
            _visibility!.Transition(cmd, ResourceStates.UnorderedAccess);

            // Depth collision reads this frame's depth and normals. The pass runs inside the G-buffer
            // fill, where both are still bound as render targets: make them readable for the dispatch.
            var camera = Camera.Main;
            bool depth = simulate && _anyDepthCollision && camera != null
                && renderer.DepthGBuffer != null && renderer.Normals != null;
            var frameConstants = _frameConstants[Engine.FrameIndex % _frameConstants.Length]!;
            if (depth)
            {
                unsafe { *frameConstants.WritePtr<Matrix4x4>() = camera!.ViewProjection; }
                cmd.ResourceBarrierTransition(renderer.DepthGBuffer!.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
                cmd.ResourceBarrierTransition(renderer.Normals!.Native, ResourceStates.RenderTarget, ResourceStates.NonPixelShaderResource);
            }

            cmd.SetComputeRootConstantBufferView(1, renderer.FrustumConstantsAddress);            // b0 FrustumPlanes
            cmd.SetComputeRootConstantBufferView(2, frameConstants.Native.GPUVirtualAddress);     // b1 ParticleFrame

            var cs = _compute!;
            cs.SetPushConstant("Emitters", _emitters!.SrvIndex);
            cs.SetPushConstant("ChunkMap", _chunkMap!.SrvIndex);
            cs.SetPushConstant("ParticleCoreUAV", _core.UavIndex);
            cs.SetPushConstant("ParticleVisualUAV", _visual.UavIndex);
            cs.SetPushConstant("VisibilityUAV", _visibility.UavIndex);
            cs.SetPushConstant("DrawListUAV", _drawList.UavIndex);
            cs.SetPushConstant("DrawArgsUAV", _drawArgs.UavIndex);
            cs.SetPushConstant("EmitterCount", (uint)_uploadedRows);
            cs.SetPushConstant("PoolSlots", (uint)_poolSlots);
            cs.SetPushConstant("SimulateFlag", simulate ? 1u : 0u);
            cs.SetPushConstant("DeltaTime", BitConverter.SingleToUInt32Bits((float)Time.Delta));
            cs.SetPushConstant("DepthTex", depth ? renderer.DepthGBuffer!.BindlessIndex : 0u);
            cs.SetPushConstant("NormalTex", depth ? renderer.Normals!.BindlessIndex : 0u);

            cs.Dispatch(_kCull, cmd, ((uint)_uploadedRows + 63) / 64);
            _visibility.UAVBarrier(cmd);
            _drawArgs.UAVBarrier(cmd);

            cs.Dispatch(_kSimulate, cmd, (uint)_uploadedHighWater);

            if (depth)
            {
                cmd.ResourceBarrierTransition(renderer.DepthGBuffer!.Native, ResourceStates.NonPixelShaderResource, ResourceStates.RenderTarget);
                cmd.ResourceBarrierTransition(renderer.Normals!.Native, ResourceStates.NonPixelShaderResource, ResourceStates.RenderTarget);
            }

            // Hand everything to the draws (these transitions also order them after the dispatch)
            _core.Transition(cmd, ResourceStates.NonPixelShaderResource);
            _visual.Transition(cmd, ResourceStates.NonPixelShaderResource);
            _drawList.Transition(cmd, ResourceStates.NonPixelShaderResource);
            _drawArgs.Transition(cmd, ResourceStates.IndirectArgument);

            if (simulate) _simulatedSerial = _uploadedSerial;
            return true;
        }

        private void CopyGrownPool(ID3D12GraphicsCommandList cmd)
        {
            if (_copyCore == null) return;

            _copyCore.Transition(cmd, ResourceStates.CopySource);
            _copyVisual!.Transition(cmd, ResourceStates.CopySource);
            _core!.Transition(cmd, ResourceStates.CopyDest);
            _visual!.Transition(cmd, ResourceStates.CopyDest);

            cmd.CopyBufferRegion(_core.Native, 0, _copyCore.Native, 0, (ulong)_copySlots * CoreStride);
            cmd.CopyBufferRegion(_visual.Native, 0, _copyVisual.Native, 0, (ulong)_copySlots * VisualStride);

            Engine.Device.DeferDispose(_copyCore);
            Engine.Device.DeferDispose(_copyVisual);
            _copyCore = null;
            _copyVisual = null;
        }

        private void Draw(ID3D12GraphicsCommandList cmd, int mode)
        {
            if (_modeCount[mode] == 0) return;

            var device = Engine.Device;

            // Material.Apply handles PSO, root signature, SceneConstants and the effect's own push
            // constants; the pool's buffer indices go on top.
            _drawMaterial!.Apply(cmd, device);

            cmd.SetGraphicsRoot32BitConstant(0, _core!.SrvIndex, 0);                    // ParticleCoreIdx
            cmd.SetGraphicsRoot32BitConstant(0, _visual!.SrvIndex, 1);                  // ParticleVisualIdx
            cmd.SetGraphicsRoot32BitConstant(0, _drawList!.SrvIndex, 2);                // DrawListIdx
            cmd.SetGraphicsRoot32BitConstant(0, (uint)(mode * _poolSlots), 3);          // DrawListOffset
            cmd.SetGraphicsRoot32BitConstant(0, _emitters!.SrvIndex, 4);                // EmittersIdx
            cmd.SetGraphicsRoot32BitConstant(0, _chunkMap!.SrvIndex, 5);                // ChunkMapIdx
            cmd.SetGraphicsRoot32BitConstant(0, DeferredRenderer.Current?.DepthGBuffer?.BindlessIndex ?? 0u, 6); // DepthGBufIdx

            cmd.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);

            // The simulate dispatch counted the instances
            cmd.ExecuteIndirect(device.DrawInstancedSignature, 1, _drawArgs!.Native, (ulong)(mode * 16), null, 0);
        }
    }

    /// <summary>
    /// The render half of <see cref="ParticleSystem"/>: runs once per rendered view while draws are being
    /// enqueued. Compute and both draws are custom actions, executed in enqueue order inside their pass:
    /// running after the IDraw components keeps particles on top of what those draw (the ocean).
    /// </summary>
    [UpdateInEditor]
    [UpdateInGroup(typeof(RenderGroup))]
    [UpdateAfter(typeof(ScriptDrawSystem))]
    public sealed class ParticleRenderSystem : EntitySystem
    {
        private ParticleSystem? _particles;
        private Action<ID3D12GraphicsCommandList>? _transparent;
        private Action<ID3D12GraphicsCommandList>? _forward;

        protected internal override void Initialize()
        {
            _particles = Systems.Get<ParticleSystem>();
            if (_particles == null) return;
            _transparent = _particles.RenderTransparent;
            _forward = _particles.RenderForward;
        }

        public override void Update()
        {
            if (_particles == null || !_particles.PrepareFrame()) return;

            CommandBuffer.Enqueue(RenderPass.Transparent, _transparent!);
            CommandBuffer.Enqueue(RenderPass.Forward, _forward!);
        }
    }
}
