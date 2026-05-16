using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Direct3D;
using Vortice.Direct3D12;

namespace Freefall.Components
{
    public enum ParticleRenderMode
    {
        Forward,      // Renders to Composite after composition — gets fog automatically
        Transparent   // Renders to GBuffer in transparent pass
    }

    /// <summary>
    /// GPU-driven particle emitter. All simulation happens on the GPU via compute shaders.
    /// CPU only uploads emitter configuration per frame.
    /// Uses dead-list + alive-list (ping-pong) pool management.
    /// </summary>
    [Icon("icon_particle.png")]
    public class ParticleEmitter : Component, IDraw, IUpdate
    {
        // ── Emission ──

        [Category("Emission")]
        [Description("Maximum number of particles in the pool")]
        public int MaxParticles = 10000;

        [Category("Emission")]
        [Description("Particles emitted per second")]
        [ValueRange(0f, 1000f)]
        public float EmitRate = 100f;

        [Category("Emission")]
        [Description("Particle lifetime in seconds")]
        [ValueRange(0.1f, 30f)]
        public float Lifetime = 2.0f;

        // ── Motion ──

        [Category("Motion")]
        [Description("Initial emission velocity (world space)")]
        public Vector3 EmitVelocity = new(0, 2, 0);

        [Category("Motion")]
        [Description("Random velocity variation (0 = uniform, 1 = full spread)")]
        [ValueRange(0f, 2f)]
        public float VelocityRandomness = 0.5f;

        [Category("Motion")]
        [Description("Gravity force applied each frame")]
        public Vector3 Gravity = new(0, -9.81f, 0);

        // ── Appearance ──

        [Category("Appearance")]
        [Description("Min/max particle size (randomized on emit, interpolated over lifetime)")]
        public Vector2 SizeRange = new(0.1f, 0.5f);

        [Category("Appearance")]
        [Description("RGBA color at birth")]
        public Vector4 ColorStart = Vector4.One;

        [Category("Appearance")]
        [Description("RGBA color at death")]
        public Vector4 ColorEnd = new(1, 1, 1, 0);

        [Category("Appearance")]
        [Description("Maximum rotation speed (radians/sec)")]
        [ValueRange(0f, 10f)]
        public float RotationRange = 1.0f;

        [Category("Appearance")]
        public Texture? ParticleTexture;

        // ── Flipbook ──

        [Category("Flipbook")]
        [Description("Number of frames in the flipbook atlas (0 = no animation)")]
        public int FlipbookFrameCount = 0;

        [Category("Flipbook")]
        [Description("Flipbook playback speed (frames/sec)")]
        public float FlipbookAnimSpeed = 10f;

        [Category("Flipbook")]
        [Description("Number of columns in the flipbook atlas")]
        public int FlipbookColumns = 1;

        [Category("Flipbook")]
        [Description("Number of rows in the flipbook atlas")]
        public int FlipbookRows = 1;

        // ── Rendering ──

        [Category("Rendering")]
        [Description("Which render pass to use")]
        public ParticleRenderMode RenderMode = ParticleRenderMode.Forward;

        [Category("Rendering")]
        [Description("Fade particles near opaque surfaces")]
        public bool SoftParticles = true;

        [Category("Rendering")]
        [Description("Depth range for soft particle fade (meters)")]
        [ValueRange(0.01f, 10f)]
        public float SoftRange = 0.5f;

        // ── GPU Buffers ──
        private GraphicsBuffer? _particleCore;
        private GraphicsBuffer? _particleVisual;
        private GraphicsBuffer? _deadList;
        private GraphicsBuffer? _aliveListA;
        private GraphicsBuffer? _aliveListB;
        private GraphicsBuffer? _counters;
        private GraphicsBuffer? _drawArgs;
        private GraphicsBuffer? _emitterParamsCB;

        // ── Shaders ──
        private ComputeShader? _computeShader;
        private int _kInit, _kEmit, _kSimulate, _kBuildArgs;
        private Material? _drawMaterial;
        private Effect? _drawEffect;

        // ── State ──
        private bool _initialized;
        private bool _readFromA = true; // ping-pong: true = read A write B
        private float _emitAccumulator;
        private uint _frameCounter;

        // ── Cached heap array ──
        private ID3D12DescriptorHeap[]? _cachedHeapArray;

        // ── GPU-mirrored structs ──

        [StructLayout(LayoutKind.Sequential)]
        private struct ParticleCoreGPU
        {
            public Vector3 Position;
            public float Age;
            public Vector3 Velocity;
            public float Lifetime;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ParticleVisualGPU
        {
            public Vector2 SizeStartEnd;
            public Vector4 ColorStart;
            public float Rotation;
            public float RotationSpeed;
            public uint FlipbookFrame;
            public uint FlipbookCount;
            public float AnimSpeed;
            public float _pad0;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct EmitterParams
        {
            public Vector3 EmitterPosition;
            public float DeltaTime;
            public Vector3 EmitVelocity;
            public float VelocityRandomness;
            public Vector3 Gravity;
            public float LifetimeParam;
            public Vector2 SizeRange;
            public float RotationRange;
            public uint RandomSeed;
            public Vector4 ColorStart;
            public Vector4 ColorEnd;
            public float FlipbookFrameCount;
            public float FlipbookAnimSpeed;
            public Vector2 _pad;
        }

        // ────────────── Lifecycle ──────────────

        protected override void Awake()
        {
            try
            {
                // Load compute shader (multi-kernel, reflection-based)
                _computeShader = new ComputeShader("particle_compute.hlsl");
                _kInit = _computeShader.FindKernel("CSInit");
                _kEmit = _computeShader.FindKernel("CSEmit");
                _kSimulate = _computeShader.FindKernel("CSSimulate");
                _kBuildArgs = _computeShader.FindKernel("CSBuildDrawArgs");

                // Load draw effect
                _drawEffect = new Effect("particle_draw");
                _drawMaterial = new Material(_drawEffect);

                // Allocate GPU buffers
                AllocateBuffers();

                Debug.Log("ParticleEmitter", $"Initialized: {MaxParticles} max particles");
            }
            catch (Exception ex)
            {
                Debug.LogError("ParticleEmitter", $"Init failed: {ex.Message}");
            }
        }

        private void AllocateBuffers()
        {
            int coreStride = Marshal.SizeOf<ParticleCoreGPU>();
            int visualStride = Marshal.SizeOf<ParticleVisualGPU>();

            _particleCore = GraphicsBuffer.CreateStructured(MaxParticles, coreStride, srv: true, uav: true);
            _particleVisual = GraphicsBuffer.CreateStructured(MaxParticles, visualStride, srv: true, uav: true);
            _deadList = GraphicsBuffer.CreateStructured<uint>(MaxParticles, srv: true, uav: true);
            _aliveListA = GraphicsBuffer.CreateStructured<uint>(MaxParticles, srv: true, uav: true);
            _aliveListB = GraphicsBuffer.CreateStructured<uint>(MaxParticles, srv: true, uav: true);
            _counters = GraphicsBuffer.CreateRaw(4, srv: true, uav: true, clearable: true);
            _drawArgs = GraphicsBuffer.CreateRaw(4, uav: true);
            _emitterParamsCB = GraphicsBuffer.CreateConstantBuffer<EmitterParams>();
        }

        public override void Destroy()
        {
            _particleCore?.Dispose();
            _particleVisual?.Dispose();
            _deadList?.Dispose();
            _aliveListA?.Dispose();
            _aliveListB?.Dispose();
            _counters?.Dispose();
            _drawArgs?.Dispose();
            _emitterParamsCB?.Dispose();
            _computeShader?.Dispose();
            _emitCountUpload?.Release();
        }

        // ────────────── Update ──────────────

        public void Update()
        {
            if (_computeShader == null || _emitterParamsCB == null) return;

            // Accumulate emit count
            _emitAccumulator += EmitRate * (float)Time.Delta;
            uint emitThisFrame = (uint)_emitAccumulator;
            _emitAccumulator -= emitThisFrame;

            // Upload emitter parameters
            unsafe
            {
                var p = _emitterParamsCB.WritePtr<EmitterParams>();
                *p = new EmitterParams
                {
                    EmitterPosition = Transform.WorldPosition,
                    DeltaTime = (float)Time.Delta,
                    EmitVelocity = EmitVelocity,
                    VelocityRandomness = VelocityRandomness,
                    Gravity = Gravity,
                    LifetimeParam = Lifetime,
                    SizeRange = SizeRange,
                    RotationRange = RotationRange,
                    RandomSeed = pcg_hash(++_frameCounter),
                    ColorStart = ColorStart,
                    ColorEnd = ColorEnd,
                    FlipbookFrameCount = FlipbookFrameCount > 0
                        ? FlipbookFrameCount
                        : FlipbookColumns * FlipbookRows,
                    FlipbookAnimSpeed = FlipbookAnimSpeed,
                };
            }

            _emitThisFrame = emitThisFrame;
        }

        private uint _emitThisFrame;

        // CPU-side PCG hash for seed generation
        private static uint pcg_hash(uint input)
        {
            uint state = input * 747796405u + 2891336453u;
            uint word = ((state >> (int)((state >> 28) + 4)) ^ state) * 277803737u;
            return (word >> 22) ^ word;
        }

        // ────────────── Draw ──────────────

        public void Draw()
        {
            if (_computeShader == null || _drawMaterial == null) return;
            if (_particleCore == null || _drawArgs == null) return;

            // Determine alive list read/write for this frame
            var aliveRead = _readFromA ? _aliveListA! : _aliveListB!;
            var aliveWrite = _readFromA ? _aliveListB! : _aliveListA!;

            // Enqueue compute dispatch (Opaque pass — runs earliest)
            CommandBuffer.Enqueue(RenderPass.Opaque, (cmd) =>
            {
                DispatchCompute(cmd, aliveRead, aliveWrite);
            });

            // Enqueue draw (Forward or Transparent pass)
            var drawPass = RenderMode == ParticleRenderMode.Forward
                ? RenderPass.Forward
                : RenderPass.Transparent;

            // Capture references for the closure
            var drawAliveList = aliveWrite; // draw reads the list that was just written
            CommandBuffer.Enqueue(drawPass, (cmd) =>
            {
                DrawParticles(cmd, drawAliveList);
            });

            // Swap alive lists for next frame
            _readFromA = !_readFromA;
        }

        // ────────────── Compute Dispatch ──────────────

        private void DispatchCompute(ID3D12GraphicsCommandList cmd, GraphicsBuffer aliveRead, GraphicsBuffer aliveWrite)
        {
            var device = Engine.Device;
            _cachedHeapArray ??= new[] { device.SrvHeap };

            cmd.SetComputeRootSignature(device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, _cachedHeapArray);

            // Bind emitter params cbuffer at root slot 2 (register b1)
            cmd.SetComputeRootConstantBufferView(2, _emitterParamsCB!.Native.GPUVirtualAddress);

            // Set push constants shared by all kernels
            SetComputePushConstants(aliveRead, aliveWrite);

            // One-time initialization
            if (!_initialized)
            {
                _initialized = true;
                uint initGroups = ((uint)MaxParticles + 255) / 256;
                _computeShader!.Dispatch(_kInit, cmd, initGroups);

                // UAV barrier — ensure init writes complete
                cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
                return; // skip simulation on first frame
            }

            // Write emit count to counters[3] via a small upload
            // Actually, we set emit count via the compute CSEmit dispatch size — EmitCount is read from Counters[3]
            // But CSBuildDrawArgs clears Counters[3] each frame, so we need to write it.
            // Use a copy from a staging buffer, or...
            // Simpler: poke it via a tiny compute, or just use a push constant.
            // For now: use ClearUAV pattern or an Emit-count push constant approach.
            // Let's use a copy. Actually simplest: just write EmitCount to the counters buffer
            // via the compute shader push constant MaxParticles slot (reuse slot 7 for EmitCount in CSEmit).
            // Actually, CSEmit reads Counters[3] for emit count. We need to write it somehow.
            // Best approach: Clear counters[3] was done by CSBuildDrawArgs, then we need to write _emitThisFrame.
            // We'll use a small structured upload buffer write.

            // Write EmitCount to Counters[3] via copy from upload
            WriteEmitCount(cmd, _emitThisFrame);

            // UAV barrier after emit count write
            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // CSSimulate
            uint simGroups = ((uint)MaxParticles + 255) / 256; // Oversized is fine, shader checks bounds
            _computeShader!.Dispatch(_kSimulate, cmd, simGroups);
            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // CSEmit
            if (_emitThisFrame > 0)
            {
                uint emitGroups = (_emitThisFrame + 63) / 64;
                _computeShader.Dispatch(_kEmit, cmd, emitGroups);
                cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));
            }

            // CSBuildDrawArgs
            _computeShader.Dispatch(_kBuildArgs, cmd, 1);
            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // Transition draw args to IndirectArgument for ExecuteIndirect
            _drawArgs!.Transition(cmd, Vortice.Direct3D12.ResourceStates.IndirectArgument);
        }

        private void SetComputePushConstants(GraphicsBuffer aliveRead, GraphicsBuffer aliveWrite)
        {
            var cs = _computeShader!;

            // Push constants (all kernels)
            cs.SetPushConstant("ParticleCoreUAV", _particleCore!.UavIndex);
            cs.SetPushConstant("ParticleVisualUAV", _particleVisual!.UavIndex);
            cs.SetPushConstant("DeadListUAV", _deadList!.UavIndex);
            cs.SetPushConstant("AliveListRead", aliveRead.SrvIndex);
            cs.SetPushConstant("AliveListWriteUAV", aliveWrite.UavIndex);
            cs.SetPushConstant("CountersUAV", _counters!.UavIndex);
            cs.SetPushConstant("DrawArgsUAV", _drawArgs!.UavIndex);
            cs.SetParam("MaxParticles", (uint)MaxParticles);
        }

        // Write emitCount to counters[3] via a CPU-visible upload + copy
        private unsafe void WriteEmitCount(ID3D12GraphicsCommandList cmd, uint emitCount)
        {
            // Use a tiny upload buffer to poke the value into the UAV buffer
            // Counters is raw buffer: offset 12 = byte offset for [3]
            // We'll use direct counter clear/write approach:
            // Since Counters is in UAV state, we can't map it (it's default heap).
            // Use CopyBufferRegion from a scratch upload buffer.

            // For efficiency, we reuse the emitter params upload buffer's last bytes
            // Actually simplest: just embed EmitCount in a push constant and have CSEmit read it.
            // But CSEmit currently reads from Counters[3].
            //
            // Pragmatic: create a 16-byte upload buffer once, reuse it.
            if (_emitCountUpload == null)
            {
                _emitCountUpload = Engine.Device.CreateUploadBuffer(16);
            }

            // Write emit count
            void* ptr;
            _emitCountUpload.Map(0, null, &ptr);
            *(uint*)ptr = emitCount;
            _emitCountUpload.Unmap(0);

            // Transition counters to CopyDest
            _counters!.Transition(cmd, Vortice.Direct3D12.ResourceStates.CopyDest);

            // CopyBufferRegion: 4 bytes from upload offset 0 to counters offset 12 (slot[3])
            cmd.CopyBufferRegion(_counters.Native, 12, _emitCountUpload, 0, 4);

            // Transition back to UAV
            _counters.Transition(cmd, Vortice.Direct3D12.ResourceStates.UnorderedAccess);
        }

        private ID3D12Resource? _emitCountUpload;

        // ────────────── Draw ──────────────

        private void DrawParticles(ID3D12GraphicsCommandList cmd, GraphicsBuffer aliveList)
        {
            var device = Engine.Device;

            // Material.Apply handles: PSO, root signature, cbuffer binding (SceneConstants b0),
            // and push constants from effect resource bindings. Our push constants set AFTER
            // override the effect's defaults for the particle buffer indices.
            _drawMaterial!.Apply(cmd, device);

            // Override push constants with particle buffer indices (post-Apply)
            uint texIdx = ParticleTexture?.BindlessIndex ?? 0;
            uint depthIdx = DeferredRenderer.Current?.DepthGBuffer?.BindlessIndex ?? 0;

            cmd.SetGraphicsRoot32BitConstant(0, _particleCore!.SrvIndex, 0);    // ParticleCoreIdx
            cmd.SetGraphicsRoot32BitConstant(0, _particleVisual!.SrvIndex, 1);  // ParticleVisualIdx
            cmd.SetGraphicsRoot32BitConstant(0, aliveList.SrvIndex, 2);         // AliveListIdx
            cmd.SetGraphicsRoot32BitConstant(0, texIdx, 3);                     // TextureIdx
            cmd.SetGraphicsRoot32BitConstant(0, depthIdx, 4);                   // DepthGBufIdx
            cmd.SetGraphicsRoot32BitConstant(0, SoftParticles ? 1u : 0u, 5);   // SoftEnabled
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(SoftRange), 6); // SoftRange

            // ColorEnd (float4 = 4 dwords at slots 7-10)
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(ColorEnd.X), 7);
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(ColorEnd.Y), 8);
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(ColorEnd.Z), 9);
            cmd.SetGraphicsRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(ColorEnd.W), 10);

            // Flipbook atlas layout
            cmd.SetGraphicsRoot32BitConstant(0, (uint)Math.Max(1, FlipbookColumns), 11);
            cmd.SetGraphicsRoot32BitConstant(0, (uint)Math.Max(1, FlipbookRows), 12);

            // Topology — triangle list for the 6-vertex quads
            cmd.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);

            // ExecuteIndirect — GPU wrote the instance count
            cmd.ExecuteIndirect(
                device.DrawInstancedSignature,
                1,
                _drawArgs!.Native,
                0,
                null,
                0);

            // Transition draw args back to Common for next frame
            _drawArgs.Transition(cmd, Vortice.Direct3D12.ResourceStates.Common);
        }
    }
}
