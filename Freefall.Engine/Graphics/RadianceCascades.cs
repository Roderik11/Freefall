using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall.Components;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Sparse 3D Radiance Cascades global illumination.
    /// Per-cascade angular scaling with interval storage and front-to-back merging.
    /// </summary>
    public class RadianceCascades : IDisposable
    {
        // Configuration — must match HLSL defines
        private const int CascadeCount = 4;
        private const int BaseOctRes = 4;
        private const int BaseGridSize = 1; // 0.5 represented as shift
        private const int HashMapCapacity = 1 << 18; // 256K per level

        // Per-level max tiles (sized so each pool is ~8MB)
        // L0: 16 dirs × 8B = 128B/tile → 65K tiles = 8MB
        // L1: 64 dirs × 8B = 512B/tile → 16K tiles = 8MB
        // L2: 256 dirs × 8B = 2KB/tile  → 4K tiles  = 8MB
        // L3: 1024 dirs × 8B = 8KB/tile → 1K tiles  = 8MB
        private static readonly int[] MaxTilesPerLevel = { 1 << 16, 1 << 14, 1 << 12, 1 << 10 };

        // Per-level directions: (BaseOctRes << level)^2
        private static int DirsForLevel(int level)
        {
            int r = BaseOctRes << level;
            return r * r;
        }

        // Per-level tile bytes: 2 uints per direction (radiance + transmittance)
        private static int TileBytesForLevel(int level) => DirsForLevel(level) * 8;

        // Per-cascade GPU resources
        private GPUHashMap[] _hashMaps = new GPUHashMap[CascadeCount];
        private GraphicsBuffer[] _tilePools = new GraphicsBuffer[CascadeCount];
        private GraphicsBuffer[] _tileInfos = new GraphicsBuffer[CascadeCount];
        private GraphicsBuffer[] _indirectArgs = new GraphicsBuffer[CascadeCount];

        // Screen-res output
        private RenderTexture2D _giBuffer;

        // Compute shader + kernels
        private ComputeShader _shader;
        private int _kMark, _kShade, _kPrepareIndirect;
        private int[] _kTrace = new int[CascadeCount];

        // State
        private int _width, _height;
        private bool _firstFrame = true;

        // Public API
        public uint GIBufferSrvIndex => _giBuffer?.BindlessIndex ?? 0;

        public RadianceCascades()
        {
            _shader = new ComputeShader("radiance_cascades.hlsl");
            _kMark = _shader.FindKernel("CSMark");
            _kPrepareIndirect = _shader.FindKernel("CSPrepareIndirect");
            _kTrace[0] = _shader.FindKernel("CSTrace0");
            _kTrace[1] = _shader.FindKernel("CSTrace1");
            _kTrace[2] = _shader.FindKernel("CSTrace2");
            _kTrace[3] = _shader.FindKernel("CSTrace3");
            _kShade = _shader.FindKernel("CSShade");
        }

        public void Initialize(int width, int height)
        {
            for (int i = 0; i < CascadeCount; i++)
            {
                _hashMaps[i] = GPUHashMap.Create(HashMapCapacity);
                int maxTiles = MaxTilesPerLevel[i];
                int tileUints = TileBytesForLevel(i) / 4; // raw buffer in uint count
                _tilePools[i] = GraphicsBuffer.CreateRaw(maxTiles * tileUints, uav: true, clearable: true);
                _tileInfos[i] = GraphicsBuffer.CreateStructured(maxTiles, 16, srv: true, uav: true);
                _indirectArgs[i] = GraphicsBuffer.CreateRaw(3, uav: true);
            }

            _giBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float, randomWrite: true);
            _width = width;
            _height = height;
        }

        public void Resize(int width, int height)
        {
            _giBuffer?.Dispose();
            _giBuffer = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float, randomWrite: true);
            _width = width;
            _height = height;
            _firstFrame = true;
        }

        /// <summary>
        /// Main entry point called by DeferredRenderer each frame.
        /// Mark (all levels) → PrepareIndirect × 4 → Trace × 4 → Shade.
        /// </summary>
        public void Execute(ID3D12GraphicsCommandList cmd, Camera camera, DeferredRenderer renderer)
        {
            if (_giBuffer == null) return;

            // Bind compute root sig + descriptor heap
            cmd.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            cmd.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });

            // Bind SceneConstants (b0) for View, Projection, CamPos, etc.
            BindSceneConstants(cmd, renderer);

            // Transition GI buffer to UAV
            var giFromState = _firstFrame ? ResourceStates.Common : ResourceStates.NonPixelShaderResource;
            cmd.ResourceBarrierTransition(_giBuffer.Native, giFromState, ResourceStates.UnorderedAccess);
            _firstFrame = false;

            // 1. Clear all hash maps + counters
            for (int i = 0; i < CascadeCount; i++)
                _hashMaps[i].Clear(cmd);

            // 2. Set per-level push constants
            var desc = _giBuffer.Native.Description;
            _shader.SetPushConstant("HEntries0", _hashMaps[0].EntriesUavIndex);
            _shader.SetPushConstant("HEntries1", _hashMaps[1].EntriesUavIndex);
            _shader.SetPushConstant("HEntries2", _hashMaps[2].EntriesUavIndex);
            _shader.SetPushConstant("HEntries3", _hashMaps[3].EntriesUavIndex);
            _shader.SetPushConstant("HCounter0", _hashMaps[0].CounterUavIndex);
            _shader.SetPushConstant("HCounter1", _hashMaps[1].CounterUavIndex);
            _shader.SetPushConstant("HCounter2", _hashMaps[2].CounterUavIndex);
            _shader.SetPushConstant("HCounter3", _hashMaps[3].CounterUavIndex);
            _shader.SetPushConstant("TPool0", _tilePools[0].UavIndex);
            _shader.SetPushConstant("TPool1", _tilePools[1].UavIndex);
            _shader.SetPushConstant("TPool2", _tilePools[2].UavIndex);
            _shader.SetPushConstant("TPool3", _tilePools[3].UavIndex);
            _shader.SetPushConstant("TInfo0", _tileInfos[0].UavIndex);
            _shader.SetPushConstant("TInfo1", _tileInfos[1].UavIndex);
            _shader.SetPushConstant("TInfo2", _tileInfos[2].UavIndex);
            _shader.SetPushConstant("TInfo3", _tileInfos[3].UavIndex);
            _shader.SetPushConstant("HCapMask", _hashMaps[0].CapacityMask);
            _shader.SetPushConstant("ScreenW", (uint)desc.Width);
            _shader.SetPushConstant("ScreenH", (uint)desc.Height);
            _shader.SetPushConstant("DepthGBuf", renderer.DepthGBuffer.BindlessIndex);
            _shader.SetPushConstant("NormalTex", renderer.Normals.BindlessIndex);
            _shader.SetPushConstant("AlbedoTex", renderer.Albedo.BindlessIndex);
            _shader.SetPushConstant("LightTex", renderer.LightBuffer.BindlessIndex);
            _shader.SetPushConstant("GIOutput", _giBuffer.UavIndex);
            _shader.SetPushConstant("RCIntensity", BitConverter.SingleToUInt32Bits(Engine.Settings.RCIntensity));
            _shader.SetPushConstant("CurLevel", 0u);
            _shader.SetPushConstant("IndArgs", _indirectArgs[0].UavIndex);

            uint groupsX = ((uint)desc.Width + 7) / 8;
            uint groupsY = ((uint)desc.Height + 7) / 8;

            // 3. CSMark — screen-res dispatch (inserts tiles at all 4 levels)
            PixMarker.Begin(cmd, "RC Mark");
            _shader.Dispatch(_kMark, cmd, groupsX, groupsY);
            PixMarker.End(cmd);

            // UAV barrier after mark
            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // 4. CSPrepareIndirect × 4 (one per level)
            PixMarker.Begin(cmd, "RC Prepare");
            for (int i = 0; i < CascadeCount; i++)
            {
                _shader.SetPushConstant(_kPrepareIndirect, "CurLevel", (uint)i);
                _shader.SetPushConstant(_kPrepareIndirect, "IndArgs", _indirectArgs[i].UavIndex);
                _shader.Dispatch(_kPrepareIndirect, cmd, 1);
            }
            PixMarker.End(cmd);

            // UAV barrier after prepare
            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // 5. CSTrace × 4 — indirect dispatch per level
            PixMarker.Begin(cmd, "RC Trace");
            for (int i = 0; i < CascadeCount; i++)
            {
                // Set current level for this trace dispatch
                _shader.SetPushConstant(_kTrace[i], "CurLevel", (uint)i);
                
                // Pass debug mode to trace (for diagnostic bypass)
                uint traceDebug = 0;
                if (Engine.Settings.DebugVisualizationMode == DebugVizMode.RCDebug)
                    traceDebug = (uint)Engine.Settings.RCDebugSubMode;
                _shader.SetPushConstant(_kTrace[i], "RCDebugMode", traceDebug);

                // Transition indirect args to argument state
                _indirectArgs[i].Transition(cmd, ResourceStates.IndirectArgument);

                // Bind kernel PSO + push constants
                _shader.BindKernel(_kTrace[i], cmd);

                // Re-bind SceneConstants since BindKernel overwrites root params
                BindSceneConstants(cmd, renderer);

                // Indirect dispatch
                cmd.ExecuteIndirect(
                    Engine.Device.DispatchSignature,
                    1,
                    _indirectArgs[i].Native,
                    0,
                    null,
                    0);

                _indirectArgs[i].Transition(cmd, ResourceStates.UnorderedAccess);
            }
            PixMarker.End(cmd);

            // UAV barrier after all traces
            cmd.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(null)));

            // 6. CSShade — For the shade pass, swap hash map entries to SRV indices
            _shader.SetPushConstant(_kShade, "HEntries0", _hashMaps[0].EntriesSrvIndex);
            _shader.SetPushConstant(_kShade, "HEntries1", _hashMaps[1].EntriesSrvIndex);
            _shader.SetPushConstant(_kShade, "HEntries2", _hashMaps[2].EntriesSrvIndex);
            _shader.SetPushConstant(_kShade, "HEntries3", _hashMaps[3].EntriesSrvIndex);

            // Debug mode: 0=off, 1=tile lookup, 2=per-level radiance, 3=transmittance
            uint rcDebug = 0;
            if (Engine.Settings.DebugVisualizationMode == DebugVizMode.RCDebug)
                rcDebug = (uint)Engine.Settings.RCDebugSubMode;
            _shader.SetPushConstant(_kShade, "RCDebugMode", rcDebug);

            PixMarker.Begin(cmd, "RC Shade");
            _shader.Dispatch(_kShade, cmd, groupsX, groupsY);
            PixMarker.End(cmd);

            // Transition GI buffer to SRV for composition
            cmd.ResourceBarrierTransition(_giBuffer.Native,
                ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
        }

        /// <summary>
        /// Bind SceneConstants (CameraInverse, View, Projection, etc.) at root parameter 1 (b0).
        /// </summary>
        private void BindSceneConstants(ID3D12GraphicsCommandList cmd, DeferredRenderer renderer)
        {
            var mat = renderer.DirectionalLightMaterial;
            if (mat == null) return;

            foreach (var cb in mat.ConstantBuffers)
            {
                if (cb.Slot == 1) // Root param 1 = SceneConstants (b0)
                {
                    mat.Effect!.GetMaterialBlock().Apply(cb);
                    cb.Commit();
                    cmd.SetComputeRootConstantBufferView((uint)cb.Slot, cb.GpuAddress);
                    break;
                }
            }
        }

        public void Dispose()
        {
            for (int i = 0; i < CascadeCount; i++)
            {
                _hashMaps[i]?.Dispose();
                _tilePools[i]?.Dispose();
                _tileInfos[i]?.Dispose();
                _indirectArgs[i]?.Dispose();
            }
            _giBuffer?.Dispose();
            _shader?.Dispose();
        }
    }
}
