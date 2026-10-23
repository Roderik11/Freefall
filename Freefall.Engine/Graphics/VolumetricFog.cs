using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// The near fog volume (god rays): the atmosphere's air over the first <see cref="EngineSettings.VolumetricFogRange"/>
    /// metres of view depth as a froxel grid, lit with the sun's and the moon's shadows (cascades, clouds, terrain) and the
    /// sky's light only where the sky is seen, so light falls through the trees and the windows in shafts and the air in
    /// a room or a cave stays dark. The aerial perspective volume (<see cref="Atmosphere"/>) goes on past it; every pass
    /// reads the two together through sky_common.fx AtmosphereFog. One per <see cref="DeferredRenderer"/>.
    ///
    /// Each frame: CSFogScatter lights every froxel at a place jittered in it and blends it with its reprojected history,
    /// CSFogIntegrate sums the columns front to back into the volume the passes read (volumetric_fog_cs.hlsl).
    /// Docs: .agent/knowledge/atmospheric_effects/physical_atmosphere.md §13.
    /// </summary>
    public class VolumetricFog : IDisposable
    {
        public const int TileSize = 16;         // pixels a froxel spans each way (sky_common.fx FOG_TILE)
        public const int Slices = 64;           // volumetric_fog_cs.hlsl FOG_SLICES
        private const float HistoryWeight = 0.9f;
        private const float TeleportDistance = 50f;     // metres in one frame: the history is dropped

        [StructLayout(LayoutKind.Sequential)]
        private struct UInt3 { public uint X, Y, Z; }

        private readonly ComputeShader _cs;
        private readonly int _kInject, _kScatter, _kIntegrate;
        // Volumetric particles' density per froxel (σe, σs, σs·g as uints, CSFogInject); cleared and filled each frame
        private GraphicsBuffer? _density;
        private ParticleSystem? _particles;
        private readonly RenderTexture3D?[] _scatter = new RenderTexture3D?[2];
        private readonly ResourceStates[] _scatterState = new ResourceStates[2];
        // Per froxel the lamps' light (rgb, per km × Exposure) and the share of the open sky's, the sun's and the moon's
        // light that reaches it (a): what the far rain and snow are lit by (LitLamps)
        private readonly RenderTexture3D?[] _lamps = new RenderTexture3D?[2];
        private readonly ResourceStates[] _lampsState = new ResourceStates[2];
        private RenderTexture3D? _integrated;
        private ResourceStates _integratedState;
        private int _gridW, _gridH, _width = 1, _height = 1;
        private int _frame;
        private bool _historyValid;
        private Matrix4x4 _prevRelativeVP;
        private Vector3 _prevCameraPos;
        private float _prevExposure;
        private float _prevRange;

        private const ResourceStates ShaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;

        /// <summary>Metres of view depth the volume covers this frame (SceneConstants.FogVolumeRange).</summary>
        public float Range { get; private set; }

        /// <summary>
        /// Bindless index of this frame's lit froxels once <see cref="Render"/> has run (Texture3D, rgb = the light the air
        /// scatters toward the camera per km × Exposure, a = its extinction per km; readable by compute); 0 before.
        /// </summary>
        public uint LitFroxels { get; private set; }

        /// <summary>
        /// Bindless index of this frame's froxel lamps once <see cref="Render"/> has run (Texture3D, the same grid: rgb = the
        /// point lights' light the air scatters toward the camera per km × Exposure, a = the share of the open sky's, the
        /// sun's and the moon's light that reaches the place past walls, roofs, crowns, shadows and the terrain); 0 before.
        /// The far rain and snow are lit by it (precipitation_layers_cs.hlsl).
        /// </summary>
        public uint LitLamps { get; private set; }

        public VolumetricFog()
        {
            _cs = new ComputeShader("volumetric_fog_cs.hlsl");
            _kInject = _cs.FindKernel("CSFogInject");
            _kScatter = _cs.FindKernel("CSFogScatter");
            _kIntegrate = _cs.FindKernel("CSFogIntegrate");
        }

        /// <summary>
        /// Make sure the volumes fit a target of this size and return the bindless index of the one the passes read
        /// (SceneConstants.FogVolumeIdx), for the whole frame: <see cref="Render"/> fills it before anything reads it.
        /// </summary>
        public uint Prepare(int width, int height)
        {
            int gw = Math.Max(1, (width + TileSize - 1) / TileSize);
            int gh = Math.Max(1, (height + TileSize - 1) / TileSize);
            if (_integrated == null || gw != _gridW || gh != _gridH)
            {
                DisposeTextures();
                var device = Engine.Device;
                // Light per km × Exposure and σe per km: in half floats from a moonless night to dense fog at noon
                for (int i = 0; i < 2; i++)
                {
                    _scatter[i] = new RenderTexture3D(device, gw, gh, Slices, Format.R16G16B16A16_Float);
                    _scatterState[i] = ResourceStates.Common;
                    _lamps[i] = new RenderTexture3D(device, gw, gh, Slices, Format.R16G16B16A16_Float);
                    _lampsState[i] = ResourceStates.Common;
                }
                _integrated = new RenderTexture3D(device, gw, gh, Slices, Format.R16G16B16A16_Float);
                _integratedState = ResourceStates.Common;
                _density = GraphicsBuffer.CreateRaw(gw * gh * Slices * 3, uav: true, clearable: true);
                _gridW = gw;
                _gridH = gh;
                _historyValid = false;
            }
            _width = Math.Max(1, width);
            _height = Math.Max(1, height);
            Range = Math.Clamp(Engine.Settings.VolumetricFogRange, 16f, 1024f);
            return _integrated.BindlessIndex;
        }

        /// <summary>
        /// Light and integrate the volume for this view: after the shadow maps (and the sky captures) of the frame, before
        /// anything draws fog. The scene constants (camera, AtmosphereIdx, Exposure) must be bound for compute by
        /// <paramref name="bindSceneConstants"/>. <paramref name="relativeVP"/> is the view-projection without translation.
        /// </summary>
        public void Render(ID3D12GraphicsCommandList list, Vector3 cameraPos, Matrix4x4 relativeVP, uint shadowMapIdx,
            Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            if (_integrated == null) return;
            var device = Engine.Device;
            int cur = _frame & 1, prev = cur ^ 1;
            var current = _scatter[cur]!;
            var history = _scatter[prev]!;
            var lamps = _lamps[cur]!;
            var lampsHistory = _lamps[prev]!;

            float exposure = Atmosphere.Exposure;
            var move = cameraPos - _prevCameraPos;
            bool useHistory = _historyValid && _prevRange == Range && move.LengthSquared() < TeleportDistance * TeleportDistance
                && _prevExposure > 0f;

            PixMarker.Begin(list, "Volumetric fog");
            list.SetDescriptorHeaps(1, new[] { device.SrvHeap });
            list.SetComputeRootSignature(device.GlobalRootSignature);
            bindSceneConstants(list);

            // Halton (2, 3, 5) over 16 frames: where in each froxel this frame's sample lies
            int j = (_frame % 16) + 1;
            var jitter = new Vector3(Halton(j, 2), Halton(j, 3), Halton(j, 5));

            _cs.SetPushConstant("ScatterUAV", current.UavIndex);
            _cs.SetPushConstant("HistoryTex", useHistory ? history.BindlessIndex : 0u);
            _cs.SetPushConstant("ScatterTex", current.BindlessIndex);
            _cs.SetPushConstant("LampsUAV", lamps.UavIndex);
            _cs.SetPushConstant("LampsHistory", useHistory ? lampsHistory.BindlessIndex : 0u);
            _cs.SetPushConstant("IntegratedUAV", _integrated.UavIndex);
            _cs.SetPushConstant("ShadowMap", shadowMapIdx);
            _cs.SetPushConstant("CascadeBuffer", shadowMapIdx != 0 ? DirectionalLight.CurrentCascadeSrvIndex : 0u);
            _cs.SetParam("PrevCameraRelativeVP", _prevRelativeVP);
            _cs.SetParam("CameraMove", move);
            _cs.SetParam("HistoryWeight", useHistory ? HistoryWeight : 0f);
            _cs.SetParam("Jitter", jitter);
            _cs.SetParam("FogRange", Range);
            _cs.SetParam("ShadowLightDir", -DirectionalLight.GetLightDirection());
            _cs.SetParam("CascadeCount", (uint)Math.Clamp(DirectionalLight.CascadeCount, 0, 8));
            _cs.SetParam("GridSize", new UInt3 { X = (uint)_gridW, Y = (uint)_gridH, Z = Slices });
            _cs.SetParam("ExposureRatio", useHistory ? exposure / _prevExposure : 1f);
            _cs.SetParam("TileUV", new Vector2((float)TileSize / _width, (float)TileSize / _height));

            uint gx = (uint)(_gridW + 7) / 8, gy = (uint)(_gridH + 7) / 8;

            // Volumetric particles (smoke) into the froxels they cover, from the pool the last simulation left
            _particles ??= Systems.Get<ParticleSystem>();
            uint densityUav = 0;
            if (_density != null && _particles != null
                && _particles.VolumeSources(list, out uint core, out uint visual, out uint emitters, out uint chunkMap, out uint slots))
            {
                _density.ClearUAV(list, new Vortice.Mathematics.Int4(0, 0, 0, 0));
                _density.UAVBarrier(list);
                _cs.SetPushConstant("DensityUAV", _density.UavIndex);
                _cs.SetPushConstant("ParticleCore", core);
                _cs.SetPushConstant("ParticleVisual", visual);
                _cs.SetPushConstant("ParticleEmitters", emitters);
                _cs.SetPushConstant("ParticleChunkMap", chunkMap);
                _cs.SetPushConstant("ParticleSlots", slots);
                _cs.Dispatch(_kInject, list, (slots + 63) / 64);
                _density.UAVBarrier(list);
                densityUav = _density.UavIndex;
            }
            _cs.SetPushConstant("DensityUAV", densityUav);

            // Light the froxels
            Transition(list, current, ref _scatterState[cur], ResourceStates.UnorderedAccess);
            Transition(list, lamps, ref _lampsState[cur], ResourceStates.UnorderedAccess);
            if (useHistory)
            {
                Transition(list, history, ref _scatterState[prev], ResourceStates.NonPixelShaderResource);
                Transition(list, lampsHistory, ref _lampsState[prev], ResourceStates.NonPixelShaderResource);
            }
            _cs.Dispatch(_kScatter, list, gx, gy, Slices);
            Transition(list, current, ref _scatterState[cur], ResourceStates.NonPixelShaderResource);
            Transition(list, lamps, ref _lampsState[cur], ResourceStates.NonPixelShaderResource);
            LitFroxels = current.BindlessIndex;
            LitLamps = lamps.BindlessIndex;

            // Sum the columns
            Transition(list, _integrated, ref _integratedState, ResourceStates.UnorderedAccess);
            _cs.Dispatch(_kIntegrate, list, gx, gy, 1);
            Transition(list, _integrated, ref _integratedState, ShaderRead);
            PixMarker.End(list);

            _prevRelativeVP = relativeVP;
            _prevCameraPos = cameraPos;
            _prevExposure = exposure;
            _prevRange = Range;
            _historyValid = true;
            _frame++;
        }

        private static void Transition(ID3D12GraphicsCommandList list, RenderTexture3D texture, ref ResourceStates state,
            ResourceStates to)
        {
            if (state == to)
            {
                if (to == ResourceStates.UnorderedAccess)
                    list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(texture.Native)));
                return;
            }
            list.ResourceBarrierTransition(texture.Native, state, to);
            state = to;
        }

        private static float Halton(int index, int b)
        {
            float f = 1f, r = 0f;
            while (index > 0)
            {
                f /= b;
                r += f * (index % b);
                index /= b;
            }
            return r;
        }

        private void DisposeTextures()
        {
            LitFroxels = 0;
            LitLamps = 0;
            for (int i = 0; i < 2; i++)
            {
                _scatter[i]?.Dispose();
                _scatter[i] = null;
                _lamps[i]?.Dispose();
                _lamps[i] = null;
            }
            _integrated?.Dispose();
            _integrated = null;
            if (_density != null)
                Engine.Device.DeferDispose(_density);
            _density = null;
        }

        public void Dispose()
        {
            DisposeTextures();
            _cs.Dispose();
        }
    }
}
