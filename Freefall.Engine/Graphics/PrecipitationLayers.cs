using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Freefall.Components;

namespace Freefall.Graphics
{
    /// <summary>
    /// The falling rain and snow past the near particles (precipitation_layers_cs.hlsl): seven layers of streaks and
    /// flakes on spheres round the camera from 10 to 70 m, over the HDR composite after the forward pass (before the
    /// exposure is measured and the bloom), cut by the scene's depth, kept off covered places by the cover map
    /// (ShelterMap) and faded into the haze. The near particles (the emitters' boxes, ~12 m) give the parallax close by;
    /// these give the mass of the fall in depth, which particles out to 70 m would cost too many of. Only while something
    /// falls at the camera (Weather.Rain, Weather.Snow) and EngineSettings.PrecipitationLayers is on.
    /// Docs: atmospheric_effects/weather.md §3.
    /// </summary>
    public sealed class PrecipitationLayers : IDisposable
    {
        /// <summary>The weather the layers draw. Must match PrecipFrame in precipitation_layers_cs.hlsl (96 bytes).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct Frame
        {
            public ShelterMap.LevelData Shelter0, Shelter1, Shelter2;
            public float ShelterMinY, ShelterRangeY, Rain, Snow;
            public Vector2 Wind; public float HazeKm, PixelAngle;
            public float RainDensity, RainFall, SnowDensity, SnowFall;
        }

        private const int FrameCount = 3;
        private readonly ComputeShader _shader;
        private readonly int _kernel;
        private readonly GraphicsBuffer[] _frames = new GraphicsBuffer[FrameCount];

        public PrecipitationLayers()
        {
            _shader = new ComputeShader("precipitation_layers_cs.hlsl");
            _kernel = _shader.FindKernel("CSLayers");
            for (int i = 0; i < FrameCount; i++)
                _frames[i] = GraphicsBuffer.CreateUpload<Frame>(1, mapped: true);
        }

        /// <summary>Whether anything falls to draw this frame.</summary>
        public static bool Due => Engine.Settings.PrecipitationLayers && (Weather.Rain > 0.002f || Weather.Snow > 0.002f);

        /// <summary>
        /// Draw the layers over <paramref name="composite"/> (in NonPixelShaderResource on entry and on return); the
        /// linear depth readable by compute. The scene constants are bound by <paramref name="bindSceneConstants"/>.
        /// <paramref name="litFroxels"/>, <paramref name="litLamps"/>: this frame's lit froxels and froxel lamps of the fog
        /// volume (VolumetricFog.LitFroxels, LitLamps), what lights the drops and flakes: the share of the sky's, the sun's
        /// and the moon's light that reaches them, and the lamps' (0: the fog volume is off, they take the sky's light).
        /// </summary>
        public unsafe void Execute(ID3D12GraphicsCommandList list, RenderTexture2D composite, uint linearDepthSrv,
            uint litFroxels, uint litLamps, Camera camera, Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            var desc = composite.Native.Description;
            int width = (int)desc.Width, height = (int)desc.Height;
            int slot = Engine.FrameIndex % FrameCount;
            var shelter = ShelterMap.Current;
            // The steady wind, not the gusts: the far field's lean moves its cells by lean × height, so a gust swept it
            var wind = Wind.Current;
            *_frames[slot].WritePtr<Frame>() = new Frame
            {
                Shelter0 = shelter.L0, Shelter1 = shelter.L1, Shelter2 = shelter.L2,
                ShelterMinY = shelter.MinY, ShelterRangeY = shelter.RangeY,
                Rain = Weather.Rain, Snow = Weather.Snow,
                Wind = new Vector2(wind.X, wind.Z),
                HazeKm = Weather.HazeKm,
                RainDensity = Weather.RainDensity, RainFall = Weather.RainFallSpeed,
                SnowDensity = Weather.SnowDensity, SnowFall = Weather.SnowFallSpeed,
                PixelAngle = 2f / (camera.Projection.M22 * height),
            };

            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            bindSceneConstants(list);
            _shader.SetPushConstant("CompositeUAV", composite.UavIndex);
            _shader.SetPushConstant("LinearDepth", linearDepthSrv);
            _shader.SetPushConstant("Frame", _frames[slot].SrvIndex);
            _shader.SetPushConstant("LitFroxels", litFroxels);
            _shader.SetPushConstant("LitLamps", litLamps);

            PixMarker.Begin(list, "Precipitation layers");
            list.ResourceBarrierTransition(composite.Native, ResourceStates.NonPixelShaderResource, ResourceStates.UnorderedAccess);
            _shader.Dispatch(_kernel, list, (uint)(width + 7) / 8, (uint)(height + 7) / 8);
            list.ResourceBarrierTransition(composite.Native, ResourceStates.UnorderedAccess, ResourceStates.NonPixelShaderResource);
            PixMarker.End(list);
        }

        public void Dispose()
        {
            foreach (var f in _frames) f?.Dispose();
            _shader.Dispose();
        }
    }
}
