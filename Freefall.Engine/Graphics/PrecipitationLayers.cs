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
        /// <summary>The weather the layers draw. Must match PrecipFrame in precipitation_layers_cs.hlsl (80 bytes).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct Frame
        {
            public ShelterMap.LevelData Shelter0, Shelter1, Shelter2;
            public float ShelterMinY, ShelterRangeY, Rain, Snow;
            public float WindAcross, _pad, HazeKm, PixelAngle;
        }

        // The steady wind across the view, eased: the far field's lean shifts its cells by lean × height, so a gust or a
        // turn of the camera taken at once swept the far rain sideways
        private float _windAcross;
        private bool _windPrimed;
        private const float WindEaseSeconds = 1.5f;

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
        /// </summary>
        public unsafe void Execute(ID3D12GraphicsCommandList list, RenderTexture2D composite, uint linearDepthSrv,
            Camera camera, Action<ID3D12GraphicsCommandList> bindSceneConstants)
        {
            var desc = composite.Native.Description;
            int width = (int)desc.Width, height = (int)desc.Height;
            int slot = Engine.FrameIndex % FrameCount;
            var shelter = ShelterMap.Current;
            // Along +azimuth (the shader's x on the sphere) at the view's middle: (forward.z, -forward.x)
            var forward = camera.Forward;
            var across = new Vector2(forward.Z, -forward.X);
            float target = across.LengthSquared() > 1e-6f
                ? Vector2.Dot(new Vector2(Wind.Current.X, Wind.Current.Z), Vector2.Normalize(across)) : 0f;
            float dt = Math.Clamp((float)Base.Time.Delta, 0f, 0.1f);
            _windAcross = _windPrimed ? _windAcross + (target - _windAcross) * (1f - MathF.Exp(-dt / WindEaseSeconds)) : target;
            _windPrimed = true;
            *_frames[slot].WritePtr<Frame>() = new Frame
            {
                Shelter0 = shelter.L0, Shelter1 = shelter.L1, Shelter2 = shelter.L2,
                ShelterMinY = shelter.MinY, ShelterRangeY = shelter.RangeY,
                Rain = Weather.Rain, Snow = Weather.Snow,
                WindAcross = _windAcross,
                HazeKm = Weather.HazeKm,
                PixelAngle = 2f / (camera.Projection.M22 * height),
            };

            list.SetDescriptorHeaps(1, new[] { Engine.Device.SrvHeap });
            list.SetComputeRootSignature(Engine.Device.GlobalRootSignature);
            bindSceneConstants(list);
            _shader.SetPushConstant("CompositeUAV", composite.UavIndex);
            _shader.SetPushConstant("LinearDepth", linearDepthSrv);
            _shader.SetPushConstant("Frame", _frames[slot].SrvIndex);

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
