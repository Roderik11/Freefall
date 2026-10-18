using System;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// A view's aerial perspective volume (atmosphere_cs.hlsl CSAerialPerspective): the light the air adds and its
    /// transmittance between the camera and each of <see cref="Slices"/> depth slices out to
    /// <see cref="Atmosphere.APMaxDepth"/>, in cells of about <see cref="CellPixels"/> across the screen. Sized to the
    /// view, so the terrain's shadow in the haze keeps the same small steps on a wide screen as on a narrow one (a fixed
    /// 32×32 drew them 107 px wide at 3440 pixels). One per <see cref="DeferredRenderer"/>; <see cref="Atmosphere.Render"/>
    /// fills it, the passes find it through SceneConstants.AerialPerspectiveIdx.
    /// </summary>
    public sealed class AerialPerspectiveVolume : IDisposable
    {
        public const int CellPixels = 20;
        public const int Slices = 32;               // atmosphere_common.hlsli AP_SLICES

        private RenderTexture3D? _texture;
        internal ResourceStates State = ResourceStates.Common;

        public int Width { get; private set; }
        public int Height { get; private set; }
        internal RenderTexture3D? Texture => _texture;

        /// <summary>Fit the volume to a target of this size (made again only when its cell count changes) and return
        /// its bindless index, the same until the next resize.</summary>
        public uint Prepare(int targetWidth, int targetHeight)
        {
            int w = Math.Max(1, (targetWidth + CellPixels - 1) / CellPixels);
            int h = Math.Max(1, (targetHeight + CellPixels - 1) / CellPixels);
            if (_texture == null || w != Width || h != Height)
            {
                _texture?.Dispose();
                // Pre-exposed light and a transmittance: half floats, as the 32³ volume before it
                _texture = new RenderTexture3D(Engine.Device, w, h, Slices, Format.R16G16B16A16_Float);
                State = ResourceStates.Common;
                Width = w;
                Height = h;
            }
            return _texture.BindlessIndex;
        }

        public void Dispose()
        {
            _texture?.Dispose();
            _texture = null;
        }
    }
}
