using System;
using Freefall.Components;

namespace Freefall.Graphics
{
    public abstract class RenderPipeline : IDisposable
    {
        public abstract void Initialize(int width, int height);

        public abstract void Resize(int width, int height);

        public abstract void Clear(Camera camera);
        public abstract void Render(Camera camera, Vortice.Direct3D12.ID3D12GraphicsCommandList commandList);

        public abstract void Dispose();
    }
}
