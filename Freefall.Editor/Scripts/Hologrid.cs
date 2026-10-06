using System.Numerics;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// Infinite editor grid on the Y=0 ground plane.
    /// LOD scales logarithmically with camera height.
    /// Rendered as a fullscreen post-process in the Forward pass.
    /// </summary>
    [UpdateInEditor]
    public class Hologrid : Component, IDraw
    {
        // ── Exposed settings ──
        public Color3 GridColor = new Color3(0.7f, 0.7f, 0.7f);
        public Color3 XAxisColor = new Color3(0.9f, 0.2f, 0.2f);
        public Color3 ZAxisColor = new Color3(0.2f, 0.4f, 0.9f);

        [ValueRange(0.1f, 1.0f)]
        public float Opacity = 0.6f;

        [ValueRange(1f, 100f)]
        public float FadeRange = 40f;

        public float PlaneY = 0f;

        private Material _material;
        private MaterialBlock _params = new MaterialBlock();

        protected override void Awake()
        {
            _material = new Material(new Effect("hologrid"));
        }

        public void Draw()
        {
            if (!Engine.Settings.ShowGrid || _material == null) return;

            CommandBuffer.Enqueue(RenderPass.PostProcess, DrawGrid);
        }

        private void DrawGrid(ID3D12GraphicsCommandList cmd)
        {
            var renderer = DeferredRenderer.Current;
            if (renderer == null) return;

            _material.SetTexture("DepthTex", renderer.DepthGBuffer);
            _material.SetTexture("CompositeTex", renderer.CompositeSnapshot);
            _material.SetParameter("GridColor", GridColor.ToVector3());
            _material.SetParameter("XAxisColor", XAxisColor.ToVector3());
            _material.SetParameter("ZAxisColor", ZAxisColor.ToVector3());
            _material.SetParameter("Opacity", Opacity);
            _material.SetParameter("FadeRange", FadeRange);
            _material.SetParameter("PlaneY", PlaneY);
            _material.Apply(cmd, Engine.Device, _params);
            cmd.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleStrip);
            cmd.DrawInstanced(4, 1, 0, 0);
        }
    }
}
