using System;
using System.Numerics;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    public class PointLight : Component, IDraw
    {
        public Color3 Color = new Color3(1, 1, 1);
        public float Intensity = 1;
        public float Range = 10;

        private static Mesh? _sphereMesh;
        private static Effect? _sharedEffect;
        private static Material? _sharedMaterial;

        public BoundingSphere BoundingSphere { get; private set; }
        public BoundingBox BoundingBox { get; private set; }

        public PointLight()
        {
        }

        protected override void Awake()
        {
            // Create shared sphere mesh (16 segments for smoother volume)
            _sphereMesh ??= Mesh.CreateSphere(Engine.Device, 1f, 16, 16);
            
            // Create shared effect with additive blend
            if (_sharedEffect == null)
            {
                _sharedEffect = new Effect("light_point");
                _sharedEffect.BlendState = BlendState.Additive;
                _sharedEffect.DepthStencilState = DepthStencilState.ZReadNoWrite;
            }
            
            // Create shared material for all point lights
            _sharedMaterial ??= new Material(_sharedEffect);
        }

        public void Draw()
        {
            // TODO: Light batching removed - will be reimplemented
            // UpdateBounds();
            // 
            // Transform.Scale = Vector3.One * Range;
            // var lightData = new LightInstanceData { ... };
            // CommandBuffer.EnqueueLight(lightData, _sphereMesh, _sharedMaterial);
        }

        private void UpdateBounds()
        {
            BoundingSphere = new BoundingSphere(Transform.Position, Range);
            BoundingBox = BoundingBox.CreateFromSphere(BoundingSphere);
        }
    }
}
