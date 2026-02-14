using System;
using System.Numerics;
using System.Runtime.InteropServices;
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

        private MaterialBlock _params = new MaterialBlock();

        public BoundingSphere BoundingSphere { get; private set; }
        public BoundingBox BoundingBox { get; private set; }

        /// <summary>
        /// Per-instance light data ÔÇö matches HLSL PointLightData struct.
        /// Sent to GPU via the generic per-instance buffer system (same pattern as TerrainPatchData).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct PointLightData
        {
            public Vector3 Color;
            public float Intensity;
            public Vector3 Position; // camera-relative
            public float Range;
        }

        public PointLight()
        {
        }

        protected override void Awake()
        {
            // Create shared sphere mesh (16 segments for smoother volume)
            _sphereMesh ??= Mesh.CreateSphere(Engine.Device, 1f, 16, 16);
            
            // Create shared effect/material
            if (_sharedEffect == null)
            {
                _sharedEffect = new Effect("light_point");
            }
            
            // Create shared material ÔÇö G-Buffer textures are set here (same for all point lights)
            _sharedMaterial ??= new Material(_sharedEffect);
        }

        public void Draw()
        {
            if (_sphereMesh == null || _sharedMaterial == null) return;

            // Scale the sphere to match light range
            Entity.Transform.Scale = Vector3.One * Range;

            UpdateBounds();

            var slot = Entity.Transform.TransformSlot;
            if (slot >= 0)
                TransformBuffer.Instance.SetTransform(slot, Entity.Transform.WorldMatrix);

            // Set G-Buffer textures on the shared Material (slots 17-20, same for all lights)
            if (DeferredRenderer.Current != null)
            {
                _sharedMaterial.SetTexture("NormalTex", DeferredRenderer.Current.Normals);
                _sharedMaterial.SetTexture("DepthTex", DeferredRenderer.Current.Depth);
                _sharedMaterial.SetTexture("AlbedoTex", DeferredRenderer.Current.Albedo);
                _sharedMaterial.SetTexture("DataTex", DeferredRenderer.Current.Data);
            }

            // LightPosition must be camera-relative: posFromDepth uses zero-translation
            // CameraInverse, so reconstructed world positions are relative to camera.
            var camPos = Camera.Main?.Position ?? Vector3.Zero;

            // Set per-instance light data via MaterialBlock ÔåÆ per-instance staging buffer
            // Same pattern as Terrain's TerrainPatchData
            _params.Clear();
            _params.SetParameter("LightData", new PointLightData
            {
                Color = new Vector3(Color.R, Color.G, Color.B),
                Intensity = Intensity,
                Position = Entity.Transform.Position - camPos,
                Range = Range
            });

            // Enqueue ÔÇö pass is inferred from shader technique ("Light")
            CommandBuffer.Enqueue(_sphereMesh, _sharedMaterial, _params, slot);
        }

        private void UpdateBounds()
        {
            BoundingSphere = new BoundingSphere(Entity.Transform.Position, Range);
            BoundingBox = BoundingBox.CreateFromSphere(BoundingSphere);
        }
    }
}
