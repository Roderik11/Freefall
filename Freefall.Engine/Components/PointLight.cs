using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    public class PointLight : Component, IDraw, IUpdate
    {
        public Color3 Color = new Color3(1, 1, 1);
        public float Intensity = 1;
        public float Range = 10;

        private static Mesh? _sphereMesh;
        private static Effect? _sharedEffect;
        private static Material? _sharedMaterial;

        private MaterialBlock _params = new MaterialBlock();

        /// <summary>
        /// Per-instance light data matches HLSL PointLightData struct.
        /// Sent to GPU via the generic per-instance buffer system
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct PointLightData
        {
            public Vector3 Color;
            public float Intensity;
            public Vector3 Position; // camera-relative
            public float Range;
        }

        protected override void Awake()
        {
            _sphereMesh ??= Mesh.CreateSphere(Engine.Device, 1f, 16, 16);
            _sharedEffect ??= new Effect("light_point");
            _sharedMaterial ??= new Material(_sharedEffect);
        }

        public void Update()
        {
            Transform.Scale = Vector3.One * Range;
        }

        public void Draw()
        {
            if (_sphereMesh == null || _sharedMaterial == null) return;

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

            // Set per-instance light data via MaterialBlock per-instance staging buffer
            //_params.Clear();
            _params.SetParameter("LightData", new PointLightData
            {
                Color = new Vector3(Color.R, Color.G, Color.B),
                Intensity = Intensity,
                Position = Entity.Transform.WorldPosition - camPos,
                Range = Range
            });

            // Enqueue pass is inferred from shader technique ("Light")
            CommandBuffer.Enqueue(_sphereMesh, _sharedMaterial, _params, Transform.TransformSlot);
        }
    }
}
