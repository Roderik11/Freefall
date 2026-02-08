using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Graphics;
using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    public class SkyboxRenderer : Component, IUpdate, IDraw
    {
        public Mesh Mesh;
        public Material Material;

        // Sky mode
        public bool UseProceduralSky = true;

        // Procedural sky parameters
        public float TimeOfDay = 16f;         // 0-24 hours
        public float CloudCoverage = 0.5f;      // 0-1
        public float CloudSpeed = 1.0f;         // Speed multiplier
        public float SunIntensity = 1.2f;       // Sun brightness multiplier
        public Vector3 SunDirection = new Vector3(0, 1, 0);

        // Star parameters
        public float StarDensity = 0.5f;        // 0-1, controls how many stars
        public float StarBrightness = 1.0f;     // Star intensity multiplier

        // Animation
        public bool AnimateTimeOfDay = false;
        public float TimeOfDaySpeed = 0.1f; // Hours per second (tweak for testing)

        // Sun light control
        public DirectionalLight SunLight;
        public bool ControlSunLight = true;     // Enable/disable automatic sun light control
        public float DayIntensity = 1.0f;       // Sun intensity at noon
        public float NightIntensity = 0.1f;     // Ambient intensity at night

        private float CloudTime = 0.0f;
        private int _transformSlot = -1;
        private MaterialBlock _materialBlock = new MaterialBlock();

        public SkyboxRenderer()
        {
        }

        public void Update()
        {
            if (Camera.Main == null) return;
            
            // Move with camera to simulate infinite distance
            Entity.Transform.Position = Camera.Main.Position;

            // Animate time of day if enabled
            if (UseProceduralSky && AnimateTimeOfDay)
            {
                TimeOfDay += (float)Time.Delta * TimeOfDaySpeed;
                if (TimeOfDay >= 24.0f) TimeOfDay -= 24.0f;
                if (TimeOfDay < 0.0f) TimeOfDay += 24.0f;

                CloudTime += (float)Time.Delta;
            }

            // Update sun light
            if (UseProceduralSky && ControlSunLight && SunLight != null)
            {
                UpdateSunLight();
            }
        }

        private void UpdateSunLight()
        {
            // Calculate sun angle (0 at sunrise, PI at sunset)
            float sunAngle = (TimeOfDay / 24.0f) * MathF.PI * 2.0f - MathF.PI * 0.5f;

            // Calculate sun direction - pure XY arc (no Z component)
            float elevation = MathF.Sin(sunAngle);
            float azimuth = MathF.Cos(sunAngle);
            SunDirection = new Vector3(azimuth, elevation, 0.0f); 

            // Clamp light direction to prevent it from shining from below ground
            Vector3 lightDirection = new Vector3(azimuth, MathF.Max(0.1f, elevation), 0.0f);

            // Light should point FROM the sun (opposite of sun direction)
            Vector3 lightDir = Vector3.Normalize(lightDirection);

            // Set directional light rotation using CreateWorld
            // Transform.Forward is what DirectionalLight uses as LightDirection
            // The shader already negates this (L = -LightDirection), so we pass the direction light is pointing (toward ground)
            SunLight.Entity.Transform.Rotation = Quaternion.CreateFromRotationMatrix(Matrix4x4.CreateWorld(Vector3.Zero, lightDir, Vector3.UnitY));

            // intensity logic
            float dayFactor = Math.Clamp((elevation - 0.0f) / 0.8f, 0.0f, 1.0f);
            float intensity = MathHelper.Lerp(NightIntensity, DayIntensity, dayFactor);
            SunLight.Intensity = intensity * SunIntensity;
            
            // Color logic (simple blend)
            if (elevation > 0)
                SunLight.Color = new Color3(1.0f, 0.95f, 0.85f);
            else
                SunLight.Color = new Color3(0.1f, 0.15f, 0.3f);
        }

        public void Draw()
        {
            if (Mesh == null)
            {
                Mesh = Mesh.CreateCube(Engine.Device, 100.0f);
            }

            if (Material == null)
            {
                Material = new Material(new Effect("mesh_skybox"));
            }

            // Allocate transform slot if needed
            if (_transformSlot < 0 && TransformBuffer.Instance != null)
            {
                _transformSlot = TransformBuffer.Instance.AllocateSlot();
            }

            // Update transform in the global buffer
            if (_transformSlot >= 0)
            {
                TransformBuffer.Instance.SetTransform(_transformSlot, Entity.Transform.WorldMatrix);
            }

            // Set sky parameters on the MaterialBlock
            Material.SetParameter("World", Entity.Transform.WorldMatrix);

            if (UseProceduralSky)
            {
                Material.SetParameter("SunDirection", SunDirection);
                Material.SetParameter("TimeOfDay", TimeOfDay);
                Material.SetParameter("CloudCoverage", CloudCoverage);
                Material.SetParameter("CloudTime", CloudTime);
                Material.SetParameter("CloudSpeed", CloudSpeed);
                Material.SetParameter("SunIntensity", SunIntensity);
                Material.SetParameter("StarDensity", StarDensity);
                Material.SetParameter("StarBrightness", StarBrightness);
            }

            // Enqueue — RenderPass is inferred from the Effect's shader passes.
            // InstanceBatch pipeline handles push constants, transform buffers, and draw calls.
            CommandBuffer.Enqueue(Mesh, Material, _materialBlock, _transformSlot);

            if (Engine.FrameIndex % 60 == 0)
            {
                Debug.Log("SkyboxRenderer", $"SunDir={SunDirection}, TimeOfDay={TimeOfDay}, CloudCoverage={CloudCoverage}, SunIntensity={SunIntensity}");
                Debug.Log("SkyboxRenderer", $"TransformSlot={_transformSlot}, MeshParts={Mesh.MeshParts.Count}, MeshPartId={Mesh.GetMeshPartId(0)}");
                Debug.Log("SkyboxRenderer", $"PosBuffer={Mesh.PosBufferIndex}, NormBuffer={Mesh.NormBufferIndex}, UVBuffer={Mesh.UVBufferIndex}, IndexBuffer={Mesh.IndexBufferIndex}");
                Debug.Log("SkyboxRenderer", $"BoundingBox={Mesh.BoundingBox}, LocalSphere={Mesh.LocalBoundingSphere}");
            }
        }
    }
}
