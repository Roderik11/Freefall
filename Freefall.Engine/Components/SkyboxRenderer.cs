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
        public MaterialBlock Params = new MaterialBlock();

        // Sky mode
        public bool UseProceduralSky = true;

        // Procedural sky parameters
        public float TimeOfDay = 16f;         // 0-24 hours
        public float CloudCoverage = 0.5f;      // 0-1
        public float CloudSpeed = 2.0f;         // Speed multiplier
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

        // Static ambient scale accessible by composition pass
        public static float AmbientScale { get; private set; } = 1.0f;

        private float CloudTime = 0.0f;


        public SkyboxRenderer()
        {
        }

        protected override void Awake()
        {
            Mesh = Mesh.CreateCube(Engine.Device, 100.0f);
            Material = new Material(new Effect("mesh_skybox"));
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

            }

            CloudTime += (float)Time.Delta * CloudSpeed;

            // Update sun light
            if (UseProceduralSky && ControlSunLight && SunLight != null)
                UpdateSunLight();
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
            
            // Ambient tracks daylight ÔÇö min 0.05 at night, full 1.0 at day
            AmbientScale = MathHelper.Lerp(0.05f, 1.0f, dayFactor);
            
            // Color logic (simple blend)
            if (elevation > 0)
                SunLight.Color = new Color3(1.0f, 0.95f, 0.85f);
            else
                SunLight.Color = new Color3(0.1f, 0.15f, 0.3f);
        }

        public void Draw()
        {
            var slot = Entity.Transform.TransformSlot;

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

            CommandBuffer.Enqueue(Mesh, Material, Params, slot);
        }
    }
}
