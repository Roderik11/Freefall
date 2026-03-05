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
        [ValueRange(0f, 24f)]
        public float TimeOfDay = 16f;         // 0-24 hours
        [ValueRange(0f, 1f)]
        public float CloudCoverage = 0.5f;      // 0-1
        [ValueRange(0f, 10f)]
        public float CloudSpeed = 1.0f;         // Speed multiplier
        [ValueRange(0f, 10f)]
        public float SunIntensity = 1.2f;       // Sun brightness multiplier

        public Vector3 SunDirection = new Vector3(0, 1, 0);

        // Star parameters
        [ValueRange(0f, 1f)]
        public float StarDensity = 0.5f;        // 0-1, controls how many stars
        [ValueRange(0f, 10f)]
        public float StarBrightness = 1.0f;     // Star intensity multiplier

        // Animation
        public bool AnimateTimeOfDay = false;
        public float TimeOfDaySpeed = 0.1f; // Hours per second (tweak for testing)

        // Sun light control
        public DirectionalLight SunLight;
        public bool ControlSunLight = true;     // Enable/disable automatic sun light control
      
        [ValueRange(0f, 10f)]
        public float DayIntensity = 3.14159f;       // Sun intensity at noon
        [ValueRange(0f, 10f)]
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

            // ── Three-state blend matching sky_common.fx GetSkyColor ──
            // Day: non-linear ramp (same pow(0.7) as shader)
            float dayFactor = MathF.Pow(Math.Clamp(elevation / 0.8f, 0.0f, 1.0f), 0.7f);

            // Sunset: triangular peak centered at elevation ≈ -0.025
            float sunsetFactor = 0.0f;
            if (elevation < 0.15f && elevation > -0.2f)
            {
                sunsetFactor = 1.0f - MathF.Abs((elevation - (-0.025f)) / 0.175f);
                sunsetFactor = MathF.Max(0.0f, sunsetFactor);
            }

            // Night: smooth onset below horizon
            float nightFactor = Math.Clamp((-elevation - 0.15f) / 0.3f, 0.0f, 1.0f);

            // Normalize weights (same as shader)
            float total = dayFactor + sunsetFactor + nightFactor;
            if (total > 0.0f)
            {
                dayFactor /= total;
                sunsetFactor /= total;
                nightFactor /= total;
            }

            // Blended intensity
            const float sunsetIntensity = 1.5f; // warm glow between day and night
            float intensity = dayFactor * DayIntensity
                            + sunsetFactor * sunsetIntensity
                            + nightFactor * NightIntensity;
            SunLight.Intensity = intensity * SunIntensity;

            // Ambient tracks blended sky brightness
            AmbientScale = dayFactor * 1.0f + sunsetFactor * 0.4f + nightFactor * 0.05f;
            
            // Blended light color (day/sunset/night palettes matching shader)
            SunLight.Color = new Color3(
                dayFactor * 1.0f  + sunsetFactor * 1.0f + nightFactor * 0.1f,
                dayFactor * 0.95f + sunsetFactor * 0.7f + nightFactor * 0.15f,
                dayFactor * 0.85f + sunsetFactor * 0.4f + nightFactor * 0.3f);
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
