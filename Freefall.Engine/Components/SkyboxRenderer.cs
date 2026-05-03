using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Graphics;
using Freefall.Base;
using Freefall.Assets;
using Vortice.Mathematics;

namespace Freefall.Components
{
    // TODO: Sky shader is ~1ms on RTX 4080 — excessive for a flat-dome sky.
    // Root cause: mesh_skybox.fx GetClouds() computes all noise procedurally per pixel:
    //   - fbm(8 octaves) + fbm(4) + worley(27 iterations) + fbm(3) + fbm(3) + domain warp fbm(3)x2
    //   - Total: ~24 octaves of sin()-based noise + 27-iter Worley per sky pixel at full res
    // Fix: Replace procedural noise with precomputed 3D noise textures (128³ Perlin-Worley + 32³ detail).
    //   - Single texture fetch vs 100+ trig ops per pixel
    //   - Budget freed up could support volumetric ray-marched clouds (32-64 steps at quarter-res)
    //     with proper volumetric lighting, god rays, and camera parallax — still under 1ms.
    public class SkyboxRenderer : Component, IUpdate, IDraw
    {
        private Mesh Mesh;
        private Material Material = InternalAssets.SkyboxMaterial;
        private MaterialBlock Params = new MaterialBlock();

        public bool AnimateTimeOfDay = false;

        [ValueRange(0f, 24f)]
        public float TimeOfDay = 16f;         // 0-24 hours
  
        [ValueRange(0.1f, 1f)]
        public float TimeOfDaySpeed = 0.1f; // Hours per second (tweak for testing)

        [ValueRange(0f, 1f)]
        public float CloudCoverage = 0.5f;      // 0-1

        [ValueRange(0f, 10f)]
        public float CloudSpeed = 1.0f;         // Speed multiplier

        [ValueRange(0f, 10f)]
        public float SunIntensity = 1.2f;       // Sun brightness multiplier

        [ValueRange(0f, 10f)]
        public float DayIntensity = 3.14159f;       // Sun intensity at noon

        [ValueRange(0f, 10f)]
        public float NightIntensity = 0.1f;     // Ambient intensity at night

        [ValueRange(0f, 1f)]
        public float StarDensity = 0.5f;        // 0-1, controls how many stars

        [ValueRange(0f, 10f)]
        public float StarBrightness = 1.0f;     // Star intensity multiplier

        [ValueRange(0f, 360f)]
        public float SunAzimuthAngle = 30.0f;    // Compass heading for sunrise in degrees (0=+X, 90=+Z, 180=-X, 270=-Z)

        public DirectionalLight SunLight;

        public Vector3 SunDirection = new Vector3(0, 1, 0);


        // Static ambient scale accessible by composition pass
        public static float AmbientScale { get; private set; } = 1.0f;
        public static Vector3 CurrentSunDirection { get; private set; } = new Vector3(0, 1, 0);

        private float CloudTime = 0.0f;


        public SkyboxRenderer()
        {
        }

        protected override void Awake()
        {
            Mesh = Mesh.CreateCube(Engine.Device, 100.0f);
            SunLight ??= EntityManager.FindComponent<DirectionalLight>();
        }

        public void Update()
        {
            if (Camera.Main == null) return;
            
            // Move with camera to simulate infinite distance
            Transform.Position = Camera.Main.Position;

            // Animate time of day if enabled
            if (AnimateTimeOfDay)
            {
                TimeOfDay += (float)Time.Delta * TimeOfDaySpeed;
                if (TimeOfDay >= 24.0f) TimeOfDay -= 24.0f;
                if (TimeOfDay < 0.0f) TimeOfDay += 24.0f;

            }

            CloudTime += (float)Time.Delta * CloudSpeed;

            UpdateSunLight();
        }

        private void UpdateSunLight()
        {
            if (SunLight == null) return;

            // Calculate sun angle (0 at sunrise, PI at sunset)
            float sunAngle = (TimeOfDay / 24.0f) * MathF.PI * 2.0f - MathF.PI * 0.5f;

            // Calculate sun direction - rotated around Y by SunAzimuthAngle
            float elevation = MathF.Sin(sunAngle);
            float horizontal = MathF.Cos(sunAngle);
            float azimuthRad = SunAzimuthAngle * (MathF.PI / 180.0f);
            SunDirection = new Vector3(horizontal * MathF.Cos(azimuthRad), elevation, horizontal * MathF.Sin(azimuthRad));

            // Clamp light direction to prevent it from shining from below ground
            float clampedElevation = MathF.Max(0.1f, elevation);
            Vector3 lightDirection = new Vector3(horizontal * MathF.Cos(azimuthRad), clampedElevation, horizontal * MathF.Sin(azimuthRad));

            // Light should point FROM the sun (opposite of sun direction)
            Vector3 lightDir = Vector3.Normalize(lightDirection);

            // Set directional light rotation using CreateWorld
            // Transform.Forward is what DirectionalLight uses as LightDirection
            // The shader already negates this (L = -LightDirection), so we pass the direction light is pointing (toward ground)
            SunLight.Transform.Rotation = Quaternion.CreateFromRotationMatrix(Matrix4x4.CreateWorld(Vector3.Zero, lightDir, Vector3.UnitY));

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
            CurrentSunDirection = SunDirection;
            
            // Blended light color (day/sunset/night palettes matching shader)
            SunLight.Color = new Color3(
                dayFactor * 1.0f  + sunsetFactor * 1.0f + nightFactor * 0.1f,
                dayFactor * 0.95f + sunsetFactor * 0.7f + nightFactor * 0.15f,
                dayFactor * 0.85f + sunsetFactor * 0.4f + nightFactor * 0.3f);
        }

        public void Draw()
        {
            if (Material == null || Mesh == null) return;

            var slot = Entity.Transform.TransformSlot;

            // Set sky parameters on the MaterialBlock
            Material.SetParameter("World", Entity.Transform.WorldMatrix);
            Material.SetParameter("SunDirection", SunDirection);
            Material.SetParameter("TimeOfDay", TimeOfDay);
            Material.SetParameter("CloudCoverage", CloudCoverage);
            Material.SetParameter("CloudTime", CloudTime);
            Material.SetParameter("CloudSpeed", CloudSpeed);
            Material.SetParameter("SunIntensity", SunIntensity);
            Material.SetParameter("StarDensity", StarDensity);
            Material.SetParameter("StarBrightness", StarBrightness);

            CommandBuffer.Enqueue(Mesh, Material, Params, slot);
        }
    }
}
