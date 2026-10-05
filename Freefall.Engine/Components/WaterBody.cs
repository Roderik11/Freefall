using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// A lake or river: draws the mesh a sibling RuntimeMesh generated from the entity's Spline with the
    /// water shader (water.fx). Closed spline = still water (lake, pond), open spline = flowing water
    /// whose current follows the spline from its first point to its last.
    ///
    /// Setup: Spline + RuntimeMesh + WaterBody (no MeshRenderer: this component draws the generated mesh).
    /// A river usually shares the entity with the HeightStamp that carves its bed: the spline carries the
    /// bed heights and RuntimeMesh.Offset lifts the surface by the water depth.
    ///
    /// Shading is shared with the ocean (water_common.fx), so all water follows time of day, sky and
    /// clouds the same way. Ripples borrow the ocean simulation's slope maps when the scene has an
    /// OceanRenderer; without one the surface is flat.
    /// </summary>
    [Icon("icon_ocean.png")]
    public class WaterBody : Component, IDraw
    {
        [Category("Color")]
        [Description(@"Color of deep water (light scattered back from the water itself)")]
        public Color3 WaterColor = new Color3(0.004f, 0.02f, 0.018f);

        [Description(@"Tint the bed takes on through shallow water")]
        public Color3 ShallowColor = new Color3(0.45f, 0.7f, 0.55f);

        [ValueRange(0.3f, 30f)]
        [Description(@"Underwater visibility in meters.
         At this much water the bed is tinted by ShallowColor and mostly faded out")]
        public float Visibility = 3.5f;

        [ValueRange(0f, 0.1f)]
        [Description(@"How much the ripples bend the view of the bed")]
        public float RefractionStrength = 0.02f;

        [Category("Surface")]
        [ValueRange(0f, 2f)]
        [Description(@"Ripple height. Lakes and slow rivers want far less than the open sea")]
        public float RippleStrength = 0.35f;

        [ValueRange(0.2f, 5f)]
        [Description(@"Ripple size multiplier (1 = the ocean's finest wave bands as they are)")]
        public float RippleSize = 1f;

        [ValueRange(0f, 6f)]
        [Description(@"Current speed in m/s along the spline (open splines only).
         Steep stretches run faster on their own")]
        public float FlowSpeed = 1.2f;

        [Category("Foam")]
        [ValueRange(0f, 1f)]
        [Description(@"Foam along the banks and around anything standing in the water")]
        public float EdgeFoam = 0.4f;

        [ValueRange(0f, 2f)]
        [Description(@"White water on steep stretches (open splines only)")]
        public float RapidsFoam = 1f;

        private Material _material = null!;
        private readonly MaterialBlock _params = new MaterialBlock();
        private RuntimeMesh? _runtimeMesh;
        private Spline? _spline;
        private OceanRenderer? _ocean;
        private DirectionalLight? _sunLight;

        [StructLayout(LayoutKind.Sequential)]
        public struct WaterData
        {
            public float Time;
            public float FlowSpeed;
            public float RippleStrength;
            public float RippleSize;
            public Vector3 WaterColor;
            public float Visibility;
            public Vector3 ShallowColor;
            public float RefractionStrength;
            public Vector3 SunDirection;
            public float SunIntensity;
            public Vector3 SunColor;
            public float EdgeFoam;
            public Vector3 CloudColor;
            public float RapidsFoam;
            public uint SlopeSRV;
            public uint NoiseSRV;
            public uint DepthGBufferSRV;
            public uint CompositeSRV;
            public float InvViewportWidth;
            public float InvViewportHeight;
            public float TileScaleA;
            public float TileScaleB;
            public uint BandA;
            public uint BandB;
            public float Flowing;       // 1 = open spline (river), 0 = closed (lake)
            public float _pad0;
        }

        protected override void Awake()
        {
            _material = new Material(Assets.InternalAssets.WaterEffect);
            _runtimeMesh = Entity.GetComponent<RuntimeMesh>();
            _spline = Entity.GetComponent<Spline>();
            _sunLight = EntityManager.FindComponent<DirectionalLight>();
        }

        public void Draw()
        {
            if (_material == null) return;
            _runtimeMesh ??= Entity.GetComponent<RuntimeMesh>();
            var mesh = _runtimeMesh?.GeneratedMesh;
            if (mesh == null) return;

            // Lazy lookups: the ocean (ripple textures) may be created after this component
            _ocean ??= EntityManager.FindComponent<OceanRenderer>();
            _sunLight ??= EntityManager.FindComponent<DirectionalLight>();
            _spline ??= Entity.GetComponent<Spline>();

            var sunDir = Vector3.UnitY;
            var sunColor = Vector3.One;
            float sunIntensity = 1.0f;
            if (_sunLight != null)
            {
                sunDir = Vector3.Transform(Vector3.UnitZ, _sunLight.Entity.Transform.Rotation);
                sunColor = new Vector3(_sunLight.Color.R, _sunLight.Color.G, _sunLight.Color.B);
                sunIntensity = _sunLight.Intensity;
            }

            // Ripples: the two finest bands of the ocean simulation
            var fft = _ocean?.FFT;
            int bandCount = _ocean?.Bands.Count ?? 0;
            uint bandA = 0, bandB = 0;
            float tileA = 0f, tileB = 0f;
            if (fft != null && bandCount > 0)
            {
                bandB = (uint)(bandCount - 1);
                bandA = (uint)Math.Max(0, bandCount - 2);
                tileA = 1.0f / _ocean!.Bands[(int)bandA].LengthScale;
                tileB = 1.0f / _ocean.Bands[(int)bandB].LengthScale;
            }

            var depth = DeferredRenderer.Current?.DepthGBuffer;

            _params.SetParameter("WaterData", new WaterData
            {
                Time = Time.TotalTime,
                FlowSpeed = FlowSpeed,
                RippleStrength = RippleStrength,
                RippleSize = RippleSize,
                WaterColor = new Vector3(WaterColor.R, WaterColor.G, WaterColor.B),
                Visibility = Visibility,
                ShallowColor = new Vector3(ShallowColor.R, ShallowColor.G, ShallowColor.B),
                RefractionStrength = RefractionStrength,
                SunDirection = sunDir,
                SunIntensity = sunIntensity,
                SunColor = sunColor,
                EdgeFoam = EdgeFoam,
                CloudColor = SkyboxRenderer.CurrentCloudColor,
                RapidsFoam = RapidsFoam,
                SlopeSRV = fft?.SlopeSRV ?? 0,
                NoiseSRV = fft?.NoiseSRV ?? 0,
                DepthGBufferSRV = depth?.BindlessIndex ?? 0,
                CompositeSRV = DeferredRenderer.Current?.CompositeSnapshot?.BindlessIndex ?? 0,
                InvViewportWidth = 1.0f / MathF.Max(1, depth?.Native.Description.Width ?? 1920),
                InvViewportHeight = 1.0f / MathF.Max(1, depth?.Native.Description.Height ?? 1080),
                TileScaleA = tileA,
                TileScaleB = tileB,
                BandA = bandA,
                BandB = bandB,
                Flowing = _spline != null && !_spline.Closed ? 1f : 0f,
            });

            CommandBuffer.Enqueue(mesh, _material, _params, Transform.TransformSlot);
        }
    }
}
