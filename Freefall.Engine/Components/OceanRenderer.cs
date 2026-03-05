using System;
using System.Collections.Generic;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Freefall;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.Direct3D12;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Freefall.Components
{
    /// <summary>
    /// Serializable per-band spectrum configuration.
    /// Each band has 2 spectrum layers (primary + secondary) with Vector2 params.
    /// </summary>
    [Serializable]
    public class SpectrumBand
    {
        [Description(@"FFT domain size in meters.
         Larger values produce longer-wavelength waves that repeat less visibly")]
        public uint LengthScale = 256;

        [Description(@"Amplitude multiplier for primary (X) and secondary (Y) spectrum layers")]
        public Vector2 Scale = new Vector2(0.5f, 0.3f);

        [Description(@"Wind speed in m/s driving each spectrum layer.
         Faster wind produces taller, longer waves")]
        public Vector2 WindSpeed = new Vector2(10f, 8f);

        [Description(@"Wind direction in degrees for each layer.
         Offset directions break up uniformity")]
        public Vector2 WindDirection = new Vector2(0f, 30f);

        [Description(@"Fetch distance in meters — how far wind blows over open water.
         Longer fetch produces more developed seas")]
        public Vector2 Fetch = new Vector2(100000f, 50000f);

        [Description(@"Blend between cosine-power (0) and
         Donelan-Banner (1) directional spreading")]
        public Vector2 SpreadBlend = new Vector2(0.5f, 0.3f);

        [Description(@"Swell factor.
         Higher values add long-period energy that persists beyond local wind")]
        public Vector2 Swell = new Vector2(1.0f, 0.5f);

        [Description(@"JONSWAP gamma — peak enhancement factor.
         Higher values concentrate energy at the peak frequency.
         3.3 = fully developed sea")]
        public Vector2 PeakEnhancement = new Vector2(3.3f, 1.0f);

        [Description(@"High-frequency rolloff.
         Suppresses short wavelengths to avoid aliasing at this band's resolution")]
        public Vector2 ShortWavesFade = new Vector2(0.01f, 0.01f);

        [Description(@"Whether this band's displacement affects vertex positions.
         Disable for fine-detail bands that only contribute normals")]
        public bool ContributeDisplacement = true;
    }

    /// <summary>
    /// Component: ocean surface with FFT wave simulation.
    /// Wave displacement and normals come from GPU-computed FFT textures.
    /// </summary>
    public class OceanRenderer : Component, IDraw, IUpdate
    {
        // ── Visual parameters ──
        public Color3 OceanColor = new Color3(0.006f, 0.035f, 0.055f);
        public Color3 DeepColor = new Color3(0.003f, 0.012f, 0.03f);

        // ── Simulation parameters (runtime-tweakable) ──

        [Category("Settings")]
        [ValueRange(0.5f, 200f)]
        [Description(@"Water depth in meters.
         Affects wave dispersion and TMA spectral correction.
         Shallower water produces shorter, steeper waves")]
        public float Depth = 40.0f;
        
        [Category("Settings")]
        [ValueRange(0f, 1f)]
        [Description(@"Horizontal displacement scale (Lambda).
         Higher values push wave crests apart,
         creating sharper peaks but risking self-intersection artifacts")]
        public float Choppiness = 1.0f;
        
        [Category("Settings")]
        [ValueRange(0f, 5f)]
        [Description(@"Time multiplier for wave animation speed.
         1.0 = real-time")]
        public float WaveSpeed = 1.0f;
        
        [Category("Settings")]
        [ValueRange(10f, 1000f)]
        [Description(@"Period in seconds before the wave pattern repeats.
         Higher values delay visible looping")]
        public float RepeatTime = 200.0f;

        // ── Foam parameters ──
        [Category("Foam")]
        [ValueRange(0f, 2f)]
        [Description(@"Jacobian threshold for foam generation.
         Lower values produce foam on gentler waves,
         higher values restrict foam to steep crests only")]
        public float FoamBias = 0.67f;
        
        [Category("Foam")]
        [ValueRange(0f, 0.1f)]
        [Description(@"Exponential decay rate per frame.
         Controls how quickly foam fades after forming.
         Lower values make foam linger longer")]
        public float FoamDecayRate = 0.0175f;
        
        [Category("Foam")]
        [ValueRange(0f, 1f)]
        [Description(@"Minimum Jacobian deviation required before any foam is added.
         Acts as a noise gate to suppress faint foam")]
        public float FoamThreshold = 0.1f;
        
        [Category("Foam")]
        [ValueRange(0f, 1f)]
        [Description(@"Amount of foam injected per frame when the Jacobian exceeds the threshold.
         Higher values create thicker, more opaque foam")]
        public float FoamAdd = 0.1f;

        [Category("Shore")]
        [ValueRange(1f, 100f)]
        [Description(@"Water depth in meters at which waves reach full strength.
         Shallower water than this will have progressively calmer waves")]
        public float ShoreDepth = 35f;

        [Category("Shore")]
        [ValueRange(0f, 0.5f)]
        [Description(@"Minimum wave displacement at the shoreline.
         0 = completely flat at shore, higher = more residual ripple")]
        public float ShoreMinWave = 0.1f;

        [Category("Shore")]
        public Color3 ShallowColor = new Color3(0.4f, 0.75f, 0.7f);

        [Category("Shore")]
        [ValueRange(1f, 30f)]
        [Description(@"Linear depth range in meters for PS shore effects.
         Controls how wide the soft intersection and shallow color zones are")]
        public float ShoreFadeDepth = 8f;

        [Category("Shore")]
        [ValueRange(0f, 0.1f)]
        [Description(@"Refraction distortion strength for terrain show-through.
         How much the wave normal bends the view of the seabed")]
        public float RefractionStrength = 0.02f;

        // ── Spectrum bands (editable) ──
        public List<SpectrumBand> Bands = CreateDefaultBands();

        private Mesh _mesh = null!;
        private Material _material = null!;
        private MaterialBlock _params = new MaterialBlock();
        private OceanFFT? _oceanFFT;
        private float _waveTime;
        private DirectionalLight? _sunLight;
        private TerrainRenderer? _terrain;

        // GPU buffer for tile scales (1/lengthScale per band)
        private ID3D12Resource? _tileScalesBuffer;
        private uint _tileScalesSRV;

        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct OceanData
        {
            public float WaveTime;
            public float Choppiness;
            public float FoamDecayRate;
            public float FoamThreshold;
            public Vector3 OceanColor;
            public float _pad0;
            public Vector3 DeepColor;
            public float _pad1;
            public Vector3 SunDirection;
            public float SunIntensity;
            public Vector3 SunColor;
            public float _pad2;
            public uint DisplacementSRV;
            public uint SlopeSRV;
            public uint NumBands;
            public uint TileScalesSRV;
            public float DisplacementAtten;
            public float NormalAtten;
            public uint HeightmapSRV;
            public float ShoreDepth;
            public float MaxTerrainHeight;
            public float OceanPlaneY;
            public float TerrainBaseY;
            public float TerrainOriginX;
            public float TerrainOriginZ;
            public float TerrainSizeX;
            public float TerrainSizeZ;
            public float ShoreMinWave;
            public uint DepthGBufferSRV;
            public uint CompositeSRV;
            public float ShoreFadeDepth;
            public Vector3 ShallowColor;
            public float RefractionStrength;
            public uint NoiseSRV;
            public uint _pad3, _pad4, _pad5;
        }

        private const int GridSize = 128;

        protected override void Awake()
        {
            _mesh = CreateOceanGrid(Engine.Device, GridSize);
            _material = new Material(new Effect("ocean"));
            _sunLight = EntityManager.FindComponent<DirectionalLight>()!;
            _terrain = EntityManager.FindComponent<TerrainRenderer>();

            // Build spectrum params from bands
            var spectrums = BuildSpectrumParams();
            var lengthScales = BuildLengthScales();

            _oceanFFT = new OceanFFT();
            _oceanFFT.Create(Engine.Device, new OceanFFT.OceanParams
            {
                Gravity = 9.81f,
                Depth = Depth,
                RepeatTime = RepeatTime,
                LowCutoff = 0.0001f,
                HighCutoff = 9000.0f,
                Seed = 42,
                Lambda = new Vector2(Choppiness, Choppiness),
                FoamBias = FoamBias,
                FoamDecayRate = FoamDecayRate,
                FoamThreshold = FoamThreshold,
                FoamAdd = FoamAdd,
                LengthScales = lengthScales,
                Spectrums = spectrums,
            });

            // Create tile scales buffer for the render shader
            CreateTileScalesBuffer();
        }

        private unsafe void CreateTileScalesBuffer()
        {
            int count = Bands.Count;
            int bufSize = Math.Max(sizeof(float) * count, 16);
            _tileScalesBuffer = Engine.Device.CreateUploadBuffer(bufSize);
            _tileScalesSRV = Engine.Device.AllocateBindlessIndex();
            Engine.Device.CreateStructuredBufferSRV(_tileScalesBuffer, (uint)count, (uint)sizeof(float), _tileScalesSRV);

            void* ptr;
            _tileScalesBuffer.Map(0, null, &ptr);
            var span = new Span<float>(ptr, count);
            for (int i = 0; i < count; i++)
                span[i] = 1.0f / Bands[i].LengthScale;
            _tileScalesBuffer.Unmap(0);
        }

        private unsafe void UpdateTileScalesBuffer()
        {
            int count = Bands.Count;
            void* ptr;
            _tileScalesBuffer.Map(0, null, &ptr);
            var span = new Span<float>(ptr, count);
            for (int i = 0; i < count; i++)
                span[i] = 1.0f / Bands[i].LengthScale;
            _tileScalesBuffer.Unmap(0);
        }

        private uint[] BuildLengthScales()
        {
            var scales = new uint[Bands.Count];
            for (int i = 0; i < Bands.Count; i++)
                scales[i] = Bands[i].LengthScale;
            return scales;
        }

        private OceanFFT.SpectrumParameters[] BuildSpectrumParams()
        {
            float gravity = 9.81f;
            float JonswapAlpha(float fetch, float windSpeed) =>
                0.076f * MathF.Pow(gravity * fetch / windSpeed / windSpeed, -0.22f);
            float JonswapPeakOmega(float fetch, float windSpeed) =>
                22f * MathF.Pow(windSpeed * fetch / gravity / gravity, -0.33f);
            float Rad(float deg) => deg / 180f * MathF.PI;

            var spectrums = new OceanFFT.SpectrumParameters[Bands.Count * 2];

            for (int i = 0; i < Bands.Count; i++)
            {
                var b = Bands[i];

                spectrums[i * 2] = new OceanFFT.SpectrumParameters
                {
                    Scale = b.Scale.X,
                    Angle = Rad(b.WindDirection.X),
                    SpreadBlend = b.SpreadBlend.X,
                    Swell = b.Swell.X,
                    Alpha = JonswapAlpha(b.Fetch.X, b.WindSpeed.X),
                    PeakOmega = JonswapPeakOmega(b.Fetch.X, b.WindSpeed.X),
                    Gamma = b.PeakEnhancement.X,
                    ShortWavesFade = b.ShortWavesFade.X,
                };

                spectrums[i * 2 + 1] = new OceanFFT.SpectrumParameters
                {
                    Scale = b.Scale.Y,
                    Angle = Rad(b.WindDirection.Y),
                    SpreadBlend = b.SpreadBlend.Y,
                    Swell = b.Swell.Y,
                    Alpha = JonswapAlpha(b.Fetch.Y, b.WindSpeed.Y),
                    PeakOmega = JonswapPeakOmega(b.Fetch.Y, b.WindSpeed.Y),
                    Gamma = b.PeakEnhancement.Y,
                    ShortWavesFade = b.ShortWavesFade.Y,
                };
            }

            return spectrums;
        }

        /// <summary>
        /// Create 4 default bands with logarithmic length scale spread.
        /// </summary>
        private static List<SpectrumBand> CreateDefaultBands()
        {
            return new List<SpectrumBand>
            {
                // Band 0: 1000m — large swell
                new SpectrumBand
                {
                    LengthScale = 1000,
                    Scale = new Vector2(0.5f, 0.3f),
                    WindSpeed = new Vector2(15f, 12f),
                    WindDirection = new Vector2(22f, 59f),
                    Fetch = new Vector2(800000f, 300000f),
                    SpreadBlend = new Vector2(0.78f, 0.5f),
                    Swell = new Vector2(1.0f, 0.9f),
                    PeakEnhancement = new Vector2(3.3f, 2.5f),
                    ShortWavesFade = new Vector2(0.01f, 0.01f),
                },
                // Band 1: 250m — medium waves
                new SpectrumBand
                {
                    LengthScale = 250,
                    Scale = new Vector2(0.25f, 0.25f),
                    WindSpeed = new Vector2(20f, 20f),
                    WindDirection = new Vector2(97f, 67f),
                    Fetch = new Vector2(100000000f, 1000000f),
                    SpreadBlend = new Vector2(0.14f, 0.47f),
                    Swell = new Vector2(1.0f, 1.0f),
                    PeakEnhancement = new Vector2(1.0f, 1.0f),
                    ShortWavesFade = new Vector2(0.5f, 0.5f),
                },
                // Band 2: 64m — close-range chop
                new SpectrumBand
                {
                    LengthScale = 64,
                    Scale = new Vector2(0.15f, 0.1f),
                    WindSpeed = new Vector2(5f, 1f),
                    WindDirection = new Vector2(105f, 19f),
                    Fetch = new Vector2(1000000f, 10000f),
                    SpreadBlend = new Vector2(0.2f, 0.298f),
                    Swell = new Vector2(1.0f, 0.695f),
                    PeakEnhancement = new Vector2(1.0f, 1.0f),
                    ShortWavesFade = new Vector2(0.5f, 0.5f),
                },
                // Band 3: 17m — fine ripples near camera
                new SpectrumBand
                {
                    LengthScale = 17,
                    Scale = new Vector2(1.0f, 0.23f),
                    WindSpeed = new Vector2(1f, 1f),
                    WindDirection = new Vector2(209f, 0f),
                    Fetch = new Vector2(200000f, 1000f),
                    SpreadBlend = new Vector2(0.56f, 0.0f),
                    Swell = new Vector2(1.0f, 0.0f),
                    PeakEnhancement = new Vector2(1.0f, 1.0f),
                    ShortWavesFade = new Vector2(0.0001f, 0.15f),
                },
            };
        }

        public void Update()
        {
            if (_mesh == null || _material == null) return;
            _waveTime += (float)Time.Delta * WaveSpeed;

            // Rebuild spectrum params from bands every frame (upload is tiny)
            var spectrums = BuildSpectrumParams();
            var lengthScales = BuildLengthScales();
            _oceanFFT.RequestSpectrumReinit(spectrums, lengthScales);

            // Push runtime-tweakable params to OceanFFT each frame
            _oceanFFT.UpdateParams(Depth, FoamBias, FoamDecayRate, FoamThreshold, FoamAdd,
                new Vector2(Choppiness, Choppiness));

            // Update tile scales buffer (in case LengthScale changed)
            UpdateTileScalesBuffer();

            // Dispatch FFT compute before enqueueing draw
            CommandBuffer.Enqueue(RenderPass.Opaque, (cmd) =>
            {
                _oceanFFT.GenerateNoise(cmd);
                _oceanFFT.InitSpectrum(cmd);
                _oceanFFT.Update(cmd, _waveTime, (float)Time.Delta);
            });
        }

        public void Draw()
        {
            if (_mesh == null || _material == null || _oceanFFT == null) return;

            // Lazy terrain lookup (terrain may not exist at Awake time)
            if (_terrain == null)
                _terrain = EntityManager.FindComponent<TerrainRenderer>();

            // Sun light
            var sunDir = Vector3.UnitY;
            var sunColor = Vector3.One;
            float sunIntensity = 1.0f;
            if (_sunLight != null)
            {
                sunDir = Vector3.Transform(Vector3.UnitZ, _sunLight.Entity.Transform.Rotation);
                sunColor = new Vector3(_sunLight.Color.R, _sunLight.Color.G, _sunLight.Color.B);
                sunIntensity = _sunLight.Intensity;
            }

            _params.SetParameter("OceanData", new OceanData
            {
                WaveTime = _waveTime,
                Choppiness = Choppiness,
                FoamDecayRate = FoamDecayRate,
                FoamThreshold = FoamThreshold,
                OceanColor = new Vector3(OceanColor.R, OceanColor.G, OceanColor.B),
                DeepColor = new Vector3(DeepColor.R, DeepColor.G, DeepColor.B),
                SunDirection = sunDir,
                SunColor = sunColor,
                SunIntensity = sunIntensity,
                DisplacementSRV = _oceanFFT.DisplacementSRV,
                SlopeSRV = _oceanFFT.SlopeSRV,
                NumBands = (uint)Bands.Count,
                TileScalesSRV = _tileScalesSRV,
                DisplacementAtten = 1.0f,
                NormalAtten = 1.0f,
                HeightmapSRV = _terrain?.Terrain?.Heightmap?.BindlessIndex ?? 0,
                ShoreDepth = ShoreDepth,
                MaxTerrainHeight = _terrain?.Terrain?.MaxHeight ?? 0,
                OceanPlaneY = Transform.WorldPosition.Y,
                TerrainBaseY = _terrain?.Transform?.WorldPosition.Y ?? 0,
                TerrainOriginX = _terrain?.Transform?.WorldPosition.X ?? 0,
                TerrainOriginZ = _terrain?.Transform?.WorldPosition.Z ?? 0,
                TerrainSizeX = _terrain?.Terrain?.TerrainSize.X ?? 1,
                TerrainSizeZ = _terrain?.Terrain?.TerrainSize.Y ?? 1,
                ShoreMinWave = ShoreMinWave,
                DepthGBufferSRV = DeferredRenderer.Current?.DepthGBuffer?.BindlessIndex ?? 0,
                CompositeSRV = DeferredRenderer.Current?.CompositeSnapshot?.BindlessIndex ?? 0,
                ShoreFadeDepth = ShoreFadeDepth,
                ShallowColor = new Vector3(ShallowColor.R, ShallowColor.G, ShallowColor.B),
                RefractionStrength = RefractionStrength,
                NoiseSRV = _oceanFFT.NoiseSRV,
            });


            CommandBuffer.Enqueue(_mesh, _material, _params, Transform.TransformSlot);
        }

        private static Mesh CreateOceanGrid(GraphicsDevice device, int gridSize)
        {
            int vertsPerSide = gridSize + 1;
            int vertCount = vertsPerSide * vertsPerSide;
            int quadCount = gridSize * gridSize;
            int indexCount = quadCount * 6;

            var positions = new Vector3[vertCount];
            var normals = new Vector3[vertCount];
            var uvs = new Vector2[vertCount];
            var indices = new uint[indexCount];

            for (int z = 0; z <= gridSize; z++)
            {
                for (int x = 0; x <= gridSize; x++)
                {
                    int i = z * vertsPerSide + x;
                    float fx = (float)x / gridSize * 2.0f - 1.0f;
                    float fz = (float)z / gridSize * 2.0f - 1.0f;
                    positions[i] = new Vector3(fx, 0, fz);
                    normals[i] = Vector3.UnitY;
                    uvs[i] = new Vector2((float)x / gridSize, (float)z / gridSize);
                }
            }

            int idx = 0;
            for (int z = 0; z < gridSize; z++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    uint v0 = (uint)(z * vertsPerSide + x);
                    uint v1 = (uint)(z * vertsPerSide + x + 1);
                    uint v2 = (uint)((z + 1) * vertsPerSide + x);
                    uint v3 = (uint)((z + 1) * vertsPerSide + x + 1);

                    indices[idx++] = v2; indices[idx++] = v1; indices[idx++] = v0;
                    indices[idx++] = v2; indices[idx++] = v3; indices[idx++] = v1;
                }
            }

            var mesh = new Mesh(device, positions, normals, uvs, indices);
            mesh.BoundingBox = new BoundingBox(
                new Vector3(-1, -10, -1),
                new Vector3(1, 10, 1)
            );
            mesh.MeshParts.Add(new MeshPart
            {
                NumIndices = indexCount,
                BoundingBox = mesh.BoundingBox,
                BoundingSphere = mesh.LocalBoundingSphere
            });
            return mesh;
        }
    }
}
