using System;
using System.Collections.Generic;
using Description = System.ComponentModel.DescriptionAttribute;
using Category = System.ComponentModel.CategoryAttribute;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
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
    [UpdateInEditor]
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

        [Description(@"Part of the sea the weather's wind raises (the environment preset's wind, Graphics.Wind): its waves come
         from the wind's speed and heading and the ocean's Wind Sea settings, and the wind bands share the wavelengths
         between them instead of each carrying all of them. Off = authored by the fields above (a swell from afar)")]
        public bool FollowWind = false;
    }

    /// <summary>
    /// The sea the wind raises, shared by every band with <see cref="SpectrumBand.FollowWind"/>: one JONSWAP spectrum at
    /// the weather's wind speed, split over those bands by wavelength. A calm preset gives a calm sea, a storm a rough one.
    /// </summary>
    [Serializable]
    public class WindSeaSettings
    {
        [Description("Distance in metres the wind has blown over open water: longer = a more developed sea, longer waves")]
        [ValueRange(1000f, 1000000f)]
        public float Fetch = 100000f;

        [Description("Energy multiplier on the spectrum (1 = the sea this wind really makes: its slopes match Cox-Munk's)")]
        [ValueRange(0f, 4f)]
        public float Scale = 1f;

        [Description("Directional spreading: cosine-power (0) to Donelan-Banner (1, measured on wind seas)")]
        [ValueRange(0f, 1f)]
        public float SpreadBlend = 1f;

        [Description("Narrows the spread round the wind's heading: higher = longer-crested waves")]
        [ValueRange(0f, 2f)]
        public float Swell = 0.5f;

        [Description("JONSWAP gamma: 3.3 = a growing sea, 1 = fully developed (Pierson-Moskowitz)")]
        [ValueRange(1f, 7f)]
        public float PeakEnhancement = 3.3f;

        [Description("High-frequency rolloff of the finest waves (against aliasing at the finest band's resolution)")]
        [ValueRange(0f, 1f)]
        public float ShortWavesFade = 0.01f;
    }

    /// <summary>
    /// Component: ocean surface with FFT wave simulation.
    /// Wave displacement and normals come from GPU-computed FFT textures.
    /// </summary>
    [Icon("icon_ocean.png")]
    public class OceanRenderer : Component, IDraw, IUpdate
    {
        [Category("Color")]
        public Color3 OceanColor = new Color3(0.002f, 0.012f, 0.022f);
        public Color3 DeepColor = new Color3(0.001f, 0.004f, 0.012f);

        [Category("Simulation")]
        [ValueRange(0.5f, 200f)]
        [Description(@"Water depth in meters.
         Affects wave dispersion and TMA spectral correction.
         Shallower water produces shorter, steeper waves")]
        public float Depth = 40.0f;
        
        [ValueRange(0f, 1f)]
        [Description(@"Horizontal displacement scale (Lambda): squeezes the crests together into sharp peaks and
         stretches the troughs (1 = what real waves do). The squeezed crests are where whitecaps form")]
        public float Choppiness = 1.0f;
        
        [ValueRange(0f, 5f)]
        [Description(@"Time multiplier for wave animation speed.
         1.0 = real-time")]
        public float WaveSpeed = 1.0f;
        
        [ValueRange(10f, 1000f)]
        [Description(@"Period in seconds before the wave pattern repeats.
         Higher values delay visible looping")]
        public float RepeatTime = 200.0f;

        // Whitecaps: foam where a crest is squeezed, the Jacobian J of the choppy displacement under FoamBias −
        // FoamThreshold. The defaults (2026-10-18) give Monahan's whitecap share within ~1.5× from 3 to 15 m/s of
        // wind (none at 3, ~1% at 10, ~3.5% at 15) and somewhat less above (7.5% at 20 m/s against 10.5%).
        [Category("Foam")]
        [ValueRange(0f, 2f)]
        [Description(@"Whitecaps form where a crest is squeezed below this Jacobian (minus the threshold).
         Higher = more crests foam, at lighter winds")]
        public float FoamBias = 0.88f;

        [ValueRange(0f, 0.1f)]
        [Description(@"Exponential decay rate per frame.
         Controls how quickly foam fades after forming.
         Lower values make foam linger longer")]
        public float FoamDecayRate = 0.005f;

        [ValueRange(0f, 1f)]
        [Description(@"Minimum Jacobian deviation required before any foam is added.
         Acts as a noise gate to suppress faint foam")]
        public float FoamThreshold = 0.1f;

        [ValueRange(0f, 1f)]
        [Description(@"Amount of foam injected per frame when the Jacobian exceeds the threshold.
         Higher values create thicker, more opaque foam")]
        public float FoamAdd = 0.35f;

        [Category("Shore")]
        [ValueRange(1f, 100f)]
        [Description(@"Water depth in meters at which waves reach full strength.
         Shallower water than this will have progressively calmer waves")]
        public float ShoreDepth = 35f;

        [ValueRange(0f, 0.5f)]
        [Description(@"Minimum wave displacement at the shoreline.
         0 = completely flat at shore, higher = more residual ripple")]
        public float ShoreMinWave = 0.1f;

        public Color3 ShallowColor = new Color3(0.4f, 0.75f, 0.7f);

        [ValueRange(1f, 30f)]
        [Description(@"Underwater visibility in meters.
         At this much water the seabed is tinted by ShallowColor and mostly faded out")]
        public float ShoreFadeDepth = 8f;

        [ValueRange(0f, 0.1f)]
        [Description(@"Refraction distortion strength for terrain show-through.
         How much the wave normal bends the view of the seabed")]
        public float RefractionStrength = 0.02f;

        [ValueRange(0.5f, 10f)]
        [Description(@"Water depth in meters below which foam bands roll in toward the beach.
         Larger values start the swash further out")]
        public float ShoreFoamDepth = 2.5f;

        [ValueRange(0f, 1f)]
        [Description(@"Amount of shore foam: swash bands plus the lace along the waterline
         and around objects standing in the water")]
        public float ShoreFoam = 0.8f;

        [ValueRange(0f, 0.5f)]
        [Description(@"How far the water level rises and falls at the beach, in meters, in step with the foam bands.
         Makes the waterline travel up and down the sand; on a flat beach a few centimeters go a long way")]
        public float ShoreSurge = 0.1f;

        [Category("Spectrum")]
        public List<SpectrumBand> Bands = CreateDefaultBands();

        [Description("The sea the weather's wind raises, for the bands marked Follow Wind")]
        public WindSeaSettings WindSea = new();

        // Must match ocean.fx BandAngles: each band's texture is turned by this much (radians) against tiling, so a
        // wave heading in the world is this much more in the band's own frame
        private static readonly float[] BandAngles = { 0.0f, 0.297f, 0.645f, 0.925f, 1.17f, 1.42f, 1.73f, 2.05f };

        // What the spectrum was last initialised with: it is only rebuilt when this changes (an edit, the wind)
        private OceanFFT.SpectrumParameters[] _appliedSpectrums = Array.Empty<OceanFFT.SpectrumParameters>();
        private uint[] _appliedLengthScales = Array.Empty<uint>();
        private float _appliedDepth = float.NaN;

        private Mesh _mesh = null!;
        private Material _material = null!;
        private MaterialBlock _params = new MaterialBlock();
        private OceanFFT? _oceanFFT;

        /// <summary>The wave simulation. WaterBody borrows its slope and noise textures for ripples.</summary>
        public OceanFFT? FFT => _oceanFFT;

        // The ocean that queued its surface this tick, for the sun shadow receivers (ShadowReceivers marks the band
        // its waves can reach along each view ray)
        private static OceanRenderer? _drawn;
        private int _drawnTick = -1;

        /// <summary>
        /// The ocean that queued its surface this tick (Draw), with its rest height, its displacement texture (one slice
        /// per band, vertical displacement in .y) and the shore surge it adds on top; null when none did.
        /// </summary>
        internal static OceanRenderer? DrawnThisTick =>
            _drawn != null && _drawn._drawnTick == Engine.TickCount && _drawn._oceanFFT != null ? _drawn : null;

        internal float PlaneY => Transform.WorldPosition.Y;
        internal uint DisplacementSrv => _oceanFFT?.DisplacementSRV ?? 0;
        internal int BandCount => _oceanFFT?.NumBands ?? 0;
        internal Effect SurfaceEffect => _material.Effect;
        private float _waveTime;
        private DirectionalLight? _sunLight;
        private TerrainRenderer? _terrain;

        // Where the open sea is: rebuilt on a worker when the terrain's heights come back or the sea level moves
        private readonly SeaMask _seaMask = new();
        private SeaMask.Result? _builtSeaMask;   // finished on a worker, waiting for Draw to upload it
        private int _seaMaskRequest;
        private float _seaMaskLevel = float.NaN;

        // GPU buffer for tile scales (1/lengthScale per band)
        private GraphicsBuffer? _tileScalesBuffer;

        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct OceanData
        {
            public float WaveTime;
            public float Choppiness;
            public float FoamDecayRate;
            public float FoamThreshold;
            public Vector3 OceanColor;
            public uint SeaMaskSRV;      // where the open sea is (SeaMask); 0 = everywhere. Was padding: an older build writes 0
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
            public float InvViewportWidth;
            public float InvViewportHeight;
            public float ShoreFoamDepth;
            public float ShoreFoam;
            public float ShoreSurge;
            public uint HiZSRV;          // the view's Hi-Z pyramid: hull shader skips hidden patches (0 = off)
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

            MessageDispatcher.AddListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            Transform?.OnChanged += OnTransformChanged;
        }

        private void OnTerrainHeightsChanged(Message msg)
        {
            if (msg.Data is TerrainHeightsChange change && change.Terrain == _terrain?.Terrain)
                RequestSeaMask();
        }

        // The sea level is the entity's height
        private void OnTransformChanged()
        {
            if (Transform.WorldPosition.Y != _seaMaskLevel)
                RequestSeaMask();
        }

        /// <summary>
        /// Flood the terrain's current heights into a new sea mask on a worker; Draw uploads it. A request made while
        /// an older one runs wins: the older result is dropped.
        /// </summary>
        private void RequestSeaMask()
        {
            var terrain = _terrain?.Terrain;
            var heights = terrain?.HeightField;
            if (terrain == null || heights == null || heights.GetLength(0) < 2) return;

            float seaLevel = Transform.WorldPosition.Y;
            float baseY = _terrain!.Transform?.WorldPosition.Y ?? 0f;
            float seaLevel01 = (seaLevel - baseY) / MathF.Max(terrain.MaxHeight, 0.001f);
            float spacing = terrain.TerrainSize.X / (heights.GetLength(0) - 1);
            _seaMaskLevel = seaLevel;

            int request = Interlocked.Increment(ref _seaMaskRequest);
            Task.Run(() =>
            {
                var mask = SeaMask.Build(heights, seaLevel01, spacing, seaLevel);
                if (Volatile.Read(ref _seaMaskRequest) == request)
                    Volatile.Write(ref _builtSeaMask, mask);
            });
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            Transform?.OnChanged -= OnTransformChanged;
            Interlocked.Increment(ref _seaMaskRequest);   // a build still running is dropped
            _seaMask.Dispose();

            // Deferred: the frames in flight still draw the grid, read the tile scales and the FFT textures
            // through their bindless slots and run the FFT kernels. OceanFFT.Dispose releases only GPU objects
            // (textures, buffers, compute pipeline states), so the whole simulation waits with them.
            var device = Engine.Device;
            device.DeferDispose(_tileScalesBuffer);
            device.DeferDispose(_oceanFFT);
            device.DeferDispose(_mesh);
            if (_drawn == this)
                _drawn = null;
        }

        private unsafe void CreateTileScalesBuffer()
        {
            int count = Bands.Count;
            _tileScalesBuffer = GraphicsBuffer.CreateUpload<float>(count, mapped: true);
            var pData = _tileScalesBuffer.WritePtr<float>();
            for (int i = 0; i < count; i++)
                pData[i] = 1.0f / Bands[i].LengthScale;
        }

        private unsafe void UpdateTileScalesBuffer()
        {
            if (_tileScalesBuffer == null) return;
            int count = Bands.Count;
            var pData = _tileScalesBuffer.WritePtr<float>();
            for (int i = 0; i < count; i++)
                pData[i] = 1.0f / Bands[i].LengthScale;
        }

        // Bands that make no whitecaps. A swell from afar does not break in deep water: with wind bands for the local sea
        // the authored bands (swells) make none; their long crests still squeezed past the foam threshold now and then.
        // Nor does a band whose tile is short: its foam repeats every tile, and the 17 m band's spots laid a grid of
        // dashes over a storm sea (its squeeze comes from waves under 4 m, which break without whitecaps anyway).
        private const uint MinWhitecapTile = 32;
        private uint NoWhitecapBands()
        {
            uint mask = 0;
            bool windSea = false;
            for (int i = 0; i < Bands.Count; i++)
                windSea |= Bands[i].FollowWind;
            for (int i = 0; i < Bands.Count && i < 32; i++)
                if ((windSea && !Bands[i].FollowWind) || Bands[i].LengthScale < MinWhitecapTile) mask |= 1u << i;
            return mask;
        }

        private uint[] BuildLengthScales()
        {
            var scales = new uint[Bands.Count];
            for (int i = 0; i < Bands.Count; i++)
                scales[i] = Bands[i].LengthScale;
            return scales;
        }

        private const float Gravity = 9.81f;
        private static float JonswapAlpha(float fetch, float windSpeed) =>
            0.076f * MathF.Pow(Gravity * DevelopedFetch(fetch, windSpeed) / windSpeed / windSpeed, -0.22f);
        private static float JonswapPeakOmega(float fetch, float windSpeed) =>
            22f * MathF.Pow(windSpeed * DevelopedFetch(fetch, windSpeed) / Gravity / Gravity, -0.33f);

        // JONSWAP's fetch laws hold until the sea is fully developed, at a dimensionless fetch gF/U² of about 22 000,
        // where they meet Pierson-Moskowitz (α 0.0081, peak 0.88 g/U). A longer fetch at a light wind raised a sea the
        // wind cannot hold: 3 m/s over 100 km gave waves 0.5 m high and 25 m long instead of 0.2 m and 8 m.
        private static float DevelopedFetch(float fetch, float windSpeed) =>
            MathF.Min(fetch, 22000f * windSpeed * windSpeed / Gravity);

        // A swell from afar: what reaches the coast of a distant storm's sea is its long waves; the short ones were
        // spent on the way. Its spectrum keeps the frequencies below twice its peak (wavelengths over a quarter of the
        // peak's, deep water: k = ω² / g). Its JONSWAP tail is where the slopes and the crests' squeeze live, so an
        // authored 15 m/s band kept a rough, foaming sea on a calm day whatever the weather.
        private static float SwellCutoff(float fetch, float windSpeed)
        {
            float omega = 2f * JonswapPeakOmega(fetch, windSpeed);
            return omega * omega / Gravity;
        }

        /// <summary>
        /// The two spectrum layers of every band. Authored bands use their own fields over all wavelengths. Bands that
        /// follow the wind carry one layer, the wind sea (the weather's wind speed, its heading turned into the band's
        /// frame, the WindSea settings), over their share of the wavelengths: each wind band from the next finer wind
        /// band's quarter tile up to its own tile, so the bands together are one spectrum, not one per band.
        /// </summary>
        private OceanFFT.SpectrumParameters[] BuildSpectrumParams()
        {
            static float Rad(float deg) => deg / 180f * MathF.PI;

            var spectrums = new OceanFFT.SpectrumParameters[Bands.Count * 2];

            // The weather's wind: never quite calm (JONSWAP divides by the wind speed)
            float windSpeed = MathF.Max(Wind.Strength, 0.5f);
            var dir = Wind.Direction;
            float heading = dir.LengthSquared() > 0f ? MathF.Atan2(dir.Z, dir.X) : 0f;

            // The wind bands' share of the wavelengths, coarsest first
            var windBands = new List<int>();
            for (int i = 0; i < Bands.Count; i++)
                if (Bands[i].FollowWind) windBands.Add(i);
            windBands.Sort((a, b) => Bands[b].LengthScale.CompareTo(Bands[a].LengthScale));

            for (int i = 0; i < Bands.Count; i++)
            {
                var b = Bands[i];

                if (b.FollowWind)
                {
                    // A boundary between two wind bands lies at a quarter of the finer one's tile (wavenumber 8π / tile):
                    // longer waves go to the coarser band, so no fine band repeats them tile after tile
                    int rank = windBands.IndexOf(i);
                    float low = rank > 0 ? 8f * MathF.PI / b.LengthScale : 0f;
                    float high = rank + 1 < windBands.Count
                        ? 8f * MathF.PI / Bands[windBands[rank + 1]].LengthScale
                        : float.MaxValue;

                    var sea = WindSea;
                    spectrums[i * 2] = new OceanFFT.SpectrumParameters
                    {
                        Scale = sea.Scale,
                        Angle = heading + BandAngles[i % BandAngles.Length],
                        SpreadBlend = sea.SpreadBlend,
                        Swell = sea.Swell,
                        Alpha = JonswapAlpha(sea.Fetch, windSpeed),
                        PeakOmega = JonswapPeakOmega(sea.Fetch, windSpeed),
                        Gamma = sea.PeakEnhancement,
                        ShortWavesFade = sea.ShortWavesFade,
                        LowCutoff = low,
                        HighCutoff = high,
                    };
                    spectrums[i * 2 + 1] = new OceanFFT.SpectrumParameters { Scale = 0f, HighCutoff = float.MaxValue };
                    continue;
                }

                // With wind bands for the local sea an authored band is a swell from afar: its long waves only
                bool swell = windBands.Count > 0;
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
                    LowCutoff = 0f,
                    HighCutoff = swell ? SwellCutoff(b.Fetch.X, b.WindSpeed.X) : float.MaxValue,
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
                    LowCutoff = 0f,
                    HighCutoff = swell ? SwellCutoff(b.Fetch.Y, b.WindSpeed.Y) : float.MaxValue,
                };
            }

            return spectrums;
        }

        /// <summary>
        /// Create 4 default bands with logarithmic length scale spread: an authored swell, and three bands the weather's
        /// wind raises (their own wind fields only matter if Follow Wind is turned off).
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
                    FollowWind = true,
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
                    FollowWind = true,
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
                    FollowWind = true,
                },
            };
        }

        public void Update()
        {
            if (_mesh == null || _material == null) return;
            _waveTime += (float)Time.Delta * WaveSpeed;

            // The spectrum's parameters (the bands, the weather's wind): the initial spectrum is rebuilt only when they
            // changed, e.g. while a preset transition turns the wind (until 2026-10-17 it was rebuilt every frame)
            var spectrums = BuildSpectrumParams();
            var lengthScales = BuildLengthScales();
            // (the initial spectrum also depends on the water depth: dispersion and the shallow-water TMA factor)
            if (!MemoryExtensions.SequenceEqual<byte>(MemoryMarshal.AsBytes(spectrums.AsSpan()), MemoryMarshal.AsBytes(_appliedSpectrums.AsSpan()))
                || !lengthScales.AsSpan().SequenceEqual(_appliedLengthScales) || Depth != _appliedDepth)
            {
                _oceanFFT.RequestSpectrumReinit(spectrums, lengthScales);
                _appliedSpectrums = spectrums;
                _appliedLengthScales = lengthScales;
                _appliedDepth = Depth;
            }

            // Push runtime-tweakable params to OceanFFT each frame
            _oceanFFT.UpdateParams(Depth, FoamBias, FoamDecayRate, FoamThreshold, FoamAdd,
                new Vector2(Choppiness, Choppiness), NoWhitecapBands());

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
            _drawn = this;
            _drawnTick = Engine.TickCount;

            // Lazy terrain lookup (terrain may not exist at Awake time)
            if (_terrain == null)
                _terrain = EntityManager.FindComponent<TerrainRenderer>();

            // The first sea mask, as soon as the terrain has heights (at load they can come from its cache with no
            // readback message); later ones follow TerrainHeightsChanged and the sea level
            if (_seaMaskRequest == 0)
                RequestSeaMask();
            if (Interlocked.Exchange(ref _builtSeaMask, null) is { } seaMask)
                _seaMask.Upload(seaMask);

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
                SeaMaskSRV = Engine.Settings.DisableSeaMask ? 0 : _seaMask.Srv,
                DeepColor = new Vector3(DeepColor.R, DeepColor.G, DeepColor.B),
                SunDirection = sunDir,
                SunColor = sunColor,
                SunIntensity = sunIntensity,
                DisplacementSRV = _oceanFFT.DisplacementSRV,
                SlopeSRV = _oceanFFT.SlopeSRV,
                NumBands = (uint)Bands.Count,
                TileScalesSRV = _tileScalesBuffer?.SrvIndex ?? 0,
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
                InvViewportWidth = 1.0f / MathF.Max(1, DeferredRenderer.Current?.DepthGBuffer?.Native.Description.Width ?? 1920),
                InvViewportHeight = 1.0f / MathF.Max(1, DeferredRenderer.Current?.DepthGBuffer?.Native.Description.Height ?? 1080),
                ShoreFoamDepth = ShoreFoamDepth,
                ShoreFoam = ShoreFoam,
                ShoreSurge = ShoreSurge,
                // Built from this frame's opaque depth before the forward pass draws the water; not while
                // frozen (stale) or with Hi-Z switched off
                HiZSRV = DeferredRenderer.Current?.HiZPyramid is { Ready: true } pyramid
                         && !Engine.Settings.FreezeFrustum && !Engine.Settings.DisableHiZ ? pyramid.FullSRV : 0,
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
