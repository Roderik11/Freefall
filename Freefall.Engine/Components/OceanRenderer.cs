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

        [Description(@"How much taller than the wind really raises it the sea stands as the wind rises: the wave height × this
         from 20 m/s of wind, ramping in from 5 m/s (calm seas stay as the wind makes them; a 10 m/s sea stands a third
         taller, enough for white horses). A real storm sea is long and not very steep; 1 = real")]
        [ValueRange(1f, 3f)]
        public float StormWaveScale = 2f;

        /// <summary>The wind sea's height as a multiple of the real one at this wind</summary>
        public float WaveScaleAt(float windSpeed) =>
            1f + (StormWaveScale - 1f) * Math.Clamp((windSpeed - 5f) / 15f, 0f, 1f);
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
        
        [ValueRange(0f, 2f)]
        [Description(@"Horizontal displacement scale (Lambda): squeezes the crests together into sharp peaks and
         stretches the troughs (1 = linear theory). The squeezed crests are where whitecaps form. 1.4 gives the peaked
         crests of a storm sea; much more folds them over")]
        public float Choppiness = 1.4f;
        
        [ValueRange(0f, 5f)]
        [Description(@"Time multiplier for wave animation speed.
         1.0 = real-time")]
        public float WaveSpeed = 1.0f;
        
        [ValueRange(10f, 1000f)]
        [Description(@"Period in seconds before the wave pattern repeats.
         Higher values delay visible looping")]
        public float RepeatTime = 200.0f;

        // Whitecaps: foam where a crest of the whitecap band (WhitecapBand) is squeezed (the Jacobian J of the displacement)
        // under FoamBias − FoamThreshold; the finer bands' crests give it its structure (ocean.fx). The defaults
        // (2026-10-18, tuned by eye against a storm reference with WindSea.StormWaveScale 2): calm and moderate seas stay
        // clean (their crests barely squeeze), a storm's taller crests foam all over and leave trails.
        [Category("Foam")]
        [ValueRange(0f, 2f)]
        [Description(@"Whitecaps form where a crest is squeezed below this Jacobian (minus the threshold).
         Higher = more crests foam, at lighter winds")]
        public float FoamBias = 0.82f;

        [ValueRange(0f, 0.1f)]
        [Description(@"Exponential decay rate per frame.
         Controls how quickly foam fades after forming.
         Lower values make foam linger longer")]
        public float FoamDecayRate = 0.012f;

        [ValueRange(0f, 1f)]
        [Description(@"Minimum Jacobian deviation required before any foam is added.
         Acts as a noise gate to suppress faint foam")]
        public float FoamThreshold = 0.02f;

        [ValueRange(0f, 1f)]
        [Description(@"Amount of foam injected per frame when the Jacobian exceeds the threshold.
         Higher values create thicker, more opaque foam")]
        public float FoamAdd = 0.1f;

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

        [ValueRange(0f, 1f)]
        [Description(@"Amount of foam along the waterline and around objects standing in the water.
         The surf's white water comes from the breaking waves themselves")]
        public float ShoreFoam = 0.8f;

        [ValueRange(0f, 1.5f)]
        [Description(@"How high the swash runs up the beach at most, in meters over the still sea: each wave's bore runs up as far
         as its height carries it (a third of it), up to this. On a flat beach a few decimeters go a long way")]
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

        // The surf (ShoreWarp): baked on the sea mask's worker right after it, and on its own when a warped band's heading
        // or frequency moves (the weather); the newest bake wins, and a band change waits for the running bake to finish
        private readonly ShoreWarp _shoreWarp = new();
        private ShoreWarp.Result? _builtShoreWarp;
        private ShoreWarp.Band[] _warpRequested = Array.Empty<ShoreWarp.Band>();
        private int _warpRequest;
        private int _warpRunning;
        private float[,]? _warpHeights;       // what the last sea mask was flooded from, for a bake of the warp alone
        private SeaMask.Result? _warpMask;

        // This frame's OceanData for the particle kernel (ParticleSystem, shape Surf), one buffer per frame in flight
        private readonly GraphicsBuffer?[] _surfData = new GraphicsBuffer?[3];
        private uint _surfDataSrv;

        /// <summary>A StructuredBuffer of this frame's OceanData (ocean_surf.hlsli), for the surf's spray and mist; 0 = none yet.</summary>
        internal uint SurfDataSrv => _surfDataSrv;

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
            public uint WindBands;       // bit per band carrying the wind sea (FollowWind): its crests glow (was padding)
            public Vector3 SunDirection;
            public float SunIntensity;
            public Vector3 SunColor;
            public uint WhitecapBand;    // the band whose foam marks the whitecaps (WhitecapBand) + 1; 0 = none (was padding)
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
            public float _shorePad;      // was ShoreFoamDepth (the clock-driven swash bands, gone with the surf 2026-10-19)
            public float ShoreFoam;
            public float ShoreSurge;
            public uint HiZSRV;          // the view's Hi-Z pyramid: hull shader skips hidden patches (0 = off)
            public uint ShoreWarpSRV;    // the surf (ShoreWarp): per warped band Δu, amplitude, c/c0; 0 = no warp
            public uint ShoreWarpJacobianSRV;   // its ∂u/∂x
            public uint WarpSlots;       // band + 1 of warp slice 0 (bits 0-7) and slice 1 (bits 8-15); 0 = none
            public float _warpPad;
            public Vector4 WarpDirs;     // the deep heading (world xz) of slice 0 (xy) and slice 1 (zw)
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
            var warp = BeginShoreWarp();
            Task.Run(() =>
            {
                try
                {
                    var mask = SeaMask.Build(heights, seaLevel01, spacing, seaLevel, terrain.MaxHeight);
                    if (Volatile.Read(ref _seaMaskRequest) != request) return;
                    Volatile.Write(ref _builtSeaMask, mask);
                    BakeShoreWarp(warp, heights, mask);
                }
                finally { Interlocked.Decrement(ref _warpRunning); }
            });
        }

        // What a bake of the surf's warp needs from the main thread: the bands and where the terrain is
        private struct ShoreWarpJob
        {
            public int Request;
            public ShoreWarp.Band[] Bands;
            public float SeaLevel01, MaxHeight;
            public Vector2 TerrainSize, Origin;
        }

        private ShoreWarpJob BeginShoreWarp()
        {
            var terrain = _terrain!.Terrain!;
            var origin = _terrain.Transform?.WorldPosition ?? Vector3.Zero;
            var bands = WarpBands();
            _warpRequested = bands;
            Interlocked.Increment(ref _warpRunning);
            return new ShoreWarpJob
            {
                Request = Interlocked.Increment(ref _warpRequest),
                Bands = bands,
                SeaLevel01 = (Transform.WorldPosition.Y - origin.Y) / MathF.Max(terrain.MaxHeight, 0.001f),
                MaxHeight = terrain.MaxHeight,
                TerrainSize = terrain.TerrainSize,
                Origin = new Vector2(origin.X, origin.Z),
            };
        }

        // On a worker. The warp's own heights and mask are kept for the next bake of the warp alone
        private void BakeShoreWarp(ShoreWarpJob job, float[,] heights, SeaMask.Result mask)
        {
            if (Volatile.Read(ref _warpRequest) != job.Request) return;
            var warp = ShoreWarp.Build(heights, mask, job.SeaLevel01, job.MaxHeight, job.TerrainSize, job.Origin, job.Bands);
            if (Volatile.Read(ref _warpRequest) != job.Request) return;
            Volatile.Write(ref _warpHeights, heights);
            Volatile.Write(ref _warpMask, mask);
            Volatile.Write(ref _builtShoreWarp, warp);
        }

        // A warped band's heading or frequency moved (a preset turning the wind): bake the warp again on its own, from the
        // heights and mask of the last bake, once no bake is running
        private void UpdateShoreWarpBands()
        {
            var heights = Volatile.Read(ref _warpHeights);
            var mask = Volatile.Read(ref _warpMask);
            if (heights == null || mask == null || _terrain?.Terrain == null || Volatile.Read(ref _warpRunning) > 0) return;

            var bands = WarpBands();
            bool same = bands.Length == _warpRequested.Length;
            for (int i = 0; same && i < bands.Length; i++)
                same = bands[i].SameAs(_warpRequested[i]);
            if (same) return;

            var job = BeginShoreWarp();
            Task.Run(() =>
            {
                try { BakeShoreWarp(job, heights, mask); }
                finally { Interlocked.Decrement(ref _warpRunning); }
            });
        }

        /// <summary>
        /// The bands the surf carries in from the open sea (ShoreWarp), at most two: a swell from afar (the first authored
        /// band, when wind bands make the local sea; without them the coarsest band) and the wind sea's coarsest band,
        /// which holds its long waves. Each with the world heading its waves travel along and the spectrum's peak.
        /// </summary>
        private ShoreWarp.Band[] WarpBands()
        {
            static float Rad(float deg) => deg / 180f * MathF.PI;
            static Vector2 Heading(float angle) => new(MathF.Cos(angle), MathF.Sin(angle));

            var bands = new List<ShoreWarp.Band>(ShoreWarp.MaxBands);
            int coarsestWind = -1, finerWind = -1;
            for (int i = 0; i < Bands.Count; i++)
                if (Bands[i].FollowWind && (coarsestWind < 0 || Bands[i].LengthScale > Bands[coarsestWind].LengthScale))
                    coarsestWind = i;
            for (int i = 0; i < Bands.Count; i++)
                if (Bands[i].FollowWind && i != coarsestWind && (finerWind < 0 || Bands[i].LengthScale > Bands[finerWind].LengthScale))
                    finerWind = i;

            // The swell: its layer's angle is in the band's turned frame (BandAngles), the world heading that much less
            int swell = -1;
            for (int i = 0; i < Bands.Count && swell < 0; i++)
                if (!Bands[i].FollowWind && Bands[i].Scale.X > 0f && (coarsestWind >= 0 || i == CoarsestBand()))
                    swell = i;
            if (swell >= 0)
            {
                var b = Bands[swell];
                bands.Add(new ShoreWarp.Band
                {
                    Index = swell,
                    Direction = Heading(Rad(b.WindDirection.X) - BandAngles[swell % BandAngles.Length]),
                    Omega = JonswapPeakOmega(b.Fetch.X, b.WindSpeed.X),
                });
            }

            // The wind sea: its peak, kept to the wavelengths its coarsest band holds (from a quarter of the next finer
            // wind band's tile up to its own tile)
            if (coarsestWind >= 0)
            {
                var dir = Wind.Direction;
                float shortest = finerWind >= 0 ? Bands[finerWind].LengthScale / 4f : Bands[coarsestWind].LengthScale / 16f;
                float longest = Bands[coarsestWind].LengthScale;
                float omega = JonswapPeakOmega(WindSea.Fetch, MathF.Max(Wind.Strength, 0.5f));
                omega = Math.Clamp(omega, MathF.Sqrt(2f * MathF.PI * Gravity / longest), MathF.Sqrt(2f * MathF.PI * Gravity / shortest));
                bands.Add(new ShoreWarp.Band
                {
                    Index = coarsestWind,
                    Direction = dir.LengthSquared() > 0f ? Vector2.Normalize(new Vector2(dir.X, dir.Z)) : Vector2.UnitX,
                    Omega = omega,
                });
            }
            return bands.ToArray();
        }

        private int CoarsestBand()
        {
            int c = -1;
            for (int i = 0; i < Bands.Count; i++)
                if (c < 0 || Bands[i].LengthScale > Bands[c].LengthScale) c = i;
            return c;
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(EngineMsg.TerrainHeightsChanged, OnTerrainHeightsChanged);
            Transform?.OnChanged -= OnTransformChanged;
            Interlocked.Increment(ref _seaMaskRequest);   // a build still running is dropped
            Interlocked.Increment(ref _warpRequest);
            _seaMask.Dispose();
            _shoreWarp.Dispose();

            // Deferred: the frames in flight still draw the grid, read the tile scales and the FFT textures
            // through their bindless slots and run the FFT kernels. OceanFFT.Dispose releases only GPU objects
            // (textures, buffers, compute pipeline states), so the whole simulation waits with them.
            var device = Engine.Device;
            device.DeferDispose(_tileScalesBuffer);
            foreach (var b in _surfData) device.DeferDispose(b);
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

        // The band whose foam says where the whitecaps are: the wind band that holds the wind sea's peak, the dominant waves
        // one sees break. They shorten as the wind drops, so the calmer the sea the finer that band (the user's idea,
        // 2026-10-18): on Eldermoor the 250 m band from ~4 m/s up (peaks 16-100 m), the 64 m band below. The finer wind
        // bands give the whitecaps their structure (ocean.fx); on their own their crests foamed where no crest is seen:
        // random blobs. -1 without wind bands.
        private int WhitecapBand()
        {
            var windBands = new List<int>();
            for (int i = 0; i < Bands.Count; i++)
                if (Bands[i].FollowWind) windBands.Add(i);
            if (windBands.Count == 0) return -1;
            windBands.Sort((a, b) => Bands[b].LengthScale.CompareTo(Bands[a].LengthScale));

            // Deep-water wavelength of the spectrum's peak; each wind band holds from a quarter of the next finer one's
            // tile up to a quarter of its own (the coarsest: everything longer), as BuildSpectrumParams shares them out
            float omega = JonswapPeakOmega(WindSea.Fetch, MathF.Max(Wind.Strength, 0.5f));
            float peak = 2f * MathF.PI * Gravity / (omega * omega);
            for (int r = windBands.Count - 1; r > 0; r--)
                if (peak < Bands[windBands[r]].LengthScale / 4f)
                    return windBands[r];
            return windBands[0];
        }

        // Bands that make no whitecaps of their own: with wind bands all but the whitecap band (WhitecapBand); a swell
        // from afar does not break in deep water. Without wind bands (authored seas) every band foams, as in the
        // reference implementation.
        private uint NoWhitecapBands()
        {
            int capBand = WhitecapBand();
            if (capBand < 0) return 0;
            uint mask = 0;
            for (int i = 0; i < Bands.Count && i < 32; i++)
                if (i != capBand) mask |= 1u << i;
            return mask;
        }


        private uint WindBandMask()
        {
            uint mask = 0;
            for (int i = 0; i < Bands.Count && i < 32; i++)
                if (Bands[i].FollowWind) mask |= 1u << i;
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
            float stormScale = WindSea.WaveScaleAt(windSpeed);

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
                        // Storms stand taller than physics (WindSea.StormWaveScale): wave height × that, energy × its square
                        Scale = sea.Scale * stormScale * stormScale,
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
            if (Interlocked.Exchange(ref _builtShoreWarp, null) is { } shoreWarp)
                _shoreWarp.Upload(shoreWarp);
            UpdateShoreWarpBands();

            // The uploaded warp's slices: which band each carries and its deep heading
            var warpBands = _shoreWarp.Bands;
            uint warpSlots = 0;
            var warpDirs = Vector4.Zero;
            for (int s = 0; s < warpBands.Length && s < ShoreWarp.MaxBands; s++)
            {
                warpSlots |= (uint)(warpBands[s].Index + 1) << (8 * s);
                if (s == 0) { warpDirs.X = warpBands[s].Direction.X; warpDirs.Y = warpBands[s].Direction.Y; }
                else { warpDirs.Z = warpBands[s].Direction.X; warpDirs.W = warpBands[s].Direction.Y; }
            }

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

            var data = new OceanData
            {
                WaveTime = _waveTime,
                Choppiness = Choppiness,
                FoamDecayRate = FoamDecayRate,
                FoamThreshold = FoamThreshold,
                OceanColor = new Vector3(OceanColor.R, OceanColor.G, OceanColor.B),
                SeaMaskSRV = Engine.Settings.DisableSeaMask ? 0 : _seaMask.Srv,
                DeepColor = new Vector3(DeepColor.R, DeepColor.G, DeepColor.B),
                WindBands = WindBandMask(),
                WhitecapBand = (uint)(WhitecapBand() + 1),
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
                ShoreFoam = ShoreFoam,
                ShoreSurge = ShoreSurge,
                // Built from this frame's opaque depth before the forward pass draws the water; not while
                // frozen (stale) or with Hi-Z switched off
                HiZSRV = DeferredRenderer.Current?.HiZPyramid is { Ready: true } pyramid
                         && !Engine.Settings.FreezeFrustum && !Engine.Settings.DisableHiZ ? pyramid.FullSRV : 0,
                ShoreWarpSRV = warpSlots != 0 && !Engine.Settings.DisableShoreWarp ? _shoreWarp.WarpSrv : 0,
                ShoreWarpJacobianSRV = _shoreWarp.JacobianSrv,
                WarpSlots = warpSlots,
                WarpDirs = warpDirs,
            };
            _params.SetParameter("OceanData", data);

            // The same data for the particle kernel (ParticleEmitter shape Surf: spray and mist where the surf breaks), in
            // a buffer of its own per frame in flight
            int frame = Engine.FrameIndex % _surfData.Length;
            _surfData[frame] ??= GraphicsBuffer.CreateUpload<OceanData>(1, mapped: true);
            unsafe { *_surfData[frame]!.WritePtr<OceanData>() = data; }
            _surfDataSrv = _surfData[frame]!.SrvIndex;

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
