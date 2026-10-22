using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>What the clouds are this frame. <see cref="Components.SkyboxRenderer"/> sets it from its fields (an
    /// EnvironmentPreset drives them); without a sky there are none.</summary>
    public struct CloudSettings
    {
        public bool Enabled;
        public float Coverage;          // 0..1
        public float Type;              // 0 stratus … 0.5 stratocumulus … 1 cumulus
        public float Base;              // m above sea level
        public float Thickness;         // m: the ordinary layer's depth
        public float TowerShare;        // 0..1: the share of storm cells that tower
        public float TowerTop;          // m above sea level: the tallest tower
        public float Density;           // extinction per metre at full density
        public float Albedo;            // single-scattering albedo
        public float Speed;             // CloudSpeed: the drift (~32 m/s × it) and the clouds' own change
        public float LightningRate;     // flashes a minute in the raining cells near the camera (0 = none)
        public float FreezingLevel;     // m above sea level: the 0 °C level (Weather: what precipitates, and as what)

        public static CloudSettings Default => new()
        {
            Enabled = false, Coverage = 0.5f, Type = 1f, Base = 1800f, Thickness = 1000f, TowerShare = 0f,
            TowerTop = 6000f, Density = 0.05f, Albedo = 0.99f, Speed = 1f, LightningRate = 0f, FreezingLevel = 2500f,
        };
    }

    /// <summary>
    /// The volumetric clouds' field (VOLUMETRIC_CLOUDS_PLAN.md): the weather, the storm cells and the cloud shapes baked
    /// into a clipmap of 3D textures that live in cloud space (the world's xz minus the wind's offset), so the clouds are
    /// modelled once and the wind carries them for free. Five levels of 256² texels, 16 to 256 km wide, centred on the
    /// camera and addressed modulo 256: a camera move bakes only the brick columns that enter, and every frame one more
    /// column of each level is baked again for the clouds' own change (a level a ~32 frames). The weather repeats every
    /// <see cref="Period"/> in cloud space (its noise tiles and cells), a whole number of every level's texture, so the
    /// offset's wrap changes nothing. Each view marches it (<see cref="CloudView"/>, clouds_cs.hlsl). Shaders find it
    /// through SceneConstants.CloudParamsIdx (cloud_common.hlsli), on while SceneConstants.CloudsOn is 1. Setting:
    /// Engine.Settings.VolumetricClouds.
    /// Docs: .agent/knowledge/atmospheric_effects/volumetric_clouds.md.
    /// </summary>
    public static class Clouds
    {
        public const int LevelCount = 5;
        public const int TexelsPerLevel = 256;          // cloud_common.hlsli CLOUD_TEXELS
        public const int Brick = 8;                     // CLOUD_BRICK
        public const double Period = 256000.0;          // CLOUD_PERIOD
        public const int CellsPerSide = 20;             // storm cells over the period (12.8 km each)
        private const float Level0Voxel = 62.5f;
        private const float ReachTexels = 100f;         // a level serves the march out to this many of its texels
        private const float DriftSpeed = 32f;           // m/s at CloudSpeed 1, as the 2D layer drifted
        private const float BaseDip = 450f;             // clouds_cs.hlsl CLOUD_BASE_DIP
        private static readonly int[] Layers = { 128, 64, 64, 64, 64 };
        private const int FrameCount = 3;
        private const int BricksPerSide = TexelsPerLevel / Brick;

        public static CloudSettings Settings = CloudSettings.Default;

        /// <summary>The clouds this frame: on in the settings and given by a sky.</summary>
        public static bool Active => Engine.Settings.VolumetricClouds && Settings.Enabled;

        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct GpuLevel
        {
            public Vector2 Origin; public float VoxelSize; public uint Layers;
            public uint DensityTex, BrickTex, LightTex, SkyTex;
            public float Reach, _pad0, _pad1, _pad2;
        }

        // Must match CloudParams in cloud_common.hlsli, field for field
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct GpuParams
        {
            public GpuLevel L0, L1, L2, L3, L4;
            public Vector2 CameraCloud; public float SlabBottom, SlabTop;
            public float Base, Thickness, Coverage, Type;
            public float TowerShare, TowerTop, Density, Albedo;
            public Vector2 WindDir; public float Evolution; public uint NoiseLut;
            public uint Cells; public float CellSize; public uint CellsPerSide; public float Reach;
            public Vector3 LightDir; public uint LightIsMoon;
            public uint ProbeClouds; public uint RainShafts, RainCount; public float FlashIntensity;
            public Vector3 FlashPos; public float FlashSpread;
            public uint ShadowMap; public float ShadowTexel; public Vector2 ShadowCentre;
            public float FreezingLevel, _pad0, _pad1, _pad2;
        }

        // CloudRainShaft in cloud_common.hlsli
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct GpuRainShaft
        {
            public Vector2 Center; public float Radius, Rain, Bottom, Seed, Fade, _pad;
        }

        // CloudCell in cloud_common.hlsli
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct GpuCell
        {
            public Vector2 Center; public float Radius, Top, Anvil, Rain, Growth, _pad;
        }

        private sealed class Level
        {
            public RenderTexture3D Density = null!, Bricks = null!;
            public ResourceStates State = ResourceStates.Common;
            // The light volume (A, the one read; B the doubling's other half) and the sky volume: half the density's
            // texels each way
            public RenderTexture3D LightA = null!, LightB = null!, Sky = null!;
            public ResourceStates LightAState = ResourceStates.Common, LightBState = ResourceStates.Common,
                SkyState = ResourceStates.Common;
            public float Voxel;
            public int Layers;
            public long CenterX, CenterZ;       // the brick (in cloud space) the camera is in
            public bool Valid;
            public int Rolling;                 // the next brick column baked again
        }

        private readonly record struct Bake(int Level, long X0, long Z0, int BricksX, int BricksZ);

        private static ComputeShader? _cs;
        private static int _kBake, _kLightInit, _kLightStep, _kLightChain, _kSkyScan, _kProbeFace, _kShadowMap;

        // The ground shadow map (clouds_cs.hlsl CSCloudShadowMap, VOLUMETRIC_CLOUDS_PLAN §8): the optical depth toward the
        // light from where a ray toward it crosses the slab's bottom, 1024² over 32 km of cloud space round the camera's
        // own crossing, toroidal. A low light crosses the slab kilometres away, where the light volume's texels are
        // 0.25-1 km: its shadows came out as blocks that flipped as the levels followed the camera.
        public const int ShadowMapTexels = 1024;                // cloud_common.hlsli CLOUD_SHADOW_TEXELS
        private const float ShadowMapTexel = 31.25f;            // m: 32 km across
        private const float EarthRadius = 6360000f;             // cloud_common.hlsli CLOUD_EARTH_RADIUS
        // All of it every 16 frames: ~0.075 ms at 64 rows (0.15 ms at 128); the map is in cloud space, so only the clouds'
        // own slow change and a moving light age it
        private const int ShadowRowsPerFrame = ShadowMapTexels / 16;
        private static RenderTexture2D? _shadowMap;
        private static ResourceStates _shadowMapState = ResourceStates.Common;
        private static int _shadowRow;
        private static bool _shadowAll = true;                  // every row this frame (a rebake, the light jumped, a teleport)
        private static Vector3 _shadowLightDir;                 // last frame's
        private static Vector2 _shadowCentre;                   // cloud space, snapped to texels
        private static int _shadowRowsLast;

        // The sky probe's clouds: 64² × 6 (SkyProbe's size), a face a frame round the five that see the sky
        private const int ProbeSize = 64;
        private static ID3D12Resource? _probeClouds;
        private static uint _probeCloudsSrv, _probeCloudsUav;
        private static ResourceStates _probeCloudsState = ResourceStates.Common;
        private static int _probeTurn;
        private static bool _probeAll = true;
        private static readonly int[] ProbeFaces = { 0, 1, 2, 4, 5 };     // +X −X +Y +Z −Z (−Y sees the ground)
        private static readonly int[] ProbeFacesAbove = { 0, 1, 2, 3, 4, 5 };   // over the clouds' base: −Y sees them too
        private static bool _cameraOverBase;      // the camera above the slab's bottom: clouds can lie below it

        /// <summary>How far the wind carried the clouds this frame (world xz, m): the views' reprojection.</summary>
        public static Vector2 LastStep { get; private set; }
        private const int LightPasses = 8;              // doubling passes: 256 steps along the light cross a level
        private static readonly List<int> _lightLevels = new();     // levels whose light and sky volumes are made this frame
        private static int _lightTurn;
        private static Vector3 _lightDir = Vector3.UnitY;
        private static bool _lightIsMoon;
        private static readonly Level[] _levels = new Level[LevelCount];
        private static readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];
        private static GraphicsBuffer? _offParams;      // an empty frame for CloudParamsIdx while the clouds are off
        private static readonly GraphicsBuffer?[] _cells = new GraphicsBuffer?[FrameCount];
        private static readonly GpuCell[] _cellData = new GpuCell[CellsPerSide * CellsPerSide];
        private const int MaxRainShafts = 16;
        private const float RainShaftReach = 80000f;   // m: raining cells farther than this are not marched
        private static readonly GraphicsBuffer?[] _rainShafts = new GraphicsBuffer?[FrameCount];
        private static readonly GpuRainShaft[] _shaftData = new GpuRainShaft[MaxRainShafts];
        private static readonly float[] _shaftDistance = new float[MaxRainShafts];
        private static int _shaftCount;

        // Lightning: one flash at a time in a raining cell, a few strokes over 0.05-0.3 s
        private const float FlashPeakCandela = 5e9f;   // a stroke's luminous intensity (lightning: ~1e9-1e10 cd)
        private static readonly Random _flashRandom = new(20261019);
        private static double _flashClock = 1e9;       // seconds since the current flash began
        private static float _flashLength, _flashPeak, _nextFlashIn = 5f, _lastDt;
        private static int _flashStrokes;
        private static bool _flashRequested;
        private static double _describeCx, _describeCz;     // this frame's camera in cloud space and flash, for Describe
        private static Vector3 _describeFlash;
        private static Vector3 _flashCloud;            // cloud-space x, height, z
        private static readonly List<Bake> _bakes = new();
        private static readonly GpuLevel[] _gpuLevels = new GpuLevel[LevelCount];
        private static int _preparedTick = -1, _renderedTick = -1;
        private static uint _paramsIdx;
        private static float _slabBottom = float.NaN, _slabTop = float.NaN;
        private static double _lastHours = double.NaN;
        private static double _offsetX, _offsetZ;       // the wind's offset (m), wrapped by Period
        private static double _evolution;               // seconds × CloudSpeed
        private static double _gameHours;
        private static int _voxelsBaked;

        internal static ComputeShader Shader
        {
            get { EnsureShader(); return _cs!; }
        }

        private static void EnsureShader()
        {
            if (_cs != null) return;
            _cs = new ComputeShader("clouds_cs.hlsl");
            _kBake = _cs.FindKernel("CSCloudBake");
            _kLightInit = _cs.FindKernel("CSCloudLightInit");
            _kLightStep = _cs.FindKernel("CSCloudLightStep");
            _kLightChain = _cs.FindKernel("CSCloudLightChain");
            _kSkyScan = _cs.FindKernel("CSCloudSkyScan");
            _kProbeFace = _cs.FindKernel("CSCloudProbeFace");
            _kShadowMap = _cs.FindKernel("CSCloudShadowMap");
        }

        private static void EnsureResources()
        {
            EnsureShader();
            if (_levels[0] != null) return;
            var device = Engine.Device;
            for (int i = 0; i < LevelCount; i++)
            {
                _levels[i] = new Level
                {
                    Voxel = Level0Voxel * (1 << i),
                    Layers = Layers[i],
                    // Density and profile, 8 bits each: the field before the detail, smooth over a texel
                    Density = new RenderTexture3D(device, TexelsPerLevel, TexelsPerLevel, Layers[i], Format.R8G8_UNorm),
                    Bricks = new RenderTexture3D(device, BricksPerSide, BricksPerSide, Layers[i] / Brick, Format.R8_UNorm),
                    // Optical depths: tens to thousands through a storm, half floats hold them
                    LightA = new RenderTexture3D(device, TexelsPerLevel / 2, TexelsPerLevel / 2, Layers[i] / 2, Format.R16_Float),
                    LightB = new RenderTexture3D(device, TexelsPerLevel / 2, TexelsPerLevel / 2, Layers[i] / 2, Format.R16_Float),
                    Sky = new RenderTexture3D(device, TexelsPerLevel / 2, TexelsPerLevel / 2, Layers[i] / 2, Format.R16G16_Float),
                };
            }
            for (int i = 0; i < FrameCount; i++)
            {
                _params[i] = GraphicsBuffer.CreateUpload<GpuParams>(1, mapped: true);
                _cells[i] = GraphicsBuffer.CreateUpload<GpuCell>(_cellData.Length, mapped: true);
                _rainShafts[i] = GraphicsBuffer.CreateUpload<GpuRainShaft>(MaxRainShafts, mapped: true);
            }

            // Optical depths, half floats as the light volumes
            _shadowMap = new RenderTexture2D(device, ShadowMapTexels, ShadowMapTexels, Format.R16_Float, randomWrite: true);

            // The probe's clouds: pre-exposed light and a transmittance, as the probe holds light
            const Format probeFormat = Format.R16G16B16A16_Float;
            _probeClouds = device.CreateTexture2D(probeFormat, ProbeSize, ProbeSize, 6, 1, ResourceFlags.AllowUnorderedAccess, ResourceStates.Common);
            _probeCloudsUav = device.AllocateBindlessIndex();
            device.NativeDevice.CreateUnorderedAccessView(_probeClouds, null, new UnorderedAccessViewDescription
            {
                Format = probeFormat,
                ViewDimension = UnorderedAccessViewDimension.Texture2DArray,
                Texture2DArray = new Texture2DArrayUnorderedAccessView { MipSlice = 0, FirstArraySlice = 0, ArraySize = 6 }
            }, device.GetCpuHandle(_probeCloudsUav));
            _probeCloudsSrv = device.AllocateBindlessIndex();
            device.NativeDevice.CreateShaderResourceView(_probeClouds, new ShaderResourceViewDescription
            {
                Format = probeFormat,
                ViewDimension = ShaderResourceViewDimension.Texture2DArray,
                Shader4ComponentMapping = ShaderComponentMapping.Default,
                Texture2DArray = new Texture2DArrayShaderResourceView { MostDetailedMip = 0, MipLevels = 1, FirstArraySlice = 0, ArraySize = 6 }
            }, device.GetCpuHandle(_probeCloudsSrv));
        }

        /// <summary>
        /// Once per frame (SkyboxRenderer.Update): the wind carries the clouds (cloud space moves with it) and they change
        /// on their own. <paramref name="windDirection"/> is where the wind blows (xz, unit); <paramref name="gameHours"/>
        /// (Day × 24 + TimeOfDay) drives the storm cells' lives.
        /// </summary>
        public static void Advance(float dt, Vector2 windDirection, double gameHours)
        {
            double speed = DriftSpeed * Settings.Speed;
            LastStep = new Vector2((float)(windDirection.X * speed * dt), (float)(windDirection.Y * speed * dt));
            _offsetX = Wrap(_offsetX + windDirection.X * speed * dt);
            _offsetZ = Wrap(_offsetZ + windDirection.Y * speed * dt);
            _evolution += dt * Settings.Speed;
            if (_evolution > 1e6) _evolution -= 1e6;
            _gameHours = gameHours;
            _lastDt = dt;
        }

        private static double Wrap(double v) => v - Period * Math.Floor(v / Period);

        /// <summary>
        /// This frame's parameters (once per frame) and the bindless index of their buffer for SceneConstants.CloudParamsIdx
        /// (SceneConstants.CloudsOn = <see cref="Active"/>): the levels follow the camera, the brick columns to bake are
        /// chosen (<see cref="Render"/>), the storm cells are placed. Without clouds, an empty frame's buffer, never 0:
        /// shaders test CloudsOn, but the driver may load ahead of that test, and a structured load at descriptor 0 (the
        /// first resource the bindless allocator handed out, a texture) hung the GPU.
        /// </summary>
        public static uint PrepareFrame(Vector3 cameraPos)
        {
            if (!Active)
            {
                _paramsIdx = 0;
                _shaftCount = 0;
                Weather.Update(cameraPos, cloudsActive: false);
                if (_offParams == null)
                {
                    _offParams = GraphicsBuffer.CreateUpload<GpuParams>(1, mapped: true);
                    unsafe { *_offParams.WritePtr<GpuParams>() = default; }
                }
                return _offParams.SrvIndex;
            }
            if (_preparedTick == Engine.TickCount)
                return _paramsIdx;
            _preparedTick = Engine.TickCount;
            EnsureResources();
            int slot = Engine.FrameIndex % FrameCount;
            var s = Settings;

            // The heights the volumes span: the layer and the towers, snapped so a preset blend rebakes rarely
            // (a storm's rough base hangs up to BaseDip lower in places: clouds_cs.hlsl CloudBaseRough, CLOUD_BASE_DIP)
            float top = MathF.Max(s.Base + s.Thickness, s.TowerShare > 0f ? s.TowerTop * 1.12f : 0f);
            // and a tower's ragged base up to 250 m lower: clouds_cs.hlsl CloudTowerEnvelope)
            float rough = Math.Clamp((s.Coverage - 0.6f) / 0.25f, 0f, 1f) * Math.Clamp((s.Type - 0.4f) / 0.3f, 0f, 1f);
            float dip = MathF.Max(BaseDip * rough, s.TowerShare > 0f ? 250f : 0f);
            float slabBottom = MathF.Floor(MathF.Max(s.Base - dip, 0f) / 250f) * 250f;
            float slabTop = MathF.Max(MathF.Ceiling(top / 250f) * 250f, slabBottom + 250f);
            bool rebuild = slabBottom != _slabBottom || slabTop != _slabTop
                || double.IsNaN(_lastHours) || Math.Abs(_gameHours - _lastHours) > 0.25;
            _slabBottom = slabBottom;
            bool overBase = cameraPos.Y > slabBottom;
            if (overBase != _cameraOverBase)
                _probeAll = true;                   // −Y joins or leaves the round: march it once more either way
            _cameraOverBase = overBase;
            _slabTop = slabTop;
            _lastHours = _gameHours;

            // The light the volumes are made along: the sun until it is ~3.4° under the horizon (a 12 km anvil still
            // sees it), then the brightest moon
            var atm = Atmosphere.Settings;
            bool moon = atm.SunDirection.Y < -0.06f && atm.MoonIlluminance != Vector3.Zero;
            var lightDir = Vector3.Normalize(moon ? atm.MoonDirection : atm.SunDirection);
            bool relight = rebuild || moon != _lightIsMoon;
            if (relight) _probeAll = true;
            _lightIsMoon = moon;
            _lightDir = lightDir;
            // A level's light and sky volumes a frame, coarse to fine (each adds the coarser level's light where its path
            // leaves it); all of them when everything is baked again or the light changed source
            _lightLevels.Clear();
            if (relight)
            {
                for (int i = LevelCount - 1; i >= 0; i--) _lightLevels.Add(i);
            }
            else
            {
                _lightLevels.Add(LevelCount - 1 - _lightTurn);
                _lightTurn = (_lightTurn + 1) % LevelCount;
            }

            // The camera in cloud space
            double cx = Wrap(cameraPos.X - _offsetX), cz = Wrap(cameraPos.Z - _offsetZ);
            _bakes.Clear();
            var gpuLevels = _gpuLevels;
            for (int i = 0; i < LevelCount; i++)
            {
                var level = _levels[i];
                double brick = level.Voxel * Brick;
                long bricksPerPeriod = (long)Math.Round(Period / brick);
                long centerX = (long)Math.Floor(cx / brick), centerZ = (long)Math.Floor(cz / brick);
                long originX = centerX - BricksPerSide / 2, originZ = centerZ - BricksPerSide / 2;
                long dx = WrapDelta(centerX - level.CenterX, bricksPerPeriod);
                long dz = WrapDelta(centerZ - level.CenterZ, bricksPerPeriod);

                if (rebuild || !level.Valid || Math.Abs(dx) >= BricksPerSide || Math.Abs(dz) >= BricksPerSide)
                {
                    _bakes.Add(new Bake(i, originX, originZ, BricksPerSide, BricksPerSide));
                }
                else
                {
                    // The columns that entered the level as the camera moved
                    if (dx > 0) _bakes.Add(new Bake(i, originX + BricksPerSide - dx, originZ, (int)dx, BricksPerSide));
                    if (dx < 0) _bakes.Add(new Bake(i, originX, originZ, (int)-dx, BricksPerSide));
                    if (dz > 0) _bakes.Add(new Bake(i, originX, originZ + BricksPerSide - dz, BricksPerSide, (int)dz));
                    if (dz < 0) _bakes.Add(new Bake(i, originX, originZ, BricksPerSide, (int)-dz));
                    // ... and one column again for the clouds' own change
                    _bakes.Add(new Bake(i, originX + level.Rolling, originZ, 1, BricksPerSide));
                    level.Rolling = (level.Rolling + 1) % BricksPerSide;
                }
                level.CenterX = centerX;
                level.CenterZ = centerZ;
                level.Valid = true;

                gpuLevels[i] = new GpuLevel
                {
                    Origin = new Vector2((float)(originX * brick), (float)(originZ * brick)),
                    VoxelSize = level.Voxel,
                    Layers = (uint)level.Layers,
                    DensityTex = level.Density.BindlessIndex,
                    BrickTex = level.Bricks.BindlessIndex,
                    LightTex = level.LightA.BindlessIndex,
                    SkyTex = level.Sky.BindlessIndex,
                    Reach = level.Voxel * ReachTexels,
                };
            }

            PlaceCells(s, slot);
            PlaceRainShafts(s, slot, cx, cz);
            UpdateFlash(s, cameraPos, cx, cz);
            Weather.Update(cameraPos, cloudsActive: true);
            _describeCx = cx;
            _describeCz = cz;
            _describeFlash = FlashRelative(cameraPos, cx, cz);

            // The ground shadow map's window: centred on where the camera's ray toward the light crosses the slab's bottom
            // (cloud_common.hlsli CloudShadowDepth's formula), snapped to texels. Every row again when the field was baked
            // again, the light changed source or jumped over 0.1° in a frame (a time-of-day drag; the day's own slow turn
            // is followed by the rolling rows), or the window jumped (a teleport); otherwise a share of the rows a frame,
            // round and round (the clouds change slowly; the window moves with them in cloud space)
            float below = MathF.Max(slabBottom - cameraPos.Y, 0f);
            float ly = MathF.Max(lightDir.Y, 0.02f);
            float cross = below > 0f ? MathF.Min(2f * below / (ly + MathF.Sqrt(ly * ly + 2f * below / EarthRadius)), 100000f) : 0f;
            var centre = new Vector2(
                (float)(Math.Round(Wrap(cx + lightDir.X * cross) / ShadowMapTexel) * ShadowMapTexel),
                (float)(Math.Round(Wrap(cz + lightDir.Z * cross) / ShadowMapTexel) * ShadowMapTexel));
            double jumpX = WrapDelta(centre.X - _shadowCentre.X), jumpZ = WrapDelta(centre.Y - _shadowCentre.Y);
            if (relight || Vector3.Dot(lightDir, _shadowLightDir) < MathF.Cos(0.1f * MathF.PI / 180f)
                || jumpX * jumpX + jumpZ * jumpZ > 4000.0 * 4000.0)
                _shadowAll = true;
            _shadowLightDir = lightDir;
            _shadowCentre = centre;

            var wind = Wind.GeneratedDirection;
            var p = new GpuParams
            {
                L0 = gpuLevels[0], L1 = gpuLevels[1], L2 = gpuLevels[2], L3 = gpuLevels[3], L4 = gpuLevels[4],
                CameraCloud = new Vector2((float)cx, (float)cz),
                SlabBottom = slabBottom,
                SlabTop = slabTop,
                Base = s.Base,
                Thickness = MathF.Max(s.Thickness, 1f),
                Coverage = Math.Clamp(s.Coverage, 0f, 1f),
                Type = Math.Clamp(s.Type, 0f, 1f),
                TowerShare = Math.Clamp(s.TowerShare, 0f, 1f),
                TowerTop = MathF.Max(s.TowerTop, s.Base + s.Thickness),
                Density = MathF.Max(s.Density, 0f),
                Albedo = Math.Clamp(s.Albedo, 0f, 1f),
                WindDir = wind,
                Evolution = (float)_evolution,
                NoiseLut = Components.SkyboxRenderer.CurrentCloudNoiseLUTIdx,
                Cells = _cells[slot]!.SrvIndex,
                CellSize = (float)(Period / CellsPerSide),
                CellsPerSide = CellsPerSide,
                Reach = Level0Voxel * (1 << (LevelCount - 1)) * ReachTexels,
                LightDir = lightDir,
                LightIsMoon = moon ? 1u : 0u,
                ProbeClouds = _probeCloudsState != ResourceStates.Common ? _probeCloudsSrv : 0u,
                RainShafts = _rainShafts[slot]!.SrvIndex,
                RainCount = (uint)_shaftCount,
                FlashIntensity = FlashIntensityNow(),
                FlashPos = FlashRelative(cameraPos, cx, cz),
                // The light spreads through the cloud's mean extinction round the flash, a tenth of a full cloud's: at
                // 0.4 only the core within ~1 km lit, where a flash lights a whole storm cell
                FlashSpread = MathF.Max(s.Density, 0f) * 0.1f,
                ShadowMap = _shadowMap!.BindlessIndex,
                ShadowTexel = ShadowMapTexel,
                ShadowCentre = centre,
                FreezingLevel = s.FreezingLevel,
            };
            unsafe { *_params[slot]!.WritePtr<GpuParams>() = p; }
            _paramsIdx = _params[slot]!.SrvIndex;
            return _paramsIdx;
        }

        private static long WrapDelta(long d, long period)
        {
            d %= period;
            if (d >= period / 2) d -= period;
            if (d < -period / 2) d += period;
            return d;
        }

        /// <summary>
        /// The storm cells over the period, from a seed and the game time: a share of them (<see cref="CloudSettings.TowerShare"/>)
        /// towers. Each lives 1.5–3 game hours: it grows to its top over the first third, a tall one spreads its anvil and
        /// rains through the middle, then it shrinks away and a new one starts in its place.
        /// </summary>
        private static void PlaceCells(in CloudSettings s, int slot)
        {
            float cellSize = (float)(Period / CellsPerSide);
            float layerTop = s.Base + s.Thickness;
            for (int j = 0; j < CellsPerSide; j++)
            {
                for (int i = 0; i < CellsPerSide; i++)
                {
                    uint h = Hash((uint)i, (uint)j);
                    float r0 = Rand(ref h), r1 = Rand(ref h), r2 = Rand(ref h), r3 = Rand(ref h);
                    float r4 = Rand(ref h), r5 = Rand(ref h), r6 = Rand(ref h);
                    ref var cell = ref _cellData[j * CellsPerSide + i];
                    if (r0 >= s.TowerShare || s.TowerTop <= layerTop)
                    {
                        cell = default;
                        continue;
                    }
                    double period = 1.5 + 1.5 * r4;
                    double life = _gameHours / period + r5;
                    float f = (float)(life - Math.Floor(life));
                    float growth = SmoothStep(0f, 0.35f, f);
                    float decay = 1f - SmoothStep(0.82f, 1f, f);
                    float potential = layerTop + (s.TowerTop - layerTop) * (0.55f + 0.45f * r6);
                    cell = new GpuCell
                    {
                        Center = new Vector2((0.2f + 0.6f * r1) * cellSize, (0.2f + 0.6f * r2) * cellSize),
                        Radius = (1800f + 2200f * r3) * (0.55f + 0.45f * growth) * MathF.Sqrt(decay),
                        Top = s.Base + s.Thickness * 0.5f + (potential - s.Base - s.Thickness * 0.5f) * growth,
                        // Only a tall tower spreads an anvil, against the tropopause it does not reach here
                        Anvil = SmoothStep(0.4f, 0.65f, f) * decay * Math.Clamp((potential - 6000f) / 2500f, 0f, 1f),
                        // Rain from a cumulonimbus (a potential top over ~5.5 km): fair-weather towers stay dry
                        Rain = SmoothStep(0.45f, 0.55f, f) * (1f - SmoothStep(0.8f, 0.92f, f))
                             * Math.Clamp((potential - 5500f) / 1500f, 0f, 1f),
                        Growth = f,
                    };
                }
            }
            unsafe
            {
                var dst = _cells[slot]!.WritePtr<GpuCell>();
                for (int k = 0; k < _cellData.Length; k++) dst[k] = _cellData[k];
            }
        }

        /// <summary>
        /// The raining cells nearest the camera (at most <see cref="MaxRainShafts"/> within <see cref="RainShaftReach"/>),
        /// for the march's rain shafts (clouds_cs.hlsl CloudRain): a column under the cell from the clouds' base, down to
        /// the ground or the sea or, for about a third of them, thinning out in the air (virga). The ground ends a shaft
        /// through the scene's depth; the sea here (the clouds are marched before the ocean is drawn: rain under the sea
        /// counted as in front of it, and its fog dropped the shaft over the water).
        /// </summary>
        private static void PlaceRainShafts(in CloudSettings s, int slot, double cx, double cz)
        {
            float cellSize = (float)(Period / CellsPerSide);
            float floor = Components.OceanRenderer.SeaLevel ?? -1000f;
            _shaftCount = 0;
            for (int j = 0; j < CellsPerSide; j++)
            {
                for (int i = 0; i < CellsPerSide; i++)
                {
                    ref var cell = ref _cellData[j * CellsPerSide + i];
                    if (cell.Rain <= 0.02f || cell.Radius <= 0f)
                        continue;
                    float dx = (float)WrapDelta(i * cellSize + cell.Center.X - cx), dz = (float)WrapDelta(j * cellSize + cell.Center.Y - cz);
                    float dist = MathF.Sqrt(dx * dx + dz * dz);
                    if (dist > RainShaftReach)
                        continue;
                    uint h = Hash((uint)i + 977u, (uint)j + 131u);
                    float virga = Rand(ref h), depth = Rand(ref h), seed = Rand(ref h);
                    float bottom = virga < 0.35f ? MathF.Max(s.Base * (0.3f + 0.4f * depth), floor) : floor;
                    var shaft = new GpuRainShaft
                    {
                        Center = new Vector2(dx, dz),
                        Radius = cell.Radius * 0.8f,
                        Rain = cell.Rain,
                        Bottom = bottom,
                        Seed = seed,
                        // Virga thins out over its lower 30 %; rain that reaches the ground or the sea ends there
                        Fade = virga < 0.35f ? 0.3f * (s.Base - bottom) : 0f,
                    };
                    // Keep the nearest: insert in order of distance
                    int k = Math.Min(_shaftCount, MaxRainShafts - 1);
                    if (_shaftCount == MaxRainShafts && dist >= _shaftDistance[k])
                        continue;
                    while (k > 0 && _shaftDistance[k - 1] > dist)
                    {
                        _shaftData[k] = _shaftData[k - 1];
                        _shaftDistance[k] = _shaftDistance[k - 1];
                        k--;
                    }
                    _shaftData[k] = shaft;
                    _shaftDistance[k] = dist;
                    if (_shaftCount < MaxRainShafts) _shaftCount++;
                }
            }
            unsafe
            {
                var dst = _rainShafts[slot]!.WritePtr<GpuRainShaft>();
                for (int k = 0; k < _shaftCount; k++) dst[k] = _shaftData[k];
            }
        }

        private static double WrapDelta(double d)
        {
            d -= Period * Math.Floor(d / Period);
            return d >= Period / 2 ? d - Period : d;
        }

        private const float RainLean = 0.25f;           // clouds_cs.hlsl RAIN_LEAN: metres downwind per metre fallen
        private const float OutflowPeak = 14f;          // m/s at the edge of the heaviest rain

        /// <summary>
        /// The wind a storm's downdrafts make at the ground under the camera (xz, m/s; Weather adds it to Wind.Gust): the
        /// rain drags cold air down its shaft, which spreads out along the ground as a gust front, calm in the core,
        /// strongest at the rain's edge, dying out a few radii away. Virga's air sinks too (a dry downburst), weaker.
        /// The shafts are this frame's (PlaceRainShafts), leaned downwind as the march draws them.
        /// </summary>
        public static Vector2 StormOutflow(float groundY)
        {
            var sum = Vector2.Zero;
            var lean = Wind.GeneratedDirection * ((Settings.Base - groundY) * RainLean);
            for (int k = 0; k < _shaftCount; k++)
            {
                ref var s = ref _shaftData[k];
                var away = -(s.Center + lean);                  // from the shaft's axis at the ground to the camera
                float r = away.Length(), radius = MathF.Max(s.Radius, 1f);
                if (r < 1f || r > 6f * radius) continue;
                float x = r / radius;
                sum += away / r * (OutflowPeak * s.Rain * x * MathF.Exp(1f - x) * (s.Fade > 0f ? 0.7f : 1f));
            }
            return sum;
        }

        /// <summary>
        /// Lightning (<see cref="CloudSettings.LightningRate"/> flashes a minute, at random intervals): a flash in one of
        /// the raining cells near the camera, at a random place inside its column, 1-4 strokes over 0.05-0.3 s. Its place
        /// stays put in cloud space (the wind carries it with the cloud).
        /// </summary>
        private static void UpdateFlash(in CloudSettings s, Vector3 cameraPos, double cx, double cz)
        {
            _flashClock += _lastDt;
            if (_flashRequested && _shaftCount > 0)
            {
                _flashRequested = false;
                StartFlash(s, cx, cz, 0);           // the nearest raining cell (the shafts' list is by distance)
                return;
            }
            if (s.LightningRate <= 0f || _shaftCount == 0)
                return;
            _nextFlashIn -= _lastDt;
            if (_nextFlashIn > 0f)
                return;
            _nextFlashIn = -MathF.Log(1f - (float)_flashRandom.NextDouble() * 0.999f) * 60f / s.LightningRate;
            // A raining cell, the harder it rains the likelier (the nearest few are the shafts' list)
            float total = 0f;
            for (int k = 0; k < _shaftCount; k++) total += _shaftData[k].Rain;
            float pick = (float)_flashRandom.NextDouble() * total;
            int chosen = 0;
            for (int k = 0; k < _shaftCount; k++)
            {
                pick -= _shaftData[k].Rain;
                if (pick <= 0f) { chosen = k; break; }
            }
            StartFlash(s, cx, cz, chosen);
        }

        /// <summary>A flash on the next frame in the raining cell nearest the camera (SkyboxRenderer's button; tests).</summary>
        public static void FlashNow() => _flashRequested = true;

        /// <summary>A flash is lighting the clouds, or did so recently enough to be in the exposure's metering still
        /// (its readback lags a few frames): AutoExposure holds still meanwhile.</summary>
        public static bool FlashInMetering => _flashRequested || _flashClock < _flashLength + 0.4;

        private static void StartFlash(in CloudSettings s, double cx, double cz, int chosen)
        {
            var shaft = _shaftData[chosen];
            float angle = (float)_flashRandom.NextDouble() * MathF.Tau, r = shaft.Radius * 0.6f * MathF.Sqrt((float)_flashRandom.NextDouble());
            float height = s.Base + (MathF.Max(s.TowerTop, s.Base + s.Thickness) - s.Base) * (0.2f + 0.4f * (float)_flashRandom.NextDouble());
            _flashCloud = new Vector3((float)(cx + shaft.Center.X + r * MathF.Cos(angle)), height, (float)(cz + shaft.Center.Y + r * MathF.Sin(angle)));
            _flashClock = 0.0;
            _flashLength = 0.05f + 0.25f * (float)_flashRandom.NextDouble();
            _flashStrokes = 1 + _flashRandom.Next(4);
            _flashPeak = FlashPeakCandela * (0.5f + (float)_flashRandom.NextDouble());
        }

        // The flash's luminous intensity now: each stroke a jump that fades over ~25 ms
        private static float FlashIntensityNow()
        {
            if (_flashClock > _flashLength + 0.15)
                return 0f;
            float t = (float)_flashClock, sum = 0f;
            for (int k = 0; k < _flashStrokes; k++)
            {
                float tk = _flashLength * k / _flashStrokes;
                if (t >= tk) sum += MathF.Exp(-(t - tk) / 0.025f);
            }
            return _flashPeak * MathF.Min(sum, 1.5f);
        }

        private static Vector3 FlashRelative(Vector3 cameraPos, double cx, double cz) =>
            new((float)WrapDelta(_flashCloud.X - cx), _flashCloud.Y - cameraPos.Y, (float)WrapDelta(_flashCloud.Z - cz));

        private static uint Hash(uint x, uint y)
        {
            uint h = x * 0x8DA6B343u ^ y * 0xD8163841u ^ 0x2C1B3C6Du;
            h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
            return h;
        }

        private static float Rand(ref uint h)
        {
            h ^= h << 13; h ^= h >> 17; h ^= h << 5;
            return (h & 0xFFFFFF) / 16777216f;
        }

        private static float SmoothStep(float a, float b, float x)
        {
            float t = Math.Clamp((x - a) / (b - a), 0f, 1f);
            return t * t * (3f - 2f * t);
        }

        /// <summary>
        /// Bake this frame's brick columns (once per frame, from Atmosphere.Render, before anything samples the clouds). The
        /// scene constants (CloudParamsIdx) must be bound for compute.
        /// </summary>
        public static void Render(ID3D12GraphicsCommandList list)
        {
            if (!Active || _paramsIdx == 0 || _renderedTick == Engine.TickCount)
                return;
            _renderedTick = Engine.TickCount;
            var cs = Shader;
            var shaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;

            PixMarker.Begin(list, "Cloud bake");
            for (int i = 0; i < LevelCount; i++)
            {
                var level = _levels[i];
                if (level.State != ResourceStates.UnorderedAccess)
                {
                    list.ResourceBarrierTransition(level.Density.Native, level.State, ResourceStates.UnorderedAccess);
                    list.ResourceBarrierTransition(level.Bricks.Native, level.State, ResourceStates.UnorderedAccess);
                    level.State = ResourceStates.UnorderedAccess;
                }
            }
            _voxelsBaked = 0;
            foreach (var bake in _bakes)
            {
                var level = _levels[bake.Level];
                cs.SetPushConstant(_kBake, "DensityUAV", level.Density.UavIndex);
                cs.SetPushConstant(_kBake, "BrickUAV", level.Bricks.UavIndex);
                cs.SetPushConstant(_kBake, "BakeLevel", (uint)bake.Level);
                cs.SetPushConstant(_kBake, "RegionX0Bits", unchecked((uint)(int)(bake.X0 * Brick)));
                cs.SetPushConstant(_kBake, "RegionZ0Bits", unchecked((uint)(int)(bake.Z0 * Brick)));
                cs.Dispatch(_kBake, list, (uint)bake.BricksX, (uint)bake.BricksZ, (uint)(level.Layers / Brick));
                _voxelsBaked += bake.BricksX * bake.BricksZ * Brick * Brick * level.Layers;
            }
            for (int i = 0; i < LevelCount; i++)
            {
                var level = _levels[i];
                list.ResourceBarrierTransition(level.Density.Native, ResourceStates.UnorderedAccess, shaderRead);
                list.ResourceBarrierTransition(level.Bricks.Native, ResourceStates.UnorderedAccess, shaderRead);
                level.State = shaderRead;
            }
            PixMarker.End(list);

            // What reaches the ground round the camera (the column over each place, the shafts), from the field just baked
            Weather.Render(list);

            PixMarker.Begin(list, "Cloud light");
            foreach (int i in _lightLevels)
                MakeLight(list, i);
            PixMarker.End(list);

            // The ground shadow map: a share of its rows (all of them after a rebake, a change of light or a jump), after
            // the light volumes (it takes the rest of a storm's depth from them)
            PixMarker.Begin(list, "Cloud shadow map");
            if (_shadowMapState != ResourceStates.UnorderedAccess)
            {
                list.ResourceBarrierTransition(_shadowMap!.Native, _shadowMapState, ResourceStates.UnorderedAccess);
                _shadowMapState = ResourceStates.UnorderedAccess;
            }
            int firstRow = _shadowAll ? 0 : _shadowRow, rows = _shadowAll ? ShadowMapTexels : ShadowRowsPerFrame;
            cs.SetPushConstant(_kShadowMap, "TargetUAV", _shadowMap!.UavIndex);
            cs.SetPushConstant(_kShadowMap, "TargetWidth", (uint)ShadowMapTexels);
            cs.SetPushConstant(_kShadowMap, "TargetHeight", (uint)rows);
            cs.SetPushConstant(_kShadowMap, "RegionZ0Bits", (uint)firstRow);
            cs.Dispatch(_kShadowMap, list, ShadowMapTexels / 8, (uint)(rows + 7) / 8, 1);
            _shadowRowsLast = rows;
            if (!_shadowAll) _shadowRow = (_shadowRow + rows) % ShadowMapTexels;
            _shadowAll = false;
            list.ResourceBarrierTransition(_shadowMap.Native, ResourceStates.UnorderedAccess, shaderRead);
            _shadowMapState = shaderRead;
            PixMarker.End(list);

            // The sky probe's clouds: a face a frame (all five after a rebake or a change of light); the probe composes
            // them over its sky every frame (sky_probe_cs.hlsl), so it follows the time and the exposure as before
            PixMarker.Begin(list, "Cloud probe");
            if (_probeCloudsState != ResourceStates.UnorderedAccess)
            {
                list.ResourceBarrierTransition(_probeClouds!, _probeCloudsState, ResourceStates.UnorderedAccess);
                _probeCloudsState = ResourceStates.UnorderedAccess;
            }
            int faces = _probeAll ? 6 : 1;          // all six once (−Y too: it must hold "no cloud", not garbage)
            for (int f = 0; f < faces; f++)
            {
                var round = _cameraOverBase ? ProbeFacesAbove : ProbeFaces;
                int face = _probeAll ? f : round[_probeTurn % round.Length];
                cs.SetPushConstant(_kProbeFace, "TargetUAV", _probeCloudsUav);
                cs.SetPushConstant(_kProbeFace, "TargetWidth", (uint)ProbeSize);
                cs.SetPushConstant(_kProbeFace, "ProbeFace", (uint)face);
                cs.Dispatch(_kProbeFace, list, ProbeSize / 8, ProbeSize / 8, 1);
            }
            if (!_probeAll) _probeTurn = (_probeTurn + 1) % (_cameraOverBase ? ProbeFacesAbove.Length : ProbeFaces.Length);
            _probeAll = false;
            list.ResourceBarrierTransition(_probeClouds!, ResourceStates.UnorderedAccess, shaderRead);
            _probeCloudsState = shaderRead;
            PixMarker.End(list);
        }

        /// <summary>
        /// A level's light volume (each texel's optical depth toward the light out of the slab, by doubling, the coarser
        /// level's added where the path leaves this one) and its sky volume (straight up and down). The coarser level's
        /// light must be made already (this frame or before): levels go coarse to fine.
        /// </summary>
        private static void MakeLight(ID3D12GraphicsCommandList list, int i)
        {
            var cs = _cs!;
            var level = _levels[i];
            var shaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;
            uint groupsXY = (uint)(TexelsPerLevel / 2 / 8), groupsZ = (uint)(level.Layers / 2 / 8);

            // The step along the light that moves at most one light texel each way (CloudLightStepLength)
            float across = level.Voxel * 2f, up = (_slabTop - _slabBottom) / (level.Layers / 2);
            var d = Vector3.Abs(_lightDir);
            float step = MathF.Min(MathF.Min(across / MathF.Max(d.X, 1e-4f), across / MathF.Max(d.Z, 1e-4f)), up / MathF.Max(d.Y, 1e-4f));

            Transition(list, level.LightA, ref level.LightAState, ResourceStates.UnorderedAccess);
            cs.SetPushConstant(_kLightInit, "BakeLevel", (uint)i);
            cs.SetPushConstant(_kLightInit, "TargetUAV", level.LightA.UavIndex);
            cs.Dispatch(_kLightInit, list, groupsXY, groupsXY, groupsZ);

            // Doubling: A → B → A ... the eighth pass ends in A
            var src = level.LightA; var dst = level.LightB;
            for (int k = 0; k < LightPasses; k++)
            {
                if (src == level.LightA) Transition(list, src, ref level.LightAState, shaderRead);
                else Transition(list, src, ref level.LightBState, shaderRead);
                if (dst == level.LightA) Transition(list, dst, ref level.LightAState, ResourceStates.UnorderedAccess);
                else Transition(list, dst, ref level.LightBState, ResourceStates.UnorderedAccess);
                cs.SetPushConstant(_kLightStep, "BakeLevel", (uint)i);
                cs.SetPushConstant(_kLightStep, "SourceTex", src.BindlessIndex);
                cs.SetPushConstant(_kLightStep, "TargetUAV", dst.UavIndex);
                cs.SetPushConstant(_kLightStep, "LightStepBits", BitConverter.SingleToUInt32Bits(step * (1 << k)));
                cs.Dispatch(_kLightStep, list, groupsXY, groupsXY, groupsZ);
                (src, dst) = (dst, src);
            }

            // The coarser level's light past this level's box
            Transition(list, level.LightA, ref level.LightAState, ResourceStates.UnorderedAccess);
            if (i < LevelCount - 1)
            {
                cs.SetPushConstant(_kLightChain, "BakeLevel", (uint)i);
                cs.SetPushConstant(_kLightChain, "TargetUAV", level.LightA.UavIndex);
                cs.Dispatch(_kLightChain, list, groupsXY, groupsXY, groupsZ);
            }
            Transition(list, level.LightA, ref level.LightAState, shaderRead);

            // The sky volume: a thread per column
            Transition(list, level.Sky, ref level.SkyState, ResourceStates.UnorderedAccess);
            cs.SetPushConstant(_kSkyScan, "BakeLevel", (uint)i);
            cs.SetPushConstant(_kSkyScan, "TargetUAV", level.Sky.UavIndex);
            cs.Dispatch(_kSkyScan, list, groupsXY, groupsXY, 1);
            Transition(list, level.Sky, ref level.SkyState, shaderRead);
        }

        private static void Transition(ID3D12GraphicsCommandList list, RenderTexture3D texture, ref ResourceStates state,
            ResourceStates to)
        {
            if (state == to)
            {
                if (to == ResourceStates.UnorderedAccess)
                    list.ResourceBarrier(new ResourceBarrier(new ResourceUnorderedAccessViewBarrier(texture.Native)));
                return;
            }
            list.ResourceBarrierTransition(texture.Native, state, to);
            state = to;
        }

        /// <summary>For engine_stats.</summary>
        public static object Describe() => new
        {
            active = Active,
            slab = float.IsNaN(_slabBottom) ? null : new[] { _slabBottom, _slabTop },     // NaN until the clouds first ran
            bakesThisFrame = _bakes.Count,
            voxelsBaked = _voxelsBaked,
            offset = new[] { _offsetX, _offsetZ },
            wind = new[] { Wind.GeneratedDirection.X, Wind.GeneratedDirection.Y },
            gameHours = _gameHours,
            towers = CountTowers(),
            lightLevelsThisFrame = _lightLevels.Count,
            litBy = _lightIsMoon ? "moon" : "sun",
            rainShafts = _shaftCount,
            nearestRainKm = _shaftCount > 0 ? _shaftDistance[0] / 1000f : -1f,
            flashCandela = FlashIntensityNow(),
            nextFlashInSeconds = Settings.LightningRate > 0f ? _nextFlashIn : -1f,
            // Camera-relative xz (m) of the nearest raining cell and the nearest mature anvil, and the last flash (x, height
            // over the camera, z): where to look for rain, mammatus, a flash
            nearestRainAt = _shaftCount > 0 ? new[] { _shaftData[0].Center.X, _shaftData[0].Center.Y } : null,
            nearestAnvilAt = NearestAnvil(),
            lastFlashAt = new[] { _describeFlash.X, _describeFlash.Y, _describeFlash.Z },
            // The ground shadow map: its window's centre relative to the camera (m, where the camera's ray toward the light
            // crosses the slab's bottom) and the rows made last frame
            shadowMap = new
            {
                centre = new[] { (float)WrapDelta(_shadowCentre.X - _describeCx), (float)WrapDelta(_shadowCentre.Y - _describeCz) },
                rowsLastFrame = _shadowRowsLast,
            },
        };

        private static float[]? NearestAnvil()
        {
            float cellSize = (float)(Period / CellsPerSide), best = float.MaxValue;
            float[]? at = null;
            for (int j = 0; j < CellsPerSide; j++)
                for (int i = 0; i < CellsPerSide; i++)
                {
                    var c = _cellData[j * CellsPerSide + i];
                    if (c.Anvil < 0.5f) continue;
                    float dx = (float)WrapDelta(i * cellSize + c.Center.X - _describeCx), dz = (float)WrapDelta(j * cellSize + c.Center.Y - _describeCz);
                    float d = dx * dx + dz * dz;
                    if (d < best) { best = d; at = new[] { dx, dz, c.Anvil, c.Top }; }
                }
            return at;
        }

        private static int CountTowers()
        {
            int n = 0;
            foreach (var c in _cellData) if (c.Radius > 0f) n++;
            return n;
        }
    }
}
