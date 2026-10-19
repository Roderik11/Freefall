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

        public static CloudSettings Default => new()
        {
            Enabled = false, Coverage = 0.5f, Type = 1f, Base = 1800f, Thickness = 1000f, TowerShare = 0f,
            TowerTop = 6000f, Density = 0.05f, Albedo = 0.99f, Speed = 1f,
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
    /// through SceneConstants.CloudParamsIdx (cloud_common.hlsli; 0 = off). Setting: Engine.Settings.VolumetricClouds.
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
        private static int _kBake, _kLightInit, _kLightStep, _kLightChain, _kSkyScan;
        private const int LightPasses = 8;              // doubling passes: 256 steps along the light cross a level
        private static readonly List<int> _lightLevels = new();     // levels whose light and sky volumes are made this frame
        private static int _lightTurn;
        private static Vector3 _lightDir = Vector3.UnitY;
        private static bool _lightIsMoon;
        private static readonly Level[] _levels = new Level[LevelCount];
        private static readonly GraphicsBuffer?[] _params = new GraphicsBuffer?[FrameCount];
        private static readonly GraphicsBuffer?[] _cells = new GraphicsBuffer?[FrameCount];
        private static readonly GpuCell[] _cellData = new GpuCell[CellsPerSide * CellsPerSide];
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
            }
        }

        /// <summary>
        /// Once per frame (SkyboxRenderer.Update): the wind carries the clouds (cloud space moves with it) and they change
        /// on their own. <paramref name="windDirection"/> is where the wind blows (xz, unit); <paramref name="gameHours"/>
        /// (Day × 24 + TimeOfDay) drives the storm cells' lives.
        /// </summary>
        public static void Advance(float dt, Vector2 windDirection, double gameHours)
        {
            double speed = DriftSpeed * Settings.Speed;
            _offsetX = Wrap(_offsetX + windDirection.X * speed * dt);
            _offsetZ = Wrap(_offsetZ + windDirection.Y * speed * dt);
            _evolution += dt * Settings.Speed;
            if (_evolution > 1e6) _evolution -= 1e6;
            _gameHours = gameHours;
        }

        private static double Wrap(double v) => v - Period * Math.Floor(v / Period);

        /// <summary>
        /// This frame's parameters (once per frame) and the bindless index of their buffer for SceneConstants.CloudParamsIdx
        /// (0 = no clouds): the levels follow the camera, the brick columns to bake are chosen (<see cref="Render"/>), the
        /// storm cells are placed.
        /// </summary>
        public static uint PrepareFrame(Vector3 cameraPos)
        {
            if (!Active)
            {
                _paramsIdx = 0;
                return 0;
            }
            if (_preparedTick == Engine.TickCount)
                return _paramsIdx;
            _preparedTick = Engine.TickCount;
            EnsureResources();
            int slot = Engine.FrameIndex % FrameCount;
            var s = Settings;

            // The heights the volumes span: the layer and the towers, snapped so a preset blend rebakes rarely
            float top = MathF.Max(s.Base + s.Thickness, s.TowerShare > 0f ? s.TowerTop * 1.12f : 0f);
            float slabBottom = MathF.Floor(MathF.Max(s.Base, 0f) / 250f) * 250f;
            float slabTop = MathF.Max(MathF.Ceiling(top / 250f) * 250f, slabBottom + 250f);
            bool rebuild = slabBottom != _slabBottom || slabTop != _slabTop
                || double.IsNaN(_lastHours) || Math.Abs(_gameHours - _lastHours) > 0.25;
            _slabBottom = slabBottom;
            _slabTop = slabTop;
            _lastHours = _gameHours;

            // The light the volumes are made along: the sun until it is ~3.4° under the horizon (a 12 km anvil still
            // sees it), then the brightest moon
            var atm = Atmosphere.Settings;
            bool moon = atm.SunDirection.Y < -0.06f && atm.MoonIlluminance != Vector3.Zero;
            var lightDir = Vector3.Normalize(moon ? atm.MoonDirection : atm.SunDirection);
            bool relight = rebuild || moon != _lightIsMoon;
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
                        Rain = SmoothStep(0.45f, 0.55f, f) * (1f - SmoothStep(0.8f, 0.92f, f)),
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

            PixMarker.Begin(list, "Cloud light");
            foreach (int i in _lightLevels)
                MakeLight(list, i);
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
            slab = new[] { _slabBottom, _slabTop },
            bakesThisFrame = _bakes.Count,
            voxelsBaked = _voxelsBaked,
            offset = new[] { _offsetX, _offsetZ },
            wind = new[] { Wind.GeneratedDirection.X, Wind.GeneratedDirection.Y },
            gameHours = _gameHours,
            towers = CountTowers(),
            lightLevelsThisFrame = _lightLevels.Count,
            litBy = _lightIsMoon ? "moon" : "sun",
        };

        private static int CountTowers()
        {
            int n = 0;
            foreach (var c in _cellData) if (c.Radius > 0f) n++;
            return n;
        }
    }
}
