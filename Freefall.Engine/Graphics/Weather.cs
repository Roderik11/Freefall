using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// The weather at the camera, made by the clouds (WEATHER_PLAN.md stage 1). Nothing here is authored: a preset gives
    /// the clouds and the freezing level, and the clouds decide where it rains or snows.
    ///   • A precipitation map round the camera (clouds_cs.hlsl CSPrecipitation, <see cref="MapTexels"/>² texels of
    ///     <see cref="MapTexel"/> m, RG16F: the rate, hail): what reaches the ground at each place, from the cloud column
    ///     over it (a deep deck whose top reaches the ice rains or snows; fair-weather cumulus and a thin overcast do not)
    ///     and from the storm cells' rain shafts (showers; hail in the cores of the heaviest), the same shafts the march draws.
    ///   • Its value at the camera (CSPrecipitationProbe) comes back to the CPU a few frames late and is eased in over a
    ///     few seconds: <see cref="Rain"/>, <see cref="Snow"/> and <see cref="Hail"/> (1 = heavy) drive the weather emitters
    ///     (EnvironmentController), split by the freezing level (<see cref="CloudSettings.FreezingLevel"/>): snow above it,
    ///     sleet in the <see cref="MeltDepth"/> under it, rain below.
    ///   • The wind: the raining cells' outflow (Clouds.StormOutflow) and gusts that grow with the rain, in
    ///     <see cref="Wind.Gust"/>.
    /// Without volumetric clouds the deck's rule is applied to the preset's clouds as a whole (no breaks, no showers).
    /// Docs: .agent/knowledge/atmospheric_effects/weather.md.
    /// </summary>
    public static class Weather
    {
        public const int MapTexels = 128;               // clouds_cs.hlsl: the map's texels each way
        public const float MapTexel = 32f;              // m: the map spans 4 km
        public const float MeltDepth = 300f;            // m under the freezing level over which falling snow melts into rain

        // The deck's rule (clouds_cs.hlsl PRECIP_*): a column precipitates when its cloud is deep and its top reaches the ice
        private const float IceLift = 1000f, IceRange = 1500f, DepthMin = 500f, DepthFull = 2000f;
        private const float EaseSeconds = 4f;           // rain sets in and stops over a few seconds, not in a frame
        private const int FrameCount = 3;

        /// <summary>What falls at the camera (eased; 1 = heavy, the emitters' EmitRate), all kinds together.</summary>
        public static float Precipitation { get; private set; }

        /// <summary>The part of it that falls as rain, as snow, and the hail (0..~1).</summary>
        public static float Rain { get; private set; }
        public static float Snow { get; private set; }
        public static float Hail { get; private set; }

        /// <summary>0..1: how much of the precipitation at the camera's height falls as snow.</summary>
        public static float SnowShare { get; private set; }

        /// <summary>
        /// The haze the falling rain and snow make of the air: extinction per km at the ground, which Atmosphere adds to
        /// the aerosols (their ~1.2 km scale height fills the air under the cloud base, their forward phase suits drops and
        /// flakes, and every pass's fog, the sky and the sun's light take it). Visibility is 3.9 / extinction: heavy rain
        /// ~1.3 km, heavy snow ~400 m (flakes scatter ~5× what drops of the same water do); light falls already show (the
        /// rate's 0.63 power, as Marshall-Palmer's for rain). In 0.02 steps, so the atmosphere's tables are made again only
        /// when it changes a step. Before it, a downpour or a snowstorm left the air as clear as a dry day (user, 2026-10-23).
        /// </summary>
        public static float HazeKm { get; private set; }

        private const float RainHazeKm = 3f, SnowHazeKm = 10f, HailHazeKm = 1f;

        /// <summary>
        /// What the near rain and snow emitters make of a heavy fall (rate 1): drops per m³ (EmitRate over the emitter's
        /// box area × its fall speed) and the fall speed (m/s, gravity over drag). EnvironmentController sets them from its
        /// emitters every frame; the far layers (PrecipitationLayers) hold as many drops and lean alike. Constants in the
        /// layers' shader held six to ten times the near rain, and went wrong with every retune of the emitters.
        /// </summary>
        public static float RainDensity { get; set; } = 6.5f;
        public static float RainFallSpeed { get; set; } = 8f;
        public static float SnowDensity { get; set; } = 10f;
        public static float SnowFallSpeed { get; set; } = 1f;

        /// <summary>
        /// The wind falling rain, snow and hail drift in, for something that falls at <paramref name="fallSpeed"/> m/s:
        /// the steady wind (not the gusts), at most 0.8 × the fall speed, as precipitation_layers_cs.hlsl leans the far
        /// field (it must: its cells fold past that). The near drops in the gusting wind, uncapped, fell almost sideways
        /// next to the far field's (user, 2026-10-23).
        /// </summary>
        public static Vector3 FallWind(float fallSpeed)
        {
            var w = Wind.Current;
            var flat = new Vector2(w.X, w.Z);
            float speed = flat.Length(), limit = 0.8f * MathF.Max(fallSpeed, 0f);
            if (speed > limit && speed > 1e-4f) flat *= limit / speed;
            return new Vector3(flat.X, 0f, flat.Y);
        }

        /// <summary>
        /// What the weather has left on the open surfaces round the camera (0..1), for the decal pass (decals_cs.hlsl):
        /// how wet they are (rising in the rain, a minute to soak in heavy rain), how full rain-filled puddles are (filling
        /// once the ground is soaked, ~5 minutes in heavy rain), how much snow lies (~10 minutes of heavy snow to cover).
        /// Once the rain stops they dry by evaporation: the sun on the ground and the wind speed it up (open ground ~1.7
        /// minutes at a sunny noon with a breeze, ~5 under overcast, ~10 on a still night; puddles four times as long);
        /// snow melts where it is warmer than the freezing level, faster in sun and rain. Real seconds, not game time.
        /// Settings › OverrideWeatherSurfaces sets them by hand.
        /// </summary>
        public static float SurfaceWetness { get; private set; }
        public static float PuddleFill { get; private set; }
        public static float SnowCover { get; private set; }

        /// <summary>
        /// How much snow has fallen at the camera since start, in covers' worth (a cover's worth: what lays SnowCover 0 → 1,
        /// CoverSeconds of heavy snow); only grows. A footprint keeps the value it was made at, and fills with what falls
        /// after it (decals_cs.hlsl: Print).
        /// </summary>
        public static float SnowFallen { get; private set; }

        private const float SoakSeconds = 60f, DrySeconds = 600f;
        private const float FillSeconds = 300f, DrainSeconds = 2400f;
        private const float CoverSeconds = 600f, MeltSeconds = 1200f;
        private const float SunLux = 60000f;    // lux of sun on level ground that counts as 1 (a clear noon gives ~1.4)
        private static float _wetness, _puddles, _snow;     // the weather's own, kept while the override is on

        /// <summary>The precipitation map (Texture2D RG16F: rate, hail; 0 = none yet) and the world xz of its corner.</summary>
        public static uint MapIndex => _mapValid ? _map!.BindlessIndex : 0u;
        public static Vector2 MapOrigin { get; private set; }

        private static RenderTexture2D? _map;
        private static ResourceStates _mapState = ResourceStates.Common;
        private static bool _mapValid;
        private static GraphicsBuffer? _probe;
        private static readonly GraphicsBuffer?[] _readback = new GraphicsBuffer?[FrameCount];
        private static readonly bool[] _written = new bool[FrameCount];
        private static int _kMap = -1, _kProbe = -1;
        private static Vector2 _originRelative;         // the map's corner relative to the camera (m)
        private static Vector4 _column;                 // at the camera: rate, hail, the column's cloud depth (m), its top (m)
        private static float _targetHail;
        private static int _updatedTick = -1;
        private static double _gustClock;

        /// <summary>
        /// Once per tick (Clouds.PrepareFrame, after the rain shafts are placed): read what the probe found a few frames
        /// ago, ease the camera's weather toward it, split it by the freezing level, and set the gusts.
        /// </summary>
        public static unsafe void Update(Vector3 cameraPos, bool cloudsActive)
        {
            if (_updatedTick == Engine.TickCount) return;
            _updatedTick = Engine.TickCount;
            float dt = Math.Clamp((float)Base.Time.Delta, 0f, 0.1f);
            var s = Clouds.Settings;

            // The map's window: snapped to its texels in world space so a camera move does not shimmer its edges
            var corner = new Vector2(
                MathF.Floor(cameraPos.X / MapTexel) * MapTexel - MapTexels / 2 * MapTexel,
                MathF.Floor(cameraPos.Z / MapTexel) * MapTexel - MapTexels / 2 * MapTexel);
            MapOrigin = corner;
            _originRelative = corner - new Vector2(cameraPos.X, cameraPos.Z);

            float target;
            if (cloudsActive)
            {
                int slot = Engine.FrameIndex % FrameCount;
                if (_written[slot] && _readback[slot] != null)
                    _column = *_readback[slot]!.ReadPtr<Vector4>();
                target = _column.X;
                _targetHail = _column.Y;
            }
            else
            {
                _mapValid = false;
                // The preset's deck as a whole: as deep as its coverage fills it, its top the layer's
                float depth = s.Enabled ? s.Thickness * SmoothStep(0.6f, 0.95f, s.Coverage) : 0f;
                target = DeckRate(depth, s.Base + s.Thickness, s.FreezingLevel);
                _targetHail = 0f;
                _column = new Vector4(target, 0f, depth, s.Base + s.Thickness);
            }

            float ease = 1f - MathF.Exp(-dt / EaseSeconds);
            Precipitation += (target - Precipitation) * ease;
            Hail += (_targetHail - Hail) * ease;
            if (Precipitation < 1e-4f) Precipitation = 0f;
            if (Hail < 1e-4f) Hail = 0f;

            // Snow above the freezing level, rain under its melting layer, sleet (both) in between
            SnowShare = Math.Clamp((cameraPos.Y - (s.FreezingLevel - MeltDepth)) / MeltDepth, 0f, 1f);
            Snow = Precipitation * SnowShare;
            Rain = Precipitation * (1f - SnowShare);
            Hail *= 1f - SnowShare;
            float haze = RainHazeKm * MathF.Pow(Rain, 0.63f) + SnowHazeKm * MathF.Pow(Snow, 0.63f) + HailHazeKm * Hail;
            HazeKm = MathF.Round(haze * 50f) / 50f;

            // What it leaves on the ground: rain soaks it and fills puddles once soaked; dry weather evaporates it, the faster
            // the more sun reaches the ground (its light, the clouds' shade included) and the stronger the wind (the two
            // terms of evaporation, crudely); snow builds a cover where it is cold enough and melts where it is not (sun and
            // rain melt it faster). Before 2026-10-22 evening drying took a fixed 15 minutes, puddles 40, whatever the
            // weather: a sunny noon after a storm stayed soaked (user: "the ground now seems to stay wet forever")
            float sun = Math.Clamp(Daylight.SunIlluminance() * Daylight.SunHeight() / SunLux, 0f, 1.5f);
            float evaporation = (1f + 3f * sun + 0.2f * MathF.Min(Wind.Strength, 15f)) * (1f - MathF.Min(Rain * 5f, 1f));
            float wetting = Rain / SoakSeconds, drying = evaporation / DrySeconds;
            _wetness = Math.Clamp(_wetness + (wetting - drying) * dt, 0f, 1f);
            float filling = _wetness > 0.6f ? Rain / FillSeconds : 0f, draining = evaporation / DrainSeconds;
            _puddles = Math.Clamp(_puddles + (filling - draining) * dt, 0f, 1f);
            float covering = Snow / CoverSeconds;
            SnowFallen += covering * dt;
            float melting = SnowShare < 0.5f ? (1f + 3f * Rain + 2f * sun) / MeltSeconds : 0f;
            _snow = Math.Clamp(_snow + (covering - melting) * dt, 0f, 1f);
            var settings = Engine.Settings;
            bool manual = settings != null && settings.OverrideWeatherSurfaces;
            SurfaceWetness = manual ? Math.Clamp(settings!.OverrideWetness, 0f, 1f) : _wetness;
            PuddleFill = manual ? Math.Clamp(settings!.OverridePuddleFill, 0f, 1f) : _puddles;
            SnowCover = manual ? Math.Clamp(settings!.OverrideSnowCover, 0f, 1f) : _snow;

            // The wind: a storm's outflow, and gusts on the steady wind that grow with the rain (gust factor ~1.2 in fair
            // weather, ~1.6 in a downpour); a little sideways too, so a gust does not only push along one line
            _gustClock += dt;
            var outflow = cloudsActive ? Clouds.StormOutflow(cameraPos.Y) : Vector2.Zero;
            var steady = Wind.Current;
            var mean = new Vector2(steady.X, steady.Z) + outflow;
            float speed = mean.Length();
            var along = speed > 1e-3f ? mean / speed : Vector2.UnitX;
            var side = new Vector2(-along.Y, along.X);
            float gustiness = 0.2f + 0.4f * MathF.Min(Precipitation, 1f);
            float surge = (float)(0.7 * Noise(_gustClock / 4.0) + 0.3 * Noise(_gustClock / 1.3 + 17.0)) * 2f;
            float sway = (float)Noise(_gustClock / 6.0 + 41.0);
            Wind.Gust = outflow + along * (speed * gustiness * surge) + side * (speed * 0.15f * sway);
        }

        /// <summary>The deck's rule, as clouds_cs.hlsl PrecipitationAt applies it to a column: deep enough, and its top in
        /// the ice (over the freezing level by <see cref="IceLift"/>, the Bergeron process).</summary>
        private static float DeckRate(float depth, float top, float freezingLevel)
            => Math.Clamp((top - freezingLevel - IceLift) / IceRange, 0f, 1f)
             * Math.Clamp((depth - DepthMin) / (DepthFull - DepthMin), 0f, 1f);

        /// <summary>
        /// Once per frame (Clouds.Render, after the bake): the map round the camera and the probe at the camera, then the
        /// probe's copy for the CPU. The scene constants (CloudParamsIdx) are bound for compute.
        /// </summary>
        public static void Render(ID3D12GraphicsCommandList list)
        {
            var cs = Clouds.Shader;
            EnsureResources(cs);
            var shaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;
            PixMarker.Begin(list, "Precipitation");

            if (_mapState != ResourceStates.UnorderedAccess)
            {
                list.ResourceBarrierTransition(_map!.Native, _mapState, ResourceStates.UnorderedAccess);
                _mapState = ResourceStates.UnorderedAccess;
            }
            SetConstants(cs, _kMap);
            SetConstants(cs, _kProbe);
            cs.Dispatch(_kMap, list, MapTexels / 8, MapTexels / 8, 1);
            list.ResourceBarrierTransition(_map!.Native, ResourceStates.UnorderedAccess, shaderRead);
            _mapState = shaderRead;
            _mapValid = true;

            _probe!.Transition(list, ResourceStates.UnorderedAccess);
            cs.Dispatch(_kProbe, list, 1, 1, 1);
            _probe.Transition(list, ResourceStates.CopySource);
            int slot = Engine.FrameIndex % FrameCount;
            list.CopyBufferRegion(_readback[slot]!.Native, 0, _probe.Native, 0, 16);
            _written[slot] = true;
            PixMarker.End(list);
        }

        private static void SetConstants(ComputeShader cs, int kernel)
        {
            cs.SetPushConstant(kernel, "TargetUAV", _map!.UavIndex);
            cs.SetPushConstant(kernel, "TargetWidth", MapTexels);
            cs.SetPushConstant(kernel, "PrecipProbeUAV", _probe!.UavIndex);
            cs.SetPushConstant(kernel, "PrecipOriginXBits", BitConverter.SingleToUInt32Bits(_originRelative.X));
            cs.SetPushConstant(kernel, "PrecipOriginZBits", BitConverter.SingleToUInt32Bits(_originRelative.Y));
            cs.SetPushConstant(kernel, "PrecipTexelBits", BitConverter.SingleToUInt32Bits(MapTexel));
        }

        private static void EnsureResources(ComputeShader cs)
        {
            if (_map != null) return;
            _kMap = cs.FindKernel("CSPrecipitation");
            _kProbe = cs.FindKernel("CSPrecipitationProbe");
            _map = new RenderTexture2D(Engine.Device, MapTexels, MapTexels, Format.R16G16_Float, randomWrite: true);
            _probe = GraphicsBuffer.CreateStructured<Vector4>(1, srv: false, uav: true);
            for (int i = 0; i < FrameCount; i++)
                _readback[i] = GraphicsBuffer.CreateReadback<Vector4>(1);
        }

        // Smooth value noise, -0.5..0.5
        private static double Noise(double x)
        {
            double i = Math.Floor(x), f = x - i;
            double u = f * f * (3.0 - 2.0 * f);
            return Lattice(i) + (Lattice(i + 1.0) - Lattice(i)) * u;
        }

        private static double Lattice(double i)
        {
            ulong h = unchecked((ulong)(long)i * 0x9E3779B97F4A7C15UL);
            h ^= h >> 29; h = unchecked(h * 0xBF58476D1CE4E5B9UL); h ^= h >> 32;
            return (h & 0xFFFFFF) / (double)0xFFFFFF - 0.5;
        }

        private static float SmoothStep(float a, float b, float x)
        {
            float t = Math.Clamp((x - a) / (b - a), 0f, 1f);
            return t * t * (3f - 2f * t);
        }

        /// <summary>For engine_stats.</summary>
        public static object Describe() => new
        {
            precipitation = Precipitation,
            rain = Rain,
            snow = Snow,
            hail = Hail,
            snowShare = SnowShare,
            hazeKm = HazeKm,        // the falling rain and snow's extinction per km (visibility 3.9 / it)
            // What the near emitters make of a heavy fall, which the far layers copy: drops per m³, fall speed (m/s)
            rainFall = new[] { RainDensity, RainFallSpeed },
            snowFall = new[] { SnowDensity, SnowFallSpeed },
            surfaces = new { wetness = SurfaceWetness, puddleFill = PuddleFill, snowCover = SnowCover, snowFallen = SnowFallen },
            // What the probe measured at the camera (not eased): the rate, hail, the cloud column's depth and top (m)
            column = new { rate = _column.X, hail = _column.Y, depth = _column.Z, top = _column.W },
            freezingLevel = Clouds.Settings.FreezingLevel,
            gust = new[] { Wind.Gust.X, Wind.Gust.Y },
            wind = new[] { Wind.Local.X, Wind.Local.Z },
            mapOrigin = new[] { MapOrigin.X, MapOrigin.Y },
        };
    }
}
