using System;
using System.Numerics;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// The terrain's shadow in the air, one map per light (the sun, the brightest moon): over the terrain and a margin
    /// round it, the height below which the terrain hides the light (terrain_light_shadow_cs.hlsl). The aerial perspective
    /// volume and the near fog volume read it (atmosphere_common.hlsli AtmTerrainLightVisibility), so a mountain the sun
    /// stands behind casts its shadow into the haze in front of it, and shafts of light pass its ridges; the ground and
    /// water past the shadow cascades take it too (sky_common.fx TerrainLightShadow).
    ///
    /// On a terrain whose map would be coarser than <see cref="WindowTexel"/> (16 km at 1024² is 20 m a texel) a second map
    /// per light, the window, covers <see cref="WindowSpan"/> round the main camera at that size (TERRAIN_PAGES_PLAN.md
    /// stage 6, 2026-11-08). Its march reads the whole map where a step leaves the window, so a mountain beyond it still
    /// shades what is in it; readers take the window where it reaches and fade to the whole map over its outer sixteenth
    /// (atmosphere_common.hlsli AtmTerrainShadowTop). It follows the camera in steps of an eighth of its span, its texels on
    /// a grid fixed in the world, so a step moves nothing that stays inside.
    ///
    /// A map is made again only when its light has moved (<see cref="RebuildCos"/>), the terrain was baked again
    /// (TerrainBaker.HeightVersion) or (the window) the camera left it: 1 + <see cref="Steps"/> dispatches over
    /// <see cref="Resolution"/>² texels. <see cref="Prepare"/> (from Atmosphere.PrepareFrame) decides and returns the
    /// indices, <see cref="Render"/> (from Atmosphere.Render, before the aerial perspective) makes what is due. Docs:
    /// .agent/knowledge/atmospheric_effects/physical_atmosphere.md §13.
    /// </summary>
    internal static class TerrainLightShadow
    {
        private const int Resolution = 1024;
        private const float Margin = 2048f;         // metres round the terrain the maps reach (the sea in its shadow)
        private const int Steps = 11;               // reach 2^11 texels: across the map's diagonal
        private const float RebuildCos = 0.99999985f;   // ~0.03° of the light's movement

        /// <summary>Metres a window texel spans; a terrain whose whole map is not more than a quarter coarser has none.</summary>
        private const float WindowTexel = 8f;
        /// <summary>Metres the window spans each way.</summary>
        private const float WindowSpan = WindowTexel * Resolution;

        private sealed class Map
        {
            public RenderTexture2D? A, B;
            public ResourceStates StateA = ResourceStates.Common, StateB = ResourceStates.Common;
            public bool Valid;              // made for Direction and Version
            public bool Due;                // to be made in this frame's Render
            public int Settle;              // frames to make it again after a bake (the bake may land after it)
            public Vector3 Direction;
            public int Version = -1;        // TerrainBaker.HeightVersion it was made for: every bake and upload moves it
            public Vector2 Origin;          // the window's corner it was made for (the whole map's is the terrain's)

            // The first step writes A, each doubling the other one
            public uint Result => Valid ? (Steps % 2 == 0 ? A!.BindlessIndex : B!.BindlessIndex) : 0u;
        }

        private static ComputeShader? _cs;
        private static int _kInit, _kStep;
        private static readonly Map _sun = new(), _moon = new(), _sunWindow = new(), _moonWindow = new();
        private static (uint Pages, float MaxHeight, Vector3 Origin, Vector2 Size) _terrain;
        private static Vector2 _origin;
        private static float _size;
        private static Vector2 _windowOrigin = new(float.NaN);
        private static bool _windowed;

        private const ResourceStates ShaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;

        /// <summary>
        /// Decide which maps this frame needs made and return what the atmosphere's parameters carry: the sun's and the
        /// moon's whole map (0 = none: no terrain, the light below the horizon), its corner and size, and the same for the
        /// window round the main camera (0 = none: a terrain small enough for the whole map to be as fine).
        /// </summary>
        public static (uint Sun, uint Moon, Vector2 Origin, float Size, uint SunWindow, uint MoonWindow, Vector2 WindowOrigin, float WindowSize) Prepare(
            (uint Pages, float MaxHeight, Vector3 Origin, Vector2 Size) terrain, Vector3 sunDir, Vector3 moonDir, bool moonLit)
        {
            _terrain = terrain;
            if (terrain.Pages == 0)
                return default;
            _size = MathF.Max(terrain.Size.X, terrain.Size.Y) + 2f * Margin;
            _origin = new Vector2(terrain.Origin.X, terrain.Origin.Z) - new Vector2(Margin);

            uint sun = PrepareMap(_sun, sunDir, null);
            uint moon = moonLit ? PrepareMap(_moon, moonDir, null) : 0u;

            // The window: on a terrain whose whole map is coarser than a window texel by more than a quarter, round the main
            // camera, its corner on the world's grid of window texels, moved once the camera is an eighth of its span off
            // its middle
            _windowed = _size / Resolution > WindowTexel * 1.25f && Components.Camera.Main != null;
            if (!_windowed)
                return (sun, moon, _origin, _size, 0, 0, Vector2.Zero, 0f);
            var camera = Components.Camera.Main!.Position;
            var middle = new Vector2(camera.X, camera.Z);
            if (float.IsNaN(_windowOrigin.X) || Vector2.Distance(middle, _windowOrigin + new Vector2(WindowSpan * 0.5f)) > WindowSpan / 8f)
            {
                var corner = middle - new Vector2(WindowSpan * 0.5f);
                _windowOrigin = new Vector2(MathF.Round(corner.X / WindowTexel), MathF.Round(corner.Y / WindowTexel)) * WindowTexel;
            }
            uint sunWindow = sun != 0 ? PrepareMap(_sunWindow, sunDir, _sun) : 0u;
            uint moonWindow = moon != 0 ? PrepareMap(_moonWindow, moonDir, _moon) : 0u;
            return (sun, moon, _origin, _size, sunWindow, moonWindow, _windowOrigin, WindowSpan);
        }

        // A map for this light: due when the light moved, the terrain changed, (a window) it moved or its whole map is due
        private static uint PrepareMap(Map map, Vector3 dir, Map? whole)
        {
            dir = Vector3.Normalize(dir);
            if (dir.Y <= 0f)
            {
                map.Due = false;
                return 0;
            }
            int version = Assets.TerrainBaker.HeightVersion;
            if (version != map.Version)
                map.Settle = 2;
            // The terrain's pages index changes every frame (one params buffer per frame in flight): a new terrain or new
            // heights show as a new HeightVersion instead
            bool moved = !map.Valid || Vector3.Dot(dir, map.Direction) < RebuildCos;
            bool shifted = whole != null && (map.Origin != _windowOrigin || whole.Due);
            map.Due = moved || shifted || map.Settle > 0;
            if (map.Due)
            {
                EnsureTextures(map);
                map.Direction = dir;
                map.Version = version;
                map.Valid = true;           // made in Render before anything reads it
                if (whole != null) map.Origin = _windowOrigin;
                if (map.Settle > 0) map.Settle--;
            }
            return map.Result;
        }

        private static void EnsureTextures(Map map)
        {
            if (map.A != null) return;
            var device = Engine.Device;
            // Heights in metres: half floats hold the terrain's 600 m to half a metre
            map.A = new RenderTexture2D(device, Resolution, Resolution, Format.R16_Float, randomWrite: true);
            map.B = new RenderTexture2D(device, Resolution, Resolution, Format.R16_Float, randomWrite: true);
            map.StateA = map.StateB = ResourceStates.Common;
        }

        /// <summary>Make the maps <see cref="Prepare"/> found due, the whole maps before the windows that read them. Compute
        /// root signature and descriptor heap bound.</summary>
        public static void Render(ID3D12GraphicsCommandList list)
        {
            if (!_sun.Due && !_moon.Due && !_sunWindow.Due && !_moonWindow.Due) return;
            if (_cs == null)
            {
                _cs = new ComputeShader("terrain_light_shadow_cs.hlsl");
                _kInit = _cs.FindKernel("CSShadowInit");
                _kStep = _cs.FindKernel("CSShadowStep");
            }
            PixMarker.Begin(list, "Terrain light shadow");
            if (_sun.Due) Make(list, _sun, _origin, _size, null);
            if (_moon.Due) Make(list, _moon, _origin, _size, null);
            if (_windowed)
            {
                if (_sunWindow.Due) Make(list, _sunWindow, _windowOrigin, WindowSpan, _sun);
                if (_moonWindow.Due) Make(list, _moonWindow, _windowOrigin, WindowSpan, _moon);
            }
            PixMarker.End(list);
        }

        private static void Make(ID3D12GraphicsCommandList list, Map map, Vector2 origin, float size, Map? whole)
        {
            var cs = _cs!;
            var dir = map.Direction;
            var across = new Vector2(dir.X, dir.Z);
            float horizontal = across.Length();
            // A light straight overhead casts no shadow across the ground: the first step finds nothing but the ground below
            var dirXZ = horizontal > 1e-4f ? across / horizontal : new Vector2(1f, 0f);
            float tan = horizontal > 1e-4f ? dir.Y / horizontal : 1e4f;

            // Params: the same for every dispatch of the frame (one copy per frame); the map, the light and the reach are
            // pushed (a frame can make the whole map and the window)
            cs.SetPushConstant("TerrainPages", _terrain.Pages);
            cs.SetParam("WholeOrigin", _origin);
            cs.SetParam("WholeSize", _size);
            cs.SetParam("MapResolution", (uint)Resolution);
            cs.SetParam("TerrainOrigin", _terrain.Origin);
            cs.SetParam("TerrainMaxHeight", _terrain.MaxHeight);
            cs.SetParam("TerrainSize", _terrain.Size);
            cs.SetPushConstant("MapOriginXBits", BitConverter.SingleToUInt32Bits(origin.X));
            cs.SetPushConstant("MapOriginZBits", BitConverter.SingleToUInt32Bits(origin.Y));
            cs.SetPushConstant("MapSizeBits", BitConverter.SingleToUInt32Bits(size));
            cs.SetPushConstant("WholeTex", whole?.Result ?? 0u);
            cs.SetPushConstant("LightDirXBits", BitConverter.SingleToUInt32Bits(dirXZ.X));
            cs.SetPushConstant("LightDirZBits", BitConverter.SingleToUInt32Bits(dirXZ.Y));
            cs.SetPushConstant("LightTanBits", BitConverter.SingleToUInt32Bits(tan));

            uint groups = (uint)(Resolution + 7) / 8;
            // First step into A, then each doubling from one to the other
            Transition(list, map.A!, ref map.StateA, ResourceStates.UnorderedAccess);
            cs.SetPushConstant("TargetUAV", map.A!.UavIndex);
            cs.Dispatch(_kInit, list, groups, groups);
            bool inA = true;
            for (int k = 0; k < Steps; k++)
            {
                var source = inA ? map.A! : map.B!;
                var target = inA ? map.B! : map.A!;
                if (inA)
                {
                    Transition(list, map.A!, ref map.StateA, ResourceStates.NonPixelShaderResource);
                    Transition(list, map.B!, ref map.StateB, ResourceStates.UnorderedAccess);
                }
                else
                {
                    Transition(list, map.B!, ref map.StateB, ResourceStates.NonPixelShaderResource);
                    Transition(list, map.A!, ref map.StateA, ResourceStates.UnorderedAccess);
                }
                cs.SetPushConstant("SourceTex", source.BindlessIndex);
                cs.SetPushConstant("TargetUAV", target.UavIndex);
                cs.SetPushConstant("StepTexelsBits", BitConverter.SingleToUInt32Bits(MathF.Pow(2f, k)));
                cs.Dispatch(_kStep, list, groups, groups);
                inA = !inA;
            }
            if (inA) Transition(list, map.A!, ref map.StateA, ShaderRead);
            else Transition(list, map.B!, ref map.StateB, ShaderRead);
            map.Due = false;
        }

        private static void Transition(ID3D12GraphicsCommandList list, RenderTexture2D texture, ref ResourceStates state,
            ResourceStates to)
        {
            if (state == to) return;
            list.ResourceBarrierTransition(texture.Native, state, to);
            state = to;
        }
    }
}
