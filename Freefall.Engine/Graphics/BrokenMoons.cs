using System;
using System.Numerics;
using Freefall.Assets;
using Vortice.Direct3D12;
using Vortice.DXGI;

namespace Freefall.Graphics
{
    /// <summary>
    /// Broken moons (<see cref="Moon.Broken"/>): one side taken apart by a glancing blow, its crust broken into plates
    /// carried off along the blow, the core showing, cracks, chunks, grit and dust. Ray-marched by moon_broken_cs.hlsl
    /// into a picture of the moon's disc frame (the rectangle that holds the moon and everything that left it), which the
    /// sky draws in place of the disc (mesh_skybox.fx MoonsAndStars, <c>MoonBrokenTex</c> / <c>MoonBrokenRect</c>).
    ///
    /// The march costs ~20 ms for a picture of 750k texels, so it is made again only when needed: all at once when
    /// the moon's settings or its size on screen changed, a band of rows a frame (<see cref="TexelsPerFrame"/>) when the
    /// sun has moved across it by more than <see cref="RelightCos"/>. The picture is sized to the moon on screen
    /// (<see cref="Prepare"/>'s pixels per moon radius), at most <see cref="MaxSize"/> a side. The plates' layout comes
    /// from <see cref="Moon.Seed"/> on the CPU (<see cref="Layout"/>), so a texel only marches.
    ///
    /// <see cref="Prepare"/> (from SkyboxRenderer.UpdateMoons) decides and returns what the sky shows, <see cref="Render"/>
    /// (from Atmosphere.Render, before the sky is drawn) makes what is due. Docs:
    /// .agent/knowledge/atmospheric_effects/sky_and_environment.md §8 "Broken moons".
    /// </summary>
    internal static class BrokenMoons
    {
        private const int MaxSize = 2048;
        private const int TexelsPerFrame = 24_000;
        private const float RelightCos = 0.99998f;      // ~0.36° of the sun's movement across the moon
        private const int FrameCount = 3;

        // Keep in step with moon_broken_cs.hlsl
        private const int Header = 10;
        private const int MaxSeeds = 64;
        private const int MaxChips = 48;
        private const int Ring = 12;
        private const int DataSize = Header + 3 * MaxSeeds + MaxChips;

        private const ResourceStates ShaderRead = ResourceStates.NonPixelShaderResource | ResourceStates.PixelShaderResource;

        private sealed class Bake
        {
            public RenderTexture2D? Texture;
            public ResourceStates State = ResourceStates.Common;
            public int Width, Height;
            public float PixelsPerRadius;
            public int LayoutHash;
            public Vector4 Rect;                         // corner xy, size zw: disc units, y up
            public float MipTexels;                      // the moon's textures' texels per moon radius
            public readonly Vector4[] Data = new Vector4[DataSize];
            public Vector3 Sun;                          // the sun across the moon the picture was (or is being) made for
            public bool Made;                            // the picture is whole: shown
            public int NextRow = -1;                     // the next band to make, -1: nothing due
            public readonly GraphicsBuffer?[] Upload = new GraphicsBuffer?[FrameCount];
        }

        private static readonly Bake?[] _bakes = new Bake?[Components.SkyboxRenderer.MaxMoons];
        private static ComputeShader? _cs;
        private static int _kernel;
        private static int _renderedTick = -1;

        /// <summary>
        /// What the sky shows for moon <paramref name="index"/>: its picture's bindless index (0 = draw the disc whole) and
        /// the rectangle it covers. <paramref name="sun"/> is the sun's direction in the moon's disc frame (x right, y up,
        /// z away from the viewer); <paramref name="pixelsPerRadius"/> how many pixels a moon radius spans on screen;
        /// <paramref name="up"/> whether it can be seen (well below the horizon its light is not kept up to date).
        /// </summary>
        public static (uint Texture, Vector4 Rect) Prepare(int index, Moon moon, Vector3 sun, float pixelsPerRadius, bool up)
        {
            if (!moon.Broken)
            {
                Release(index);
                return (0u, Vector4.Zero);
            }
            var bake = _bakes[index] ??= new Bake();

            int hash = LayoutHashOf(moon);
            bool full = false;
            if (hash != bake.LayoutHash || !bake.Made && bake.NextRow < 0)
            {
                Layout(moon, bake);
                bake.LayoutHash = hash;
                full = true;
            }

            // Sized to the moon on screen: made again when that changed by a quarter
            float ppr = Math.Clamp(pixelsPerRadius, 16f, 4096f);
            if (bake.Texture == null || MathF.Abs(ppr / bake.PixelsPerRadius - 1f) > 0.25f || full)
            {
                float scale = MathF.Min(ppr, MathF.Min(MaxSize / bake.Rect.Z, MaxSize / bake.Rect.W));
                int width = Math.Clamp((int)MathF.Ceiling(bake.Rect.Z * scale), 16, MaxSize);
                int height = Math.Clamp((int)MathF.Ceiling(bake.Rect.W * scale), 16, MaxSize);
                if (bake.Texture == null || width != bake.Width || height != bake.Height)
                {
                    if (bake.Texture != null)
                        Engine.Device.DeferDispose(bake.Texture);
                    bake.Texture = new RenderTexture2D(Engine.Device, width, height, Format.R16G16B16A16_Float, randomWrite: true);
                    bake.State = ResourceStates.Common;
                    bake.Width = width;
                    bake.Height = height;
                    full = true;
                }
                bake.PixelsPerRadius = ppr;
            }

            if (full)
            {
                // All of it this frame, before the sky draws it
                bake.Sun = sun;
                bake.NextRow = 0;
                bake.Made = true;
                WriteLight(moon, bake);
            }
            else if (up && bake.NextRow < 0 && Vector3.Dot(sun, bake.Sun) < RelightCos)
            {
                // The sun has moved across it: made again a band a frame, over the old picture (the change is too
                // small to show a seam)
                bake.Sun = sun;
                bake.NextRow = 0;
                WriteLight(moon, bake);
            }
            return (bake.Made ? bake.Texture!.BindlessIndex : 0u, bake.Rect);
        }

        /// <summary>Make the bands due this frame. Compute root signature and descriptor heap bound.</summary>
        public static void Render(ID3D12GraphicsCommandList list)
        {
            if (_renderedTick == Engine.TickCount)
                return;
            _renderedTick = Engine.TickCount;

            bool any = false;
            foreach (var bake in _bakes)
                any |= bake != null && bake.NextRow >= 0;
            if (!any)
                return;

            if (_cs == null)
            {
                _cs = new ComputeShader("moon_broken_cs.hlsl");
                _kernel = _cs.FindKernel("CSBakeMoon");
            }
            PixMarker.Begin(list, "Broken moons");
            int slot = Engine.FrameIndex % FrameCount;
            foreach (var bake in _bakes)
            {
                if (bake == null || bake.NextRow < 0 || bake.Texture == null)
                    continue;

                // A whole picture at once the first time (bake.Made was set with it), bands after
                int rows = bake.NextRow == 0 && bake.State == ResourceStates.Common
                    ? bake.Height
                    : Math.Max(8, TexelsPerFrame / bake.Width / 8 * 8);
                rows = Math.Min(rows, bake.Height - bake.NextRow);

                var upload = bake.Upload[slot] ??= GraphicsBuffer.CreateUpload<Vector4>(DataSize, mapped: true);
                unsafe
                {
                    var dst = upload.WritePtr<Vector4>();
                    for (int i = 0; i < DataSize; i++)
                        dst[i] = bake.Data[i];
                }

                if (bake.State != ResourceStates.UnorderedAccess)
                {
                    list.ResourceBarrierTransition(bake.Texture.Native, bake.State, ResourceStates.UnorderedAccess);
                    bake.State = ResourceStates.UnorderedAccess;
                }
                _cs.SetPushConstant(_kernel, "Data", upload.SrvIndex);
                _cs.SetPushConstant(_kernel, "TargetUAV", bake.Texture.UavIndex);
                _cs.SetPushConstant(_kernel, "RowStart", (uint)bake.NextRow);
                _cs.SetPushConstant(_kernel, "TargetWidth", (uint)bake.Width);
                _cs.SetPushConstant(_kernel, "TargetHeight", (uint)bake.Height);
                _cs.Dispatch(_kernel, list, (uint)(bake.Width + 7) / 8, (uint)(rows + 7) / 8);
                list.ResourceBarrierTransition(bake.Texture.Native, ResourceStates.UnorderedAccess, ShaderRead);
                bake.State = ShaderRead;

                bake.NextRow += rows;
                if (bake.NextRow >= bake.Height)
                    bake.NextRow = -1;
            }
            PixMarker.End(list);
        }

        /// <summary>Forget moon <paramref name="index"/>'s picture (not broken any more, or the sky is gone).</summary>
        public static void Release(int index)
        {
            var bake = _bakes[index];
            if (bake == null) return;
            if (bake.Texture != null)
                Engine.Device?.DeferDispose(bake.Texture);
            foreach (var upload in bake.Upload)
                if (upload != null)
                    Engine.Device?.DeferDispose(upload);
            _bakes[index] = null;
        }

        public static void ReleaseAll()
        {
            for (int i = 0; i < _bakes.Length; i++)
                Release(i);
        }

        // ── Layout ─────────────────────────────────────

        private static int LayoutHashOf(Moon moon)
        {
            var h = new HashCode();
            h.Add(moon.BreakBearing); h.Add(moon.BreakTilt); h.Add(moon.BreakSize);
            h.Add(moon.BlowBearing); h.Add(moon.BlowTilt);
            h.Add(moon.Plates); h.Add(moon.Spread); h.Add(moon.CrustThickness); h.Add(moon.Chunks);
            h.Add(moon.Cracks); h.Add(moon.Dust); h.Add(moon.Grit);
            h.Add(moon.InnerColor); h.Add(moon.CoreColor); h.Add(moon.DustColor); h.Add(moon.Seed);
            h.Add(moon.Texture?.BindlessIndex ?? 0u); h.Add(moon.NormalMap?.BindlessIndex ?? 0u);
            return h.ToHashCode();
        }

        /// <summary>A direction in the disc frame from degrees round the disc and towards the viewer.</summary>
        private static Vector3 DiscDirection(float bearingDeg, float tiltDeg)
        {
            float b = bearingDeg * (MathF.PI / 180f), t = tiltDeg * (MathF.PI / 180f);
            return Vector3.Normalize(new Vector3(MathF.Cos(b) * MathF.Cos(t), MathF.Sin(b) * MathF.Cos(t), -MathF.Sin(t)));
        }

        /// <summary>
        /// The seeds of the plates and where the blow put them, the chunks, the rectangle that holds it all, and the
        /// header the bake reads. Seeds in the cap lie denser towards its middle, so the plates round its edge come out
        /// bigger; the middle is carried furthest, the edge only lifts and cracks open.
        /// </summary>
        private static void Layout(Moon moon, Bake bake)
        {
            var rng = new Random(moon.Seed * 7919 + 17);
            float R() => (float)rng.NextDouble();

            Vector3 breakDir = DiscDirection(moon.BreakBearing, moon.BreakTilt);
            Vector3 blowDir = DiscDirection(moon.BlowBearing, moon.BlowTilt);
            float cap = moon.BreakSize * (MathF.PI / 180f);
            float thick = moon.CrustThickness;
            int capCount = Math.Clamp(moon.Plates, 1, MaxSeeds - Ring - 1);
            int seedCount = capCount + Ring + 1;
            int chipCount = Math.Clamp(moon.Chunks, 0, MaxChips);

            Vector3 t1 = Vector3.Normalize(Vector3.Cross(breakDir, MathF.Abs(breakDir.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitX));
            Vector3 t2 = Vector3.Cross(breakDir, t1);
            Span<Vector4> seed = stackalloc Vector4[seedCount];
            Span<Vector4> move = stackalloc Vector4[seedCount];
            Span<Vector4> axis = stackalloc Vector4[seedCount];
            for (int k = 0; k < seedCount; k++)
            {
                float angle, bearing, carried = 0f;
                if (k < capCount)
                {
                    float u = (k + R()) / capCount;
                    angle = cap * MathF.Pow(u, 0.75f);
                    bearing = k * 2.39996f + (R() - 0.5f) * 2.2f;
                    float rim = Math.Clamp(1f - angle / cap, 0f, 1f);
                    carried = MathF.Pow(rim, 1.2f) * (0.25f + 1.3f * R());
                }
                else if (k < capCount + Ring)
                {
                    angle = cap + 0.28f + (R() - 0.5f) * 0.12f;
                    bearing = (k - capCount + 0.5f) / Ring * MathF.Tau + (R() - 0.5f) * 0.3f;
                }
                else
                {
                    angle = MathF.PI;
                    bearing = 0f;
                }
                Vector3 dir = Vector3.Normalize(breakDir * MathF.Cos(angle) + (t1 * MathF.Cos(bearing) + t2 * MathF.Sin(bearing)) * MathF.Sin(angle));
                seed[k] = new Vector4(dir, carried);

                // Out from the surface and along the blow, more the further it was carried; turned as it tumbles
                var jitter = new Vector3(R(), R(), R()) - new Vector3(0.5f);
                Vector3 offset = carried * moon.Spread * (Vector3.Normalize(dir * 0.55f + blowDir) + jitter * 0.5f) + dir * carried * 0.08f;
                move[k] = new Vector4(offset, carried * 2.2f * (R() - 0.5f));
                var turn = new Vector3(R(), R(), R()) - new Vector3(0.5f);
                axis[k] = new Vector4(Vector3.Normalize(turn.LengthSquared() > 1e-6f ? turn : Vector3.UnitX), 0f);
            }

            // Each cap cell's reach (the plate's bound): the furthest of a few thousand directions that fall in it
            var dirs = new Vector3[seedCount];
            for (int k = 0; k < seedCount; k++)
                dirs[k] = new Vector3(seed[k].X, seed[k].Y, seed[k].Z);
            var reach = new float[seedCount];
            var sample = new Random(12345);
            for (int s = 0; s < 6000; s++)
            {
                // Uniform over the cap and a margin round it
                float cosMax = MathF.Cos(MathF.Min(cap + 0.6f, MathF.PI));
                float z = 1f - (float)sample.NextDouble() * (1f - cosMax);
                float phi = (float)sample.NextDouble() * MathF.Tau;
                float r = MathF.Sqrt(MathF.Max(0f, 1f - z * z));
                Vector3 dir = breakDir * z + (t1 * MathF.Cos(phi) + t2 * MathF.Sin(phi)) * r;
                int best = 0;
                float bestDot = -2f;
                for (int k = 0; k < seedCount; k++)
                {
                    float dd = Vector3.Dot(dir, dirs[k]);
                    if (dd > bestDot) { bestDot = dd; best = k; }
                }
                reach[best] = MathF.Max(reach[best], MathF.Acos(Math.Clamp(bestDot, -1f, 1f)));
            }
            for (int k = 0; k < seedCount; k++)
                axis[k].W = reach[k] * 1.08f + 0.02f;

            // Chunks: out of the break and along the blow, scattering wider the further they went
            Vector3 e1 = Vector3.Cross(blowDir, Vector3.UnitZ);
            e1 = e1.LengthSquared() > 1e-6f ? Vector3.Normalize(e1) : Vector3.UnitX;
            Vector3 e2 = Vector3.Cross(blowDir, e1);
            Span<Vector4> chips = stackalloc Vector4[Math.Max(chipCount, 1)];
            for (int c = 0; c < chipCount; c++)
            {
                float u = (c + R()) / chipCount;
                float along = 0.3f + moon.Spread * 1.55f * MathF.Pow(u, 0.85f);
                float wide = 0.15f + 0.55f * u;
                float a = R(), b = R(), size = R();
                Vector3 at = breakDir * 1.05f + blowDir * along + (e1 * (a - 0.5f) + e2 * (b - 0.5f)) * 2f * wide;
                chips[c] = new Vector4(at, 0.11f * (1f - 0.6f * u) * (0.35f + 0.9f * size * b));
            }

            // The rectangle: the moon, the moved plates, the chunks, and the dust and grit along the blow
            float dustReach = moon.Spread * 1.6f + 1f;
            Vector2 min = new(-1.05f), max = new(1.05f);
            void Grow(Vector2 p, float r) { min = Vector2.Min(min, p - new Vector2(r)); max = Vector2.Max(max, p + new Vector2(r)); }
            for (int k = 0; k < capCount; k++)
            {
                if (seed[k].W < 0.02f) continue;
                Vector3 c = new Vector3(seed[k].X, seed[k].Y, seed[k].Z) * (1f - thick * 0.5f) + new Vector3(move[k].X, move[k].Y, move[k].Z);
                Grow(new Vector2(c.X, c.Y), axis[k].W + thick);
            }
            for (int c = 0; c < chipCount; c++)
                Grow(new Vector2(chips[c].X, chips[c].Y), chips[c].W);
            if (moon.Dust > 0f || moon.Grit > 0f)
            {
                Vector2 b = new Vector2(breakDir.X, breakDir.Y) * 0.92f;
                Vector2 a = new Vector2(blowDir.X, blowDir.Y);
                a = a.LengthSquared() > 1e-8f ? Vector2.Normalize(a) : Vector2.UnitX;
                Vector2 across = new(-a.Y, a.X);
                for (float s = -0.6f; s <= dustReach + 1e-3f; s += dustReach / 8f)
                {
                    float w = 1.2f * (0.45f + 0.55f * MathF.Max(s, 0f));
                    Grow(b + a * s + across * w, 0f);
                    Grow(b + a * s - across * w, 0f);
                }
                Grow(b, 0.6f);
            }
            // A little margin for the fade at the edges
            min -= new Vector2(0.05f);
            max += new Vector2(0.05f);
            bake.Rect = new Vector4(min, max.X - min.X, max.Y - min.Y);

            var d = bake.Data;
            Array.Clear(d);
            d[1] = bake.Rect;
            d[2] = new Vector4(thick, moon.Cracks, moon.Dust, moon.Grit);
            d[3] = new Vector4(moon.InnerColor.R, moon.InnerColor.G, moon.InnerColor.B, 0f);
            d[4] = new Vector4(moon.CoreColor.R, moon.CoreColor.G, moon.CoreColor.B, 0f);
            d[5] = new Vector4(moon.DustColor.R, moon.DustColor.G, moon.DustColor.B, 0.25f + 0.35f * cap);
            d[6] = new Vector4(breakDir, cap);
            d[7] = new Vector4(blowDir, dustReach);
            d[8] = new Vector4(seedCount, capCount, chipCount, BitConverter.UInt32BitsToSingle(moon.Texture?.BindlessIndex ?? 0u));
            // The textures cover the disc (two radii) with their width; a texel of the picture spans Rect.Z / width
            float texWidth = moon.Texture != null ? moon.Texture.Native.Description.Width : 1024f;
            d[9] = new Vector4(BitConverter.UInt32BitsToSingle(moon.NormalMap?.BindlessIndex ?? 0u), 0f, 0f, 0f);
            bake.MipTexels = texWidth * 0.5f;
            for (int k = 0; k < seedCount; k++)
            {
                d[Header + 3 * k] = seed[k];
                d[Header + 3 * k + 1] = move[k];
                d[Header + 3 * k + 2] = axis[k];
            }
            for (int c = 0; c < chipCount; c++)
                d[Header + 3 * seedCount + c] = chips[c];
        }

        /// <summary>The header's light: the sun across the moon, the earthshine, the dust's light; and the textures' mip.</summary>
        private static void WriteLight(Moon moon, Bake bake)
        {
            var sun = bake.Sun;
            // Earthshine: the dark side lit by the world (full behind a crescent, none behind a full moon), as the disc has it
            bake.Data[0] = new Vector4(sun, 0.005f * (1f + sun.Z));
            // Dust: the part of a grain the sun lights as seen from here, and the glow of dust lit from behind
            bake.Data[3].W = 0.5f * (1f - sun.Z) + 0.8f * MathF.Pow(MathF.Max(sun.Z, 0f), 6f);
            float texelsPerTexel = bake.MipTexels * bake.Rect.Z / Math.Max(bake.Width, 1);
            bake.Data[9].Y = MathF.Max(0f, MathF.Log2(MathF.Max(texelsPerTexel, 1f)));
        }
    }
}
