using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    [Icon("icon_pointlight.png")]
    public class PointLight : Component, ISceneGizmo
    {
        public Color3 Color = new Color3(1, 1, 1);
        public float Intensity = 1;
        public float Range = 10;

        [System.ComponentModel.Description("How much the brightness wavers, like a flame (0 = steady)")]
        [ValueRange(0f, 1f)]
        public float Flicker = 0;

        [System.ComponentModel.Description("How fast it wavers")]
        [ValueRange(0.1f, 30f)]
        public float FlickerSpeed = 8;

        /// <summary>
        /// This frame's intensity and position of a flickering light. Value noise only (no trig): the
        /// renderer calls this for every light with Flicker > 0 while it uploads the light table.
        /// </summary>
        internal void ApplyFlicker(float time, ref float intensity, ref Vector3 position)
        {
            // Offset by the id so neighbouring lights do not waver in step
            float t = time * FlickerSpeed + (Id & 1023) * 0.37f;
            float wave = 0.65f * Noise(t) + 0.35f * Noise(t * 2.7f + 11.3f);
            intensity *= 1f + Flicker * (wave * 1.4f - 0.7f);

            // The flame also moves: a few centimetres are enough to make highlights shift
            float s = t * 0.6f;
            position += new Vector3(Noise(s + 5.1f) - 0.5f, Noise(s + 17.7f) - 0.5f, Noise(s + 31.3f) - 0.5f) * (0.15f * Flicker);
        }

        // 1D value noise, 0..1
        private static float Noise(float x)
        {
            float floor = MathF.Floor(x);
            int i = (int)floor;
            float f = x - floor;
            f = f * f * (3f - 2f * f);
            float a = Hash(i), b = Hash(i + 1);
            return a + (b - a) * f;
        }

        private static float Hash(int i)
        {
            uint state = (uint)i * 747796405u + 2891336453u;
            uint word = ((state >> (int)((state >> 28) + 4)) ^ state) * 277803737u;
            return ((word >> 22) ^ word) * (1f / 4294967295f);
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            // Static visualization: 3-axis wire sphere
            ctx.Color = new Color4(1f, 0.9f, 0.3f, 1f);
            //ctx.LineWidth = 0.02f;
            ctx.DrawWireSphere(Vector3.Zero, Range);

            // Interactive handle: drag circle edge to change range
            ctx.Color = new Color4(1f, 0.7f, 0.1f, 1f);
            ctx.LineWidth = 2f;
            Range = ctx.RadiusHandle(Vector3.Zero, Range);
        }

        /// <summary>
        /// Per-instance light data matches HLSL PointLightData struct.
        /// Uploaded by DeferredRenderer via ComponentCache&lt;PointLight&gt;.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 16)]
        public struct PointLightData
        {
            public Vector3 Color;
            public float Intensity;
            public Vector3 Position; // camera-relative
            public float Range;
        }
    }
}

