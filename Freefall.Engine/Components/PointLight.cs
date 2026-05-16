using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    [Icon("icon_pointlight.png")]
    public class PointLight : Component, /* IDraw, */ ISceneGizmo
    {
        public Color3 Color = new Color3(1, 1, 1);
        public float Intensity = 1;
        public float Range = 10;

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

