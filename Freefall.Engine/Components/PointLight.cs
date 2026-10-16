using System;
using System.Numerics;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    /// <summary>
    /// A point light. It lives in a slot of the GPU light registry (PointLightBuffer): its settings are properties
    /// whose setters write that slot, so nothing reads them per frame; the position comes from the entity's
    /// transform on the GPU and the flicker is computed there (point_lights.hlsli).
    /// </summary>
    [Icon("icon_pointlight.png")]
    public class PointLight : Component, ISceneGizmo
    {
        private Color3 _color = new Color3(1, 1, 1);
        private float _intensity = 1;
        private float _range = 10;
        private float _flicker;
        private float _flickerSpeed = 8;
        private bool _onlyWhenDark;

        /// <summary>Registry slot (PointLightBuffer), -1 while not registered.</summary>
        internal int GpuSlot = -1;

        public Color3 Color
        {
            get => _color;
            set { if (_color.Equals(value)) return; _color = value; Changed(); }
        }

        public float Intensity
        {
            get => _intensity;
            set { if (_intensity == value) return; _intensity = value; Changed(); }
        }

        public float Range
        {
            get => _range;
            set { if (_range == value) return; _range = value; Changed(); }
        }

        [System.ComponentModel.Description("How much the brightness wavers, like a flame (0 = steady)")]
        [ValueRange(0f, 1f)]
        public float Flicker
        {
            get => _flicker;
            set { if (_flicker == value) return; _flicker = value; Changed(); }
        }

        [System.ComponentModel.Description("How fast it wavers")]
        [ValueRange(0.1f, 30f)]
        public float FlickerSpeed
        {
            get => _flickerSpeed;
            set { if (_flickerSpeed == value) return; _flickerSpeed = value; Changed(); }
        }

        [System.ComponentModel.Description("Lit only when it is dark (Daylight.Lamps): at night, in a thunderstorm, on an " +
                                           "overcast dusk. Street lamps and lanterns; lights indoors stay on.")]
        public bool OnlyWhenDark
        {
            get => _onlyWhenDark;
            set { if (_onlyWhenDark == value) return; _onlyWhenDark = value; Changed(); }
        }

        private void Changed() => PointLightBuffer.Instance?.Changed(this);

        protected override void OnEnabledChanged() => Changed();

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
    }
}
