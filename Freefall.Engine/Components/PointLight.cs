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
    ///
    /// Physical units (ATMOSPHERE_PLAN stage 3): <see cref="Lumens"/> is the luminous flux, sent equally in every
    /// direction, so a surface d metres away facing it gets lumens / (4π d²) lux, faded smoothly to nothing at
    /// <see cref="Range"/> (light_point_cs.hlsl). <see cref="Color"/> is only the hue (scaled to luminance 1 in the
    /// registry). The camera's exposure decides how bright that looks: a lantern lights
    /// the street at night and is lost in daylight with nothing switching it.
    /// </summary>
    [Icon("icon_pointlight.png")]
    public class PointLight : Component, ISceneGizmo
    {
        /// <summary>Illuminance (lux) at which <see cref="RangeFor"/> ends a light: a full-moon night gives ~0.07.</summary>
        public const float RangeCutoffLux = 0.1f;

        /// <summary>Metres at which a light of <paramref name="lumens"/> falls to <see cref="RangeCutoffLux"/>.</summary>
        public static float RangeFor(float lumens) => MathF.Sqrt(MathF.Max(lumens, 0f) / (4f * MathF.PI * RangeCutoffLux));

        private Color3 _color = new Color3(1, 1, 1);
        private float _lumens = 100;
        private float _range = 9;
        private float _flicker;
        private float _flickerSpeed = 8;
        private bool _onlyWhenDark;
        private float _innerRadius;

        /// <summary>Registry slot (PointLightBuffer), -1 while not registered.</summary>
        internal int GpuSlot = -1;

        public Color3 Color
        {
            get => _color;
            set { if (_color.Equals(value)) return; _color = value; Changed(); }
        }

        [System.ComponentModel.Description("Luminous flux in lumens: a candle ~12, an oil lantern ~100, a torch ~300, a " +
                                           "brazier or campfire 2000-5000. Lights a surface d metres away with lumens / (4π d²) lux.")]
        [ValueRange(0f, 20000f)]
        public float Lumens
        {
            get => _lumens;
            set { if (_lumens == value) return; _lumens = value; Changed(); }
        }

        [System.ComponentModel.Description("Metres to where the light ends (it fades smoothly to nothing there). Where it " +
                                           "falls to 0.1 lux: about 9 for 100 lm, 15 for 300, 49 for 3000 (PointLight.RangeFor).")]
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

        [System.ComponentModel.Description("Lit only when it is dark where it hangs: at night and at dusk, and by day in a " +
                                           "closed room or a cave that daylight does not reach. Street lamps, lanterns, " +
                                           "lamps in houses.")]
        public bool OnlyWhenDark
        {
            get => _onlyWhenDark;
            set { if (_onlyWhenDark == value) return; _onlyWhenDark = value; Changed(); }
        }

        [System.ComponentModel.Description("Metres round the light that get none of it: for a light that leaves its fitting " +
                                           "through glass (a lantern), so the fitting's own frame is not lit from within. " +
                                           "Full light from 1.5 times this. 0 = none.")]
        [ValueRange(0f, 2f)]
        public float InnerRadius
        {
            get => _innerRadius;
            set { if (_innerRadius == value) return; _innerRadius = value; Changed(); }
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
