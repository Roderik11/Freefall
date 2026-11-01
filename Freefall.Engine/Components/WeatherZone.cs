using System;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;
using Description = System.ComponentModel.DescriptionAttribute;

namespace Freefall.Components
{
    /// <summary>
    /// A place with a weather of its own: while the camera is in this volume, <see cref="Preset"/> takes over from the
    /// world's weather (EnvironmentController lays it over the climate), coming in over <see cref="Fade"/> metres from the
    /// faces and over <see cref="Seconds"/> at the least. Its box is the unit cube under the entity's transform and the
    /// size is the scale, as an <see cref="AudioVolume"/>'s. Overlapping zones are laid on in order of
    /// <see cref="Priority"/>: a higher one over a lower. What falls still comes from the clouds (Graphics.Weather), so
    /// a preset whose clouds rain or snow does so once they have gathered. From outside it shows too: the preset's clouds
    /// stand over the box, out to <see cref="CloudReach"/> past it (Graphics.Clouds), with a curtain of rain or snow
    /// under them if they give any.
    /// </summary>
    [Icon("icon_sky.png")]
    public class WeatherZone : Component, ISceneGizmo
    {
        [Description("The weather here")]
        public EnvironmentPreset Preset;

        [Description("Metres in from the box's faces over which the preset comes in")]
        public float Fade = 200f;

        [Description("Seconds it takes at the least to come in or go: the whole sky changes with it, so a ride through or a teleport does not whip it round")]
        public float Seconds = 15f;

        [Description("Where zones overlap, the higher is laid over the lower")]
        public int Priority;

        [Description("Metres past the box over which the preset's clouds fade into the sky round it, so the weather here is seen from afar. They need a few km to read as a sky. 0 = none: the zone only shows from inside (a cave)")]
        public float CloudReach = 3000f;

        /// <summary>How much of the preset shows now, 0..1.</summary>
        public float Weight { get; private set; }

        /// <summary>The share of the preset a place has: 0 outside the box, 1 past the Fade.</summary>
        internal float ShareAt(Vector3 place)
        {
            if (!Enabled || Preset == null || !Matrix4x4.Invert(Transform.WorldMatrix, out var toBox)) return 0f;
            var p = Vector3.Transform(place, toBox);
            var size = Transform.WorldScale;
            float depth = MathF.Min(MathF.Min((0.5f - MathF.Abs(p.X)) * size.X, (0.5f - MathF.Abs(p.Y)) * size.Y), (0.5f - MathF.Abs(p.Z)) * size.Z);
            if (depth <= 0f) return 0f;
            return Fade > 0f ? MathF.Min(depth / Fade, 1f) : 1f;
        }

        /// <summary>One frame toward the share of the camera's place. <paramref name="snap"/>: a scene's first frame.</summary>
        internal float Step(Vector3 place, float dt, bool snap)
        {
            float share = ShareAt(place);
            float step = Seconds > 0f ? dt / Seconds : 1f;
            Weight = snap ? share : share > Weight ? MathF.Min(Weight + step, share) : MathF.Max(Weight - step, share);
            return Weight;
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            // The gizmo matrix has no scale in it
            ctx.Color = new Color4(0.5f, 0.7f, 1f, 1f);
            ctx.DrawWireBox(Vector3.Zero, Transform.WorldScale);
        }
    }
}
