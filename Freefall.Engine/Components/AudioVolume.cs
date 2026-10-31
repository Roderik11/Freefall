using System;
using System.Numerics;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    /// <summary>
    /// A body of air shut off from the open: a complex under a hill, a cave. Its box is the unit cube under the entity's
    /// transform and the size is the scale, as a <see cref="Decal"/>'s. With the listener inside, sounds marked
    /// <see cref="AudioSource.Outdoors"/> are heard at <see cref="Outside"/>, coming down over <see cref="Fade"/> metres in
    /// from the faces, so the forest goes quiet as one walks in. Where boxes overlap the quietest counts. With
    /// <see cref="Subterranean"/> one rough box round a whole cave will do: the hill's slopes above it stay outdoors.
    /// </summary>
    [Icon("icon_audio.png")]
    public class AudioVolume : Component, ISceneGizmo
    {
        [System.ComponentModel.Description("How much of the outdoors is heard inside: 0 nothing, 1 all of it")]
        [ValueRange(0f, 1f)]
        public float Outside = 0f;

        [System.ComponentModel.Description("Metres in from the box's faces over which the outdoors fades to that")]
        public float Fade = 6f;

        [System.ComponentModel.Description("Counts only under the terrain's surface, so the box can be rough: the ground above and beside it stays outdoors. The Fade runs down from the surface too. A hole in the terrain does not lower the surface: a stair down through one fades like any other")]
        public bool Subterranean;

        /// <summary>The share of the outdoors heard at a place: 1 outside the box.</summary>
        internal float ShareAt(Vector3 place)
        {
            if (!Matrix4x4.Invert(Transform.WorldMatrix, out var toBox)) return 1f;
            var p = Vector3.Transform(place, toBox);
            var size = Transform.WorldScale;
            float depth = MathF.Min(MathF.Min((0.5f - MathF.Abs(p.X)) * size.X, (0.5f - MathF.Abs(p.Y)) * size.Y), (0.5f - MathF.Abs(p.Z)) * size.Z);
            if (depth <= 0f) return 1f;
            if (Subterranean)
            {
                var terrains = ComponentCache<TerrainRenderer>.All;
                if (terrains.Count > 0) depth = MathF.Min(depth, terrains[0].GetHeight(place) - place.Y);
                if (depth <= 0f) return 1f;
            }
            float inside = Fade > 0f ? MathF.Min(depth / Fade, 1f) : 1f;
            return 1f + (Outside - 1f) * inside;
        }

        public void DrawGizmos(GizmoContext ctx)
        {
            // The gizmo matrix has no scale in it
            ctx.Color = new Color4(0.5f, 1f, 0.6f, 1f);
            ctx.DrawWireBox(Vector3.Zero, Transform.WorldScale);
        }
    }
}
