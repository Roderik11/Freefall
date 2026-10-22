using System;
using System.Numerics;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    /// <summary>
    /// A decal (DECALS_PLAN.md): its material projected onto the surfaces inside its box, the unit cube under the entity's
    /// transform. X and Z span the decal, Y is the projection axis: it faces +Y and lands on what lies below, so one on
    /// the ground needs no tilt and one on a wall is turned to face out of it. The size is the scale.
    ///
    /// It lives in a slot of the GPU decal registry (DecalBuffer): its settings are properties whose setters write that
    /// slot, the box comes from the entity's matrix in the transform buffer, and the decal pass (DecalPass,
    /// decals_cs.hlsl) applies every decal in view to the G-buffer right after the opaque pass. The look is the material
    /// (an effect "decal" .mat: textures, modes, opacities, fades, receivers); per decal only what differs.
    /// </summary>
    [Icon("icon_splatstamp.png")]
    public class Decal : Component, ISceneGizmo
    {
        private Material? _material;
        private Color3 _tint = new Color3(1, 1, 1);
        private float _opacity = 1f;
        private int _priority;
        private int _variant;

        /// <summary>Registry slot (DecalBuffer), -1 while not registered.</summary>
        internal int GpuSlot = -1;

        [System.ComponentModel.Description("The decal's look: a material of the effect \"decal\" (textures, modes, fades, what it lands on)")]
        public Material? Material
        {
            get => _material;
            set { if (_material == value) return; _material = value; Changed(); }
        }

        [System.ComponentModel.Description("Multiplies the material's colour, for this decal alone")]
        public Color3 Tint
        {
            get => _tint;
            set { if (_tint.Equals(value)) return; _tint = value; Changed(); }
        }

        [System.ComponentModel.Description("Multiplies every opacity of the material, for this decal alone (fade it in and out)")]
        [ValueRange(0f, 1f)]
        public float Opacity
        {
            get => _opacity;
            set { if (_opacity == value) return; _opacity = value; Changed(); }
        }

        [System.ComponentModel.Description("Overlapping decals are applied in ascending priority (0-15): a higher one lies on top. " +
                                           "Of the same priority, the one placed later lies on top")]
        [ValueRange(0, 15)]
        public int Priority
        {
            get => _priority;
            set { value = Math.Clamp(value, 0, 15); if (_priority == value) return; _priority = value; Changed(); }
        }

        [System.ComponentModel.Description("Tile of the material's atlas (AtlasColumns × AtlasRows): variety without more materials")]
        public int Variant
        {
            get => _variant;
            set { value = Math.Max(0, value); if (_variant == value) return; _variant = value; Changed(); }
        }

        private void Changed() => DecalBuffer.Instance?.Changed(this);

        protected override void OnEnabledChanged() => Changed();

        public void DrawGizmos(GizmoContext ctx)
        {
            // The box, and the way it projects (down its Y axis onto what lies below)
            ctx.Color = new Color4(0.4f, 0.8f, 1f, 1f);
            ctx.DrawWireBox(Vector3.Zero, Vector3.One);
            ctx.Color = new Color4(1f, 0.85f, 0.3f, 1f);
            ctx.DrawRay(new Vector3(0, 0.5f, 0), -Vector3.UnitY, 1f);
        }
    }
}
