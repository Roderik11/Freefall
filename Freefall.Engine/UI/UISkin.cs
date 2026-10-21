using System;
using System.Collections.Generic;
using System.Numerics;
using Squid;
using Vortice.Mathematics;
using Freefall.Assets;
using Freefall.Graphics;
using Category = System.ComponentModel.CategoryAttribute;

namespace Freefall.UI
{
    /// <summary>
    /// How the game's UI looks: one atlas image and the named styles cut from it. A control picks a
    /// style by name (<c>Style = "window"</c>), as in the editor's skin; this is the same thing kept
    /// as an asset, so it is edited in the inspector and applied live while playing.
    /// </summary>
    [CreateAsset("UI Skin")]
    public class UISkin : Asset
    {
        /// <summary>The image every style cuts its pieces from.</summary>
        public Texture Atlas;

        /// <summary>
        /// How big the atlas is drawn: 0.5 = its pieces at half their pixel size (an atlas painted for
        /// 4K, on a 1080p screen). Rects and slices stay in the atlas's own pixels. Nine-slice corners
        /// keep their drawn size whatever the control's size, so this is what sets how heavy borders
        /// and ornaments look.
        /// </summary>
        [ValueRange(0.1f, 2f, 0.05f)]
        public float Scale = 1f;

        /// <summary>Text of styles that name no font of their own.</summary>
        public Typeface Font;

        [ValueRange(6, 96)]
        public int FontSize = 16;

        /// <summary>Pixel height of the text game code marks over the picture (Graphics.Overlay: a prompt), drawn in <see cref="Font"/>.</summary>
        [ValueRange(6, 96)]
        public int MarkFontSize = 18;

        public List<UIStyle> Styles = [];

        /// <summary>The renderer's name for the marks' font (<see cref="Font"/> at <see cref="MarkFontSize"/>), or null without a Font.</summary>
        public string MarkFontName(IGameUIRenderer renderer) => FontName(renderer, Font, MarkFontSize);

        /// <summary>
        /// Writes the styles into <paramref name="skin"/>. A style that is already there is changed in
        /// place, so controls that use it show the change without being built again.
        /// </summary>
        public void ApplyTo(Skin skin, IGameUIRenderer renderer)
        {
            // The renderer is told the atlas is Scale times its size; every rect and slice is then
            // given in those units, and Squid draws them at that size
            float scale = Scale > 0f ? Scale : 1f;
            string atlas = null;
            if (Atlas != null)
            {
                atlas = "uiskin:" + (Guid ?? Name) + ":atlas";
                renderer?.InsertTexture(atlas, Atlas, scale);
            }

            string defaultFont = FontName(renderer, Font, FontSize);

            foreach (var entry in Styles)
            {
                if (entry == null || string.IsNullOrEmpty(entry.Name)) continue;

                if (!skin.TryGetValue(entry.Name, out var style))
                {
                    style = new ControlStyle();
                    skin[entry.Name] = style;
                }

                entry.WriteTo(style, atlas, scale, FontName(renderer, entry.Font ?? Font, entry.FontSize > 0 ? entry.FontSize : FontSize) ?? defaultFont);
            }
        }

        private static string FontName(IGameUIRenderer renderer, Typeface face, int size)
        {
            if (face == null || renderer == null) return null;
            string name = "uifont:" + (face.Guid ?? face.Name) + ":" + size;
            renderer.RegisterFont(name, face, size);
            return name;
        }
    }

    /// <summary>
    /// One named look in a <see cref="UISkin"/>: a piece of the atlas, how it fills the control, and
    /// the text drawn on it, with changes for a control under the mouse, pressed or selected (an
    /// open tab, a chosen slot). A state field left at zero (or a colour with alpha 0) keeps the
    /// normal look. Colours are RGBA in a Vector4 (the engine's colour type with alpha; it is what
    /// the serializer and the inspector's colour picker know).
    /// </summary>
    [Serializable]
    public class UIStyle
    {
        /// <summary>What controls name to use it.</summary>
        public string Name = string.Empty;

        /// <summary>The piece of the atlas, in its pixels: x, y, width, height. Width 0 = no picture.</summary>
        [Category("Picture")]
        public Int4 Rect;

        /// <summary>
        /// How the piece fills the control. Grid: nine-slice, corners keep their size (see Slice);
        /// Stretch: the whole piece scaled; Center: at its own size in the middle; Repeat*: tiled.
        /// </summary>
        public TextureMode Tiling = TextureMode.Grid;

        /// <summary>Grid only: the border widths that keep their size, left, top, right, bottom (atlas pixels).</summary>
        public Int4 Slice;

        /// <summary>Multiplies the picture.</summary>
        public Vector4 Tint = Vector4.One;

        /// <summary>A flat fill under the picture (alpha 0 = none).</summary>
        public Vector4 Fill;

        [Category("Text")]
        public Typeface Font;

        /// <summary>Pixel height of the text; 0 = the skin's.</summary>
        [ValueRange(0, 96)]
        public int FontSize;

        public Vector4 TextColor = new(0.86f, 0.82f, 0.74f, 1f);

        public Alignment TextAlign = Alignment.MiddleLeft;

        /// <summary>Room between the edge and the text in screen pixels: left, top, right, bottom.</summary>
        public Int4 TextPadding;

        [Category("Under the mouse")]
        public Int4 HotRect;
        public Vector4 HotTint;
        public Vector4 HotTextColor;

        [Category("Pressed")]
        public Int4 PressedRect;
        public Vector4 PressedTint;
        public Vector4 PressedTextColor;

        [Category("Selected")]
        public Int4 SelectedRect;
        public Vector4 SelectedTint;
        public Vector4 SelectedTextColor;

        /// <param name="scale">The skin's Scale: atlas pixels to drawn pixels.</param>
        internal void WriteTo(ControlStyle style, string atlas, float scale, string font)
        {
            bool picture = atlas != null && Rect.Z > 0 && Rect.W > 0;

            // Every state first (the ControlStyle setters write all of them), then what differs
            style.Texture = picture ? atlas : string.Empty;
            style.TextureRect = ToRect(Rect, scale);
            style.Tiling = Tiling;
            style.Grid = ToMargin(Slice, scale);
            style.Tint = ToInt(Tint);
            style.BackColor = ToInt(Fill);
            style.TextColor = ToInt(TextColor);
            style.TextAlign = TextAlign;
            style.TextPadding = ToMargin(TextPadding, 1f);
            style.Font = font ?? "default";

            State(HotRect, HotTint, HotTextColor, picture, scale, style.Hot);
            State(PressedRect, PressedTint, PressedTextColor, picture, scale, style.Pressed);

            // Selected and checked are the same thing to a game: the open tab, the chosen slot
            foreach (var selected in new[] { style.Selected, style.SelectedHot, style.SelectedPressed, style.SelectedFocused,
                                             style.Checked, style.CheckedHot, style.CheckedPressed, style.CheckedFocused })
                State(SelectedRect, SelectedTint, SelectedTextColor, picture, scale, selected);
        }

        private static void State(Int4 rect, Vector4 tint, Vector4 text, bool picture, float scale, Style state)
        {
            if (picture && rect.Z > 0 && rect.W > 0) state.TextureRect = ToRect(rect, scale);
            if (tint.W > 0) state.Tint = ToInt(tint);
            if (text.W > 0) state.TextColor = ToInt(text);
        }

        private static Squid.Rectangle ToRect(Int4 r, float scale)
        {
            int left = (int)MathF.Round(r.X * scale), top = (int)MathF.Round(r.Y * scale);
            int right = (int)MathF.Round((r.X + r.Z) * scale), bottom = (int)MathF.Round((r.Y + r.W) * scale);
            return new(left, top, right - left, bottom - top);
        }

        private static Margin ToMargin(Int4 m, float scale)
            => new((int)MathF.Round(m.X * scale), (int)MathF.Round(m.Y * scale), (int)MathF.Round(m.Z * scale), (int)MathF.Round(m.W * scale));

        /// <summary>RGBA (x, y, z, w) to Squid's ARGB int.</summary>
        private static int ToInt(Vector4 c)
            => ColorInt.ARGB(c.W, c.X, c.Y, c.Z);
    }
}
