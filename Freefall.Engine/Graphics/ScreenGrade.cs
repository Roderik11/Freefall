using System.Numerics;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// The last thing the picture goes through, for what happens to whoever is looking: blinded by a flash, the sight
    /// closing in, frost over the eyes, a daze. A game sets these while it wants them (an effect on the player, a blow
    /// to the head) and Clear puts them back; at rest the finalize pass only passes a few branches.
    /// </summary>
    public static class ScreenGrade
    {
        /// <summary>The light is times this before the tone curve: above 1 the picture burns out as a dazzled eye's does, below 1 it sinks into the dark.</summary>
        public static float Exposure = 1f;

        /// <summary>The whole picture goes this share of the way to Tint, after the tone curve.</summary>
        public static float TintAmount;
        public static Color3 Tint = new(1f, 1f, 1f);

        /// <summary>The sight closes in from the edges of the screen: 0 not at all, 1 shut. It closes to VignetteColor.</summary>
        public static float Vignette;
        public static Color3 VignetteColor = new(0f, 0f, 0f);

        /// <summary>How far the colour goes out of the picture, 0 to 1.</summary>
        public static float Desaturate;

        /// <summary>How far the picture is smeared, as a share of the screen's height (0.01 is a soft daze).</summary>
        public static float Blur;

        /// <summary>
        /// A picture over the screen (frost on the eyes), its colour painted as it is to show, not linear: import it with
        /// sRGB off. Its alpha says how soon a pixel is covered as OverlayAmount rises from 0 to 1 (1 = at once), so it can
        /// creep in from wherever it is painted brightest. Null = none.
        /// </summary>
        public static Texture? Overlay;
        public static float OverlayAmount;

        /// <summary>Anything set, for whoever only wants to spend time when it is.</summary>
        public static bool Active => Exposure != 1f || TintAmount > 0f || Vignette > 0f || Desaturate > 0f || Blur > 0f || (Overlay != null && OverlayAmount > 0f);

        /// <summary>Back to the plain picture.</summary>
        public static void Clear()
        {
            Exposure = 1f;
            TintAmount = Vignette = Desaturate = Blur = OverlayAmount = 0f;
            Overlay = null;
        }

        // A colour and an amount in one push constant: linear 8 bits each is plenty for a tint
        internal static uint Pack(Color3 color, float amount)
        {
            static uint Byte(float v) => (uint)(System.Math.Clamp(v, 0f, 1f) * 255f + 0.5f);
            return Byte(color.R) | Byte(color.G) << 8 | Byte(color.B) << 16 | Byte(amount) << 24;
        }
    }
}
