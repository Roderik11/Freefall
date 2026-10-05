using System;
using System.Globalization;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Numeric field that is also its own slider: the value sits on a bar filled in proportion to
    /// the range. Drag left/right to scrub, click without dragging to type a value.
    /// </summary>
    public class ValueSlider : Frame
    {
        private const int DragThreshold = 3;

        private readonly TextBox editor;
        private float value;
        private bool pressed;
        private bool dragging;
        private int pressX;

        public float Minimum = 0;
        public float Maximum = 1;

        /// <summary>Whole numbers only.</summary>
        public bool Integer;

        public event Action<ValueSlider> ValueChanged;

        public float Value => value;

        public ValueSlider()
        {
            NoEvents = false;
            Style = "";
            Size = new Point(60, 20);
            Cursor = Cursors.SizeWE;

            editor = new TextBox
            {
                Dock = DockStyle.Fill,
                Style = "textbox",
                Mode = TextBoxMode.Numeric,
                Visible = false,
            };
            editor.TextCommit += (s, e) => EndEdit(apply: true);
            editor.TextCancel += (s, e) => EndEdit(apply: false);
            editor.LostFocus += s => EndEdit(apply: true);
            Controls.Add(editor);

            MouseDown += (s, e) =>
            {
                if (e.Button != 0) return;
                pressed = true;
                dragging = false;
                pressX = Gui.MousePosition.x;
            };

            MousePress += (s, e) =>
            {
                if (!pressed) return;
                if (!dragging && Math.Abs(Gui.MousePosition.x - pressX) < DragThreshold) return;

                dragging = true;
                float t = (Gui.MousePosition.x - Location.x) / (float)Math.Max(1, Size.x);
                SetValue(Minimum + Math.Clamp(t, 0f, 1f) * (Maximum - Minimum));
            };

            MouseUp += (s, e) =>
            {
                if (e.Button != 0 || !pressed) return;
                pressed = false;

                if (!dragging) BeginEdit();
                dragging = false;
            };
        }

        /// <summary>Set the value, clamped to the range and rounded to the display precision.</summary>
        public void SetValue(float newValue, bool notify = true)
        {
            newValue = Math.Clamp(newValue, Minimum, Maximum);
            newValue = Integer ? MathF.Round(newValue) : MathF.Round(newValue, Decimals);

            if (newValue == value) return;
            value = newValue;
            if (notify) ValueChanged?.Invoke(this);
        }

        // Enough decimals for ~1000 steps across the range: 3 for 0..1, 1 for 0..100, 0 beyond
        private int Decimals
        {
            get
            {
                float range = MathF.Abs(Maximum - Minimum);
                if (range <= 0) return 3;
                return Math.Clamp(3 - (int)MathF.Floor(MathF.Log10(range)), 0, 4);
            }
        }

        private string Format(float v)
        {
            return Integer ? v.ToString("0", CultureInfo.InvariantCulture) : v.ToString("0.####", CultureInfo.InvariantCulture);
        }

        private void BeginEdit()
        {
            editor.Text = Format(value);
            editor.Visible = true;
            editor.Focus();
            editor.SelectAll();
        }

        private void EndEdit(bool apply)
        {
            if (!editor.Visible) return;
            editor.Visible = false;

            if (apply && float.TryParse(editor.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float typed))
                SetValue(typed);
        }

        protected override void DrawStyle(Style style, float opacity)
        {
            if (opacity == 0 || editor.Visible) return;

            int x = Location.x, y = Location.y, w = Size.x, h = Size.y;
            bool hot = dragging || State == ControlState.Hot || State == ControlState.Pressed;

            LandingArt.Slice(hot ? EditorSkin.FieldHotArt : EditorSkin.FieldArt, x, y, w, h, EditorSkin.ArtInset, -1);

            float range = Maximum - Minimum;
            float t = range > 0 ? Math.Clamp((value - Minimum) / range, 0f, 1f) : 0f;
            int fill = (int)MathF.Round((w - 2) * t);
            if (fill > 0)
            {
                LandingArt.Slice(EditorSkin.ButtonArt, x + 1, y + 1, fill, h - 2, EditorSkin.ArtInset,
                    dragging ? ColorInt.ARGB(1f, .30f, .38f, .52f) : hot ? ColorInt.ARGB(1f, .25f, .31f, .41f) : ColorInt.ARGB(1f, .20f, .25f, .33f));
            }

            // Handle at the value, always visible so a slider at its minimum doesn't pass for a text field
            int handle = Math.Clamp(x + 1 + fill - 1, x + 3, x + w - 5);
            Gui.Renderer.DrawBox(handle, y + 3, 2, h - 6, hot ? LandingArt.Coral : ColorInt.ARGB(1f, .42f, .50f, .64f));

            string text = Format(value);
            int font = Gui.Renderer.GetFont("roboto_regular_10");
            int textHeight = Gui.Renderer.GetTextSize(text, font).y;
            Gui.Renderer.DrawText(text, x + 7, y + (h - textHeight) / 2, font, ColorInt.ARGB(1f, .88f, .88f, .88f));
        }
    }
}
