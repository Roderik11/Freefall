using Squid;
using Vortice.Mathematics;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    /// <summary>
    /// Four whole numbers in a row: a rect (x, y, width, height) or borders (left, top, right,
    /// bottom) in pixels, e.g. a UI skin style's atlas piece.
    /// </summary>
    [PropertyControl(typeof(Int4))]
    public class Int4Property : PropertyControl
    {
        private const int Gap = 4;
        private readonly TextBox[] boxes = new TextBox[4];

        public Int4Property(GUIProperty property) : base(property)
        {
            for (int i = 0; i < 4; i++)
            {
                var box = new TextBox
                {
                    Size = new Point(20, 26),
                    Style = "textbox",
                    Mode = TextBoxMode.Numeric,
                    TabIndex = i + 1,
                };
                box.TextCommit += Textbox_OnTextCommit;
                boxes[i] = box;
                Controls.Add(box);
            }

            Show(property.GetValue());
        }

        protected override void OnLayout()
        {
            base.OnLayout();

            int w = (Size.x - 3 * Gap) / 4;
            for (int i = 0; i < 4; i++)
            {
                boxes[i].Position = new Point(i * (w + Gap), 0);
                boxes[i].Size = new Point(i == 3 ? Size.x - 3 * (w + Gap) : w, Size.y);
            }
        }

        private void Show(object value)
        {
            if (value is not Int4 v) return;
            boxes[0].Text = v.X.ToString();
            boxes[1].Text = v.Y.ToString();
            boxes[2].Text = v.Z.ToString();
            boxes[3].Text = v.W.ToString();
        }

        void Textbox_OnTextCommit(object sender, EventArgs e)
        {
            var parts = new int[4];
            for (int i = 0; i < 4; i++)
            {
                // Typed as a fraction: the nearest whole number
                if (!float.TryParse(boxes[i].Text, System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var f))
                    return;
                parts[i] = (int)MathF.Round(f);
            }

            var v = new Int4(parts[0], parts[1], parts[2], parts[3]);
            if (property.GetValue() is Int4 now && now.Equals(v)) return;

            property.SetValue(v);
            NotifyChange();
        }

        protected override void OnUpdate()
        {
            foreach (var box in boxes)
                if (Desktop.FocusedControl == box) return;

            Timer += Time.Delta;
            if (Timer <= Interval) return;
            Timer = 0;

            if (property.HasMixedValue)
            {
                foreach (var box in boxes) box.Text = "---";
                return;
            }

            Show(property.GetValue());
        }
    }
}
