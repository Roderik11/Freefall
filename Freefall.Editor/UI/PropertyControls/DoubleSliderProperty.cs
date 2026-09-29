using Squid;
using Freefall.Reflection;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    public class DoubleSliderProperty : PropertyControl
    {
        public TextBox Textbox { get; private set; }
        public Slider Slider { get; private set; }

        public DoubleSliderProperty(GUIProperty property) : base(property)
        {
            Textbox = new TextBox();
            Textbox.Size = new Point(40, 20);
            Textbox.Dock = DockStyle.Left;
            Textbox.Style = "textbox";
            Textbox.Mode = TextBoxMode.Numeric;
            Controls.Add(Textbox);

            Slider = new Slider();
            Slider.Margin = new Margin(4, 0, 0, 0);
            Slider.Size = new Point(20, 20);
            Slider.Orientation = Orientation.Horizontal;
            Slider.Dock = DockStyle.Fill;
            Slider.Button.Style = "sliderThumb";
            Slider.Button.Size = new Point(16, 16);
            Slider.Style = "sliderTrack";
            Slider.Minimum = -100;
            Slider.Maximum = 100;

            ValueRangeAttribute range = property.GetAttribute<ValueRangeAttribute>();
            if (range != null)
            {
                Slider.Minimum = range.Min;
                Slider.Maximum = range.Max;
            }

            object value = property.GetValue();

            if (value != null)
            {
                var rawValue = Convert.ToSingle(value);
                Slider.SetValue(rawValue);
                Textbox.Text = Slider.Value.ToString();
            }

            Slider.ValueChanged += Slider_OnValueChanged;

            Textbox.TextCommit += (sender, e) =>
            {
                if (float.TryParse(Textbox.Text, out float newValue))
                {
                    if (newValue < Slider.Minimum)
                        newValue = Slider.Minimum;
                    else if (newValue > Slider.Maximum)
                        newValue = Slider.Maximum;

                    Slider.SetValue(newValue);
                }
                else
                {
                    Textbox.Text = Slider.Value.ToString();
                }
            };

            Controls.Add(Slider);
        }

        void Slider_OnValueChanged(Control sender)
        {
            property.SetValue(Slider.Value);
            Textbox.Text = Slider.Value.ToString();
        }
    }
}
