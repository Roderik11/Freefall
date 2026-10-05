using Squid;
using Freefall.Reflection;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    public class DoubleSliderProperty : PropertyControl
    {
        public ValueSlider Slider { get; private set; }

        public DoubleSliderProperty(GUIProperty property) : base(property)
        {
            Slider = new ValueSlider();
            Slider.Size = new Point(20, 20);
            Slider.Dock = DockStyle.Fill;
            Slider.Integer = false;
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
                Slider.SetValue(Convert.ToSingle(value), notify: false);

            Slider.ValueChanged += Slider_OnValueChanged;

            Controls.Add(Slider);
        }

        void Slider_OnValueChanged(ValueSlider sender)
        {
            property.SetValue((double)sender.Value);
        }
    }
}
