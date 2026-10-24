using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    public class IntSliderProperty : PropertyControl
    {
        public ValueSlider Slider { get; private set; }

        public IntSliderProperty(GUIProperty property) : base(property)
        {
            Slider = new ValueSlider();
            Slider.Size = new Point(20, 20);
            Slider.Dock = DockStyle.Fill;
            Slider.Integer = true;
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
            property.SetValue(Convert.ToInt32(sender.Value));
        }

        // Shows what scripts, the game's options or the MCP set (Engine.Settings in the Settings tab too), without
        // writing it back (notify: false) and not while the user drags or types
        protected override void OnUpdate()
        {
            Timer += Time.Delta;
            if (Timer <= Interval || Slider.Interacting || property.HasMixedValue) return;
            Timer = 0;
            object value = property.GetValue();
            if (value != null)
                Slider.SetValue(Convert.ToSingle(value), notify: false);
        }
    }
}
