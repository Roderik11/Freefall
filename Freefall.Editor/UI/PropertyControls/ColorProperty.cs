using Squid;
using System.Numerics;
using Vortice.Mathematics;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(Color3))]
    public class ColorProperty : PropertyControl
    {
        public Color3 Color;
        public Button Button { get; private set; }

        private static ColorPicker _picker;

        public ColorProperty(GUIProperty property) : base(property)
        {
            Color = (Color3)property.GetValue();

            Button = new Button();
            Button.Size = new Point(20, 20);
            Button.Dock = DockStyle.Fill;
            Button.Style = "color";
            Button.Margin = new Margin(0, 1, 0, 1);
            Button.Tint = (int)new Color4(Color).ToRgba();
            Controls.Add(Button);

            Button.MouseClick += (sender, e) =>
            {
                _picker ??= new ColorPicker();
                _picker.ColorChanged -= OnPickerColorChanged;
                _picker.Color = new Color4(Color);
                _picker.ColorChanged += OnPickerColorChanged;
                _picker.Open(Button);
            };
        }

        private void OnPickerColorChanged(Color4 c)
        {
            Color = new Color3(c.R, c.G, c.B);
            property.SetValue(Color);
            Button.Tint = (int)new Color4(Color).ToRgba();
            NotifyChange();
        }
    }

    [PropertyControl(typeof(Vector4))]
    public class Color4Property : PropertyControl
    {
        public Color4 Color;
        public Button Button { get; private set; }

        private static ColorPicker _picker;

        public Color4Property(GUIProperty property) : base(property)
        {
            var vector = (Vector4)property.GetValue();
            Color = new Color4(vector);

            Button = new Button();
            Button.Size = new Point(20, 20);
            Button.Dock = DockStyle.Fill;
            Button.Style = "color";
            Button.Margin = new Margin(0, 1, 0, 1);
            Button.Tint = (int)Color.ToRgba();
            Controls.Add(Button);

            Button.MouseClick += (sender, e) =>
            {
                _picker ??= new ColorPicker();
                _picker.ColorChanged -= OnPickerColorChanged;
                _picker.Color = Color;
                _picker.ColorChanged += OnPickerColorChanged;
                _picker.Open(Button);
            };
        }

        private void OnPickerColorChanged(Color4 c)
        {
            Color = c;
            property.SetValue(new Vector4(c.R, c.G, c.B, c.A));
            Button.Tint = (int)Color.ToRgba();
            NotifyChange();
        }
    }
}
