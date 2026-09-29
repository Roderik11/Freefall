using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(bool))]
    public class BoolProperty : PropertyControl
    {
        public Button CheckButton { get; private set; }
        public ImageControl Image { get; private set; }

        public BoolProperty(GUIProperty property) : base(property)
        {
            CheckButton = new Button();
            CheckButton.Dock = DockStyle.Left;
            CheckButton.Size = new Point(18, 24);
            CheckButton.CheckOnClick = true;
            CheckButton.CheckedChanged += CheckBox_CheckedChanged;
            CheckButton.Style = "checkbox";
            CheckButton.Margin = new Margin(0, 4, 4, 4);
            Controls.Add(CheckButton);

            Image = new ImageControl();
            Image.Dock = DockStyle.Fill;
            Image.Margin = new Margin(4);
            Image.Style = "checkmark";
            Image.NoEvents = true;
            CheckButton.GetElements().Add(Image);

            CheckButton.CheckedChanged += CheckBox_CheckedChanged;

            object value = property.GetValue();

            if (value != null)
                CheckButton.Checked = (bool)value;

            Image.Visible = CheckButton.Checked;
        }

        void CheckBox_CheckedChanged(Control sender)
        {
            property.SetValue(CheckButton.Checked);
            Image.Visible = CheckButton.Checked;
        }

        protected override void OnUpdate()
        {
            Timer += Time.Delta;
            if (Timer > Interval)
            {
                Timer = 0;

                object value = property.GetValue();

                if (value != null)
                    CheckButton.Checked = (bool)value;
            }
        }
    }
}
