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

        // The box is being set to the field's value: CheckedChanged fires, but that is no edit to write back.
        // Before 2026-10-16 it was, so a field a script reset (a bindstone's ImpulseNow) came back from the
        // refresh as a change, and the component rebuilt itself.
        private bool _showing;

        public BoolProperty(GUIProperty property) : base(property)
        {
            CheckButton = new Button();
            CheckButton.Dock = DockStyle.Left;
            CheckButton.Size = new Point(18, 24);
            CheckButton.CheckOnClick = true;
            CheckButton.Style = "checkbox";
            CheckButton.Margin = new Margin(0, 4, 4, 4);
            Controls.Add(CheckButton);

            Image = new ImageControl();
            Image.Dock = DockStyle.Fill;
            Image.Margin = new Margin(4);
            Image.Style = "checkmark";
            Image.NoEvents = true;
            CheckButton.GetElements().Add(Image);

            // Once: it was subscribed twice, and every click wrote the value (and notified the component) twice
            CheckButton.CheckedChanged += CheckBox_CheckedChanged;

            Show(property.GetValue());
            Image.Visible = CheckButton.Checked;
        }

        void Show(object value)
        {
            if (value is not bool on) return;
            _showing = true;
            CheckButton.Checked = on;
            _showing = false;
            Image.Visible = on;
        }

        void CheckBox_CheckedChanged(Control sender)
        {
            Image.Visible = CheckButton.Checked;
            if (_showing) return;
            property.SetValue(CheckButton.Checked);
        }

        protected override void OnUpdate()
        {
            Timer += Time.Delta;
            if (Timer > Interval)
            {
                Timer = 0;
                Show(property.GetValue());
            }
        }
    }
}
