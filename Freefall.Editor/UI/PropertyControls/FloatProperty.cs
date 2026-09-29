using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(float))]
    public class FloatProperty : PropertyControl
    {
        public TextBox Textbox { get; private set; }

        public FloatProperty(GUIProperty property) : base(property)
        {
            Textbox = new TextBox();
            Textbox.Size = new Point(20, 20);
            Textbox.Dock = DockStyle.Fill;
            Textbox.Style = "textbox";
            Textbox.TextCommit += HandleTextboxOnTextCommit;
            Textbox.Mode = TextBoxMode.Numeric;
            Controls.Add(Textbox);
        }

        void HandleTextboxOnTextCommit(object sender, EventArgs e)
        {
            try
            {
                Convert.ToSingle(Textbox.Text);
                float v = Convert.ToSingle(Textbox.Text);
                float now = (float)property.GetValue();

                if (v != now)
                {
                    property.SetValue(v);
                    NotifyChange();
                }
            }
            catch
            {
                // Revert to current value if parse fails
                Textbox.Text = ((float)property.GetValue()).ToString();
            }
        }

        protected override void OnUpdate()
        {
            if (Desktop.FocusedControl == Textbox)
                return;

            Timer += Time.Delta;

            if (Timer > Interval)
            {
                Timer = 0;

                float value = (float)property.GetValue();

                if (property.HasMixedValue)
                {
                    Textbox.Text = "---";
                    return;
                }

                string v = value.ToString();
                if (v != Textbox.Text)
                    Textbox.Text = v;
            }
        }
    }
}
