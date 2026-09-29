using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(double))]
    public class DoubleProperty : PropertyControl
    {
        public TextBox Textbox { get; private set; }

        public DoubleProperty(GUIProperty property) : base(property)
        {
            Textbox = new TextBox();
            Textbox.Size = new Point(20, 20);
            Textbox.Dock = DockStyle.Fill;
            Textbox.Style = "textbox";
            Textbox.Mode = TextBoxMode.Numeric;
            Textbox.TextCommit += HandleTextboxOnTextCommit;

            Controls.Add(Textbox);
        }

        void HandleTextboxOnTextCommit(object sender, EventArgs e)
        {
            double v = Convert.ToDouble(Textbox.Text);
            double now = (double)property.GetValue();

            if (v != now)
            {
                property.SetValue(v);
                NotifyChange();
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

                double value = (double)property.GetValue();

                string v = value.ToString();
                if (v != Textbox.Text)
                    Textbox.Text = v;
            }
        }
    }
}
