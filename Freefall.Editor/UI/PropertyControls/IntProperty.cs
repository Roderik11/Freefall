using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(int))]
    public class IntProperty : PropertyControl
    {
        public TextBox Textbox { get; private set; }

        public IntProperty(GUIProperty property) : base(property)
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
            try
            {
                int v = Convert.ToInt32(Textbox.Text);
                int now = (int)property.GetValue();

                if (v != now)
                {
                    property.SetValue(v);
                    NotifyChange();
                }
            }
            catch { }
        }

        protected override void OnUpdate()
        {
            if (Desktop.FocusedControl == Textbox)
                return;

            Timer += Time.Delta;
            if (Timer > Interval)
            {
                Timer = 0;

                if (property.HasMixedValue)
                {
                    Textbox.Text = "---";
                    return;
                }

                int value = (int)property.GetValue();

                string v = value.ToString();
                if (v != Textbox.Text)
                {
                    Textbox.Text = v;
                }
            }
        }
    }
}
