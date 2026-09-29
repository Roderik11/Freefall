using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;
    public class LabelProperty : PropertyControl
    {
        public Label Label { get; private set; }

        public LabelProperty(GUIProperty property) : base(property)
        {
            Label = new Label();
            Label.Size = new Point(20, 20);
            Label.Dock = DockStyle.Fill;

            object value = property.GetValue();

            if (value != null)
                Label.Text = value.ToString();

            Controls.Add(Label);
        }

        protected override void OnUpdate()
        {
            Timer += Time.Delta;

            if (Timer > Interval)
            {
                Timer = 0;

                object value = property.GetValue();

                if (value != null)
                {
                    string v = value.ToString();
                    if (v != Label.Text)
                    {
                        Label.Text = v;
                    }
                }
            }
        }
    }
}
