using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(string))]
    public class StringProperty : PropertyControl
    {
        public TextBox Textbox { get; private set; }

        public StringProperty(GUIProperty property) : base(property)
        {
            Textbox = new TextBox();
            Textbox.Size = new Point(20, 20);
            Textbox.Dock = DockStyle.Fill;
            Textbox.Style = "textbox";
            Textbox.TextCommit += HandleTextboxOnTextCommit;

            object value = property.GetValue();

            if (value != null)
                Textbox.Text = value.ToString();

            Controls.Add(Textbox);

            if (property.GetAttribute<FilePathAttribute>() != null)
            {
                var btn = new Button
                {
                    Style = "catbutton",
                    Size = new Point(32, 32),
                    Margin = new Margin(1),
                    Dock = DockStyle.Right,
                    Text = "...",
                };
                btn.MouseClick += (sender, args) =>
                {
                    var dialog = new System.Windows.Forms.OpenFileDialog
                    {
                        InitialDirectory =  Path.GetDirectoryName((string)property.GetValue()) ?? "",
                    };
                    if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                    {
                        property.SetValue(dialog.FileName);
                        NotifyChange();
                    }
                };
                Controls.Add(btn);
                btn.BringToBack();
            }
        }

        void HandleTextboxOnTextCommit(object sender, EventArgs e)
        {
            try
            {
                string now = (string)property.GetValue();

                if (Textbox.Text != now)
                {
                    property.SetValue(Textbox.Text);
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

                object value = property.GetValue();

                if (value != null)
                {
                    string v = value.ToString();
                    if (v != Textbox.Text)
                    {
                        Textbox.Text = v;
                    }
                }
            }
        }
    }
}
