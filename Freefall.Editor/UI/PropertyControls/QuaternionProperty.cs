using Squid;
using System;
using System.Numerics;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(Quaternion))]
    public class QuaternionProperty : PropertyControl
    {
        public TextBox Textbox1 { get; private set; }
        public TextBox Textbox2 { get; private set; }
        public TextBox Textbox3 { get; private set; }

        public QuaternionProperty(GUIProperty property) : base(property)
        {
            Textbox1 = new TextBox();
            Textbox1.Size = new Point(20, 26);
            Textbox1.Dock = DockStyle.Left;
            Textbox1.Style = "textbox";
            Textbox1.Margin = new Squid.Margin(0, 0, 4, 0);
            Textbox1.Mode = TextBoxMode.Numeric;

            Textbox2 = new TextBox();
            Textbox2.Size = new Point(20, 26);
            Textbox2.Dock = DockStyle.Fill;
            Textbox2.Style = "textbox";
            Textbox2.Margin = new Squid.Margin(0, 0, 0, 0);
            Textbox2.Mode = TextBoxMode.Numeric;

            Textbox3 = new TextBox();
            Textbox3.Size = new Point(20, 26);
            Textbox3.Dock = DockStyle.Right;
            Textbox3.Style = "textbox";
            Textbox3.Margin = new Squid.Margin(4, 0, 0, 0);
            Textbox3.Mode = TextBoxMode.Numeric;

            object value = property.GetValue();

            if (value != null)
            {
                Quaternion q = (Quaternion)value;
                Vector3 v = ToEuler(q);

                Textbox1.Text = v.X.ToString("0.###");
                Textbox2.Text = v.Y.ToString("0.###");
                Textbox3.Text = v.Z.ToString("0.###");
            }

            Textbox1.TextCommit += Textbox_OnTextCommit;
            Textbox2.TextCommit += Textbox_OnTextCommit;
            Textbox3.TextCommit += Textbox_OnTextCommit;

            Controls.Add(Textbox1);
            Controls.Add(Textbox3);
            Controls.Add(Textbox2);
        }

        protected override void OnLayout()
        {
            base.OnLayout();

            int w = (Textbox3.Location.x + Textbox3.Size.x - Textbox1.Location.x - 8) / 3;
            Textbox1.Size = new Point(w, Textbox1.Size.y);
            Textbox3.Size = new Point(w, Textbox3.Size.y);
        }

        void Textbox_OnTextCommit(object sender, EventArgs e)
        {
            try
            {
                Vector3 v = new Vector3(Convert.ToSingle(Textbox1.Text), Convert.ToSingle(Textbox2.Text), Convert.ToSingle(Textbox3.Text));

                object value = property.GetValue();
                Vector3 now = ToEuler((Quaternion)value);

                if (!v.Equals(now))
                {
                    Quaternion q = FromEuler(v);
                    property.SetValue(q);
                    NotifyChange();
                }
            }
            catch
            {
                // Ignore parse errors, user can fix them
            }
        }

        protected override void OnUpdate()
        {
            if (Desktop.FocusedControl == Textbox1) return;
            if (Desktop.FocusedControl == Textbox2) return;
            if (Desktop.FocusedControl == Textbox3) return;

            Timer += Time.Delta;

            if (Timer > Interval)
            {
                Timer = 0;

                if (property.HasMixedValue)
                {
                    Textbox1.Text = "---";
                    Textbox2.Text = "---";
                    Textbox3.Text = "---";
                    return;
                }

                object value = property.GetValue();

                if (value != null)
                {
                    Quaternion q = (Quaternion)value;
                    Vector3 v = ToEuler(q);

                    Textbox1.Text = v.X.ToString("0.###");
                    Textbox2.Text = v.Y.ToString("0.###");
                    Textbox3.Text = v.Z.ToString("0.###");
                }
            }
        }

        static Vector3 ToEuler(Quaternion q)
        {
            float deg = 180f / MathF.PI;
            float sinr_cosp = 2f * (q.W * q.X + q.Y * q.Z);
            float cosr_cosp = 1f - 2f * (q.X * q.X + q.Y * q.Y);
            float roll = MathF.Atan2(sinr_cosp, cosr_cosp) * deg;

            float sinp = 2f * (q.W * q.Y - q.Z * q.X);
            float pitch = MathF.Abs(sinp) >= 1f
                ? MathF.CopySign(90f, sinp)
                : MathF.Asin(sinp) * deg;

            float siny_cosp = 2f * (q.W * q.Z + q.X * q.Y);
            float cosy_cosp = 1f - 2f * (q.Y * q.Y + q.Z * q.Z);
            float yaw = MathF.Atan2(siny_cosp, cosy_cosp) * deg;

            return new Vector3(roll, pitch, yaw);
        }

        static Quaternion FromEuler(Vector3 euler)
        {
            float rad = MathF.PI / 180f;
            return Quaternion.CreateFromYawPitchRoll(euler.Y * rad, euler.X * rad, euler.Z * rad);
        }
    }
}
