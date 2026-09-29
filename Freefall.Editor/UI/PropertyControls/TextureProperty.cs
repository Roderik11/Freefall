using Freefall.Assets;
using Freefall.Editor;
using Freefall.Graphics;
using Freefall.Reflection;
using Squid;

namespace Freefall.Editor
{
    using Point = Squid.Point;

   // [PropertyControl(typeof(Texture))]
    public class TextureProperty : PropertyControl
    {
        public Button Button { get; private set; }

        public TextureProperty(GUIProperty property) : base(property)
        {
            RowHeight = 64;
            //AutoHeight = true;

            var asset = property.GetValue() as Asset;

            var btn = new Button
            {
                Size = new Point(16, 16),
                Dock = DockStyle.Right,
                Style = "textbox",
                Text = "o"
            };

            Button = new Button
            {
                Size = new Point(20, 20),
                Dock = DockStyle.Fill,
                Style = "textbox",
                Text = asset != null ? asset.Name : "- None -"
            };

            btn.MouseClick += Button_MouseClick;
            Button.MouseDoubleClick += (s, e) =>
            {
                var value = property.GetValue() as Asset;
                Selector.SelectedObject = value;
            };

            Frame frame = new Frame { Dock = DockStyle.Top, Size = new Point(20, 26) };
            frame.Controls.Add(btn);
            frame.Controls.Add(Button);

            var shadow = new Frame
            {
                Size = new Point(64, 64),
                Style = "dropshadow",
                Dock = DockStyle.Left,
                Margin = new Margin(1),
                Padding = new Margin(0, 0, 1, 1)
            };
            Controls.Add(shadow);

            shadow.GetElements().Add(new Frame
            {
                Style = "border",
                Dock = DockStyle.Fill,
                Margin = new Margin(0, 0, 1, 1)
            });

            var img = new ImageControl
            {
                Size = new Point(62, 62),
                Dock = DockStyle.Fill,
                Texture = AssetDatabase.GetThumbnail(asset)
            };
            shadow.Controls.Add(img);

            Controls.Add(frame);
        }

        private void Button_MouseClick(Control sender, MouseEventArgs args)
        {
          //  AssetBrowser.Open(Desktop, property.Type);
          //  AssetBrowser.SelectAsset(property.GetValue() as IAsset);
        }
    }
}
