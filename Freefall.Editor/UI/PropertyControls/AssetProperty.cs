using Squid;
using Freefall.Reflection;
using Freefall.Assets;
using Freefall.Base;
using Point = Squid.Point;

namespace Freefall.Editor
{
    [PropertyControl(typeof(Asset))]
    public class AssetProperty : PropertyControl
    {
        public Button Button { get; private set; }
        private readonly ImageControl thumbnail;

        private Button ClearButton;

        public AssetProperty(GUIProperty property) : base(property)
        {
            RowHeight = 68;

            var asset = property.GetValue() as Asset;

            Button = new Button
            {
                Size = new Point(26, 20),
                Dock = DockStyle.Fill,
                Style = "textbox",
                Text = asset != null ? asset.Name : "- None -",
                AllowDrop = true,
            };

            Button.DragResponse += Button_DragResponse;
            Button.DragDrop += Button_DragDrop;
            Button.MouseDoubleClick += Button_MouseDoubleClick;

            ClearButton = new Button
            {
                Margin = new Margin(2, 0, 0, 0),
                Size = new Point(32, 32),
                Dock = DockStyle.Right,
                Style = "close",
                Text = "",
            };

            ClearButton.MouseClick += (s, e) =>
            {
                property.SetValue(null);
                thumbnail.Texture = AssetDatabase.GetThumbnail(string.Empty);

                Button.Text = "- None -";
            };

            //Controls.Add(ClearButton);
           // Controls.Add(Button);

            Frame frame = new Frame { Dock = DockStyle.Top, Size = new Point(20, 26) };
            frame.Controls.Add(ClearButton);
            frame.Controls.Add(Button);

            var shadow = new Frame
            {
                Size = new Point(60, 60),
                Style = "dropshadow",
                Dock = DockStyle.Left,
                Margin = new Margin(0, 0, 6, 0),
                Padding = new Margin(0, 0, 1, 1)
            };
            Controls.Add(shadow);

            shadow.GetElements().Add(new Frame
            {
                Style = "border",
                Dock = DockStyle.Fill,
                Margin = new Margin(0, 0, 1, 1)
            });

            thumbnail = new ImageControl
            {
                Size = new Point(62, 62),
                Dock = DockStyle.Fill,
                Texture = AssetDatabase.GetThumbnail(asset)
            };
            shadow.Controls.Add(thumbnail);

            thumbnail.AllowDrop = true;
            thumbnail.DragResponse += Button_DragResponse;
            thumbnail.DragDrop += Button_DragDrop;
            Controls.Add(frame);
        }

        private bool IsDropCompatible(DragDropEventArgs e)
        {
            if (e.DraggedControl?.Tag is not AssetDragData data) return false;
            return property.Type.IsAssignableFrom(data.AssetType);
        }

        private void Button_DragResponse(Control sender, DragDropEventArgs e)
        {
                        
            if(IsDropCompatible(e))
                Button.State = ControlState.Hot;
        }

        private void Button_DragDrop(Control sender, DragDropEventArgs e)
        {
            Button.TextColor = -1;

            if (e.DraggedControl?.Tag is not AssetDragData data) return;
            if (!property.Type.IsAssignableFrom(data.AssetType)) return;

            var asset = Engine.Assets.LoadByGuid(data.Guid, data.AssetType) as Asset;
            if (asset != null)
            {
                property.SetValue(asset);
                Button.Text = asset.Name;
                thumbnail.Texture = AssetDatabase.GetThumbnail(asset);
            }
        }

        private void Button_MouseDoubleClick(Control sender, MouseEventArgs e)
        {
            var asset = property.GetValue();
            if (asset != null)
                Selector.SelectedObject = asset;
        }

        protected override void OnUpdate()
        {
            Timer += Time.Delta;
            if (Timer > Interval)
            {
                Timer = 0;
                var asset = property.GetValue() as Asset;
                Button.Text = asset != null ? asset.Name : "- None -";
            }
        }
    }
}
