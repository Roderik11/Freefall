using Squid;
using Freefall.Reflection;
using Freefall.Assets;
using Freefall.Base;
using Point = Squid.Point;

namespace Freefall.Editor
{
    [PropertyControl(typeof(Entity))]
    public class EntityProperty : PropertyControl
    {
        public Button Button { get; private set; }

        private Button ClearButton;

        public EntityProperty(GUIProperty property) : base(property)
        {
            RowHeight = 68;

            var entity = property.GetValue() as Entity;

            Button = new Button
            {
                Size = new Point(26, 20),
                Dock = DockStyle.Fill,
                Style = "textbox",
                Text = entity != null ? entity.Name : "- None -",
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
                Button.Text = "- None -";
            };

            Controls.Add(ClearButton);
            Controls.Add(Button);
        }

        private bool IsDropCompatible(DragDropEventArgs e)
        {
            if (e.DraggedControl?.Tag is not Entity entity) return false;
            return property.Type.IsAssignableFrom(entity.GetType());
        }

        private void Button_DragResponse(Control sender, DragDropEventArgs e)
        {
            if(IsDropCompatible(e))
                Button.State = ControlState.Hot;
        }

        private void Button_DragDrop(Control sender, DragDropEventArgs e)
        {
            Button.TextColor = -1;

            if (e.DraggedControl?.Tag is not Entity entity) return;
            if (!property.Type.IsAssignableFrom(entity.GetType())) return;

            property.SetValue(entity);
            Button.Text = entity.Name;
        }

        private void Button_MouseDoubleClick(Control sender, MouseEventArgs e)
        {
            var entity = property.GetValue() as Entity;
            if (entity != null)
                Selector.SelectedEntity = entity;
        }

        protected override void OnUpdate()
        {
            Timer += Time.Delta;
            if (Timer > Interval)
            {
                Timer = 0;
                var entity = property.GetValue() as Entity;
                Button.Text = entity != null ? entity.Name : "- None -";
            }
        }
    }
}
