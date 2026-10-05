using Squid;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(ValueSelectAttribute))]
    public class ValueSelectProperty : PropertyControl
    {
        public DropDownList Dropdown { get; private set; }

        public ValueSelectProperty(GUIProperty property) : base(property)
        {
            var provider = property.GetAttribute<ValueSelectAttribute>().Provider;

            Dropdown = new DropDownList();
            Dropdown.Padding = new Squid.Margin(0);
            Dropdown.Style = "textbox";
            Dropdown.Size = new Squid.Point(222, 32);
            Dropdown.Dock = DockStyle.Fill;
            Dropdown.Label.NoEvents = true;
            Dropdown.Button.NoEvents = true;

            Dropdown.StateChanged += () =>
            {
                Dropdown.Button.State = Dropdown.State;
                Dropdown.Label.State = Dropdown.State;
            };
            
            Dropdown.MouseClick += (sender, e) =>
            {
                if (Dropdown.IsOpen)
                    Dropdown.Close();
                else
                    Dropdown.Open();
            };

            Dropdown.Label.Style = "dropdownLabel";
            Dropdown.Label.Dock = DockStyle.Fill;
            Dropdown.Button.Size = new Point(24, 16);
            Dropdown.Button.Margin = new Margin(1, 0, 0, 0);
            Dropdown.Button.TextAlign = Alignment.MiddleCenter;
            Dropdown.Button.Dock = DockStyle.Right;
            Dropdown.Dropdown.Style = "popup";
            Dropdown.Dropdown.Padding = new Squid.Margin(4);
            Dropdown.DropdownAutoSize = false;
            Dropdown.Listbox.Size = new Point(200, 32);

            Dropdown.Listbox.Scrollbar.Size = new Point(12, 16);
            Dropdown.Listbox.Scrollbar.ButtonDown.Visible = false;
            Dropdown.Listbox.Scrollbar.ButtonUp.Visible = false;
            Dropdown.Listbox.Scrollbar.Slider.Button.Margin = new Margin(2, 4, 0, 4);
            Dropdown.Listbox.Scrollbar.Slider.Ease = false;
            Dropdown.Listbox.Scrollbar.Slider.MinHandleSize = 64;
            Dropdown.Listbox.Scrollbar.Dock = DockStyle.Right;

            Dropdown.OnOpened += HandleDropdownOnOpened;

            object value = property.GetValue();
            ListBoxItem selected = null;

            foreach (var entry in provider.GetValues(property.Target))
            {
                ListBoxItem item = new ListBoxItem();
                item.Text = entry.Name;
                item.Value = entry;
                item.Style = "item";
                item.Size = new Point(32, 26);
                item.Margin = new Margin(0, 1, 0, 0);
                Dropdown.Items.Add(item);

                if (entry.Value.Equals(value))
                    selected = item;
            }

            if (selected != null)
                Dropdown.SelectedItem = selected;

            Dropdown.SelectedItemChanged += Dropdown_SelectedItemChanged;

            ImageControl img = new ImageControl
            {
                Dock = DockStyle.Fill,
                NoEvents = true,
                Texture = "icon_down.png",
                Tiling = TextureMode.Center,
            };

            Dropdown.Button.GetElements().Add(img);
           
            Controls.Add(Dropdown);
        }

        void HandleDropdownOnOpened(Control sender, SquidEventArgs args)
        {
            DropDownList drop = sender as DropDownList;
            Window target = drop.Dropdown;
            target.Opacity = 1;

            var contentSize = drop.Listbox.ItemContainer.GetContentSize();
            int idealHeight = contentSize.y + target.Padding.Top + target.Padding.Bottom;

            int maxHeight = 400;
            int height = Math.Min(idealHeight, maxHeight);

            target.Size = new Point(drop.Size.x, height);

            int screenHeight = drop.Desktop.Size.y;
            int screenWidth = drop.Desktop.Size.x;

            Point pos = target.Position;

            if (pos.y + height > screenHeight)
                pos = new Point(pos.x, Math.Max(0, screenHeight - height));

            if (pos.x + target.Size.x > screenWidth)
                pos = new Point(Math.Max(0, screenWidth - target.Size.x), pos.y);

            target.Position = pos;
        }

        void Dropdown_SelectedItemChanged(Control sender, ListBoxItem value)
        {
            var pvalue = (ProviderValue)value.Value;
            property.SetValue(pvalue.Value);
        }
    }
}
