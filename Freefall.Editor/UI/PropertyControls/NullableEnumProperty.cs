using Squid;
using Freefall.Reflection;
using System;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    /// <summary>
    /// An enum that may be left unset (<c>HumanoidBone?</c>): the enum's dropdown with "(none)" on top.
    /// A nullable of anything else shows its value and cannot be edited here.
    /// </summary>
    [PropertyControl(typeof(Nullable<>))]
    public class NullableEnumProperty : PropertyControl
    {
        public DropDownList Dropdown { get; private set; }

        public NullableEnumProperty(GUIProperty property) : base(property)
        {
            var enumType = Nullable.GetUnderlyingType(property.Type);
            if (enumType == null || !enumType.IsEnum)
            {
                Controls.Add(new Label
                {
                    Dock = DockStyle.Fill,
                    Text = property.GetValue()?.ToString() ?? "(none)",
                });
                return;
            }

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
            Dropdown.DropdownAutoSize = true;
            Dropdown.Listbox.Size = new Point(200, 32);

            Dropdown.Listbox.Scrollbar.Size = new Point(12, 16);
            Dropdown.Listbox.Scrollbar.ButtonDown.Visible = false;
            Dropdown.Listbox.Scrollbar.ButtonUp.Visible = false;
            Dropdown.Listbox.Scrollbar.Slider.Button.Margin = new Margin(2, 4, 0, 4);
            Dropdown.Listbox.Scrollbar.Slider.Ease = false;
            Dropdown.Listbox.Scrollbar.Slider.MinHandleSize = 64;
            Dropdown.Listbox.Scrollbar.Dock = DockStyle.Right;

            Dropdown.OnOpened += (sender, args) =>
            {
                if (sender is DropDownList drop) drop.Dropdown.Opacity = 1;
            };

            object value = property.GetValue();

            var none = NewItem("(none)", null);
            Dropdown.Items.Add(none);
            ListBoxItem selected = none;

            foreach (object entry in Enum.GetValues(enumType))
            {
                // A count at the end of an enum is not one of its values
                string name = entry.ToString();
                if (name == "Count") continue;

                var item = NewItem(name.Replace("_", ""), entry);
                Dropdown.Items.Add(item);

                if (entry.Equals(value))
                    selected = item;
            }

            Dropdown.SelectedItem = selected;
            Dropdown.SelectedItemChanged += (sender, item) => property.SetValue(item?.Value);

            var img = new ImageControl
            {
                Dock = DockStyle.Fill,
                NoEvents = true,
                Texture = "icon_down.png",
                Tiling = TextureMode.Center,
            };

            Dropdown.Button.GetElements().Add(img);

            Controls.Add(Dropdown);
        }

        private static ListBoxItem NewItem(string text, object value) => new ListBoxItem
        {
            Text = text,
            Value = value,
            Style = "item",
            Size = new Point(32, 26),
            Margin = new Margin(0, 1, 0, 0),
        };
    }
}
