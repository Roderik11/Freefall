using Squid;
using Freefall.Reflection;
using Freefall.Base;
using System.Collections.Generic;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(System.Enum))]
    public class EnumProperty : PropertyControl
    {
        public DropDownList Dropdown { get; private set; }

        private readonly bool _isFlags;

        public EnumProperty(GUIProperty property) : base(property)
        {
            _isFlags = property.Type.GetCustomAttributes(typeof(System.FlagsAttribute), false).Length > 0;

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

            Dropdown.OnOpened += HandleDropdownOnOpened;

            if (_isFlags)
                BuildFlagsItems();
            else
                BuildSingleSelectItems();

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

        // ── Single-select (non-flags) ──

        private void BuildSingleSelectItems()
        {
            object value = property.GetValue();
            ListBoxItem selected = null;

            foreach (object entry in Enum.GetValues(property.Type))
            {
                ListBoxItem item = new ListBoxItem();
                item.Text = entry.ToString().Replace("_", "");
                item.Value = entry;
                item.Style = "item";
                item.Size = new Point(32, 26);
                item.Margin = new Margin(0, 1, 0, 0);
                Dropdown.Items.Add(item);

                if (entry.Equals(value))
                    selected = item;
            }

            if (selected != null)
                Dropdown.SelectedItem = selected;

            Dropdown.SelectedItemChanged += new SelectedItemChangedEventHandler(Dropdown_SelectedItemChanged);
        }

        void Dropdown_SelectedItemChanged(Control sender, ListBoxItem value)
        {
            property.SetValue(value.Value);
        }

        // ── Multi-select (flags) ──

        private void BuildFlagsItems()
        {
            Dropdown.Label.Text = BuildFlagsSummary();
            RebuildFlagsItems();
        }

        private void RebuildFlagsItems()
        {
            Dropdown.Items.Clear();

            int currentValue = Convert.ToInt32(property.GetValue());

            foreach (object entry in Enum.GetValues(property.Type))
            {
                int flagValue = Convert.ToInt32(entry);
                if (flagValue == 0) continue; // skip None

                bool selected = (currentValue & flagValue) == flagValue;
                string prefix = selected ? "[x] " : "[  ] ";
                string name = entry.ToString().Replace("_", "");

                var item = new ListBoxItem
                {
                    Text = prefix + name,
                    Value = entry,
                    Style = "item",
                    Size = new Point(32, 26),
                    Margin = new Margin(0, 1, 0, 0),
                };

                int capturedFlag = flagValue;
                item.MouseClick += (sender, args) =>
                {
                    ToggleFlag(capturedFlag);
                };

                Dropdown.Items.Add(item);
            }
        }

        private void ToggleFlag(int flagValue)
        {
            int current = Convert.ToInt32(property.GetValue());

            if ((current & flagValue) == flagValue)
                current &= ~flagValue;
            else
                current |= flagValue;

            property.SetValue(Enum.ToObject(property.Type, current));
            Dropdown.Label.Text = BuildFlagsSummary();
            RebuildFlagsItems();
        }

        private string BuildFlagsSummary()
        {
            int value = Convert.ToInt32(property.GetValue());
            if (value == 0) return "None";

            var names = new List<string>();
            foreach (object entry in Enum.GetValues(property.Type))
            {
                int flagValue = Convert.ToInt32(entry);
                if (flagValue == 0) continue;
                if ((value & flagValue) == flagValue)
                    names.Add(entry.ToString().Replace("_", ""));
            }

            if (names.Count == 0) return "None";
            if (names.Count <= 2) return string.Join(", ", names);
            return $"{names.Count} flags";
        }

        // ── Common ──

        void HandleDropdownOnOpened(Control sender, SquidEventArgs args)
        {
            DropDownList drop = sender as DropDownList;
            Window target = drop.Dropdown;
            target.Opacity = 1;

            if (_isFlags)
                RebuildFlagsItems();
        }
    }
}
