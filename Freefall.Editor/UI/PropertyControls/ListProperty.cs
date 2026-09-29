using Squid;
using System;
using System.Collections.Generic;
using System.Linq;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(List<>))]
    public class ListProperty : PropertyControl
    {
        private readonly GUIInspectorRow header;
        private readonly Frame expander;
        private readonly Label headerLabel;
        private Type elementBaseType;
        private Type[] concreteTypes;

        public ListProperty(GUIProperty property) : base(property)
        {
            Dock = DockStyle.Top;
            Expandable = true;
            AutoSize = AutoSize.Vertical;

            header = new GUIInspectorRow(true, property.Expanded, property.Name);
            if(property.Depth > 0)
                header.Indent(property.Depth);

            Controls.Add(header);

            expander = new Frame
            {
                Dock = DockStyle.Top,
                AutoSize = AutoSize.Vertical,
                Visible = property.Expanded
            };
            Controls.Add(expander);

            elementBaseType = property.GetElementType();

            // Discover concrete subclasses for abstract/interface element types
            if (elementBaseType.IsAbstract || elementBaseType.IsInterface)
            {
                concreteTypes = elementBaseType.Assembly.GetTypes()
                    .Where(t => !t.IsAbstract && elementBaseType.IsAssignableFrom(t))
                    .OrderBy(t => t.Name)
                    .ToArray();
            }

            // + button: dropdown for polymorphic lists, direct add for concrete types
            if (concreteTypes != null && concreteTypes.Length > 0)
            {
                var btnPlus = new DropDownButton
                {
                    Style = "iconplus",
                    Size = new Point(32, 32),
                    Margin = new Margin(1),
                    Dock = DockStyle.Right,
                    Align = Alignment.TopLeft, // opens to the left so it doesn't go off-screen
                };

                // Size the dropdown window to fit the items
                int itemHeight = 26;
                int dropWidth = 180;
                int dropHeight = Math.Min(concreteTypes.Length * (itemHeight + 1) + 4, 300);

                btnPlus.Dropdown.Style = "window";
                btnPlus.Dropdown.Padding = new Margin(2);
                btnPlus.Dropdown.Size = new Point(dropWidth, dropHeight);
                btnPlus.Dropdown.Resizable = false;

                var listbox = new ListBox();
                listbox.Dock = DockStyle.Fill;
                listbox.Scrollbar.Size = new Point(12, 16);
                listbox.Scrollbar.ButtonDown.Visible = false;
                listbox.Scrollbar.ButtonUp.Visible = false;
                listbox.Scrollbar.Slider.Button.Margin = new Margin(2, 4, 0, 4);
                listbox.Scrollbar.Slider.Ease = false;
                listbox.Scrollbar.Slider.MinHandleSize = 64;
                listbox.Scrollbar.Dock = DockStyle.Right;

                foreach (var type in concreteTypes)
                {
                    var item = new ListBoxItem
                    {
                        Text = type.Name,
                        Value = type,
                        Style = "item",
                        Size = new Point(dropWidth - 4, itemHeight),
                        Margin = new Margin(0, 1, 0, 0),
                    };
                    listbox.Items.Add(item);
                }

                listbox.SelectedItemChanged += (sender, item) =>
                {
                    if (item?.Value is Type selectedType)
                    {
                        var element = property.AddElement(selectedType);
                        var elm = GUIInspector.GetInspectorElement(element, $"{selectedType.Name} {element.Index}");
                        expander.Controls.Add(elm);
                        UpdateHeaderText();
                    }
                    btnPlus.Close();
                };

                btnPlus.Dropdown.Controls.Add(listbox);
                header.Content.Controls.Add(btnPlus);
            }
            else
            {
                var btnPlus = new Button
                {
                    Style = "iconplus",
                    Size = new Point(32, 32),
                    Margin = new Margin(1),
                    Dock = DockStyle.Right,
                };
                btnPlus.MouseClick += BtnPlus_MouseClick;
                header.Content.Controls.Add(btnPlus);
            }

            headerLabel = new Label
            {
                NoEvents = true,
                Margin = new Margin(8, 0, 0, 0),
                Dock = DockStyle.Fill,
            };

            header.Content.Controls.Add(headerLabel);

            header.ExpandedChanged += (expanded) =>
            {
                property.Expanded = expanded;
                expander.Visible = expanded;
            };

            for (int i = 0; i < property.GetArrayLength(); i++)
            {
                var element = property.GetArrayElementAtIndex(i);
                var value = element.GetValue();
                string label = value != null ? value.ToString() : $"{element.Type.Name} {i}";
                var elm = GUIInspector.GetInspectorElement(element, label);
                if(elm == null)
                {
                    elm = new Label
                    {
                        Text = label,
                        Margin = new Margin(4),
                        Dock = DockStyle.Top,
                    };
                }
                elm.Tooltip = label;

                if (elm is Frame frame && frame.Controls[0] is GUIInspectorRow row)
                {
                    var btnRemove = new Button
                    {
                        Style = "iconclose",
                        Size = new Point(32, 32),
                        Margin = new Margin(1),
                        Dock = DockStyle.Fill,
                    };
                    btnRemove.MouseClick += (sender, args) =>
                    {
                        property.RemoveElementAtIndex(element.Index);
                        expander.Controls.Remove(elm);
                        UpdateHeaderText();
                    };
                    row.ExtraButton.Controls.Add(btnRemove);
                }
                expander.Controls.Add(elm);
            }

            UpdateHeaderText();
        }

        void UpdateHeaderText()
        {
            var count = property.GetArrayLength();
            var text = $"{count} Element";
            if (count != 1) text += "s";
            headerLabel.Text = text;
        } 
        private void BtnPlus_MouseClick(Control sender, MouseEventArgs args)
        {
            var element = property.AddElement();
            var elm = GUIInspector.GetInspectorElement(element, $"Element {element.Index}");
            expander.Controls.Add(elm);

            if (elm is Frame frame && frame.Controls[0] is GUIInspectorRow row)
            {
                var btnRemove = new Button
                {
                    Style = "iconclose",
                    Size = new Point(32, 32),
                    Margin = new Margin(1),
                    Dock = DockStyle.Fill,
                };
                btnRemove.MouseClick += (sender, args) =>
                {
                    property.RemoveElementAtIndex(element.Index);
                    expander.Controls.Remove(elm);
                    UpdateHeaderText();
                };
                row.ExtraButton.Controls.Add(btnRemove);
            }

            UpdateHeaderText();
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();
        }
    }
}
