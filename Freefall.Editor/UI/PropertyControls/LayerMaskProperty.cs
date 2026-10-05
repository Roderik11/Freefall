using Squid;
using System.Collections.Generic;
using Freefall.Reflection;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [PropertyControl(typeof(LayerMask))]
    public class LayerMaskProperty : PropertyControl
    {
        private readonly DropDownList Dropdown;
        private Terrain terrain;

        public LayerMaskProperty(GUIProperty property) : base(property)
        {
            terrain = FindTerrain(property);

            Dropdown = new DropDownList();
            Dropdown.Padding = new Margin(0);
            Dropdown.Style = "textbox";
            Dropdown.Size = new Point(222, 32);
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
            Dropdown.Label.Text = BuildSummary();
            Dropdown.Button.Size = new Point(24, 16);
            Dropdown.Button.Margin = new Margin(1, 0, 0, 0);
            Dropdown.Button.TextAlign = Alignment.MiddleCenter;
            Dropdown.Button.Dock = DockStyle.Right;
            Dropdown.Dropdown.Style = "popup";
            Dropdown.Dropdown.Padding = new Margin(4);
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

            RebuildItems();

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

        private void RebuildItems()
        {
            Dropdown.Items.Clear();

            if (terrain?.Layers == null) return;

            var mask = property.GetValue() as LayerMask;

            for (int i = 0; i < terrain.Layers.Count; i++)
            {
                var layer = terrain.Layers[i];
                ulong layerId = layer.LayerId;
                bool selected = mask != null && mask.Contains(layerId);

                string name = layer.Diffuse != null ? layer.Diffuse.Name : $"Layer {i}";
                string prefix = selected ? "[x] " : "[  ] ";

                var item = new ListBoxItem
                {
                    Text = prefix + name,
                    Value = layerId,
                    Style = "item",
                    Size = new Point(32, 26),
                    Margin = new Margin(0, 1, 0, 0),
                };

                ulong capturedId = layerId;
                item.MouseClick += (sender, args) =>
                {
                    ToggleLayer(capturedId);
                };

                Dropdown.Items.Add(item);
            }
        }

        private void ToggleLayer(ulong layerId)
        {
            var mask = property.GetValue() as LayerMask ?? new LayerMask();

            if (mask.Contains(layerId))
                mask.Remove(layerId);
            else
                mask.Add(layerId);

            property.SetValue(mask);
            Dropdown.Label.Text = BuildSummary();
            RebuildItems();
        }

        private string BuildSummary()
        {
            var mask = property.GetValue() as LayerMask;
            if (mask == null || mask.Count == 0) return "None";
            if (terrain?.Layers == null) return $"{mask.Count} layer(s)";

            var names = new List<string>();
            foreach (var id in mask)
            {
                for (int i = 0; i < terrain.Layers.Count; i++)
                {
                    if (terrain.Layers[i].LayerId == id)
                    {
                        names.Add(terrain.Layers[i].Diffuse?.Name ?? $"Layer {i}");
                        break;
                    }
                }
            }

            if (names.Count == 0) return "None";
            if (names.Count <= 2) return string.Join(", ", names);
            return $"{names.Count} layers";
        }

        void HandleDropdownOnOpened(Control sender, SquidEventArgs args)
        {
            (sender as DropDownList).Dropdown.Opacity = 1;
            RebuildItems();
        }

        private static Terrain FindTerrain(GUIProperty property)
        {
            var parent = property.Parent;
            while (parent != null)
            {
                if (parent.Target is Terrain t) return t;
                parent = parent.Parent;
            }
            return null;
        }

        protected override void OnUpdate()
        {
            Timer += Time.Delta;
            if (Timer > Interval)
            {
                Timer = 0;
                Dropdown.Label.Text = BuildSummary();
            }
        }
    }
}
