using System.Linq;
using Squid;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Editor Preferences panel. Docked as a tab next to Settings.
    /// Left: category tree. Right: property panel with color rows.
    /// </summary>
    public class PreferencesControl : Frame
    {
        private readonly ScrollPanel _propertyPanel;
        private Frame catList;

        public PreferencesControl()
        {
            Style = "frame";
            Dock = DockStyle.Fill;

            // Split: left category list, right property panel
            var split = new SplitContainer
            {
                Dock = DockStyle.Fill,
                RetainAspect = false,
                Orientation = Orientation.Horizontal,
            };
            split.SplitFrame1.Size = new Point(140, 100);
            split.SplitButton.Size = new Point(2, 2);
            split.SplitButton.Style = "frame";

            // Left: category buttons
            catList = new Frame
            {
                Dock = DockStyle.Fill,
                AutoSize = AutoSize.Vertical,
            };

            AddButton("Asset Colors", ShowAssetColors);
            AddButton("Snapping", ShowSnapping);

            split.SplitFrame1.Controls.Add(catList);

            // Right: scrollable property panel
            _propertyPanel = new ScrollPanel
            {
                Dock = DockStyle.Fill,
                Style = "frame"
            };
            split.SplitFrame2.Controls.Add(_propertyPanel);

            Controls.Add(split);

            // Show asset colors by default
            ShowAssetColors();
        }


        void AddButton(string text, Action onClick)
        {
            var btn = new Button
            {
                Text = text,
                Dock = DockStyle.Top,
                Size = new Point(100, 28),
                Style = "button",
            };
            btn.MouseClick += (s, e) => onClick();
            catList.Controls.Add(btn);
        }

        private void ShowSnapping()
        {
            _propertyPanel.Content.Controls.Clear();

            var prefs = EditorPreferences.Instance;
            var guiObj = new GUIObject(prefs.Snapping);
            var inspector = new GenericInspector(guiObj);
            _propertyPanel.Content.Controls.Add(inspector);
        }


        private void ShowAssetColors()
        {
            _propertyPanel.Content.Controls.Clear();

            var prefs = EditorPreferences.Instance;
            if (prefs.AssetTypeColors.Count == 0)
                prefs.GetAssetColor("UNKNOWN");

            var sorted = prefs.AssetTypeColors.OrderBy(e => e.Name).ToList();

            foreach (var entry in sorted)
            {
                var guiObj = new GUIObject(entry);
                var inspector = new GUIInspector(guiObj);

                var colorProp = guiObj.GetProperty("Color");
                if (colorProp != null)
                {
                    inspector.AddProperty(colorProp, entry.Name);

                    colorProp.OnValueChanged += () =>
                    {
                        EditorPreferences.SyncStyle(entry);
                    };
                }

                _propertyPanel.Content.Controls.Add(inspector);
            }
        }
    }
}
