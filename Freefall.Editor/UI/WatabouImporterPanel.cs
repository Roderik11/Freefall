using System;
using System.IO;
using System.Threading.Tasks;
using Squid;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Panel for configuring and running Watabou village/city imports.
    /// Uses GenericInspector to expose WatabouImportSettings with automatic binding.
    /// </summary>
    public class WatabouImporterPanel : Frame
    {
        private Button importButton;
        private ScrollPanel scrollPanel;
        private readonly Tools.WatabouSettings settings = new();

        public WatabouImporterPanel()
        {
            Style = "frame";
            Dock = DockStyle.Fill;
            Size = new Point(580, 620);
            BuildLayout();
        }

        private void BuildLayout()
        {
            scrollPanel = new ScrollPanel();
            Controls.Add(scrollPanel);

            // ── Settings via GenericInspector ──
            var obj = new GUIObject(settings);
            var inspector = GUIInspector.GetInspector(obj);
            inspector.Dock = DockStyle.Top;
            inspector.Margin = new Margin(0, 8, 0, 0);
            scrollPanel.Content.Controls.Add(inspector);

            importButton = new Button
            {
                Text = "Import",
                Style = "catbutton",
                Size = new Point(100, 32),
                Dock = DockStyle.Top,
                Margin = new Margin(4, 4, 4, 4),
            };
            importButton.MouseClick += (s, e) => RunImport();

            inspector.AddControl(importButton);
        }

        private void RunImport()
        {
            var filePath = settings.FilePath;
            if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
            {
                Toast.Show("Please select a valid JSON file.");
                return;
            }

            importButton.Enabled = false;
            Toast.Show("Starting Watabou Import...");

            try
            {
                var json = File.ReadAllText(filePath);
                var name = Path.GetFileNameWithoutExtension(filePath);

                // Auto-detect format: Dwellings has "floors", Village has "features"
                bool isDwellings = json.Contains("\"floors\"");

                if (isDwellings)
                {
                    Toast.Show("Parsing Dwellings JSON...");
                    var data = Tools.DwellingsParser.Parse(json);

                    int totalCells = 0;
                    foreach (var floor in data.Floors)
                        foreach (var room in floor.Rooms)
                            totalCells += room.Cells.Count;

                    Toast.Show($"Building dwelling ({data.Floors.Count} floors, {totalCells} cells)...");
                    // Wall pieces are 2M wide × 3M tall — grid must match prefab dimensions
                    var builder = new Tools.DwellingsBuilder(
                        cellSize: 2f,
                        storyHeight: 3f,
                        seed: name.GetHashCode());
                    var root = builder.Build(data, name);

                    Debug.Log($"[Dwellings] Imported '{name}': {data.Floors.Count} floors, {totalCells} cells");
                }
                else
                {
                    Toast.Show("Parsing Village JSON...");
                    var data = Tools.WatabouParser.Parse(json);

                    Toast.Show($"Building scene ({data.Buildings.Count} buildings, {data.Roads.Count} roads, {data.Walls.Count} walls)...");
                    var root = Tools.WatabouBuilder.Build(data, settings, name);

                    Debug.Log($"[Watabou] Imported '{name}': {data.Buildings.Count} buildings, {data.Roads.Count} roads, {data.Walls.Count} walls, {data.Rivers.Count} rivers");
                }

                Toast.Show($"Done! Created '{name}'");
                importButton.Enabled = true;
            }
            catch (Exception ex)
            {
                Debug.Log($"[Watabou] Import error: {ex.Message}");
                Toast.Show($"Error: {ex.Message}");
                importButton.Enabled = true;
            }
        }
    }
}
