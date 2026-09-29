using System;
using System.Threading.Tasks;
using Squid;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Modal window for configuring and running Unity asset pack imports.
    /// Uses GenericInspector to expose ModelImporter settings with automatic binding.
    /// </summary>
    public class UnityImporterWindow : Window
    {
        private TextBox sourcePathBox;
        private TextBox packNameBox;
        private Label statusLabel;
        private Button importButton;
        private Button browseButton;

        private readonly Tools.UnityImporterSettings settings = new();

        public UnityImporterWindow()
        {
            Style = "frame";
            Size = new Point(520, 480);
            Dock = DockStyle.Center;
            Modal = true;
            Padding = new Margin(16, 16, 16, 16);

            BuildLayout();
        }

        private void BuildLayout()
        {
            // ── Title ──
            var title = new Label
            {
                Text = "Import Unity Pack",
                Style = "label",
                Size = new Point(100, 28),
                Dock = DockStyle.Top,
                TextAlign = Alignment.MiddleCenter,
            };
            Controls.Add(title);

            // ── Source directory row ──
            var sourceRow = new Frame
            {
                Style = "window",
                Size = new Point(100, 32),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 8, 0, 0),
            };
            Controls.Add(sourceRow);

            var sourceLabel = new Label
            {
                Text = "Source:",
                Style = "label",
                Size = new Point(80, 28),
                Dock = DockStyle.Left,
                TextAlign = Alignment.MiddleLeft,
            };
            sourceRow.Controls.Add(sourceLabel);

            browseButton = new Button
            {
                Text = "...",
                Style = "button",
                Size = new Point(32, 28),
                Dock = DockStyle.Right,
            };
            browseButton.MouseClick += (s, e) => BrowseSource();
            sourceRow.Controls.Add(browseButton);

            sourcePathBox = new TextBox
            {
                Style = "textbox",
                Dock = DockStyle.Fill,
                Margin = new Margin(4, 2, 4, 2),
            };
            sourceRow.Controls.Add(sourcePathBox);

            // ── Pack name row ──
            var packRow = new Frame
            {
                Style = "window",
                Size = new Point(100, 32),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 4, 0, 0),
            };
            Controls.Add(packRow);

            var packLabel = new Label
            {
                Text = "Pack Name:",
                Style = "label",
                Size = new Point(80, 28),
                Dock = DockStyle.Left,
                TextAlign = Alignment.MiddleLeft,
            };
            packRow.Controls.Add(packLabel);

            packNameBox = new TextBox
            {
                Style = "textbox",
                Dock = DockStyle.Fill,
                Margin = new Margin(4, 2, 0, 2),
            };
            packRow.Controls.Add(packNameBox);

            // ── Settings via GenericInspector ──
            var obj = new GUIObject(settings);
            var inspector = GUIInspector.GetInspector(obj);
            inspector.Dock = DockStyle.Top;
            inspector.Margin = new Margin(0, 8, 0, 0);
            Controls.Add(inspector);

            // ── Bottom bar ──
            var bottomBar = new Frame
            {
                Style = "window",
                Size = new Point(100, 40),
                Dock = DockStyle.Bottom,
            };
            Controls.Add(bottomBar);

            importButton = new Button
            {
                Text = "Import",
                Style = "button",
                Size = new Point(100, 32),
                Dock = DockStyle.Right,
                Margin = new Margin(4, 4, 0, 4),
            };
            importButton.MouseClick += (s, e) => RunImport();
            bottomBar.Controls.Add(importButton);

            var cancelButton = new Button
            {
                Text = "Cancel",
                Style = "button",
                Size = new Point(100, 32),
                Dock = DockStyle.Right,
                Margin = new Margin(0, 4, 4, 4),
            };
            cancelButton.MouseClick += (s, e) => Close();
            bottomBar.Controls.Add(cancelButton);

            // ── Status ──
            statusLabel = new Label
            {
                Text = "",
                Style = "label",
                Dock = DockStyle.Fill,
                TextAlign = Alignment.MiddleCenter,
            };
            Controls.Add(statusLabel);
        }

        private void BrowseSource()
        {
            using var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select the Unity asset source directory",
                ShowNewFolderButton = false,
            };

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                sourcePathBox.Text = dialog.SelectedPath;
                packNameBox.Text = System.IO.Path.GetFileName(dialog.SelectedPath);
            }
        }

        private void RunImport()
        {
            var sourcePath = sourcePathBox.Text?.Trim();
            var packName = packNameBox.Text?.Trim();
            var assetsRoot = Engine.Project?.AssetsDirectory;

            if (string.IsNullOrEmpty(sourcePath) || string.IsNullOrEmpty(packName) || string.IsNullOrEmpty(assetsRoot))
            {
                statusLabel.Text = "Please fill in source path and pack name.";
                return;
            }

            // Disable controls during import
            importButton.Enabled = false;
            browseButton.Enabled = false;

            statusLabel.Text = $"Importing {packName}...";

            // settings object is already bound via the inspector
            Task.Run(() =>
            {
                try
                {
                    Tools.UnityImporter.Import(sourcePath, assetsRoot, packName,
                        status => statusLabel.Text = status, settings);
                }
                catch (Exception ex)
                {
                    Debug.Log($"[UnityImporter] Error: {ex.Message}");
                    statusLabel.Text = $"Error: {ex.Message}";
                    System.Threading.Thread.Sleep(3000);
                }
                finally
                {
                    Close();
                    MessageDispatcher.Send(Msg.RefreshAssets);
                }
            });
        }
    }
}
