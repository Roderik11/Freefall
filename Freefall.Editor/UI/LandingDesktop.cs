using System;
using System.Collections.Generic;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Landing page desktop shown at editor startup before a project is opened.
    /// Displays recent projects and provides open/create project buttons.
    /// </summary>
    public class LandingDesktop : Desktop
    {
        
        /// <summary>
        /// Fired when the user selects a project to open (path to project directory or .ffproject).
        /// </summary>
        public event Action<string> OnProjectSelected;

        /// <summary>
        /// Fired when the user dismisses an error dialog.
        /// </summary>
        public event Action OnErrorDismissed;

        private VirtualList projectList;
        private Window importDialog;
        private Label importStatusLabel;
        private volatile string pendingStatus;

        public LandingDesktop()
        {
            EditorSkin.Apply(this);
            ShowCursor = true;
            ModalColor = ColorInt.ARGB(.5f, 0f, 0f, 0f);

            BuildLayout();
        }

        private void BuildLayout()
        {
            // ── Root container (full screen) ──
            var root = new Frame
            {
                Style = "window",
                Dock = DockStyle.Fill,
                Padding = new Margin(32, 24, 32, 24),
            };
            Controls.Add(root);

            // ── Title ──
            var title = new Label
            {
                Text = "Freefall Engine",
                Style = "label",
                Size = new Point(100, 48),
                Dock = DockStyle.Top,
                TextAlign = Alignment.MiddleCenter,
            };
            root.Controls.Add(title);

            // ── Subtitle ──
            var subtitle = new Label
            {
                Text = "Select a project to open",
                Style = "label",
                Size = new Point(100, 28),
                Dock = DockStyle.Top,
                TextAlign = Alignment.MiddleCenter,
            };
            root.Controls.Add(subtitle);

            // ── Button bar ──
            var buttonBar = new Frame
            {
                Style = "window",
                Size = new Point(100, 48),
                Dock = DockStyle.Bottom,
            };
            root.Controls.Add(buttonBar);

            var openButton = new Button
            {
                Text = "Open Project...",
                Style = "button",
                Size = new Point(160, 32),
                Dock = DockStyle.Left,
                Margin = new Margin(0, 8, 4, 8),
            };
            openButton.MouseClick += (s, e) => OnOpenProjectClicked();
            buttonBar.Controls.Add(openButton);

            var newButton = new Button
            {
                Text = "New Project...",
                Style = "button",
                Size = new Point(160, 32),
                Dock = DockStyle.Left,
                Margin = new Margin(4, 8, 4, 8),
            };
            newButton.MouseClick += (s, e) => OnNewProjectClicked();
            buttonBar.Controls.Add(newButton);

            // ── Recent projects (VirtualList) ──
            projectList = new VirtualList
            {
                Dock = DockStyle.Fill,
                Margin = new Margin(0, 8, 0, 8),
                ItemHeight = 32,
            };
            projectList.Scrollbar.Size = new Point(14, 14);
            projectList.Scrollbar.Slider.Style = "scrollSliderButton";
            projectList.Scrollbar.Slider.MinSize = new Point(0, 32);
            projectList.CreateItem = CreateProjectItem;
            projectList.BindItem = BindProjectItem;
            projectList.DataSource = (System.Collections.IList)RecentProjects.Entries;
            root.Controls.Add(projectList);
        }

        // ── Import progress dialog ──

        /// <summary>
        /// Show a modal import progress dialog.
        /// </summary>
        public void ShowImportDialog(string projectName)
        {
            importDialog = new Window
            {
                Style = "frame",
                Size = new Point(500, 120),
                Dock =  DockStyle.Center,
                Modal = true,
            };

            importStatusLabel = new Label
            {
                Text = $"Opening {projectName}...",
                Style = "label",
                Dock = DockStyle.Fill,
                TextAlign = Alignment.MiddleCenter,
            };
            importDialog.Controls.Add(importStatusLabel);

            importDialog.Show(this);
        }

        /// <summary>
        /// Update the import dialog status text. Thread-safe — stores pending text
        /// that gets applied on the next Update() call.
        /// </summary>
        public void UpdateImportStatus(string status)
        {
            pendingStatus = status;
        }

        /// <summary>
        /// Close the import dialog.
        /// </summary>
        public void HideImportDialog()
        {
            if (importDialog != null)
            {
                importDialog.Close();
                importDialog = null;
                importStatusLabel = null;
            }
        }

        /// <summary>
        /// Show a modal error dialog with an OK button.
        /// Clicking OK fires OnErrorDismissed.
        /// </summary>
        public void ShowErrorDialog(string message)
        {
            HideImportDialog();

            var errorWindow = new Window
            {
                Style = "frame",
                Size = new Point(500, 160),
                Anchor = AnchorStyles.None,
                Modal = true,
            };

            var errorLabel = new Label
            {
                Text = message,
                Style = "label",
                Dock = DockStyle.Fill,
                TextAlign = Alignment.MiddleCenter,
            };
            errorWindow.Controls.Add(errorLabel);

            var okButton = new Button
            {
                Text = "OK",
                Style = "button",
                Size = new Point(100, 32),
                Dock = DockStyle.Bottom,
                Margin = new Margin(0, 4, 0, 8),
            };
            okButton.MouseClick += (s, e) =>
            {
                errorWindow.Close();
                OnErrorDismissed?.Invoke();
            };
            errorWindow.Controls.Add(okButton);

            errorWindow.Show(this);
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();

            // Apply pending status from background thread
            if (pendingStatus != null && importStatusLabel != null)
            {
                importStatusLabel.Text = pendingStatus;
                pendingStatus = null;
            }
        }

        // ── VirtualList callbacks ──

        private class ProjectEntry : Button
        {
            public Button RemoveBtn;

            public ProjectEntry(int height)
            {
                Dock = DockStyle.Top;
                Size = new Point(100, height);
                Style = "item";
                TextAlign = Alignment.MiddleLeft;

                RemoveBtn = new Button
                {
                    Text = "✕",
                    Style = "button",
                    Size = new Point(24, 24),
                    Dock = DockStyle.Right,
                    Margin = new Margin(4, 4, 4, 4),
                };
                Elements.Add(RemoveBtn);
            }
        }

        private Control CreateProjectItem(int index)
        {
            var entry = new ProjectEntry(projectList.ItemHeight);

            entry.MouseDoubleClick += (s, e) =>
            {
                if (entry.Tag is string path)
                    OnProjectSelected?.Invoke(path);
            };

            entry.RemoveBtn.MouseClick += (s, e) =>
            {
                if (entry.Tag is string path)
                {
                    RecentProjects.Remove(path);
                    projectList.DataSource = (System.Collections.IList)RecentProjects.Entries;
                }
            };

            return entry;
        }

        private void BindProjectItem(Control control, int index)
        {
            if (control is not ProjectEntry entry) return;
            if (index < 0 || index >= RecentProjects.Entries.Count) return;

            var data = RecentProjects.Entries[index];
            entry.Tag = data.Path;
            entry.Text = $"{data.Name}   —   {data.Path}";
        }

        private void OnOpenProjectClicked()
        {
            using var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select a Freefall project folder",
                ShowNewFolderButton = false,
            };

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                OnProjectSelected?.Invoke(dialog.SelectedPath);
            }
        }

        private void OnNewProjectClicked()
        {
            using var dialog = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select a folder for the new project",
                ShowNewFolderButton = true,
            };

            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                var folderName = System.IO.Path.GetFileName(dialog.SelectedPath);
                var project = Freefall.Assets.FreefallProject.Create(dialog.SelectedPath, folderName);
                OnProjectSelected?.Invoke(dialog.SelectedPath);
            }
        }
    }
}
