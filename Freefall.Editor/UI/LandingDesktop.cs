using System;
using System.Collections.Generic;
using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Landing page desktop shown at editor startup before a project is opened.
    /// Features the last opened project, lists the other recent projects as cards
    /// and provides open/create project buttons.
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

        private const int CardWidth = 264;
        private const int CardHeight = 226;
        private const int CardSpacing = 16;
        private const int FeatureHeight = 300;

        private Frame root;
        private VirtualList projectGrid;
        private int columns = 1;
        private int lastGridWidth = -1;
        private bool rebuildPending;

        // Cards in the grid: every recent project except the featured one, then null for the "New project" tile.
        private readonly List<RecentProjectEntry> gridItems = new();

        private Window importDialog;
        private Label importStatusLabel;
        private volatile string pendingStatus;

        public LandingDesktop()
        {
            EditorSkin.Apply(this);
            LandingArt.Ensure();
            RegisterStyles();
            ShowCursor = true;
            ModalColor = ColorInt.ARGB(.5f, 0f, 0f, 0f);

            BuildLayout();
        }

        private void RegisterStyles()
        {
            ControlStyle Text(string font, int color, Alignment align = Alignment.MiddleLeft)
            {
                var style = new ControlStyle();
                style.Font = font;
                style.TextColor = color;
                style.TextAlign = align;
                return style;
            }

            var background = new ControlStyle();
            background.BackColor = LandingArt.Background;

            var primary = Text(LandingArt.FontBodyMedium, LandingArt.OnCoral, Alignment.MiddleCenter);
            primary.Texture = LandingArt.RoundSmall;
            primary.Tiling = TextureMode.Grid;
            primary.Grid = new Margin(LandingArt.RadiusSmall);
            primary.Tint = LandingArt.Coral;
            primary.Hot.Tint = LandingArt.CoralHot;
            primary.Pressed.Tint = LandingArt.CoralHot;

            var ghost = Text(LandingArt.FontBodyMedium, LandingArt.Text, Alignment.MiddleCenter);
            ghost.Texture = LandingArt.OutlineSmall;
            ghost.Tiling = TextureMode.Grid;
            ghost.Grid = new Margin(LandingArt.RadiusSmall);
            ghost.Tint = LandingArt.Rgb(0xffffff, .16f);
            ghost.Hot.Tint = LandingArt.Rgb(0xffffff, .4f);
            ghost.Pressed.Tint = LandingArt.Rgb(0xffffff, .4f);

            Skin["landing.background"] = background;
            Skin["landing.brand"] = Text(LandingArt.FontTitle, LandingArt.Text);
            Skin["landing.headline"] = Text(LandingArt.FontHeadline, LandingArt.Text);
            Skin["landing.sub"] = Text(LandingArt.FontBody, LandingArt.Muted);
            Skin["landing.eyebrow"] = Text(LandingArt.FontEyebrow, LandingArt.Muted);
            Skin["landing.primary"] = primary;
            Skin["landing.ghost"] = ghost;
        }

        private void BuildLayout()
        {
            var entries = RecentProjects.Entries;
            var featured = entries.Count > 0 ? entries[0] : null;

            // ── Root container (full screen) ──
            root = new Frame
            {
                Style = "landing.background",
                Dock = DockStyle.Fill,
                Padding = new Margin(40, 26, 40, 0),
            };
            Controls.Add(root);

            // ── Header: brand left, actions right ──
            var header = new Frame
            {
                Size = new Point(100, 48),
                Dock = DockStyle.Top,
            };
            root.Controls.Add(header);

            header.Controls.Add(new ImageControl
            {
                Texture = LandingArt.Logo,
                Tiling = TextureMode.Center,
                Size = new Point(48, 48),
                Dock = DockStyle.Left,
                NoEvents = true,
            });

            header.Controls.Add(new Label
            {
                Text = "Freefall Editor",
                Style = "landing.brand",
                Size = new Point(220, 48),
                Dock = DockStyle.Left,
                Margin = new Margin(12, 0, 0, 0),
                NoEvents = true,
            });

            var newButton = new Button
            {
                Text = "New project",
                Style = "landing.primary",
                Size = new Point(128, 36),
                Dock = DockStyle.Right,
                Margin = new Margin(8, 6, 0, 6),
                Cursor = Cursors.Link,
            };
            newButton.MouseClick += (s, e) => OnNewProjectClicked();
            header.Controls.Add(newButton);

            var openButton = new Button
            {
                Text = "Open project…",
                Style = "landing.ghost",
                Size = new Point(140, 36),
                Dock = DockStyle.Right,
                Margin = new Margin(0, 6, 0, 6),
                Cursor = Cursors.Link,
            };
            openButton.MouseClick += (s, e) => OnOpenProjectClicked();
            header.Controls.Add(openButton);

            // ── Headline ──
            root.Controls.Add(new Label
            {
                Text = featured != null ? "Pick up where the world left off." : "Start a new world.",
                Style = "landing.headline",
                Size = new Point(100, 44),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 26, 0, 0),
                NoEvents = true,
            });

            root.Controls.Add(new Label
            {
                Text = Summary(entries, featured),
                Style = "landing.sub",
                Size = new Point(100, 24),
                Dock = DockStyle.Top,
                NoEvents = true,
            });

            // ── Featured project + last session ──
            if (featured != null)
            {
                var feature = new Frame
                {
                    Size = new Point(100, FeatureHeight),
                    Dock = DockStyle.Top,
                    Margin = new Margin(0, 22, 0, 0),
                };
                root.Controls.Add(feature);

                var session = new SessionCard(featured)
                {
                    Size = new Point(340, FeatureHeight),
                    Dock = DockStyle.Right,
                    Margin = new Margin(16, 0, 0, 0),
                };
                session.ContinueButton.MouseClick += (s, e) => OnProjectSelected?.Invoke(featured.Path);
                feature.Controls.Add(session);

                var hero = new HeroCard(featured) { Dock = DockStyle.Fill };
                hero.MouseClick += (s, e) => OnProjectSelected?.Invoke(featured.Path);
                feature.Controls.Add(hero);
            }

            // ── Project grid ──
            gridItems.Clear();
            for (int i = 1; i < entries.Count; i++)
                gridItems.Add(entries[i]);
            gridItems.Add(null);

            root.Controls.Add(new Label
            {
                Text = featured != null ? "RECENT PROJECTS" : "GET STARTED",
                Style = "landing.eyebrow",
                Size = new Point(100, 16),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 30, 0, 12),
                NoEvents = true,
            });

            projectGrid = new VirtualList
            {
                Dock = DockStyle.Fill,
                ItemHeight = CardHeight + CardSpacing,
            };
            projectGrid.Content.Style = "";
            projectGrid.Scrollbar.Size = new Point(14, 14);
            projectGrid.Scrollbar.Slider.Style = "scrollSliderButton";
            projectGrid.Scrollbar.Slider.MinSize = new Point(0, 32);
            projectGrid.CreateItem = CreateRow;
            projectGrid.BindItem = BindRow;
            root.Controls.Add(projectGrid);

            lastGridWidth = -1;
        }

        private static string Summary(List<RecentProjectEntry> entries, RecentProjectEntry featured)
        {
            if (featured == null)
                return "No recent projects yet. Create one, or open an existing project folder.";

            var count = entries.Count == 1 ? "One recent project" : $"{entries.Count} recent projects";
            var when = LandingArt.TimeAgo(featured.LastOpened);
            return when.Length > 0 ? $"{count}. Last session was {featured.Name}, {when}." : $"{count}.";
        }

        /// <summary>
        /// Recalculate the column count from the grid width and refresh the rows.
        /// </summary>
        private void UpdateColumns()
        {
            var width = projectGrid.ClipFrame.Size.x;
            if (width <= 0 || width == lastGridWidth) return;
            lastGridWidth = width;

            columns = Math.Max(1, (width + CardSpacing) / (CardWidth + CardSpacing));

            // VirtualList needs an IList — one entry per row
            var rows = new System.Collections.ArrayList();
            int rowCount = (gridItems.Count + columns - 1) / columns;
            for (int i = 0; i < rowCount; i++)
                rows.Add(i);

            projectGrid.DataSource = rows;
            projectGrid.Refresh();
        }

        // ── VirtualList callbacks (each item is a row of cards, as in the asset browser) ──

        private Control CreateRow(int index)
        {
            var row = new Frame
            {
                Size = new Point(100, CardHeight + CardSpacing),
                Dock = DockStyle.Top,
                Style = "",
            };

            BindRow(row, index);
            return row;
        }

        private void BindRow(Control control, int index)
        {
            if (control is not Frame row) return;

            while (row.Controls.Count < columns)
                row.Controls.Add(CreateCard());

            while (row.Controls.Count > columns)
                row.Controls.Remove(row.Controls[row.Controls.Count - 1]);

            for (int c = 0; c < columns; c++)
            {
                var card = (ProjectCard)row.Controls[c];
                int dataIndex = index * columns + c;

                card.Visible = dataIndex < gridItems.Count;
                if (card.Visible)
                    card.Entry = gridItems[dataIndex];
            }
        }

        private ProjectCard CreateCard()
        {
            var card = new ProjectCard
            {
                Size = new Point(CardWidth, CardHeight),
                Dock = DockStyle.Left,
                Margin = new Margin(0, 0, CardSpacing, CardSpacing),
            };

            card.MouseClick += (s, e) =>
            {
                if (e.Button != 0) return;

                if (card.Entry == null)
                {
                    OnNewProjectClicked();
                }
                else if (card.IsOverRemove)
                {
                    RecentProjects.Remove(card.Entry.Path);
                    rebuildPending = true;
                }
                else
                {
                    OnProjectSelected?.Invoke(card.Entry.Path);
                }
            };

            return card;
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

            // Rebuild outside of the click handler that asked for it
            if (rebuildPending)
            {
                rebuildPending = false;
                Controls.Remove(root);
                BuildLayout();
            }

            UpdateColumns();

            // Apply pending status from background thread
            if (pendingStatus != null && importStatusLabel != null)
            {
                importStatusLabel.Text = pendingStatus;
                pendingStatus = null;
            }
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

        // ── Cards (drawn directly: rounded panels and cover-cropped images don't map onto skin styles) ──

        private static bool IsHot(Control control)
        {
            return control.State == ControlState.Hot || control.State == ControlState.Pressed;
        }

        /// <summary>
        /// Draw a project's viewport snapshot into a rect, or a placeholder if it has none yet.
        /// </summary>
        private static void DrawPreview(RecentProjectEntry entry, int x, int y, int w, int h)
        {
            var thumbnail = ProjectThumbnails.GetTexture(entry.Path);
            if (thumbnail != null)
            {
                LandingArt.Cover(thumbnail, x, y, w, h);
                return;
            }

            Gui.Renderer.DrawBox(x, y, w, h, LandingArt.Well);
            LandingArt.Centered(LandingArt.Logo, x + w / 2, y + h / 2, LandingArt.Rgb(0xffffff, .22f));
        }

        /// <summary>
        /// The large featured project: snapshot with the name and path over a scrim.
        /// </summary>
        private class HeroCard : Control
        {
            private readonly RecentProjectEntry entry;

            public HeroCard(RecentProjectEntry entry)
            {
                this.entry = entry;
                Cursor = Cursors.Link;
            }

            protected override void DrawStyle(Style style, float opacity)
            {
                int x = Location.x, y = Location.y, w = Size.x, h = Size.y;
                if (opacity == 0 || w <= 0 || h <= 0) return;

                DrawPreview(entry, x, y, w, h);

                int scrim = Math.Min(h, 150);
                LandingArt.Stretch(LandingArt.Scrim, x, y + h - scrim, w, scrim, ColorInt.ARGB(.85f, 0f, 0f, 0f));

                const int pad = 22;
                int textWidth = w - pad * 2;

                string stats = entry.EntityCount > 0 ? $"{entry.EntityCount:N0} entities" : "";
                int statsWidth = LandingArt.Measure(LandingArt.FontBody, stats).x;
                if (stats.Length > 0)
                {
                    LandingArt.DrawText(LandingArt.FontBody, stats, x + w - pad - statsWidth, y + h - 40, LandingArt.TextOnImage);
                    textWidth -= statsWidth + 24;
                }

                LandingArt.DrawText(LandingArt.FontHeading, LandingArt.Fit(LandingArt.FontHeading, entry.Name, textWidth),
                    x + pad, y + h - 72, LandingArt.Text);
                LandingArt.DrawText(LandingArt.FontBody, LandingArt.Fit(LandingArt.FontBody, entry.Path, textWidth, keepEnd: true),
                    x + pad, y + h - 40, LandingArt.TextOnImage);

                // Round the corners by painting the page colour over them, then outline
                LandingArt.Slice(LandingArt.Mask, x, y, w, h, LandingArt.Radius, LandingArt.Background);
                LandingArt.Slice(LandingArt.Outline, x, y, w, h, LandingArt.Radius,
                    IsHot(this) ? LandingArt.Rgb(0xff775f, .7f) : LandingArt.Line);
            }
        }

        /// <summary>
        /// "Last session" panel next to the hero: what the editor recorded when the project was last closed.
        /// </summary>
        private class SessionCard : Frame
        {
            private readonly RecentProjectEntry entry;

            public Button ContinueButton { get; }

            public SessionCard(RecentProjectEntry entry)
            {
                this.entry = entry;
                Padding = new Margin(22);

                ContinueButton = new Button
                {
                    Text = "Continue",
                    Style = "landing.primary",
                    Size = new Point(100, 42),
                    Dock = DockStyle.Bottom,
                    Cursor = Cursors.Link,
                };
                Controls.Add(ContinueButton);
            }

            protected override void DrawStyle(Style style, float opacity)
            {
                int x = Location.x, y = Location.y, w = Size.x, h = Size.y;
                if (opacity == 0 || w <= 0 || h <= 0) return;

                LandingArt.Slice(LandingArt.Round, x, y, w, h, LandingArt.Radius, LandingArt.Card);
                LandingArt.Slice(LandingArt.Outline, x, y, w, h, LandingArt.Radius, LandingArt.Line);

                const int pad = 22;
                int inner = w - pad * 2;
                int line = y + pad;

                LandingArt.DrawText(LandingArt.FontEyebrow, "LAST SESSION", x + pad, line, LandingArt.Coral);
                line += 26;
                LandingArt.DrawText(LandingArt.FontHeading, LandingArt.Fit(LandingArt.FontHeading, entry.Name, inner), x + pad, line, LandingArt.Text);
                line += 44;

                Row("Scene", string.IsNullOrEmpty(entry.Scene) ? "—" : entry.Scene);
                Row("Entities", entry.EntityCount > 0 ? entry.EntityCount.ToString("N0") : "—");
                Row("Last opened", LandingArt.TimeAgo(entry.LastOpened));

                void Row(string label, string value)
                {
                    Gui.Renderer.DrawBox(x + pad, line, inner, 1, LandingArt.Line);
                    int labelWidth = LandingArt.Measure(LandingArt.FontBody, label).x;
                    value = LandingArt.Fit(LandingArt.FontBody, value, inner - labelWidth - 16);
                    LandingArt.DrawText(LandingArt.FontBody, label, x + pad, line + 11, LandingArt.Muted);
                    LandingArt.DrawText(LandingArt.FontBody, value, x + w - pad - LandingArt.Measure(LandingArt.FontBody, value).x, line + 11, LandingArt.Text);
                    line += 38;
                }
            }
        }

        /// <summary>
        /// A grid card: snapshot, name, path and last-opened time. With no entry it is the "New project" tile.
        /// </summary>
        private class ProjectCard : Control
        {
            private const int PreviewHeight = 148;
            private const int RemoveSize = 26;

            public RecentProjectEntry Entry;

            public ProjectCard()
            {
                Cursor = Cursors.Link;
            }

            /// <summary>True while the mouse is over the "remove from list" button of a hovered card.</summary>
            public bool IsOverRemove
            {
                get
                {
                    if (Entry == null) return false;
                    var mouse = Gui.MousePosition - Location;
                    int left = Size.x - RemoveSize - 8;
                    return mouse.x >= left && mouse.x < left + RemoveSize && mouse.y >= 8 && mouse.y < 8 + RemoveSize;
                }
            }

            protected override void DrawStyle(Style style, float opacity)
            {
                int x = Location.x, y = Location.y, w = Size.x, h = Size.y;
                if (opacity == 0 || w <= 0 || h <= 0) return;

                bool hot = IsHot(this);
                Tooltip = Entry != null && hot && IsOverRemove ? "Remove from recent projects" : null;

                if (Entry == null)
                {
                    DrawNewTile(x, y, w, h, hot);
                    return;
                }

                const int pad = 14;
                int inner = w - pad * 2;

                Gui.Renderer.DrawBox(x, y, w, h, hot ? LandingArt.CardHot : LandingArt.Card);
                DrawPreview(Entry, x, y, w, PreviewHeight);

                int line = y + PreviewHeight + 12;
                LandingArt.DrawText(LandingArt.FontTitle, LandingArt.Fit(LandingArt.FontTitle, Entry.Name, inner), x + pad, line, LandingArt.Text);
                line += 24;
                LandingArt.DrawText(LandingArt.FontSmall, LandingArt.Fit(LandingArt.FontSmall, Entry.Path, inner, keepEnd: true), x + pad, line, LandingArt.Faint);
                line += 18;
                LandingArt.DrawText(LandingArt.FontSmall, LandingArt.TimeAgo(Entry.LastOpened), x + pad, line, LandingArt.Muted);

                if (hot)
                {
                    int rx = x + w - RemoveSize - 8, ry = y + 8;
                    LandingArt.Slice(LandingArt.RoundSmall, rx, ry, RemoveSize, RemoveSize, LandingArt.RadiusSmall,
                        IsOverRemove ? LandingArt.Rgb(0x000000, .85f) : LandingArt.Rgb(0x000000, .55f));
                    var cross = LandingArt.Measure(LandingArt.FontTitle, "×");
                    LandingArt.DrawText(LandingArt.FontTitle, "×", rx + (RemoveSize - cross.x) / 2, ry + (RemoveSize - cross.y) / 2, LandingArt.Text);
                }

                LandingArt.Slice(LandingArt.Mask, x, y, w, h, LandingArt.Radius, LandingArt.Background);
                LandingArt.Slice(LandingArt.Outline, x, y, w, h, LandingArt.Radius,
                    hot ? LandingArt.Rgb(0xff775f, .6f) : LandingArt.Line);
            }

            private static void DrawNewTile(int x, int y, int w, int h, bool hot)
            {
                const int pad = 18;

                LandingArt.Slice(LandingArt.Round, x, y, w, h, LandingArt.Radius, hot ? LandingArt.Card : LandingArt.Well);
                LandingArt.Slice(LandingArt.Outline, x, y, w, h, LandingArt.Radius, LandingArt.Rgb(0xff775f, hot ? .7f : .3f));

                const int plus = 36;
                LandingArt.Slice(LandingArt.RoundSmall, x + pad, y + pad, plus, plus, LandingArt.RadiusSmall, LandingArt.Coral);
                var glyph = LandingArt.Measure(LandingArt.FontHeading, "+");
                LandingArt.DrawText(LandingArt.FontHeading, "+", x + pad + (plus - glyph.x) / 2, y + pad + (plus - glyph.y) / 2, LandingArt.OnCoral);

                LandingArt.DrawText(LandingArt.FontTitle, "New project", x + pad, y + h - 58, LandingArt.Text);
                LandingArt.DrawText(LandingArt.FontSmall, "Start from an empty folder.", x + pad, y + h - 34, LandingArt.Muted);
            }
        }
    }
}
