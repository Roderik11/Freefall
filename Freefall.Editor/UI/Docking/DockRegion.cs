using Squid;
using Freefall.Base;

namespace Freefall.Editor
{
    public class DockRegion : Frame
    {
        private readonly DockGroup dockGroup;

        // A group shown alone over the whole region, the split frame it came from, and the layout hidden meanwhile
        private DockGroup? _maximized;
        private Control? _maximizedHome;
        private Control? _hiddenRoot;

        public DockRegion()
        {
            Style = "window";
            Padding = new Margin(0);

            Size = new Point(320, 300);
            Dock = DockStyle.Fill;

            dockGroup = new DockGroup();
            dockGroup.Dock = DockStyle.Fill;
            Controls.Add(dockGroup);

            // Docking rearranges the tree: put a maximized group back first
            MessageDispatcher.AddListener(Msg.StartDockDrag, _ => Maximize(null));
        }

        /// <summary>The group shown alone over the whole region, or null.</summary>
        public DockGroup? Maximized => _maximized;

        /// <summary>
        /// Show one group alone over the whole region, the rest of the layout hidden but kept; null puts it back
        /// into its split. Double-clicking a tab toggles its group.
        /// </summary>
        public void Maximize(DockGroup? group)
        {
            if (_maximized == group) return;

            if (_maximized != null)
            {
                _maximized.Parent = _maximizedHome;
                if (_hiddenRoot != null) _hiddenRoot.Visible = true;
                _maximized = null;
                _maximizedHome = null;
                _hiddenRoot = null;
            }

            // Nothing to do for a group that already is the whole region
            if (group == null || group.Parent == this || group.Parent == null) return;

            _hiddenRoot = Controls.Count > 0 ? Controls[0] : null;
            _maximizedHome = group.Parent;
            group.Parent = this;
            if (_hiddenRoot != null) _hiddenRoot.Visible = false;
            _maximized = group;
        }

        public void ToggleMaximize(DockGroup group) => Maximize(_maximized == group ? null : group);

        private void Tab_MouseDoubleClick(Control sender, MouseEventArgs args)
        {
            if (args.Button == 0 && sender.Tag is TabPage { Tag: DockGroup group })
                ToggleMaximize(group);
        }

        public DockGroup DockContent(string name, Control control, DockStyle dock = DockStyle.Fill, bool header = true)
        {
            if (control == null) return null!;
            control.Dock = DockStyle.Fill;

            var page = new TabPage();
            page.Button.Text = name;
            page.Button.Size = new Point(128, 26);
            page.Button.Margin = new Margin(0, 0, 1, 0);
            page.Button.Style = "tab";
            page.Button.MouseDrag += Button_MouseDrag;
            page.Button.MouseDoubleClick += Tab_MouseDoubleClick;
            page.Button.Tag = page;
            page.Button.GetElements().Add(new Button
            {
                Size = new Point(26, 26),
                Dock = DockStyle.Right,
                Style = "closetab"
            });
            page.Controls.Add(control);
            var result = dockGroup.DockContent(page, dock);

            if (!header)
                dockGroup.ButtonFrame.Visible = false;

            return result;
        }

        public DockGroup DockContent(DockGroup group, string name, Control control, DockStyle dock = DockStyle.Fill, bool header = true)
        {
            if (control == null) return null!;
            control.Dock = DockStyle.Fill;

            var page = new TabPage();
            page.Button.Text = name;
            page.Button.Size = new Point(128, 26);
            page.Button.Margin = new Margin(0, 0, 1, 0);
            page.Button.Style = "tab";
            page.Button.MouseDrag += Button_MouseDrag;
            page.Button.MouseDoubleClick += Tab_MouseDoubleClick;
            page.Button.Tag = page;
            page.Button.GetElements().Add(new Button
            {
                Size = new Point(26, 26),
                Dock = DockStyle.Right,
                Style = "closetab"
            });
            page.Size = control.Size;
            page.Controls.Add(control);
            var result = group.DockContent(page, dock);

            if (!header)
                group.ButtonFrame.Visible = false;

            return result;
        }

        private void Button_MouseDrag(Control sender, MouseEventArgs args)
        {
            Button image = new Button();
            image.Style = "button";
            image.Text = ((Button)sender).Text;
            image.Size = sender.Size;
            image.Position = sender.Location;
            image.TextAlign = ((Button)sender).TextAlign;
            image.Tag = sender.Tag;

            sender.DoDragDrop(image);

            MessageDispatcher.Send(Msg.StartDockDrag);
            Desktop.DragDropEnded += Desktop_DragDropEnded;
        }

        private void Desktop_DragDropEnded(Control sender, DragDropEventArgs e)
        {
            Desktop.DragDropEnded -= Desktop_DragDropEnded;
            MessageDispatcher.Send(Msg.EndDockDrag);
        }
    }
}
