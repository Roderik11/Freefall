using System;
using Squid;

namespace Freefall.Editor
{
    public static class MenuBuilder
    {
        // Gap kept between a popup and the desktop edge
        private const int ScreenMargin = 8;

        public static DropDownButton CreateMenuItem(Frame parent, string name, bool hotDrop = true)
        {
            DropDownButton btn = new DropDownButton();
            btn.Align = Alignment.BottomLeft;
            btn.HotDrop = hotDrop;
            btn.Text = name;
            btn.Size = new Point(70, 26);
            btn.MinSize = new Point(70, 20);
            btn.Dock = DockStyle.Left;
            btn.Style = "menu";
            btn.Margin = new Margin(1, 1, 0, 1);
            btn.AutoSize = AutoSize.Horizontal;
            btn.Dropdown.Size = new Point(280, 300);
            btn.Dropdown.Style = "popup";
            btn.Dropdown.Padding = new Squid.Margin(4);
            btn.OnOpened += (s, a) => FitToDesktop(btn);
            parent.Controls.Add(btn);

            return btn;
        }

        public static void AddSeparator(DropDownButton parent)
        {
            ImageControl frame = new ImageControl
            {
                Size = new Point(1, 1),
                Texture = "[]whitetex.png",
                Dock = DockStyle.Top,
                Color = ColorInt.ARGB(1f, .25f, .25f, .25f),
                Margin = new Margin(8, 4, 8, 4),
                Tiling = TextureMode.Stretch
            };

            Items(parent).Controls.Add(frame);
        }

        public static DropDownButton AddMenuItem(DropDownButton parent, string name, MouseEvent? mouseClick)
        {
            DropDownButton btn = new DropDownButton();
            btn.HotDrop = true;
            btn.Align = Alignment.TopRight;
            btn.Text = name;
            btn.Size = new Point(100, 26);
            btn.Dock = DockStyle.Top;
            btn.Style = "menuitem";
            btn.Margin = new Margin(0, 0, 0, 1);
            btn.Dropdown.MinSize = new Point();
            btn.Dropdown.Size = new Point(200, 300);
            btn.Dropdown.Style = "popup";
            btn.Dropdown.Padding = new Squid.Margin(4);
            btn.OnOpened += (s, a) => FitToDesktop(btn);
            btn.MouseClick += (s, e) => { parent.Close(); };

            if (mouseClick != null)
                btn.MouseClick += mouseClick;

            Items(parent).Controls.Add(btn);

            return btn;
        }

        /// <summary>Removes every item from a menu's popup.</summary>
        public static void ClearItems(DropDownButton menu)
        {
            menu.Dropdown.Controls.Clear();
        }

        /// <summary>
        /// The frame a menu's items go into. Items sit in a scroll panel, so a menu with more
        /// items than fit on the desktop scrolls. The panel is created with the first item:
        /// DropDownButton treats an empty Dropdown as "no submenu".
        /// </summary>
        private static Frame Items(DropDownButton menu)
        {
            var popup = menu.Dropdown.Controls;

            if (popup.Count > 0 && popup[0] is ScrollPanel existing)
                return existing.Content;

            var panel = new ScrollPanel();
            popup.Add(panel);

            return panel.Content;
        }

        /// <summary>
        /// Sizes an opened popup to its items, limited to the room the desktop has left. A submenu
        /// slides up to fit first; a menu bar popup stays below its button. What doesn't fit scrolls.
        /// </summary>
        private static void FitToDesktop(DropDownButton menu)
        {
            var dropdown = menu.Dropdown;
            var desktop = dropdown.Desktop;

            if (desktop == null || dropdown.Controls.Count == 0 || dropdown.Controls[0] is not ScrollPanel panel)
                return;

            int wanted = panel.Content.Size.y + dropdown.Padding.Top + dropdown.Padding.Bottom;
            int bottom = desktop.Size.y - ScreenMargin;
            var position = dropdown.Position;

            int top = menu.Align == Alignment.BottomLeft ? position.y : ScreenMargin;
            int height = Math.Min(wanted, bottom - top);
            position.y = Math.Max(top, Math.Min(position.y, bottom - height));

            dropdown.Position = position;
            dropdown.Size = new Point(dropdown.Size.x, height);

            // Decided here rather than by the panel's next update, so the first frame is already laid out right
            panel.VScroll.Visible = wanted > height;
            panel.Scroll(0);

            dropdown.PerformLayout();
        }
    }
}
