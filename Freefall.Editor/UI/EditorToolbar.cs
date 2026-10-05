using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Toolbar strip for editor windows (graph, animation): compact icon + label buttons that
    /// only show a background on hover, with thin separators between groups.
    /// </summary>
    public static class EditorToolbar
    {
        public const int Height = 38;

        public static Frame Create()
        {
            return new Frame
            {
                Style = "frame",
                Size = new Point(16, Height),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1),
                Padding = new Margin(5),
            };
        }

        /// <param name="icon">One of the EditorSkin.Icon* textures.</param>
        /// <param name="primary">Accent-coloured icon, for the action the window exists for.</param>
        public static ToolbarButton Add(Frame toolbar, string icon, string text, MouseEvent handler, bool primary = false)
        {
            var button = new ToolbarButton(icon, text, primary);
            button.MouseClick += handler;
            toolbar.Controls.Add(button);
            return button;
        }

        public static void AddSeparator(Frame toolbar)
        {
            toolbar.Controls.Add(new Frame
            {
                Style = "toolbarSeparator",
                Size = new Point(1, 18),
                Dock = DockStyle.Left,
                Margin = new Margin(7, 5, 7, 5),
            });
        }
    }

    public class ToolbarButton : Button
    {
        private const string LabelFont = EditorSkin.HeadingFont;
        private const int IconSize = 18;
        private const int Pad = 10;
        private const int Gap = 7;

        private readonly string icon;
        private readonly string label;
        private readonly bool primary;

        public ToolbarButton(string icon, string label, bool primary)
        {
            this.icon = icon;
            this.label = label;
            this.primary = primary;

            Style = "";
            Dock = DockStyle.Left;
            Margin = new Margin(0, 0, 2, 0);

            int textWidth = Gui.Renderer.GetTextSize(label, Gui.Renderer.GetFont(LabelFont)).x;
            Size = new Point(Pad + IconSize + Gap + textWidth + Pad, EditorToolbar.Height - 10);
        }

        protected override void DrawStyle(Style style, float opacity)
        {
            int x = Location.x, y = Location.y, w = Size.x, h = Size.y;
            if (opacity == 0) return;

            bool pressed = State == ControlState.Pressed;
            bool hot = pressed || State == ControlState.Hot;

            if (hot)
                LandingArt.Slice(EditorSkin.ButtonArt, x, y, w, h, EditorSkin.ArtInset,
                    ColorInt.ARGB(pressed ? .16f : .09f, 1f, 1f, 1f));

            int iconColor = primary ? LandingArt.Coral : ColorInt.ARGB(1f, hot ? .92f : .68f, hot ? .92f : .70f, hot ? .92f : .74f);
            LandingArt.Centered(icon, x + Pad + IconSize / 2, y + h / 2, iconColor);

            int font = Gui.Renderer.GetFont(LabelFont);
            int textHeight = Gui.Renderer.GetTextSize(label, font).y;
            Gui.Renderer.DrawText(label, x + Pad + IconSize + Gap, y + (h - textHeight) / 2, font,
                ColorInt.ARGB(1f, hot ? .95f : .80f, hot ? .95f : .80f, hot ? .95f : .80f));
        }
    }
}
