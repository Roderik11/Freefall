using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Shared skin setup for all editor desktops (landing page, editor, etc.).
    /// Extracted from EditorDesktop so multiple Desktop subclasses share consistent styling.
    /// </summary>
    public static class EditorSkin
    {
        // Generated control art (see EnsureArt)
        public const string ButtonArt = "ui_button";                // white, radius 4 — tinted per state
        private const string FieldArt = "ui_field";                 // inset well with a hairline border
        private const string FieldHotArt = "ui_field_hot";
        private const string FieldFocusArt = "ui_field_focus";      // accent border
        private const string UnderlineArt = "ui_underline";         // white 2px bottom edge
        private const string PopupArt = "ui_popup";                 // rounded panel with a light border
        public const int ArtInset = 5;
        private const int PopupInset = 8;

        /// <summary>Corner mask (opaque outside radius 6) for rounding cards; tint with the colour behind them.</summary>
        public const string CardMaskArt = "ui_card_mask";
        public const int CardMaskInset = 7;

        /// <summary>Folder icon for asset browser cards (128×128, draw centred).</summary>
        public const string FolderArt = "ui_folder";

        /// <summary>Roboto Medium, a step up from body text: panel and category headers.</summary>
        public const string HeadingFont = "ui_heading";

        // Toolbar glyphs: white, 18×18, tinted when drawn (see EditorToolbar)
        public const string IconSave = "ui_icon_save";
        public const string IconOpen = "ui_icon_open";
        public const string IconPlay = "ui_icon_play";
        public const string IconPlus = "ui_icon_plus";

        private static readonly HashSet<string> _plugArt = new();

        /// <summary>
        /// Graph plug shapes rasterized at the exact on-screen diameter, so they stay crisp at any
        /// canvas zoom. Returns the texture name: a ring, or the dot that sits inside it.
        /// </summary>
        public static string PlugArt(int diameter, bool dot)
        {
            string name = $"ui_plug_{(dot ? "dot" : "ring")}_{diameter}";
            if (_plugArt.Add(name) && Gui.Renderer is SquidRenderer renderer)
            {
                LandingArt.Insert(renderer, name, diameter + 2, diameter + 2, (g, r) =>
                {
                    if (dot)
                    {
                        float size = diameter * .46f;
                        g.FillEllipse(System.Drawing.Brushes.White, (r.Width - size) / 2, (r.Height - size) / 2, size, size);
                    }
                    else
                    {
                        float stroke = MathF.Max(1.5f, diameter * .14f);
                        using var pen = new System.Drawing.Pen(System.Drawing.Color.White, stroke);
                        g.DrawEllipse(pen, 1 + stroke / 2, 1 + stroke / 2, diameter - stroke, diameter - stroke);
                    }
                });
            }
            return name;
        }

        /// <summary>Graph editor node card colours (drawn by GraphFrame).</summary>
        public static readonly int NodeBodyColor = Cool(.085f);
        public static readonly int NodeHeaderColor = Cool(.19f);

        /// <summary>Background of docked panels ("frame" style).</summary>
        public static readonly int PanelColor = Cool(.125f);

        private static bool _artReady;

        /// <summary>
        /// Neutral grey pulled slightly towards blue, so panels share the landing page's cool tone.
        /// </summary>
        private static int Cool(float v) => ColorInt.ARGB(1f, v - .016f, v - .004f, v + .020f);

        private static void EnsureArt()
        {
            if (_artReady || Gui.Renderer is not SquidRenderer renderer) return;
            _artReady = true;

            LandingArt.Insert(renderer, ButtonArt, 16, 16, (g, r) =>
            {
                using var path = LandingArt.RoundedRect(r, 4);
                g.FillPath(System.Drawing.Brushes.White, path);
            });

            void Field(string name, System.Drawing.Color border)
            {
                LandingArt.Insert(renderer, name, 16, 16, (g, r) =>
                {
                    using (var fill = LandingArt.RoundedRect(r, 4))
                    using (var brush = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(255, 14, 17, 23)))
                        g.FillPath(brush, fill);

                    r.Inflate(-.5f, -.5f);
                    using var edge = LandingArt.RoundedRect(r, 3.5f);
                    using var pen = new System.Drawing.Pen(border, 1f);
                    g.DrawPath(pen, edge);
                });
            }

            Field(FieldArt, System.Drawing.Color.FromArgb(22, 255, 255, 255));
            Field(FieldHotArt, System.Drawing.Color.FromArgb(70, 255, 255, 255));
            Field(FieldFocusArt, System.Drawing.Color.FromArgb(230, 255, 119, 95));

            LandingArt.Insert(renderer, UnderlineArt, 8, 8, (g, r) =>
                g.FillRectangle(System.Drawing.Brushes.White, 0, 6, 8, 2));

            LandingArt.Insert(renderer, PopupArt, 24, 24, (g, r) =>
            {
                using (var fill = LandingArt.RoundedRect(r, 6))
                using (var brush = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(255, 24, 28, 36)))
                    g.FillPath(brush, fill);

                r.Inflate(-.5f, -.5f);
                using var edge = LandingArt.RoundedRect(r, 5.5f);
                using var pen = new System.Drawing.Pen(System.Drawing.Color.FromArgb(46, 255, 255, 255), 1f);
                g.DrawPath(pen, edge);
            });

            LandingArt.Insert(renderer, CardMaskArt, 16, 16, (g, r) =>
            {
                using var path = LandingArt.RoundedRect(r, 6);
                path.AddRectangle(r);       // alternate fill: the rect minus the rounded rect
                g.FillPath(System.Drawing.Brushes.White, path);
            });

            LandingArt.Insert(renderer, FolderArt, 128, 128, (g, r) =>
            {
                // Back plate with its tab, then a lighter front flap
                using var back = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(255, 74, 86, 108));
                using var tab = LandingArt.RoundedRect(new System.Drawing.RectangleF(22, 28, 40, 22), 6);
                using var plate = LandingArt.RoundedRect(new System.Drawing.RectangleF(22, 38, 84, 62), 7);
                g.FillPath(back, tab);
                g.FillPath(back, plate);

                var flap = new System.Drawing.RectangleF(22, 50, 84, 50);
                using var front = new System.Drawing.Drawing2D.LinearGradientBrush(flap,
                    System.Drawing.Color.FromArgb(255, 124, 140, 168), System.Drawing.Color.FromArgb(255, 102, 117, 144), 90f);
                using var flapPath = LandingArt.RoundedRect(flap, 7);
                g.FillPath(front, flapPath);
            });

            renderer.RegisterRuntimeFont(HeadingFont, "Roboto/Roboto-Medium.ttf", 12);

            void Icon(string name, Action<System.Drawing.Graphics, System.Drawing.Pen> paint)
            {
                LandingArt.Insert(renderer, name, 18, 18, (g, r) =>
                {
                    g.ScaleTransform(18f / 16f, 18f / 16f);     // glyphs are authored on a 16px grid
                    using var pen = new System.Drawing.Pen(System.Drawing.Color.White, 1.4f)
                    {
                        LineJoin = System.Drawing.Drawing2D.LineJoin.Round,
                        StartCap = System.Drawing.Drawing2D.LineCap.Round,
                        EndCap = System.Drawing.Drawing2D.LineCap.Round,
                    };
                    paint(g, pen);
                });
            }

            Icon(IconSave, (g, pen) =>
            {
                g.DrawPolygon(pen, new System.Drawing.PointF[] { new(2.5f, 2.5f), new(10.5f, 2.5f), new(13.5f, 5.5f), new(13.5f, 13.5f), new(2.5f, 13.5f) });
                g.DrawRectangle(pen, 5f, 2.5f, 4.5f, 3.5f);
                g.DrawRectangle(pen, 5f, 9f, 6f, 4.5f);
            });
            Icon(IconOpen, (g, pen) =>
                g.DrawPolygon(pen, new System.Drawing.PointF[] { new(1.5f, 3.5f), new(6f, 3.5f), new(7.5f, 5.5f), new(14.5f, 5.5f), new(14.5f, 12.5f), new(1.5f, 12.5f) }));
            Icon(IconPlay, (g, pen) =>
                g.FillPolygon(System.Drawing.Brushes.White, new System.Drawing.PointF[] { new(4.5f, 2.5f), new(13.5f, 8f), new(4.5f, 13.5f) }));
            Icon(IconPlus, (g, pen) =>
            {
                pen.Width = 1.7f;
                g.DrawLine(pen, 8f, 3.5f, 8f, 12.5f);
                g.DrawLine(pen, 3.5f, 8f, 12.5f, 8f);
            });
        }

        public static void Apply(Desktop desktop)
        {
            EnsureArt();

            int accent = LandingArt.Coral;

            int grey10 = Cool(.10f);
            int grey125 = Cool(.125f);
            int grey15 = Cool(.15f);
            int grey166 = Cool(.166f);
            int grey170 = Cool(.170f);
            int grey175 = Cool(.175f);
            int grey20 = Cool(.20f);
            int grey25 = Cool(.25f);
            int grey30 = Cool(.30f);
            int grey35 = Cool(.35f);
            int grey40 = Cool(.40f);
            int textColor = ColorInt.ARGB(1f, .80f, .80f, .80f);
            int darkColor = grey10;

            int windowColor = darkColor;
            int normalColor = PanelColor;
            int hotColor = grey20;
            // Selection: the old grey25 warmed ~10% towards the accent
            int selectedColor = ColorInt.ARGB(1f, .305f, .250f, .262f);

            ControlStyle baseStyle = new ControlStyle();
            baseStyle.Font = "roboto_regular_10";
            baseStyle.TextColor = textColor;

            ControlStyle colorstyle = new ControlStyle(baseStyle);
            colorstyle.BackColor = -1;
            colorstyle.Texture = "border_black.dds";
            colorstyle.Tiling = TextureMode.Grid;
            colorstyle.Grid = new Squid.Margin(4);

            ControlStyle dropshadow = new ControlStyle(baseStyle);
            dropshadow.Texture = "dropshadow.dds";
            dropshadow.Tiling = TextureMode.Grid;
            dropshadow.Grid = new Squid.Margin(4);
            dropshadow.Tint = ColorInt.ARGB(.7f, 1, 1, 1);

            ControlStyle window = new ControlStyle(baseStyle);
            window.BackColor = windowColor;

            ControlStyle frame = new ControlStyle(baseStyle);
            frame.BackColor = normalColor;
            frame.TextPadding = new Squid.Margin(8, 0, 0, 0);

            ControlStyle category = new ControlStyle(baseStyle);
            category.BackColor = grey20;
            category.Hot.BackColor = grey25;
            category.TextPadding = new Squid.Margin(0, 0, 0, 0);
            category.TextAlign = Alignment.MiddleLeft;

            var subcategory = new ControlStyle(category);
            subcategory.BackColor = grey20;
            subcategory.TextPadding = new Squid.Margin(28, 0, 0, 0);

            var catbutton = new ControlStyle(category);
            catbutton.TextAlign = Alignment.MiddleCenter;

            ControlStyle colorGrey170 = new ControlStyle();
            colorGrey170.BackColor = grey166;

            ControlStyle label = new ControlStyle(baseStyle);
            label.TextPadding = new Squid.Margin(2, 0, 0, 0);

            ControlStyle propertyLabel = new ControlStyle(category);
            propertyLabel.TextPadding = new Squid.Margin(28, 0, 0, 0);
            propertyLabel.BackColor = grey170;
            propertyLabel.Hot.BackColor = grey20;

            ControlStyle propertyElement = new ControlStyle(category);
            propertyElement.TextPadding = new Squid.Margin(0, 0, 0, 0);
            propertyElement.BackColor = grey166;
            propertyElement.Hot.BackColor = grey20;
            propertyElement.Selected.BackColor = grey25;
            propertyElement.SelectedHot.BackColor = grey25;
            propertyElement.SelectedPressed.BackColor = grey25;
            propertyElement.SelectedFocused.BackColor = grey25;
            propertyElement.Checked.BackColor = grey25;
            propertyElement.CheckedHot.BackColor = grey25;

            var header = new ControlStyle(propertyElement);
            header.TextPadding = new Squid.Margin(8, 0, 0, 0);
            header.TextAlign = Alignment.MiddleLeft;

            ControlStyle propertyIndent = new ControlStyle(category);
            propertyIndent.TextPadding = new Squid.Margin(0, 0, 0, 0);
            propertyIndent.BackColor = grey170;
            propertyIndent.Hot.BackColor = grey20;
            propertyIndent.Texture = "shadow_right.png";
            propertyIndent.Tiling = TextureMode.RepeatX;

            ControlStyle statusBarLabel = new ControlStyle(label);
            statusBarLabel.BackColor = grey15;
            statusBarLabel.TextPadding = new Margin(16, 0, 0, 0);

            var tab = new ControlStyle(baseStyle);
            tab.TextAlign = Alignment.MiddleLeft;
            tab.TextPadding = new Squid.Margin(8, 0, 0, 0);
            tab.BackColor = grey10;
            tab.Hot.BackColor = grey125;
            tab.Selected.BackColor = grey15;
            tab.SelectedHot.BackColor = grey20;
            tab.SelectedPressed.BackColor = grey20;
            tab.SelectedFocused.BackColor = grey20;
            tab.Checked.BackColor = grey15;
            tab.CheckedHot.BackColor = grey20;
            tab.Pressed.BackColor = grey20;
            tab.CheckedPressed.BackColor = grey20;
            tab.SelectedPressed.BackColor = grey20;

            // Active tab: accent underline
            foreach (var state in new[] { tab.Selected, tab.SelectedHot, tab.SelectedPressed, tab.SelectedFocused,
                                          tab.Checked, tab.CheckedHot, tab.CheckedPressed })
            {
                state.Texture = UnderlineArt;
                state.Tiling = TextureMode.Grid;
                state.Grid = new Squid.Margin(1, 1, 1, 2);
                state.Tint = accent;
            }

            // Square, flat-colour button: the base for list items and anything that tiles edge to edge
            ControlStyle flatButton = new ControlStyle(baseStyle);
            flatButton.TextAlign = Alignment.MiddleCenter;
            flatButton.BackColor = normalColor;
            flatButton.Hot.BackColor = hotColor;
            flatButton.Selected.BackColor = selectedColor;
            flatButton.SelectedHot.BackColor = selectedColor;
            flatButton.SelectedPressed.BackColor = selectedColor;
            flatButton.SelectedFocused.BackColor = selectedColor;
            flatButton.Checked.BackColor = selectedColor;
            flatButton.CheckedHot.BackColor = selectedColor;

            // Standalone button: rounded, tinted per state
            ControlStyle button = new ControlStyle(baseStyle);
            button.TextAlign = Alignment.MiddleCenter;
            button.Texture = ButtonArt;
            button.Tiling = TextureMode.Grid;
            button.Grid = new Squid.Margin(ArtInset);
            button.Tint = grey20;
            button.Hot.Tint = grey25;
            button.Pressed.Tint = grey30;
            button.Selected.Tint = grey30;
            button.SelectedHot.Tint = grey30;
            button.SelectedPressed.Tint = grey30;
            button.SelectedFocused.Tint = grey30;
            button.Checked.Tint = grey30;
            button.CheckedHot.Tint = grey30;

            ControlStyle item = new ControlStyle(flatButton);
            item.TextAlign = Alignment.MiddleLeft;
            item.TextPadding = new Squid.Margin(8, 0, 0, 0);
            item.BackColor = 0;
            item.Hot.BackColor = hotColor;
            item.Selected.BackColor = selectedColor;
            item.SelectedHot.BackColor = selectedColor;
            item.SelectedPressed.BackColor = selectedColor;
            item.SelectedFocused.BackColor = selectedColor;
            item.Checked.BackColor = selectedColor;
            item.CheckedHot.BackColor = selectedColor;


            var prefabItem = new ControlStyle(item);
            prefabItem.TextColor = ColorInt.ARGB(1f, .4f, .6f, .8f);

            var node = new ControlStyle(item);
            node.Default.BackColor = ColorInt.ARGB(0, 0, 0, 0);

            ControlStyle indent18 = new ControlStyle(item);
            indent18.Texture = "shadow_18.png";
            indent18.Tiling = TextureMode.Repeat;

            var menu = new ControlStyle(item);
            menu.Default.BackColor = ColorInt.ARGB(0, 0, 0, 0);
            menu.TextPadding = new Margin(8, 0, 8, 0);
            menu.TextAlign = Alignment.MiddleCenter;

            var menuitem = new ControlStyle(item);
            menuitem.Default.BackColor = ColorInt.ARGB(0, 0, 0, 0);
            menuitem.TextPadding = new Margin(32, 0, 8, 0);

            // Menus highlight with a rounded pill instead of a full-width bar
            foreach (var state in new[] { menu.Hot, menu.Pressed, menuitem.Hot, menuitem.Pressed })
            {
                state.BackColor = 0;
                state.Texture = ButtonArt;
                state.Tiling = TextureMode.Grid;
                state.Grid = new Squid.Margin(ArtInset);
                state.Tint = grey25;
            }

            // Dropdowns, menus and tooltips: rounded panel with a light border (pad contents by 4)
            var popup = new ControlStyle(baseStyle);
            popup.Texture = PopupArt;
            popup.Tiling = TextureMode.Grid;
            popup.Grid = new Squid.Margin(PopupInset);

            ControlStyle textbox = new ControlStyle(baseStyle);
            textbox.Texture = FieldArt;
            textbox.TextAlign = Alignment.MiddleLeft;
            textbox.TextPadding = new Squid.Margin(6, 0, 6, 0);
            textbox.Hot.Texture = FieldHotArt;
            textbox.Focused.Texture = FieldFocusArt;
            textbox.Tiling = TextureMode.Grid;
            textbox.Grid = new Squid.Margin(ArtInset);

            var dropdownLabel = new ControlStyle(baseStyle);
            dropdownLabel.TextPadding = new Squid.Margin(6, 0, 6, 0);

            var searchbox = new ControlStyle(textbox);
            searchbox.TextPadding = new Margin(20, 0, 20, 0);

            ControlStyle down = new ControlStyle();
            down.Texture = "icon_down.png";

            ControlStyle switchbutton = new ControlStyle(baseStyle);
            switchbutton.Texture = "switch_off.png";
            switchbutton.Checked.Texture = "switch_on.png";
            switchbutton.CheckedHot.Texture = "switch_on.png";
            switchbutton.Tiling = TextureMode.Center;

            ControlStyle checkbox = new ControlStyle(textbox);
            checkbox.CheckedHot.Texture = FieldHotArt;

            ControlStyle border = new ControlStyle();
            border.Texture = "border.dds";
            border.Tiling = TextureMode.Grid;
            border.Grid = new Squid.Margin(4);

            ControlStyle darkborder = new ControlStyle();
            darkborder.Texture = "border.dds";
            darkborder.Tiling = TextureMode.Grid;
            darkborder.Grid = new Squid.Margin(4);
            darkborder.Tint = ColorInt.ARGB(1f, 0.55f, 0.55f, 0.55f);

            ControlStyle scrollbutton = new ControlStyle(baseStyle);
            scrollbutton.TextAlign = Alignment.MiddleCenter;
            scrollbutton.Texture = ButtonArt;
            scrollbutton.Tiling = TextureMode.Grid;
            scrollbutton.Grid = new Squid.Margin(ArtInset);
            scrollbutton.Tint = grey30;
            scrollbutton.Hot.Tint = grey40;
            scrollbutton.Pressed.Tint = grey40;

            ControlStyle scroll = new ControlStyle(baseStyle);
            scroll.BackColor = grey125;

            var multiline = new ControlStyle(baseStyle);
            multiline.TextPadding = new Squid.Margin(4);
            multiline.TextAlign = Alignment.TopLeft;

            var dark = new ControlStyle(baseStyle);
            dark.BackColor = grey15;

            var dark2 = new ControlStyle(baseStyle);
            dark2.BackColor = grey10;

            var checkmark = new ControlStyle(baseStyle);
            checkmark.Texture = "checkmark.png";
            checkmark.Tint = ColorInt.ARGB(1f, .8f, .8f, .8f);

            var tile = new ControlStyle(baseStyle);
            tile.BackColor = grey20;
            tile.Hot.BackColor = grey25;

            var viewport = new ControlStyle(baseStyle);
            viewport.Texture = "viewport_border.dds";
            viewport.Grid = new Margin(3);
            viewport.Tiling = TextureMode.Grid;
            viewport.Tint = darkColor;

            var inport = new ControlStyle();
            inport.Tiling = TextureMode.Center;
            inport.Texture = "port.png";
            inport.Hot.Texture = "port_hot.png";
            inport.Selected.Texture = "port_hot.png";
            inport.Pressed.Texture = "port_hot.png";
            inport.SelectedPressed.Texture = "port_hot.png";
            inport.SelectedHot.Texture = "port_hot.png";
            inport.Tint = ColorInt.ARGB(1f, .42f, .66f, .95f);

            var outport = new ControlStyle(inport);
            outport.Tint = ColorInt.ARGB(1f, 1f, .52f, .40f);

            var closebutton = new ControlStyle(flatButton);
            closebutton.Texture = "close.dds";
            closebutton.Tiling = TextureMode.Center;

            var closetab = new ControlStyle();
            closetab.Texture = "close.dds";
            closetab.Tint = ColorInt.ARGB(1f, .5f, .5f, .5f);
            closetab.Hot.Tint = ColorInt.ARGB(1f, .75f, .75f, .75f);
            closetab.Tiling = TextureMode.Center;

            var alterRows = new ControlStyle(flatButton);
            alterRows.Texture = "alter_rows.png";
            alterRows.Tiling = TextureMode.RepeatY;

            var iconplus = new ControlStyle();
            iconplus.Texture = "icon_plus.png";
            iconplus.Tint = ColorInt.ARGB(1f, .5f, .5f, .5f);
            iconplus.Hot.Tint = ColorInt.ARGB(1f, .8f, .8f, .8f);

            var iconplay = new ControlStyle();
            iconplay.Texture = "icon_play.png";
            iconplay.Tint = ColorInt.ARGB(1f, .5f, .5f, .5f);
            iconplay.Hot.Tint = ColorInt.ARGB(1f, .8f, .8f, .8f);

            var iconstop = new ControlStyle();
            iconstop.Texture = "icon_stop.png";
            iconstop.Tint = ColorInt.ARGB(1f, .5f, .5f, .5f);
            iconstop.Hot.Tint = ColorInt.ARGB(1f, .8f, .8f, .8f);


            var iconclose = new ControlStyle();
            iconclose.Texture = "close.dds";
            iconclose.Tint = ColorInt.ARGB(1f, .5f, .5f, .5f);
            iconclose.Hot.Tint = ColorInt.ARGB(1f, .8f, .8f, .8f);
            iconclose.Tiling = TextureMode.Center;

            var foldout = new ControlStyle();
            foldout.Tiling = TextureMode.Center;
            foldout.Texture = "icon_right.png";
            foldout.Checked.Texture = "icon_down.png";

            var canvas = new ControlStyle(frame);
            canvas.BackColor = Cool(.105f);

            var toolbarSeparator = new ControlStyle();
            toolbarSeparator.BackColor = ColorInt.ARGB(.12f, 1f, 1f, 1f);

            // Graph node title: text only, the node card paints the band behind it
            var graphTitle = new ControlStyle(baseStyle);
            graphTitle.Font = HeadingFont;
            graphTitle.TextAlign = Alignment.MiddleLeft;
            graphTitle.TextPadding = new Squid.Margin(10, 0, 0, 0);

            // Slider track: thin line texture (transparent with 2px center stripe)
            var sliderTrack = new ControlStyle();
            sliderTrack.Texture = "slider_track.png";
            sliderTrack.Tiling = TextureMode.RepeatX;

            // Slider thumb: round circle texture
            var sliderThumb = new ControlStyle();
            sliderThumb.Texture = "slider_thumb.png";
            sliderThumb.Tiling = TextureMode.Center;
            sliderThumb.Hot.Tint = ColorInt.ARGB(1f, 1f, 1f, 1f);
            sliderThumb.Pressed.Tint = ColorInt.ARGB(1f, .85f, .85f, .85f);

            var prevtoolbutton = new ControlStyle(button);
            prevtoolbutton.TextPadding = new Squid.Margin(4, 0, 4, 0);

            // Set last: the property-row styles above are copies of category and must keep the body font
            category.Font = HeadingFont;
            catbutton.Font = HeadingFont;
            header.Font = HeadingFont;
            tab.Font = HeadingFont;

            Skin skin = new Skin
            {
                { "popup", popup },
                { "prevtoolbutton", prevtoolbutton},
                { "foldout", foldout},
                { "iconplus", iconplus },
                { "iconclose", iconclose },
                { "iconplay", iconplay },
                { "iconstop", iconstop },
                { "canvas", canvas },
                { "graphTitle", graphTitle },
                { "toolbarSeparator", toolbarSeparator },
                { "window", window },
                { "frame", frame },
                { "label", label },
                { "statusBarLabel", statusBarLabel },
                { "item", item },
                { "prefabItem", prefabItem },
                { "button", button },
                { "textbox", textbox },
                { "checkbox", checkbox },
                { "border", border },
                { "darkborder", darkborder },
                { "category", category },
                { "subcategory", subcategory },
                { "catbutton", catbutton },
                { "color", colorstyle },
                { "scrollSliderButton", scrollbutton },
                { "tab", tab },
                { "scroll", scroll },
                { "multiline", multiline },
                { "dark", dark },
                { "dark2", dark2 },
                { "checkmark", checkmark },
                { "tile", tile },
                { "node", node },
                { "searchbox", searchbox },
                { "viewport", viewport },
                { "menu", menu },
                { "menuitem", menuitem },
                { "inport", inport },
                { "outport", outport },
                { "close", closebutton },
                { "closetab", closetab },
                { "switch", switchbutton },
                { "dropshadow", dropshadow },
                { "propertyIndent", propertyIndent },
                { "propertyLabel", propertyLabel },
                { "propertyElement", propertyElement },
                { "indent18", indent18 },
                { "alterRows", alterRows },
                { "header", header },
                { "colorGrey170", colorGrey170 },
                { "dropdownLabel", dropdownLabel },
                { "sliderTrack", sliderTrack },
                { "sliderThumb", sliderThumb },
            };

            Point cursorSize = new Point(32, 32);
            Point halfSize = cursorSize / 2;

            desktop.CursorSet.Add(Cursors.Default, new Cursor { Texture = "Arrow.png", Size = cursorSize, HotSpot = Point.Zero });
            desktop.CursorSet.Add(Cursors.Link, new Cursor { Texture = "Link.png", Size = cursorSize, HotSpot = Point.Zero });
            desktop.CursorSet.Add(Cursors.Move, new Cursor { Texture = "Move.png", Size = cursorSize, HotSpot = halfSize });
            desktop.CursorSet.Add(Cursors.Select, new Cursor { Texture = "Select.png", Size = cursorSize, HotSpot = halfSize });
            desktop.CursorSet.Add(Cursors.SizeNS, new Cursor { Texture = "SizeNS.png", Size = cursorSize, HotSpot = halfSize });
            desktop.CursorSet.Add(Cursors.SizeWE, new Cursor { Texture = "SizeWE.png", Size = cursorSize, HotSpot = halfSize });
            desktop.CursorSet.Add(Cursors.HSplit, new Cursor { Texture = "SizeNS.png", Size = cursorSize, HotSpot = halfSize });
            desktop.CursorSet.Add(Cursors.VSplit, new Cursor { Texture = "SizeWE.png", Size = cursorSize, HotSpot = halfSize });
            desktop.CursorSet.Add(Cursors.SizeNESW, new Cursor { Texture = "SizeNESW.png", Size = cursorSize, HotSpot = halfSize });
            desktop.CursorSet.Add(Cursors.SizeNWSE, new Cursor { Texture = "SizeNWSE.png", Size = cursorSize, HotSpot = halfSize });

            desktop.Skin = skin;
            desktop.TooltipControl.Style = "popup";
            desktop.TooltipControl.Padding = new Margin(4);
        }
    }
}
