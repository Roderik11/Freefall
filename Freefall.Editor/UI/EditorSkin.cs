using Squid;

namespace Freefall.Editor
{
    /// <summary>
    /// Shared skin setup for all editor desktops (landing page, editor, etc.).
    /// Extracted from EditorDesktop so multiple Desktop subclasses share consistent styling.
    /// </summary>
    public static class EditorSkin
    {
        public static void Apply(Desktop desktop)
        {
            float b = .0061f;

            int grey10 = ColorInt.ARGB(1f, .10f - b, .10f - b, .10f + b);
            int grey125 = ColorInt.ARGB(1f, .125f - b, .125f - b, .125f + b);
            int grey15 = ColorInt.ARGB(1f, .15f - b, .15f - b, .15f + b);
            int grey166 = ColorInt.ARGB(1f, .166f - b, .166f - b, .166f + b);
            int grey170 = ColorInt.ARGB(1f, .170f - b, .170f - b, .170f + b);
            int grey175 = ColorInt.ARGB(1f, .175f - b, .175f - b, .175f + b);
            int grey20 = ColorInt.ARGB(1f, .20f - b, .20f - b, .20f + b);
            int grey25 = ColorInt.ARGB(1f, .25f - b, .25f - b, .25f + b);
            int grey30 = ColorInt.ARGB(1f, .30f - b, .30f - b, .30f + b);
            int grey35 = ColorInt.ARGB(1f, .35f - b, .35f - b, .35f + b);
            int grey40 = ColorInt.ARGB(1f, .40f - b, .40f - b, .40f + b);
            int textColor = ColorInt.ARGB(1f, .80f, .80f, .80f);
            int darkColor = grey10;

            int windowColor = darkColor;
            int normalColor = grey125;
            int hotColor = grey20;
            int selectedColor = grey25;

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

            ControlStyle button = new ControlStyle(baseStyle);
            button.TextAlign = Alignment.MiddleCenter;
            button.BackColor = normalColor;
            button.Hot.BackColor = hotColor;
            button.Selected.BackColor = selectedColor;
            button.SelectedHot.BackColor = selectedColor;
            button.SelectedPressed.BackColor = selectedColor;
            button.SelectedFocused.BackColor = selectedColor;
            button.Checked.BackColor = selectedColor;
            button.CheckedHot.BackColor = selectedColor;

            ControlStyle item = new ControlStyle(button);
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

            ControlStyle textbox = new ControlStyle(baseStyle);
            textbox.Texture = "shadow.png";
            textbox.TextAlign = Alignment.MiddleLeft;
            textbox.BackColor = grey10;
            textbox.TextPadding = new Squid.Margin(6, 0, 6, 0);
            textbox.Hot.Texture = "border.dds";
            textbox.Focused.Texture = "border.dds";
            textbox.Tiling = TextureMode.Grid;
            textbox.Grid = new Squid.Margin(4);

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
            checkbox.CheckedHot.Texture = "border.dds";

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
            scrollbutton.BackColor = grey30;
            scrollbutton.Hot.BackColor = grey35;
            scrollbutton.Pressed.BackColor = grey35;

            ControlStyle scroll = new ControlStyle(baseStyle);
            scroll.BackColor = grey15;

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
            inport.Tint = ColorInt.ARGB(1f, .1f, .5f, .1f);

            var outport = new ControlStyle(inport);
            outport.Tint = ColorInt.ARGB(1f, .5f, .3f, .1f);

            var closebutton = new ControlStyle(button);
            closebutton.Texture = "close.dds";
            closebutton.Tiling = TextureMode.Center;

            var closetab = new ControlStyle();
            closetab.Texture = "close.dds";
            closetab.Tint = ColorInt.ARGB(1f, .5f, .5f, .5f);
            closetab.Hot.Tint = ColorInt.ARGB(1f, .75f, .75f, .75f);
            closetab.Tiling = TextureMode.Center;

            var alterRows = new ControlStyle(button);
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

            Skin skin = new Skin
            {
                { "prevtoolbutton", prevtoolbutton},
                { "foldout", foldout},
                { "iconplus", iconplus },
                { "iconclose", iconclose },
                { "iconplay", iconplay },
                { "iconstop", iconstop },
                { "canvas", canvas },
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
            desktop.TooltipControl.Style = "frame";
            desktop.TooltipControl.Padding = new Margin(4);
        }
    }
}
