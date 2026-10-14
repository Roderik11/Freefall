using System;
using System.Collections.Generic;
using Squid;
using Freefall.Assets;
using Freefall.Graphics;

namespace Freefall.UI
{
    /// <summary>
    /// What a host's Squid renderer does for the game's UI beyond <see cref="ISquidRenderer"/>:
    /// pictures and fonts that come from assets.
    /// </summary>
    public interface IGameUIRenderer
    {
        /// <summary>
        /// Styles name <paramref name="name"/> to draw from this texture, which Squid is told is
        /// <paramref name="scale"/> times its pixel size (texture rects are given in those units).
        /// </summary>
        void InsertTexture(string name, Texture texture, float scale);

        /// <summary>Text drawn with style font <paramref name="name"/> comes from this face at this pixel height.</summary>
        void RegisterFont(string name, Typeface face, int pixelSize);
    }

    /// <summary>
    /// The game's UI: one Squid desktop over the game's picture, separate from the editor's (its
    /// own skin, focus and windows). Project scripts put their HUD and windows on it with
    /// <see cref="Add"/>; the host lays it over the picture, feeds it the mouse and draws it (the
    /// editor: over the scene viewport in play mode).
    ///
    /// Escape opens and closes the menu: the windows added with <c>menu: true</c> show, the cursor
    /// is free, and player control is off (scripts check <see cref="MenuOpen"/>). With the menu
    /// closed the cursor is locked to the picture and only the HUD shows.
    /// </summary>
    public static class GameUI
    {
        public static Desktop Desktop { get; private set; } = NewDesktop(null);

        /// <summary>The look of every control on the desktop. Re-applied when the asset is edited.</summary>
        public static UISkin Skin
        {
            get => _skin;
            set
            {
                if (_skin == value) return;
                _skin = value;
                ApplySkin();
            }
        }

        /// <summary>Windows shown, cursor free, player control off.</summary>
        public static bool MenuOpen
        {
            get => _menuOpen;
            set
            {
                if (_menuOpen == value) return;
                _menuOpen = value;
                foreach (var control in _menu)
                    control.Visible = value;
                MenuChanged?.Invoke();
            }
        }

        public static event Action MenuChanged;

        /// <summary>True from the first <see cref="Update"/> of a run to <see cref="Reset"/>.</summary>
        public static bool Running { get; private set; }

        private static UISkin _skin;
        private static bool _menuOpen;
        private static readonly List<Control> _menu = new();
        private static bool _listening;

        /// <summary>
        /// Puts a control on the desktop. A HUD control (a bar, the hotbar) always shows; a menu
        /// control (a window) only while the menu is open. Whoever adds it takes it off again in its
        /// Destroy, see <see cref="Remove"/>.
        /// </summary>
        public static void Add(Control control, bool menu = false)
        {
            if (control == null) return;
            Desktop.Controls.Add(control);
            if (menu)
            {
                _menu.Add(control);
                control.Visible = _menuOpen;
            }
        }

        /// <summary>
        /// Takes a control off the desktop. The desktop is renewed with the rest of its controls,
        /// because it remembers the control under the mouse, the focused and the last clicked
        /// window in fields nothing else can clear, and a script control left there would keep the
        /// old script assembly loaded after a reload.
        /// </summary>
        public static void Remove(Control control)
        {
            if (control == null) return;
            _menu.Remove(control);
            control.Parent = null;
            Desktop = NewDesktop(Desktop);
        }

        /// <summary>
        /// Once per frame by the host while the game runs, before the desktop updates: Escape, and
        /// the cursor lock that goes with a closed menu.
        /// </summary>
        public static void Update()
        {
            if (!Running)
            {
                Running = true;
                if (!_listening)
                {
                    Freefall.Base.MessageDispatcher.AddListener("AssetReloaded", OnAssetReloaded);
                    _listening = true;
                }
            }

            if (Input.IsKeyPressed(Keys.Escape))
                MenuOpen = !MenuOpen;

            Input.IsMouseLocked = !_menuOpen;
        }

        /// <summary>The game stopped: an empty desktop, menu closed, cursor free.</summary>
        public static void Reset()
        {
            foreach (var control in new List<Control>(Desktop.Controls))
                control.Parent = null;
            _menu.Clear();
            _menuOpen = false;
            _skin = null;
            Desktop = NewDesktop(null);
            Running = false;
            Input.IsMouseLocked = false;
            MenuChanged = null;
        }

        private static Desktop NewDesktop(Desktop old)
        {
            var desktop = new Desktop
            {
                // The host's own desktop draws the cursor over both
                ShowCursor = false,
                Skin = old?.Skin ?? new Skin(),
                Position = old?.Position ?? new Point(),
                Size = old?.Size ?? new Point(1280, 720),
            };
            desktop.TooltipControl.Style = "tooltip";

            if (old != null)
                foreach (var control in new List<Control>(old.Controls))
                    control.Parent = desktop;

            return desktop;
        }

        private static void ApplySkin()
        {
            if (_skin == null) return;
            _skin.ApplyTo(Desktop.Skin, Gui.Renderer as IGameUIRenderer);

            // Controls look their style up once, when they join a desktop: a style that is new to
            // the skin reaches them only by joining again
            Desktop = NewDesktop(Desktop);
        }

        private static void OnAssetReloaded(Freefall.Base.Message message)
        {
            if (_skin != null && message.Data is UISkin skin && skin.Guid == _skin.Guid)
            {
                _skin = skin;
                ApplySkin();
            }
        }
    }
}
