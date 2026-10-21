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
    ///
    /// The desktop's bottom layer draws <see cref="Overlay"/>'s marks (crosshair, prompts) and, with
    /// <see cref="ShowFps"/>, the frame rate: one desktop and one batch for everything over the
    /// picture. (Two batches a frame garbled each other: the sprite batch holds a frame's sprites.)
    /// </summary>
    public static class GameUI
    {
        // The bottom layer: always the desktop's first control, moved with the others to a new desktop.
        // Declared before Desktop: static fields initialize in order, and NewDesktop puts it on.
        private static readonly MarksLayer _marks = new();

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

        /// <summary>
        /// The menu stays open and Escape does nothing: a main menu, where there is no game to go back to. Whoever sets
        /// it clears it again when it goes (Reset does too).
        /// </summary>
        public static bool MenuAlwaysOpen
        {
            get => _menuAlwaysOpen;
            set
            {
                _menuAlwaysOpen = value;
                if (value) MenuOpen = true;
            }
        }

        private static bool _menuAlwaysOpen;

        /// <summary>True from the first <see cref="Update"/> of a run to <see cref="Reset"/>.</summary>
        public static bool Running { get; private set; }

        /// <summary>Frames per second and milliseconds per frame in the top right corner (the host or game code turns it on).</summary>
        public static bool ShowFps { get; set; }

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

            // Under the top layer, also when added after it (a scene's HUD built while its loading screen shows)
            int at = Desktop.Controls.Count;
            foreach (var top in _top)
                if (top.Parent == Desktop) at = Math.Min(at, Desktop.Controls.IndexOf(top));
            Desktop.Controls.Insert(at, control);

            if (menu)
            {
                _menu.Add(control);
                control.Visible = _menuOpen;
            }
        }

        /// <summary>
        /// Puts a control over every other, also over those added after it with <see cref="Add"/>: a loading screen,
        /// a fade to black. Taken off with <see cref="Remove"/>.
        /// </summary>
        public static void AddOnTop(Control control)
        {
            if (control == null) return;
            Desktop.Controls.Add(control);
            _top.Add(control);
        }

        private static readonly List<Control> _top = new();

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
            _top.Remove(control);
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

            if (MenuAlwaysOpen)
                MenuOpen = true;
            else if (Input.IsKeyPressed(Keys.Escape))
                MenuOpen = !MenuOpen;

            Input.IsMouseLocked = !_menuOpen;
        }

        /// <summary>The game stopped: an empty desktop, menu closed, cursor free.</summary>
        public static void Reset()
        {
            foreach (var control in new List<Control>(Desktop.Controls))
                if (control != _marks) control.Parent = null;
            _menu.Clear();
            _top.Clear();
            _menuOpen = false;
            _menuAlwaysOpen = false;
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
            if (_marks.Parent != desktop)
                desktop.Controls.Insert(0, _marks);

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

        /// <summary>
        /// Under every other control: the marks game code asked for this frame, and the frame rate. Takes
        /// no mouse (NoEvents), so the picture under it stays the hot control.
        /// </summary>
        private sealed class MarksLayer : Control
        {
            private readonly System.Diagnostics.Stopwatch _clock = new();
            private int _frames;
            private string _fps = "";

            public MarksLayer()
            {
                Dock = DockStyle.Fill;
                NoEvents = true;
            }

            protected override void DrawStyle(Style style, float opacity)
            {
                Overlay.Draw(Gui.Renderer, Location, Size, opacity);
                if (ShowFps) DrawFps(opacity);
            }

            // Averaged over half a second (a number that changes every frame cannot be read), on a dark box,
            // in the colour of the rate: red under 90, yellow to 120, green above
            private const int FpsFontSize = 24;
            private double _rate;

            private void DrawFps(float opacity)
            {
                // From the first frame drawn: the time before it (loading the scene) is no frame of the game's
                if (!_clock.IsRunning) _clock.Start();
                else _frames++;
                double seconds = _clock.Elapsed.TotalSeconds;
                if (seconds >= 0.5)
                {
                    _rate = _frames / seconds;
                    _fps = $"{_rate:0} fps  {seconds * 1000 / _frames:0.0} ms";
                    _frames = 0;
                    _clock.Restart();
                }
                var renderer = Gui.Renderer;
                var name = _skin?.FontNameAt(renderer as IGameUIRenderer, FpsFontSize);
                int font = name != null ? renderer.GetFont(name) : -1;
                if (_fps.Length == 0 || font < 0) return;

                var extent = renderer.GetTextSize(_fps, font);
                const int pad = 8;
                int x = Location.x + Size.x - extent.x - 16, y = Location.y + 10;
                renderer.DrawBox(x - pad, y - pad / 2, extent.x + 2 * pad, extent.y + pad, ColorInt.ARGB(0.6f * opacity, 0f, 0f, 0f));

                int color = _rate < 90 ? ColorInt.ARGB(opacity, 1f, 0.32f, 0.26f)
                          : _rate <= 120 ? ColorInt.ARGB(opacity, 1f, 0.84f, 0.25f)
                          : ColorInt.ARGB(opacity, 0.4f, 0.92f, 0.4f);
                renderer.DrawText(_fps, x + 1, y + 1, font, ColorInt.ARGB(0.8f * opacity, 0f, 0f, 0f));
                renderer.DrawText(_fps, x, y, font, color);
            }
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
