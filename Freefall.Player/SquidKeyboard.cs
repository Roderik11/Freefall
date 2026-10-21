using System.Runtime.InteropServices;
using System.Text;
using Squid;
using WinKeys = System.Windows.Forms.Keys;

namespace Freefall.Player
{
    /// <summary>
    /// The window's keys for Squid (text boxes, keyboard focus in the game's UI): key downs and ups with scancodes, and
    /// the characters they type, buffered between frames and handed over by <see cref="Feed"/>. As the editor's desktop
    /// does it (EditorUI), for a window of the player's.
    /// </summary>
    sealed class SquidKeyboard
    {
        private readonly List<KeyData> _buffer = new();

        // Extended keys need special scancodes (MapVirtualKey returns wrong values for these)
        private static readonly Dictionary<WinKeys, int> SpecialScancodes = new()
        {
            { WinKeys.Home, 0xC7 }, { WinKeys.Up, 0xC8 }, { WinKeys.Left, 0xCB }, { WinKeys.Right, 0xCD },
            { WinKeys.End, 0xCF }, { WinKeys.Down, 0xD0 }, { WinKeys.Insert, 0xD2 }, { WinKeys.Delete, 0xD3 },
        };

        [DllImport("user32.dll")] private static extern uint MapVirtualKey(uint uCode, uint uMapType);
        [DllImport("user32.dll")] private static extern int GetKeyboardLayout(int dwLayout);
        [DllImport("user32.dll")] private static extern int GetKeyboardState(byte[] lpKeyState);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int ToUnicodeEx(uint wVirtKey, uint wScanCode, byte[] lpKeyState,
            [Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pwszBuff, int cchBuff, uint wFlags, IntPtr dwhkl);

        public void Attach(System.Windows.Forms.Form form)
        {
            form.KeyPreview = true;
            form.PreviewKeyDown += (s, e) =>
            {
                // WinForms would take the navigation keys for itself
                if (e.KeyCode is WinKeys.Up or WinKeys.Down or WinKeys.Left or WinKeys.Right or WinKeys.Tab
                    or WinKeys.Home or WinKeys.End or WinKeys.Delete or WinKeys.Insert)
                    e.IsInputKey = true;
            };
            form.KeyDown += (s, e) => _buffer.Add(new KeyData { Pressed = true, Scancode = Scancode(e.KeyCode), Char = CharOf((uint)e.KeyCode) });
            form.KeyUp += (s, e) => _buffer.Add(new KeyData { Pressed = false, Released = true, Scancode = Scancode(e.KeyCode), Char = null });
            form.KeyPress += (s, e) =>
            {
                // KeyPress has the character as typed (layout, shift): it goes to the last key down that has none yet
                for (int i = _buffer.Count - 1; i >= 0; i--)
                {
                    if (_buffer[i].Pressed && _buffer[i].Char == null)
                    {
                        var key = _buffer[i];
                        key.Char = e.KeyChar;
                        _buffer[i] = key;
                        break;
                    }
                }
            };
        }

        /// <summary>This frame's keys to Squid (none is also news: the keys of the frame before are over).</summary>
        public void Feed()
        {
            Gui.SetKeyboard(_buffer.ToArray(), _buffer.Count);
            _buffer.Clear();
        }

        private static int Scancode(WinKeys key) =>
            SpecialScancodes.TryGetValue(key, out var special) ? special : (int)MapVirtualKey((uint)key, 0);

        private static char? CharOf(uint keyCode)
        {
            var state = new byte[256];
            GetKeyboardState(state);
            var sb = new StringBuilder(4);
            int result = ToUnicodeEx(keyCode, MapVirtualKey(keyCode, 0), state, sb, 4, 0, (IntPtr)GetKeyboardLayout(0));
            return result == 1 ? sb[0] : null;
        }
    }
}
