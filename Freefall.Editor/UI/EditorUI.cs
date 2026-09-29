using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Squid;
using WinKeys = System.Windows.Forms.Keys;
using WinCursor = System.Windows.Forms.Cursor;
using WinControl = System.Windows.Forms.Control;
using WinMouseButtons = System.Windows.Forms.MouseButtons;

namespace Freefall.Editor
{
    public class EditorUI : EditorDesktop
    {
        private static Desktop? staticdesk;
        private readonly System.Windows.Forms.Form form;

        // Keyboard input buffering
        private readonly List<KeyData> _keyBuffer = new();

        // Extended keys need special scancodes (MapVirtualKey returns wrong values for these)
        private static readonly Dictionary<WinKeys, int> SpecialScancodes = new()
        {
            { WinKeys.Home,   0xC7 },
            { WinKeys.Up,     0xC8 },
            { WinKeys.Left,   0xCB },
            { WinKeys.Right,  0xCD },
            { WinKeys.End,    0xCF },
            { WinKeys.Down,   0xD0 },
            { WinKeys.Insert, 0xD2 },
            { WinKeys.Delete, 0xD3 },
        };

        [DllImport("user32.dll")]
        private static extern uint MapVirtualKey(uint uCode, uint uMapType);

        [DllImport("user32.dll")]
        private static extern int GetKeyboardLayout(int dwLayout);

        [DllImport("user32.dll")]
        private static extern int GetKeyboardState(byte[] lpKeyState);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int ToUnicodeEx(uint wVirtKey, uint wScanCode, byte[] lpKeyState,
            [Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder pwszBuff,
            int cchBuff, uint wFlags, IntPtr dwhkl);

        public static bool KeyboardCaptured
        {
            get
            {
                if (staticdesk == null) return false;
                return staticdesk.FocusedControl is TextBox || staticdesk.FocusedControl is TextArea;
            }
        }

        public static bool MouseCaptured
        {
            get
            {
                if (staticdesk == null) return false;
                if (staticdesk.HotControl is InnerViewport)
                    return false;
                return staticdesk.HotControl != staticdesk || staticdesk.PressedControl != null;
            }
        }

        public EditorUI(System.Windows.Forms.Form form)
        {
            this.form = form;
            staticdesk = this;
            ShowCursor = true;

            // Hook keyboard events on the form
            form.KeyPreview = true;
            form.PreviewKeyDown += OnPreviewKeyDown;
            form.KeyDown += OnKeyDown;
            form.KeyUp += OnKeyUp;
            form.KeyPress += OnKeyPress;
        }

        private void OnPreviewKeyDown(object? sender, System.Windows.Forms.PreviewKeyDownEventArgs e)
        {
            // Prevent WinForms from swallowing navigation keys (arrows, tab, etc.)
            switch (e.KeyCode)
            {
                case WinKeys.Up:
                case WinKeys.Down:
                case WinKeys.Left:
                case WinKeys.Right:
                case WinKeys.Tab:
                case WinKeys.Home:
                case WinKeys.End:
                case WinKeys.Delete:
                case WinKeys.Insert:
                    e.IsInputKey = true;
                    break;
            }
        }

        private void OnKeyDown(object? sender, System.Windows.Forms.KeyEventArgs e)
        {
            var scancode = GetScancode(e.KeyCode);
            var ch = VirtualKeyToChar((uint)e.KeyCode);

            _keyBuffer.Add(new KeyData
            {
                Pressed = true,
                Scancode = scancode,
                Char = ch
            });
        }

        private void OnKeyUp(object? sender, System.Windows.Forms.KeyEventArgs e)
        {
            var scancode = GetScancode(e.KeyCode);

            _keyBuffer.Add(new KeyData
            {
                Pressed = false,
                Released = true,
                Scancode = scancode,
                Char = null
            });
        }

        private void OnKeyPress(object? sender, System.Windows.Forms.KeyPressEventArgs e)
        {
            // KeyPress gives the correctly translated character (respects layout, shift, etc.)
            // Retroactively patch the last pressed KeyData entry that has no char yet
            for (int i = _keyBuffer.Count - 1; i >= 0; i--)
            {
                if (_keyBuffer[i].Pressed && _keyBuffer[i].Char == null)
                {
                    var kd = _keyBuffer[i];
                    kd.Char = e.KeyChar;
                    _keyBuffer[i] = kd;
                    break;
                }
            }
        }

        private static int GetScancode(WinKeys keyCode)
        {
            if (SpecialScancodes.TryGetValue(keyCode, out var special))
                return special;
            return (int)MapVirtualKey((uint)keyCode, 0);
        }

        private static char? VirtualKeyToChar(uint keyCode)
        {
            var keyboardState = new byte[256];
            GetKeyboardState(keyboardState);

            uint scanCode = MapVirtualKey(keyCode, 0);
            var sb = new StringBuilder(4);
            var layout = (IntPtr)GetKeyboardLayout(0);
            int result = ToUnicodeEx(keyCode, scanCode, keyboardState, sb, 4, 0, layout);

            return result == 1 ? sb[0] : null;
        }

        public new void Update()
        {
            Gui.TimeElapsed = Freefall.Base.Time.DeltaMilliseconds;

            // Feed mouse position to Squid (screen → client coords)
            var cursorPos = WinCursor.Position;
            var clientPos = form.PointToClient(new System.Drawing.Point(cursorPos.X, cursorPos.Y));
            Gui.SetMouse(clientPos.X, clientPos.Y, -Input.MouseWheelDelta);

            // Feed mouse buttons to Squid
            Gui.SetButtons(
                (WinControl.MouseButtons & WinMouseButtons.Left) != 0,
                (WinControl.MouseButtons & WinMouseButtons.Right) != 0,
                (WinControl.MouseButtons & WinMouseButtons.Middle) != 0
            );

            // Feed keyboard to Squid
            if (_keyBuffer.Count > 0)
            {
                Gui.SetKeyboard(_keyBuffer.ToArray(), _keyBuffer.Count);
                _keyBuffer.Clear();
            }
            else
            {
                Gui.SetKeyboard(Array.Empty<KeyData>(), 0);
            }

            // Resize desktop to match client area
            Size = new Squid.Point(form.ClientSize.Width, form.ClientSize.Height);

            base.Update();
        }
    }
}
