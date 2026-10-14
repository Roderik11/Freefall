using System;
using System.Collections.Generic;
using Linearstar.Windows.RawInput;
using Linearstar.Windows.RawInput.Native;

namespace Freefall
{
    public static class Input
    {
        private static IntPtr _hWnd;
        private static readonly HashSet<Keys> _keysDown = new HashSet<Keys>();
        private static readonly HashSet<Keys> _keysPressed = new HashSet<Keys>();
        private static readonly HashSet<Keys> _keysReleased = new HashSet<Keys>();

        private static readonly HashSet<int> _mouseDown = new();
        private static readonly HashSet<int> _mousePressed = new();
        private static readonly HashSet<int> _mouseReleased = new();
        
        private static int _rawX;
        private static int _rawY;
        private static int _rawWheel;
        public static Vortice.Mathematics.Int2 MouseDelta { get; private set; }
        public static Vortice.Mathematics.Int2 MousePosition { get; private set; }
        public static int MouseWheelDelta { get; private set; }

        /// <summary>
        /// The cursor is hidden and held in the middle of <see cref="LockArea"/> while the window has
        /// the focus; only <see cref="MouseDelta"/> moves. The game's UI sets it (closed menu = locked).
        /// </summary>
        public static bool IsMouseLocked { get; set; } = false;

        /// <summary>Where a locked cursor is held, in client pixels: the game's picture. Empty = the whole window.</summary>
        public static Vortice.Mathematics.RectI LockArea { get; set; }

        private static bool _cursorHidden;

        public static bool IsKeyDown(Keys key) => _keysDown.Contains(key);
        public static bool IsKeyPressed(Keys key) => _keysPressed.Contains(key);
        public static bool IsKeyReleased(Keys key) => _keysReleased.Contains(key);

        /// <summary>Mouse button state. 0=Left, 1=Right, 2=Middle.</summary>
        public static bool IsMouseDown(int button) => _mouseDown.Contains(button);
        public static bool IsMousePressed(int button) => _mousePressed.Contains(button);
        public static bool IsMouseReleased(int button) => _mouseReleased.Contains(button);
        public static bool Shift => IsKeyDown(Keys.ShiftKey);
        public static bool Control => IsKeyDown(Keys.ControlKey);
        public static bool Alt => IsKeyDown(Keys.Menu);

        public static void Init(IntPtr hWnd)
        {
            _hWnd = hWnd;
            RawInputDevice.RegisterDevice(HidUsageAndPage.Mouse, RawInputDeviceFlags.None, hWnd);
        }

        public static void Update()
        {
            var hWnd = _hWnd;
            // Accumulate raw delta into the public MouseDelta
            MouseDelta = new Vortice.Mathematics.Int2(_rawX, _rawY);
            MouseWheelDelta = _rawWheel;
            
            // Reset accumulation for next frame
            _rawX = 0;
            _rawY = 0;
            _rawWheel = 0;
            
            bool lockNow = IsMouseLocked && hWnd != IntPtr.Zero && User32.GetForegroundWindow() == hWnd;
            SetCursorHidden(lockNow);

            if (lockNow)
            {
                // Held in the middle, so it never reaches the edge (raw deltas keep coming) and a click
                // never lands outside the picture
                var area = LockArea;
                POINT middle;
                if (area.Width > 0 && area.Height > 0)
                    middle = new POINT(area.X + area.Width / 2, area.Y + area.Height / 2);
                else
                {
                    User32.GetClientRect(hWnd, out var client);
                    middle = new POINT((client.Left + client.Right) / 2, (client.Top + client.Bottom) / 2);
                }
                User32.ClientToScreen(hWnd, ref middle);
                User32.SetCursorPos(middle.X, middle.Y);
            }
        }

        // ShowCursor keeps a count per thread: shown while it is 0 or more. Called every frame it ran up
        // into the thousands and a later hide took as many frames; now it is called on a change only,
        // until the count is on the wanted side.
        private static void SetCursorHidden(bool hidden)
        {
            if (hidden == _cursorHidden) return;
            _cursorHidden = hidden;

            if (hidden)
                while (User32.ShowCursor(false) >= 0) { }
            else
                while (User32.ShowCursor(true) < 0) { }
        }

        public static void ProcessRawInput(IntPtr lParam)
        {
            var data = RawInputData.FromHandle(lParam);
            if (data is RawInputMouseData mouse)
            {
                _rawX += mouse.Mouse.LastX;
                _rawY += mouse.Mouse.LastY;
            }
        }

        public static void ProcessMessage(uint msg, UIntPtr wParam, IntPtr lParam)
        {
            // WM_KEYDOWN = 0x0100, WM_KEYUP = 0x0101, WM_SYSKEYDOWN = 0x0104, WM_SYSKEYUP = 0x0105
            if (msg == 0x0100 || msg == 0x0104) // KeyDown
            {
                Keys key = (Keys)wParam.ToUInt64();
                if (!_keysDown.Contains(key))
                {
                    _keysDown.Add(key);
                    _keysPressed.Add(key);
                }
            }
            else if (msg == 0x0101 || msg == 0x0105) // KeyUp
            {
                Keys key = (Keys)wParam.ToUInt64();
                _keysDown.Remove(key);
                _keysReleased.Add(key);
            }
            else if (msg == 0x020A) // WM_MOUSEWHEEL
            {
                // High word of wParam contains wheel delta (typically ±120)
                short delta = (short)(wParam.ToUInt64() >> 16);
                _rawWheel += delta;
            }

            // Mouse buttons: WM_LBUTTONDOWN/UP = 0x0201/0x0202, WM_RBUTTONDOWN/UP = 0x0204/0x0205, WM_MBUTTONDOWN/UP = 0x0207/0x0208
            if (msg == 0x0201) { _mouseDown.Add(0); _mousePressed.Add(0); UpdateMousePos(lParam); }
            else if (msg == 0x0202) { _mouseDown.Remove(0); _mouseReleased.Add(0); }
            else if (msg == 0x0204) { _mouseDown.Add(1); _mousePressed.Add(1); UpdateMousePos(lParam); }
            else if (msg == 0x0205) { _mouseDown.Remove(1); _mouseReleased.Add(1); }
            else if (msg == 0x0207) { _mouseDown.Add(2); _mousePressed.Add(2); UpdateMousePos(lParam); }
            else if (msg == 0x0208) { _mouseDown.Remove(2); _mouseReleased.Add(2); }
            else if (msg == 0x0200) { UpdateMousePos(lParam); } // WM_MOUSEMOVE
        }

        private static void UpdateMousePos(IntPtr lParam)
        {
            int xy = lParam.ToInt32();
            MousePosition = new Vortice.Mathematics.Int2((short)(xy & 0xFFFF), (short)((xy >> 16) & 0xFFFF));
        }

        public static void ClearFrameCallbacks()
        {
            _keysPressed.Clear();
            _keysReleased.Clear();
            _mousePressed.Clear();
            _mouseReleased.Clear();
        }
    }
}
