using System;
using System.Collections.Generic;
using Linearstar.Windows.RawInput;
using Linearstar.Windows.RawInput.Native;

namespace Freefall
{
    public static class Input
    {
        private static readonly HashSet<Keys> _keysDown = new HashSet<Keys>();
        private static readonly HashSet<Keys> _keysPressed = new HashSet<Keys>();
        private static readonly HashSet<Keys> _keysReleased = new HashSet<Keys>();
        
        private static int _rawX;
        private static int _rawY;
        private static int _rawWheel;
        public static Vortice.Mathematics.Int2 MouseDelta { get; private set; }
        public static int MouseWheelDelta { get; private set; }
        public static bool IsMouseLocked { get; set; } = false;

        public static bool IsKeyDown(Keys key) => _keysDown.Contains(key);
        public static bool IsKeyPressed(Keys key) => _keysPressed.Contains(key);
        public static bool IsKeyReleased(Keys key) => _keysReleased.Contains(key);

        public static void Init(IntPtr hWnd)
        {
            RawInputDevice.RegisterDevice(HidUsageAndPage.Mouse, RawInputDeviceFlags.None, hWnd);
        }

        public static void Update(IntPtr hWnd)
        {
            // Accumulate raw delta into the public MouseDelta
            MouseDelta = new Vortice.Mathematics.Int2(_rawX, _rawY);
            MouseWheelDelta = _rawWheel;
            
            // Reset accumulation for next frame
            _rawX = 0;
            _rawY = 0;
            _rawWheel = 0;
            
            if (IsMouseLocked && User32.GetForegroundWindow() == hWnd)
            {
                User32.ShowCursor(false);
            }
            else
            {
                User32.ShowCursor(true);
            }
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

            if (msg == 0x0100 && (Keys)wParam.ToUInt64() == Keys.Escape)
            {
                IsMouseLocked = !IsMouseLocked;
            }
        }

        public static void ClearFrameCallbacks()
        {
            _keysPressed.Clear();
            _keysReleased.Clear();
        }
    }
}
