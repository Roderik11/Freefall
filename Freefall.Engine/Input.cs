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

        /// <summary>A key or a mouse button went down this frame (a title card that any of them skips).</summary>
        public static bool AnyPressed => _keysPressed.Count > 0 || _mousePressed.Count > 0;
        public static bool Shift => IsKeyDown(Keys.ShiftKey);
        public static bool Control => IsKeyDown(Keys.ControlKey);
        public static bool Alt => IsKeyDown(Keys.Menu);

        public static void Init(IntPtr hWnd)
        {
            _hWnd = hWnd;
            RawInputDevice.RegisterDevice(HidUsageAndPage.Mouse, RawInputDeviceFlags.None, hWnd);
        }

        // ---- Simulated input: what a tool plays in place of the player's hands (the editor's MCP
        // play_input). Keys and buttons go into the same sets as the devices', so the game cannot tell
        // them apart; the game UI gets a cursor of its own (SimulatedCursor), the real one is untouched.

        private static readonly HashSet<Keys> _simulatedKeys = new();
        private static readonly HashSet<int> _simulatedButtons = new();

        /// <summary>The script a tool is playing, or null.</summary>
        public static InputScript Script { get; private set; }

        /// <summary>
        /// Where a tool holds the game UI's cursor, in pixels of the game's picture from its top left;
        /// null = the game UI follows the real mouse.
        /// </summary>
        public static System.Numerics.Vector2? SimulatedCursor { get; private set; }

        /// <summary>The wheel a tool turned this frame, for the game UI under <see cref="SimulatedCursor"/>.</summary>
        public static int SimulatedWheel { get; private set; }

        /// <summary>A tool holds this mouse button down (for the game UI under <see cref="SimulatedCursor"/>).</summary>
        public static bool IsSimulatedButtonDown(int button) => _simulatedButtons.Contains(button);

        /// <summary>
        /// Plays a script from the next frame on. One still playing is stopped and what it held let go;
        /// after one that finished, the cursor, keys and buttons it left stay as they are (a drag held
        /// across calls, with a look at the screen between them).
        /// </summary>
        public static void Play(InputScript script)
        {
            if (Script != null) StopSimulation();
            Script = script;
        }

        /// <summary>Stops a script, lets go of every key and button a tool holds, gives the game UI the real mouse back.</summary>
        public static void StopSimulation()
        {
            Script = null;
            foreach (var key in new List<Keys>(_simulatedKeys)) SimulateKey(key, false);
            foreach (var button in new List<int>(_simulatedButtons)) SimulateButton(button, false);
            SimulatedCursor = null;
            SimulatedWheel = 0;
        }

        public static void SimulateKey(Keys key, bool down)
        {
            if (down)
            {
                if (_simulatedKeys.Add(key) && _keysDown.Add(key)) _keysPressed.Add(key);
            }
            else if (_simulatedKeys.Remove(key) && _keysDown.Remove(key))
                _keysReleased.Add(key);
        }

        public static void SimulateButton(int button, bool down)
        {
            if (down)
            {
                if (_simulatedButtons.Add(button) && _mouseDown.Add(button)) _mousePressed.Add(button);
            }
            else if (_simulatedButtons.Remove(button) && _mouseDown.Remove(button))
                _mouseReleased.Add(button);
        }

        /// <summary>Raw mouse movement, as the device sends it: the camera turns.</summary>
        public static void SimulateLook(int dx, int dy)
        {
            _rawX += dx;
            _rawY += dy;
        }

        public static void SimulateCursor(System.Numerics.Vector2? at) => SimulatedCursor = at;

        public static void SimulateWheel(int delta)
        {
            SimulatedWheel += delta;
            _rawWheel += delta;
        }

        public static void Update()
        {
            var hWnd = _hWnd;

            // A tool's script plays first, so this frame sees what it does as if the devices had sent it
            SimulatedWheel = 0;
            if (Script != null)
            {
                Script.Tick(Freefall.Base.Time.Delta);
                if (Script.Done) Script = null;
            }

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
