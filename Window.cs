using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using Freefall;
using static Freefall.User32;
using static Freefall.Kernel32;
using Vortice.Mathematics;

namespace Freefall
{
    public class Window : IDisposable
    {
        private const uint PM_REMOVE = 1;
        private readonly string _className = "FreefallWindowClass";
        private readonly IntPtr _hInstance;
        private IntPtr _hWnd;
        private bool _disposed;
        private WNDPROC _wndProc; // Keep delegate alive to prevent GC

        public string Title { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public IntPtr Handle => _hWnd;

        public event Action<int, int> OnResize;

        public Window(string title, int width, int height)
        {
            Title = title;
            Width = width;
            Height = height;
            _hInstance = GetModuleHandle(null);

            Initialize();
        }

        private void Initialize()
        {
            _wndProc = WndProc; // Store delegate to prevent GC
            
            var wndClass = new WNDCLASSEX
            {
                Size = Unsafe.SizeOf<WNDCLASSEX>(),
                Styles = WindowClassStyles.CS_HREDRAW | WindowClassStyles.CS_VREDRAW | WindowClassStyles.CS_OWNDC,
                WindowProc = _wndProc,
                InstanceHandle = _hInstance,
                IconHandle = LoadIcon(IntPtr.Zero, (IntPtr)((int)SystemIcon.IDI_APPLICATION)),
                CursorHandle = LoadCursor(IntPtr.Zero, (int)SystemCursor.IDC_ARROW),
                BackgroundBrushHandle = IntPtr.Zero,
                MenuName = null,
                ClassName = _className,
                SmallIconHandle = IntPtr.Zero
            };

            if (RegisterClassEx(ref wndClass) == 0)
            {
                throw new InvalidOperationException("Failed to register window class.");
            }

            // Calculate window size to match client area
            RECT windowRect = new RECT { Left = 0, Top = 0, Right = Width, Bottom = Height };
            AdjustWindowRect(ref windowRect, WindowStyles.WS_OVERLAPPEDWINDOW, false);
            int adjustedWidth = windowRect.Right - windowRect.Left;
            int adjustedHeight = windowRect.Bottom - windowRect.Top;

            _hWnd = CreateWindowEx(
                (int)WindowExStyles.WS_EX_APPWINDOW,
                _className,
                Title,
                (int)(WindowStyles.WS_OVERLAPPEDWINDOW | WindowStyles.WS_VISIBLE),
                (int)GetSystemMetrics(SystemMetrics.SM_CXSCREEN) / 2 - adjustedWidth / 2,
                (int)GetSystemMetrics(SystemMetrics.SM_CYSCREEN) / 2 - adjustedHeight / 2,
                adjustedWidth,
                adjustedHeight,
                IntPtr.Zero,
                IntPtr.Zero,
                _hInstance,
                IntPtr.Zero
            );

            if (_hWnd == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to create window.");
            }
        }

        private IntPtr WndProc(IntPtr hWnd, uint msg, UIntPtr wParam, IntPtr lParam)
        {
            switch ((WindowMessage)msg)
            {
                case WindowMessage.Destroy:
                    PostQuitMessage(0);
                    break;
                case WindowMessage.Size:
                    // LoWord is width, HiWord is height
                    int width = (int)(lParam.ToInt64() & 0xFFFF);
                    int height = (int)((lParam.ToInt64() >> 16) & 0xFFFF);
                    if (width != 0 && height != 0)
                    {
                        Width = width;
                        Height = height;
                        OnResize?.Invoke(width, height);
                    }
                    break;
                case WindowMessage.Input:
                    Input.ProcessRawInput(lParam);
                    break;
            }

            Input.ProcessMessage(msg, wParam, lParam);
            return DefWindowProc(hWnd, msg, wParam, lParam);
        }
        
        public bool ProcessEvents()
        {
            while (PeekMessage(out var msg, IntPtr.Zero, 0, 0, PM_REMOVE))
            {
                if (msg.Value == (uint)WindowMessage.Quit)
                    return false;

                TranslateMessage(ref msg);
                DispatchMessage(ref msg);
            }
            return true;
        }

        public void SetTitle(string title)
        {
            Title = title;
            User32.SetWindowText(_hWnd, title);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_hWnd != IntPtr.Zero)
                {
                    DestroyWindow(_hWnd);
                    _hWnd = IntPtr.Zero;
                }
                UnregisterClass(_className, _hInstance);
                _disposed = true;
            }
        }
    }
}
