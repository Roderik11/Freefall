using System;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;

namespace Freefall.Graphics.Upscaling
{
    /// <summary>
    /// NVIDIA DLSS Super Resolution (and DLAA, its 1:1 mode) on GeForce RTX, through NVIDIA's NGX SDK 310.9.16
    /// (UPSCALING_PLAN.md stage 4). NGX's helpers are inline C++ and its loader a static library, so the engine calls a
    /// small shim (ThirdParty/DLSS/src/freefall_dlss.cpp, built by build.cmd into freefall_dlss.dll) that NGX then has load
    /// nvngx_dlss.dll, the DLSS model, from beside it. Both are fetched or built, not committed; without them, or on a
    /// GPU without DLSS, <see cref="Available"/> is false and the renderer takes FSR or nothing.
    ///
    /// NGX is started once for the engine's device (the first time DLSS is asked for) with a project id of its own
    /// (<see cref="ProjectId"/>: a custom engine needs no application id from NVIDIA). A renderer's feature is made for
    /// its render and display size and quality on the first frame's command list after <see cref="Create"/> (NGX records
    /// its creation on one), and let go at the renderer's resize point, where the GPU is idle.
    /// </summary>
    internal sealed unsafe class Dlss : ITemporalUpscaler
    {
        private const string Library = "freefall_dlss";

        /// <summary>The project id NGX knows the engine by (GUID-like, as NVSDK_NGX_ENGINE_TYPE_CUSTOM requires).</summary>
        private const string ProjectId = "6f1c2a8e-3b7d-4e59-9a0c-5d2e7b41c8f3";
        private const string EngineVersion = "2026.11";

        // NVSDK_NGX_DLSS_Feature_Flags: HDR colour, motion vectors at the render size, reverse-Z depth. Not MVJittered
        // (they hold no jitter), not AutoExposure (an exposure texture of 1 goes in: the colour is pre-exposed).
        private const int FlagIsHDR = 1 << 0, FlagMVLowRes = 1 << 1, FlagDepthInverted = 1 << 3;

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int FfDlss_Init(IntPtr device, byte* projectId, byte* engineVersion, char* dataPath,
            delegate* unmanaged[Cdecl]<byte*, void> log);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int FfDlss_Available(int* needsDriver, int* initResult);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int FfDlss_OptimalSettings(uint displayWidth, uint displayHeight, int quality, uint* renderWidth, uint* renderHeight);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr FfDlss_Create(IntPtr list, uint renderWidth, uint renderHeight, uint displayWidth,
            uint displayHeight, int quality, int flags, uint preset, int* result);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern int FfDlss_Evaluate(IntPtr handle, IntPtr list, IntPtr color, IntPtr output, IntPtr depth,
            IntPtr motion, IntPtr exposure, float jitterX, float jitterY, float mvScaleX, float mvScaleY,
            uint renderWidth, uint renderHeight, int reset, float preExposure, float frameMs);

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern ulong FfDlss_VideoMemory();

        [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
        private static extern void FfDlss_Release(IntPtr handle);

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static void OnLog(byte* message)
        {
            if (message != null) Debug.LogWarning("DLSS", Marshal.PtrToStringUTF8((IntPtr)message) ?? "");
        }

        // ── NGX, once for the device ──

        private static int _state = -1;   // -1 not tried, 0 unavailable, 1 available
        private static string _dllVersion = "";

        /// <summary>
        /// Whether DLSS runs here: the shim and nvngx_dlss.dll are beside the engine, NGX started on this device, and
        /// it says Super Resolution is available (an RTX GPU and a recent enough driver). Tried once, and the reason
        /// logged when not.
        /// </summary>
        public static bool Available
        {
            get
            {
                if (_state >= 0) return _state == 1;
                _state = 0;
                // NGX is NVIDIA's alone: on another GPU (or Proton's translation of one) it is never started, so the
                // game's options can ask what runs without loading NVIDIA's code there
                uint vendor = Engine.Device.VendorId;
                if (vendor != 0x10DE)
                {
                    Debug.LogAlways($"[DLSS] not an NVIDIA GPU (vendor 0x{vendor:X4}): no DLSS here");
                    return false;
                }
                if (!NativeLibrary.TryLoad(Library + ".dll", typeof(Dlss).Assembly,
                        DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory, out IntPtr loaded))
                {
                    Debug.LogWarning("DLSS", "freefall_dlss.dll is not beside the engine: run ThirdParty/DLSS/fetch.ps1 and build.cmd, then build again.");
                    return false;
                }
                string folder = Path.GetDirectoryName(typeof(Dlss).Assembly.Location) ?? AppContext.BaseDirectory;
                string model = Path.Combine(folder, "nvngx_dlss.dll");
                if (!File.Exists(model))
                {
                    Debug.LogWarning("DLSS", "nvngx_dlss.dll is not beside the engine: run ThirdParty/DLSS/fetch.ps1 and build again.");
                    return false;
                }
                // "310,9,16,0" → "310.9.16"
                string[] parts = (FileVersionInfo.GetVersionInfo(model).FileVersion ?? "").Replace(',', '.').Split('.');
                _dllVersion = string.Join('.', parts.Take(3));

                // NGX's logs and temporary files: the game's user folder, else the engine's under LocalAppData
                string data = Path.Combine(Engine.UserFolder ?? Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Freefall"), "NGX");
                Directory.CreateDirectory(data);

                int init;
                fixed (byte* project = System.Text.Encoding.ASCII.GetBytes(ProjectId + "\0"))
                fixed (byte* version = System.Text.Encoding.ASCII.GetBytes(EngineVersion + "\0"))
                fixed (char* path = data + "\0")
                    init = FfDlss_Init(Engine.Device.NativeDevice.NativePointer, project, version, path, &OnLog);
                if (init != 1)
                {
                    Debug.LogWarning("DLSS", $"NGX did not start on this device (0x{init:X8}): no DLSS here.");
                    return false;
                }
                int needsDriver, initResult;
                if (FfDlss_Available(&needsDriver, &initResult) != 1)
                {
                    Debug.LogWarning("DLSS", needsDriver != 0
                        ? "DLSS needs a newer NVIDIA driver here."
                        : $"DLSS is not available on this GPU (0x{initResult:X8}).");
                    return false;
                }
                _state = 1;
                return true;
            }
        }

        // NVSDK_NGX_PerfQuality_Value for the engine's modes
        private static int Quality(UpscaleQuality mode) => mode switch
        {
            UpscaleQuality.NativeAA => 5,          // DLAA
            UpscaleQuality.Quality => 2,           // MaxQuality
            UpscaleQuality.Balanced => 1,
            UpscaleQuality.Performance => 0,       // MaxPerf
            _ => 3,                                // UltraPerformance
        };

        /// <summary>The render size DLSS wants for this display size and mode (its own rounding); false if NGX cannot say.</summary>
        public static bool RenderSize(int displayWidth, int displayHeight, UpscaleQuality mode, out int renderWidth, out int renderHeight)
        {
            renderWidth = displayWidth;
            renderHeight = displayHeight;
            if (mode == UpscaleQuality.NativeAA) return true;
            if (!Available) return false;
            uint w, h;
            if (FfDlss_OptimalSettings((uint)displayWidth, (uint)displayHeight, Quality(mode), &w, &h) != 1) return false;
            renderWidth = Math.Clamp((int)w, 1, displayWidth);
            renderHeight = Math.Clamp((int)h, 1, displayHeight);
            return true;
        }

        // ── A feature ──

        private IntPtr _feature;
        private bool _created;
        private int _renderWidth, _renderHeight, _displayWidth, _displayHeight;
        private UpscaleQuality _mode;

        public bool Created => _created;
        public string Version => _created ? "DLSS " + _dllVersion : "";
        public ulong GpuBytes { get; private set; }

        /// <summary>
        /// A feature for these sizes and this mode; made on the next frame's command list (<see cref="Dispatch"/>). False
        /// when DLSS is not available.
        /// </summary>
        public bool Create(int renderWidth, int renderHeight, int displayWidth, int displayHeight, UpscaleQuality mode)
        {
            Dispose();
            if (!Available) return false;
            _renderWidth = renderWidth;
            _renderHeight = renderHeight;
            _displayWidth = displayWidth;
            _displayHeight = displayHeight;
            _mode = mode;
            _created = true;
            return true;
        }

        /// <summary>As FSR counts them (the same Halton sequence): 8 × the upscale ratio², at least 8.</summary>
        public int JitterPhaseCount(int renderWidth, int displayWidth)
        {
            float ratio = (float)displayWidth / Math.Max(renderWidth, 1);
            return Math.Max(8, (int)(8f * ratio * ratio));
        }

        public bool Dispatch(ID3D12GraphicsCommandList list, ID3D12Resource color, ID3D12Resource depth, ID3D12Resource motion,
            ID3D12Resource exposure, ID3D12Resource output, int renderWidth, int renderHeight, int displayWidth, int displayHeight,
            Vector2 jitterPixels, bool reset, float near, float far, float fovYRadians, float frameMs, float sharpness)
        {
            if (!_created) return false;
            if (_feature == IntPtr.Zero)
            {
                int result;
                _feature = FfDlss_Create(list.NativePointer, (uint)_renderWidth, (uint)_renderHeight, (uint)_displayWidth,
                    (uint)_displayHeight, Quality(_mode), FlagIsHDR | FlagMVLowRes | FlagDepthInverted, 0, &result);
                if (_feature == IntPtr.Zero)
                {
                    Debug.LogWarning("DLSS", $"the feature was not made (0x{result:X8}) for {_renderWidth}x{_renderHeight} to {_displayWidth}x{_displayHeight}");
                    _created = false;
                    return false;
                }
                GpuBytes = FfDlss_VideoMemory();
                Debug.LogAlways($"[DLSS] {Version}: {_renderWidth}x{_renderHeight} to {_displayWidth}x{_displayHeight} ({_mode}), {GpuBytes / (1024.0 * 1024.0):F0} MiB of its own");
                reset = true;
            }
            // The jitter as FSR takes it (the projection's shift in render pixels, y down). Measured 2026-11-19 in TestWorld's
            // village wood at Performance with the camera still: negated, the picture kept a quarter less fine detail
            // (mean |Laplacian| over mean light 11.2–11.6 against 13.6–14.8, two shots each, alternated)
            int done = FfDlss_Evaluate(_feature, list.NativePointer, color.NativePointer, output.NativePointer,
                depth.NativePointer, motion.NativePointer, exposure.NativePointer,
                jitterPixels.X, jitterPixels.Y, renderWidth, renderHeight,   // UV units to the render size's pixels
                (uint)renderWidth, (uint)renderHeight, reset ? 1 : 0, 1f, frameMs);
            if (done != 1)
            {
                Debug.LogWarning("DLSS", $"evaluate failed (0x{done:X8})");
                return false;
            }
            return true;
        }

        /// <summary>Lets the feature go. Only where the GPU has finished with it (a resize point, shutdown).</summary>
        public void Dispose()
        {
            if (_feature != IntPtr.Zero) FfDlss_Release(_feature);
            _feature = IntPtr.Zero;
            _created = false;
            GpuBytes = 0;
        }
    }
}
