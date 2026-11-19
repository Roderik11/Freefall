using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Vortice.Direct3D12;

namespace Freefall.Graphics.Upscaling
{
    /// <summary>
    /// AMD FSR, a temporal upscaler, through the FidelityFX API (UPSCALING_PLAN.md stage 3): AMD's signed DLLs
    /// (amd_fidelityfx_loader_dx12.dll, which loads amd_fidelityfx_upscaler_dx12.dll beside it; ThirdParty/FidelityFX,
    /// fetched by fetch.ps1), called by P/Invoke on the plain C API of SDK v2.4.0 (ffx_api.h, ffx_upscale.h,
    /// ffx_api_dx12.h). The provider is the SDK's choice: FSR 4 on Radeon RX 9000 and newer, FSR 3.1.5 on every other DX12
    /// GPU (<see cref="Version"/> says which).
    ///
    /// One context per renderer (each view keeps its own history), made for the view's size at a resize point, where the
    /// GPU is idle, and destroyed there too. The dispatch records its own barriers, descriptor heaps and root signature
    /// on the command list: the caller sets the engine's again after it.
    /// </summary>
    internal sealed unsafe class Fsr : ITemporalUpscaler
    {
        private const string Loader = "amd_fidelityfx_loader_dx12";

        // ── ffx_api.h / ffx_upscale.h / ffx_api_dx12.h ──

        private const uint ReturnOk = 0;
        private const ulong CreateContextUpscale = 0x00010000;             // FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE
        private const ulong DispatchUpscale = 0x00010001;                  // FFX_API_DISPATCH_DESC_TYPE_UPSCALE
        private const ulong QueryJitterPhaseCount = 0x00010004;            // FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT
        private const ulong QueryGpuMemoryUsage = 0x00010008;              // FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE
        private const ulong CreateBackendDX12 = 0x00000002;                // FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12
        private const ulong QueryProviderVersion = 6;                      // FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION

        private const uint FlagHighDynamicRange = 1u << 0;                 // FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE
        private const uint FlagDepthInverted = 1u << 3;                    // FFX_UPSCALE_ENABLE_DEPTH_INVERTED
        private const uint FlagDebugChecking = 1u << 7;                    // FFX_UPSCALE_ENABLE_DEBUG_CHECKING

        private const uint ResourceTexture2D = 2;                          // FFX_API_RESOURCE_TYPE_TEXTURE2D
        private const uint FormatR16G16B16A16Float = 4;                    // FfxApiSurfaceFormat, in its order
        private const uint FormatR16G16Float = 18;
        private const uint FormatR32Float = 28;
        private const uint UsageReadOnly = 0, UsageUav = 1u << 1;          // FFX_API_RESOURCE_USAGE_*
        private const uint StateUnorderedAccess = 1u << 1;                 // FFX_API_RESOURCE_STATE_*
        private const uint StateComputeRead = 1u << 2;
        private const uint StatePixelComputeRead = (1u << 3) | (1u << 2);

        [StructLayout(LayoutKind.Sequential)]
        private struct ApiHeader
        {
            public ulong Type;
            public ApiHeader* Next;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct Dimensions2D
        {
            public uint Width, Height;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct FloatCoords2D
        {
            public float X, Y;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ResourceDescription
        {
            public uint Type, Format, Width, Height, Depth, MipCount, Flags, Usage;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ApiResource
        {
            public IntPtr Resource;
            public ResourceDescription Description;
            public uint State;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct CreateBackendDX12Desc
        {
            public ApiHeader Header;
            public IntPtr Device;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct CreateContextDescUpscale
        {
            public ApiHeader Header;
            public uint Flags;
            public Dimensions2D MaxRenderSize;
            public Dimensions2D MaxUpscaleSize;
            public delegate* unmanaged[Cdecl]<uint, char*, void> Message;
        }

        // Its two bools are one byte each, as in C
        [StructLayout(LayoutKind.Sequential)]
        private struct DispatchDescUpscale
        {
            public ApiHeader Header;
            public IntPtr CommandList;
            public ApiResource Color, Depth, MotionVectors, Exposure, Reactive, TransparencyAndComposition, Output;
            public FloatCoords2D JitterOffset, MotionVectorScale;
            public Dimensions2D RenderSize, UpscaleSize;
            public byte EnableSharpening;
            public float Sharpness, FrameTimeDelta, PreExposure;
            public byte Reset;
            public float CameraNear, CameraFar, CameraFovAngleVertical, ViewSpaceToMetersFactor;
            public uint Flags;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct QueryJitterPhaseCountDesc
        {
            public ApiHeader Header;
            public uint RenderWidth, DisplayWidth;
            public int* OutPhaseCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct EffectMemoryUsage
        {
            public ulong TotalUsageInBytes, AliasableUsageInBytes;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct QueryGpuMemoryUsageDesc
        {
            public ApiHeader Header;
            public EffectMemoryUsage* Usage;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct QueryProviderVersionDesc
        {
            public ApiHeader Header;
            public ulong VersionId;
            public byte* VersionName;
        }

        [DllImport(Loader, CallingConvention = CallingConvention.Cdecl)]
        private static extern uint ffxCreateContext(IntPtr* context, ApiHeader* desc, void* allocationCallbacks);

        [DllImport(Loader, CallingConvention = CallingConvention.Cdecl)]
        private static extern uint ffxDestroyContext(IntPtr* context, void* allocationCallbacks);

        [DllImport(Loader, CallingConvention = CallingConvention.Cdecl)]
        private static extern uint ffxQuery(IntPtr* context, ApiHeader* desc);

        [DllImport(Loader, CallingConvention = CallingConvention.Cdecl)]
        private static extern uint ffxDispatch(IntPtr* context, ApiHeader* desc);

        // ── Availability ──

        private static int _available = -1;   // -1 not looked yet, 0 no, 1 yes

        /// <summary>
        /// Whether the FidelityFX DLLs are next to the engine (ThirdParty/FidelityFX/fetch.ps1 fetches them; the engine
        /// project copies them to the output and the player when they are there). Looked up once, logged when missing.
        /// </summary>
        public static bool Available
        {
            get
            {
                if (_available < 0)
                {
                    bool found = NativeLibrary.TryLoad(Loader + ".dll", typeof(Fsr).Assembly,
                        DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.ApplicationDirectory, out _);
                    _available = found ? 1 : 0;
                    if (!found)
                        Debug.LogWarning("FSR", "amd_fidelityfx_loader_dx12.dll is not beside the engine: run ThirdParty/FidelityFX/fetch.ps1 and build again. The upscaler falls back to the plain one.");
                }
                return _available == 1;
            }
        }

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static void OnMessage(uint type, char* message)
        {
            string text = message != null ? new string(message) : "";
            if (type == 0) Debug.LogError("FSR", text);
            else Debug.LogWarning("FSR", text);
        }

        // ── A context ──

        private IntPtr _context;

        /// <summary>The provider the SDK picked for this GPU ("FSR 4.1.2", "FSR 3.1.5", ...), empty before <see cref="Create"/>.</summary>
        public string Version { get; private set; } = "";

        /// <summary>The video memory the context's own resources take (not in the GpuMemory ledger: the SDK makes them).</summary>
        public ulong GpuBytes { get; private set; }

        public bool Created => _context != IntPtr.Zero;

        /// <summary>
        /// A context for pictures up to <paramref name="displayWidth"/> × <paramref name="displayHeight"/>, drawn at any
        /// size up to that: pre-exposed HDR colour, reverse-Z depth, motion vectors in UV units without jitter. False
        /// (logged) when the SDK refuses.
        /// </summary>
        public bool Create(ID3D12Device device, int displayWidth, int displayHeight, bool debugChecks)
        {
            Dispose();
            var backend = new CreateBackendDX12Desc
            {
                Header = new ApiHeader { Type = CreateBackendDX12 },
                Device = device.NativePointer,
            };
            var desc = new CreateContextDescUpscale
            {
                Header = new ApiHeader { Type = CreateContextUpscale, Next = &backend.Header },
                Flags = FlagHighDynamicRange | FlagDepthInverted | (debugChecks ? FlagDebugChecking : 0u),
                MaxRenderSize = new Dimensions2D { Width = (uint)displayWidth, Height = (uint)displayHeight },
                MaxUpscaleSize = new Dimensions2D { Width = (uint)displayWidth, Height = (uint)displayHeight },
                Message = &OnMessage,
            };
            IntPtr context;
            uint result = ffxCreateContext(&context, &desc.Header, null);
            if (result != ReturnOk)
            {
                Debug.LogWarning("FSR", $"ffxCreateContext failed ({result}) for {displayWidth}x{displayHeight}");
                return false;
            }
            _context = context;

            var version = new QueryProviderVersionDesc { Header = new ApiHeader { Type = QueryProviderVersion } };
            if (ffxQuery(&context, &version.Header) == ReturnOk && version.VersionName != null)
                Version = "FSR " + (Marshal.PtrToStringUTF8((IntPtr)version.VersionName) ?? "");

            EffectMemoryUsage usage;
            var memory = new QueryGpuMemoryUsageDesc { Header = new ApiHeader { Type = QueryGpuMemoryUsage }, Usage = &usage };
            GpuBytes = ffxQuery(&context, &memory.Header) == ReturnOk ? usage.TotalUsageInBytes : 0;
            return true;
        }

        /// <summary>How many jitter offsets go round for this render and display width (the SDK's own count).</summary>
        public int JitterPhaseCount(int renderWidth, int displayWidth)
        {
            if (_context == IntPtr.Zero) return 8;
            int count = 0;
            var query = new QueryJitterPhaseCountDesc
            {
                Header = new ApiHeader { Type = QueryJitterPhaseCount },
                RenderWidth = (uint)renderWidth,
                DisplayWidth = (uint)displayWidth,
                OutPhaseCount = &count,
            };
            IntPtr context = _context;
            return ffxQuery(&context, &query.Header) == ReturnOk && count > 0 ? count : 8;
        }

        private static ApiResource Texture(ID3D12Resource resource, uint format, int width, int height, uint usage, uint state) => new()
        {
            Resource = resource.NativePointer,
            Description = new ResourceDescription
            {
                Type = ResourceTexture2D, Format = format, Width = (uint)width, Height = (uint)height, Depth = 1, MipCount = 1,
                Usage = usage,
            },
            State = state,
        };

        /// <summary>
        /// Upscale this frame into <paramref name="output"/> (display size, R16G16B16A16_Float, in UnorderedAccess and
        /// left there). The inputs are the render size: <paramref name="color"/> R16G16B16A16_Float in
        /// NonPixelShaderResource, <paramref name="depth"/> the hardware depth (R32 typeless, reverse Z) readable by
        /// both stages, <paramref name="motion"/> R16G16_Float (last frame's UV minus this frame's) in
        /// NonPixelShaderResource, <paramref name="exposure"/> a 1x1 R32_Float holding 1 in NonPixelShaderResource. The
        /// SDK returns each to the state it was handed in.
        /// </summary>
        public bool Dispatch(ID3D12GraphicsCommandList list, ID3D12Resource color, ID3D12Resource depth, ID3D12Resource motion,
            ID3D12Resource exposure, ID3D12Resource output, int renderWidth, int renderHeight, int displayWidth, int displayHeight,
            Vector2 jitterPixels, bool reset, float near, float far, float fovYRadians, float frameMs, float sharpness)
        {
            if (_context == IntPtr.Zero) return false;
            var desc = new DispatchDescUpscale
            {
                Header = new ApiHeader { Type = DispatchUpscale },
                CommandList = list.NativePointer,
                Color = Texture(color, FormatR16G16B16A16Float, renderWidth, renderHeight, UsageReadOnly, StateComputeRead),
                Depth = Texture(depth, FormatR32Float, renderWidth, renderHeight, UsageReadOnly, StatePixelComputeRead),
                MotionVectors = Texture(motion, FormatR16G16Float, renderWidth, renderHeight, UsageReadOnly, StateComputeRead),
                // 1x1 R32_Float holding 1: the colour is pre-exposed, so its exposure for FSR's tonemapping is one
                Exposure = Texture(exposure, FormatR32Float, 1, 1, UsageReadOnly, StateComputeRead),
                Output = Texture(output, FormatR16G16B16A16Float, displayWidth, displayHeight, UsageUav, StateUnorderedAccess),
                JitterOffset = new FloatCoords2D { X = jitterPixels.X, Y = jitterPixels.Y },
                // UV units to the render size's pixels
                MotionVectorScale = new FloatCoords2D { X = renderWidth, Y = renderHeight },
                RenderSize = new Dimensions2D { Width = (uint)renderWidth, Height = (uint)renderHeight },
                UpscaleSize = new Dimensions2D { Width = (uint)displayWidth, Height = (uint)displayHeight },
                EnableSharpening = sharpness > 0f ? (byte)1 : (byte)0,
                Sharpness = Math.Clamp(sharpness, 0f, 1f),
                FrameTimeDelta = frameMs,
                PreExposure = 1f,       // the colour is pre-exposed already; the exposure moves slowly from frame to frame
                Reset = reset ? (byte)1 : (byte)0,
                CameraNear = near,
                CameraFar = far,
                CameraFovAngleVertical = fovYRadians,
                ViewSpaceToMetersFactor = 1f,
            };
            IntPtr context = _context;
            uint result = ffxDispatch(&context, &desc.Header);
            if (result != ReturnOk)
            {
                Debug.LogWarning("FSR", $"ffxDispatch failed ({result})");
                return false;
            }
            return true;
        }

        /// <summary>Destroys the context. Only where the GPU has finished with it (a resize point, shutdown).</summary>
        public void Dispose()
        {
            if (_context == IntPtr.Zero) return;
            IntPtr context = _context;
            ffxDestroyContext(&context, null);
            _context = IntPtr.Zero;
            Version = "";
            GpuBytes = 0;
        }
    }
}
