// Freefall's DLSS shim (UPSCALING_PLAN.md stage 4): a flat C API over NVIDIA's NGX SDK for the engine's C#
// (Freefall.Engine/Graphics/Upscaling/Dlss.cs). NGX's create and evaluate helpers are inline C++ in its headers and its
// loader is a static library (nvsdk_ngx_s.lib, static CRT), so C# cannot call it directly the way it calls AMD's FFX API.
// Built by ../build.cmd into ../win-x64/freefall_dlss.dll beside nvngx_dlss.dll, the DLSS model NGX loads at run time.
//
// One NGX instance for the engine's one device (FfDlss_Init / FfDlss_Shutdown); a DLSS feature per renderer, made for
// its render and display size and quality (FfDlss_Create, FfDlss_Release), evaluated on the frame's command list
// (FfDlss_Evaluate). The capability parameters NGX hands out at init carry every create and evaluate: the engine calls
// all of this from its main thread, one renderer after the other.

#include <d3d12.h>
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers_d3d.h"

#define FF_API extern "C" __declspec(dllexport)

typedef void(__cdecl *FfDlssLog)(const char *message);

static ID3D12Device *g_device = nullptr;
static NVSDK_NGX_Parameter *g_params = nullptr;
static FfDlssLog g_log = nullptr;

static void NVSDK_CONV OnNgxLog(const char *message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
{
    if (g_log && message) g_log(message);
}

// NGX for this device. projectId: a GUID-like string (NVSDK_NGX_ENGINE_TYPE_CUSTOM needs no id from NVIDIA); dataPath: a
// writable folder for NGX's logs. log: NGX's own log lines (warnings and errors), may be null. Returns 1, or the
// NVSDK_NGX_Result that failed.
FF_API int FfDlss_Init(ID3D12Device *device, const char *projectId, const char *engineVersion, const wchar_t *dataPath, FfDlssLog log)
{
    if (g_device) return 1;
    g_log = log;
    NVSDK_NGX_FeatureCommonInfo info = {};
    info.LoggingInfo.LoggingCallback = log ? OnNgxLog : nullptr;
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
    info.LoggingInfo.DisableOtherLoggingSinks = false;
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(projectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, engineVersion,
                                                                   dataPath, device, &info);
    if (NVSDK_NGX_FAILED(result)) return (int)result;
    result = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_params);
    if (NVSDK_NGX_FAILED(result))
    {
        NVSDK_NGX_D3D12_Shutdown1(device);
        return (int)result;
    }
    g_device = device;
    return 1;
}

// Whether DLSS Super Resolution runs on this GPU and driver: 1 yes, 0 no. needsDriver: 1 when a newer driver would
// make it run. initResult: the feature's init result NGX reports (an NVSDK_NGX_Result).
FF_API int FfDlss_Available(int *needsDriver, int *initResult)
{
    int available = 0, driver = 0, init = 0;
    if (g_params)
    {
        NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
        NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &driver);
        NVSDK_NGX_Parameter_GetI(g_params, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &init);
    }
    if (needsDriver) *needsDriver = driver;
    if (initResult) *initResult = init;
    return available ? 1 : 0;
}

// The render size DLSS wants for a display size and quality (an NVSDK_NGX_PerfQuality_Value). Returns 1, or 0.
FF_API int FfDlss_OptimalSettings(unsigned displayWidth, unsigned displayHeight, int quality, unsigned *renderWidth, unsigned *renderHeight)
{
    if (!g_params) return 0;
    void *callback = nullptr;
    NVSDK_NGX_Parameter_GetVoidPointer(g_params, NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback, &callback);
    if (!callback) return 0;
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_Width, displayWidth);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_Height, displayHeight);
    NVSDK_NGX_Parameter_SetI(g_params, NVSDK_NGX_Parameter_PerfQualityValue, quality);
    NVSDK_NGX_Parameter_SetI(g_params, NVSDK_NGX_Parameter_RTXValue, 0);   // older DLLs still read it
    if (NVSDK_NGX_FAILED(((PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback)callback)(g_params))) return 0;
    unsigned width = 0, height = 0;
    NVSDK_NGX_Parameter_GetUI(g_params, NVSDK_NGX_Parameter_OutWidth, &width);
    NVSDK_NGX_Parameter_GetUI(g_params, NVSDK_NGX_Parameter_OutHeight, &height);
    if (width == 0 || height == 0) return 0;
    *renderWidth = width;
    *renderHeight = height;
    return 1;
}

// A DLSS feature, recorded on 'list' (executed with the frame). flags: NVSDK_NGX_DLSS_Feature_Flags; preset: an
// NVSDK_NGX_DLSS_Hint_Render_Preset for every mode (0 = DLSS's default). Null when NGX refuses (result in *outResult).
FF_API void *FfDlss_Create(ID3D12GraphicsCommandList *list, unsigned renderWidth, unsigned renderHeight,
                           unsigned displayWidth, unsigned displayHeight, int quality, int flags, unsigned preset, int *outResult)
{
    if (!g_params) return nullptr;
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
    NVSDK_NGX_Parameter_SetUI(g_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, preset);

    NVSDK_NGX_DLSS_Create_Params create = {};
    create.Feature.InWidth = renderWidth;
    create.Feature.InHeight = renderHeight;
    create.Feature.InTargetWidth = displayWidth;
    create.Feature.InTargetHeight = displayHeight;
    create.Feature.InPerfQualityValue = (NVSDK_NGX_PerfQuality_Value)quality;
    create.InFeatureCreateFlags = flags;

    NVSDK_NGX_Handle *handle = nullptr;
    NVSDK_NGX_Result result = NGX_D3D12_CREATE_DLSS_EXT(list, 1, 1, &handle, g_params, &create);
    if (outResult) *outResult = (int)result;
    return NVSDK_NGX_FAILED(result) ? nullptr : handle;
}

// One frame through DLSS, recorded on 'list'. Inputs at the render size readable by compute (color, depth, motion,
// exposure: a 1x1 R32_Float), output at the display size in unordered access. jitter: render pixels; mvScale: what
// turns the motion vectors into render pixels. Returns 1, or the NVSDK_NGX_Result that failed.
FF_API int FfDlss_Evaluate(void *handle, ID3D12GraphicsCommandList *list, ID3D12Resource *color, ID3D12Resource *output,
                           ID3D12Resource *depth, ID3D12Resource *motion, ID3D12Resource *exposure,
                           float jitterX, float jitterY, float mvScaleX, float mvScaleY,
                           unsigned renderWidth, unsigned renderHeight, int reset, float preExposure, float frameMs)
{
    if (!g_params || !handle) return 0;
    NVSDK_NGX_D3D12_DLSS_Eval_Params eval = {};
    eval.Feature.pInColor = color;
    eval.Feature.pInOutput = output;
    eval.pInDepth = depth;
    eval.pInMotionVectors = motion;
    eval.pInExposureTexture = exposure;
    eval.InJitterOffsetX = jitterX;
    eval.InJitterOffsetY = jitterY;
    eval.InRenderSubrectDimensions.Width = renderWidth;
    eval.InRenderSubrectDimensions.Height = renderHeight;
    eval.InReset = reset;
    eval.InMVScaleX = mvScaleX;
    eval.InMVScaleY = mvScaleY;
    eval.InPreExposure = preExposure;
    eval.InExposureScale = 1.0f;
    eval.InFrameTimeDeltaInMsec = frameMs;
    NVSDK_NGX_Result result = NGX_D3D12_EVALUATE_DLSS_EXT(list, (NVSDK_NGX_Handle *)handle, g_params, &eval);
    return NVSDK_NGX_FAILED(result) ? (int)result : 1;
}

// The video memory DLSS's features hold now (NGX's own resources), 0 if NGX cannot say.
FF_API unsigned long long FfDlss_VideoMemory()
{
    if (!g_params) return 0;
    void *callback = nullptr;
    NVSDK_NGX_Parameter_GetVoidPointer(g_params, NVSDK_NGX_Parameter_DLSSGetStatsCallback, &callback);
    if (!callback) return 0;
    if (NVSDK_NGX_FAILED(((PFN_NVSDK_NGX_DLSS_GetStatsCallback)callback)(g_params))) return 0;
    unsigned long long bytes = 0;
    NVSDK_NGX_Parameter_GetULL(g_params, NVSDK_NGX_Parameter_SizeInBytes, &bytes);
    return bytes;
}

// Lets a feature go. Only where the GPU has finished with it.
FF_API void FfDlss_Release(void *handle)
{
    if (handle) NVSDK_NGX_D3D12_ReleaseFeature((NVSDK_NGX_Handle *)handle);
}

// NGX shut down for the device (after every feature was released).
FF_API void FfDlss_Shutdown()
{
    if (!g_device) return;
    if (g_params) NVSDK_NGX_D3D12_DestroyParameters(g_params);
    g_params = nullptr;
    NVSDK_NGX_D3D12_Shutdown1(g_device);
    g_device = nullptr;
}
