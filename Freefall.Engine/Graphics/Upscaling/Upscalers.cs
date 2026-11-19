namespace Freefall.Graphics.Upscaling
{
    /// <summary>
    /// Which temporal upscalers run on this PC, for game code (the game's options) as for the renderer: FSR where AMD's
    /// DLLs are beside the engine, DLSS where its DLLs are there and NGX says it runs (an NVIDIA RTX GPU and a recent
    /// enough driver). The first ask of DLSS starts NGX on the engine's device, so ask after Engine.Initialize.
    /// </summary>
    public static class Upscalers
    {
        /// <summary>Whether <paramref name="kind"/> runs here: Off always, Auto where either FSR or DLSS does.</summary>
        public static bool Runs(UpscalerKind kind) => kind switch
        {
            UpscalerKind.Off => true,
            UpscalerKind.FSR => Fsr.Available,
            UpscalerKind.DLSS => Dlss.Available,
            _ => Dlss.Available || Fsr.Available,
        };

        /// <summary>
        /// The upscaler that runs for EngineSettings.Upscaler = <paramref name="wanted"/>: DLSS (and Auto) where DLSS
        /// runs, else FSR where it runs, else Off. A choice made on another PC falls back here the same way.
        /// </summary>
        public static UpscalerKind Resolve(UpscalerKind wanted)
        {
            if (wanted == UpscalerKind.Off) return UpscalerKind.Off;
            if ((wanted == UpscalerKind.DLSS || wanted == UpscalerKind.Auto) && Dlss.Available) return UpscalerKind.DLSS;
            return Fsr.Available ? UpscalerKind.FSR : UpscalerKind.Off;
        }

        /// <summary>Whether the upscaler <paramref name="wanted"/> resolves to sharpens by EngineSettings.UpscaleSharpness: FSR does, DLSS has no sharpening.</summary>
        public static bool Sharpens(UpscalerKind wanted) => Resolve(wanted) == UpscalerKind.FSR;
    }
}
