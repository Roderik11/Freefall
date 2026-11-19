using System.Diagnostics;

namespace Freefall.Editor.Publishing
{
    /// <summary>
    /// The temporal upscalers a published game carries (UPSCALING_PLAN.md stage 5). Their DLLs reach the player as the
    /// engine project's publish items when the checkout has them (ThirdParty/FidelityFX/fetch.ps1; ThirdParty/DLSS/fetch.ps1
    /// and build.cmd); without them the game runs, its upscaler falling back to FSR or Off. After the player is copied,
    /// <see cref="Check"/> names what the game has and warns about what it lacks; their notices go into
    /// <see cref="ThirdPartyNotices"/> with the other components'.
    /// </summary>
    public static class UpscalerFiles
    {
        public static readonly string[] Fsr = { "amd_fidelityfx_loader_dx12.dll", "amd_fidelityfx_upscaler_dx12.dll" };
        public static readonly string[] Dlss = { "freefall_dlss.dll", "nvngx_dlss.dll" };

        /// <summary>
        /// Worker, after the player is copied into <paramref name="output"/>: the upscalers it carries ("FSR (…), DLSS (…)"
        /// or "none"), each missing one as a warning.
        /// </summary>
        public static string Check(string output, List<string> warnings)
        {
            bool Has(string[] files) => files.All(f => File.Exists(Path.Combine(output, f)));
            string Version(string file) => (FileVersionInfo.GetVersionInfo(Path.Combine(output, file)).FileVersion ?? "?").Replace(',', '.');

            var names = new List<string>();
            if (Has(Fsr)) names.Add($"FSR ({Fsr[1]} {Version(Fsr[1])})");
            else warnings.Add("No FSR in this game: its DLLs were not beside the engine when the player was built (run ThirdParty/FidelityFX/fetch.ps1, then publish again)");
            if (Has(Dlss)) names.Add($"DLSS ({Dlss[1]} {Version(Dlss[1])})");
            else warnings.Add("No DLSS in this game: its DLLs were not beside the engine when the player was built (run ThirdParty/DLSS/fetch.ps1 and build.cmd, then publish again)");
            return names.Count > 0 ? string.Join(", ", names) : "none";
        }
    }
}
