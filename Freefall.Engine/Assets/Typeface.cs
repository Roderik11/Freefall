namespace Freefall.Assets
{
    /// <summary>
    /// A font file (.ttf, .otf) as it was imported. Text is drawn from it at whatever pixel size a
    /// <see cref="Freefall.UI.UIStyle"/> asks for; whoever draws rasterizes the face per size.
    /// </summary>
    [AssetTypeAlias("TypefaceData")]
    public class Typeface : Asset
    {
        /// <summary>The file's bytes, unchanged.</summary>
        public byte[] Data { get; set; }
    }
}
