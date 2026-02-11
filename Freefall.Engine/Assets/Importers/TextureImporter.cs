using Freefall.Graphics;

namespace Freefall.Assets.Importers
{
    /// <summary>
    /// Imports texture files (.png, .jpg, .dds).
    /// </summary>
    [AssetReader(".png", ".jpg", ".jpeg", ".dds")]
    public class TextureImporter : AssetReader<Texture>
    {
        public override Texture Import(string filepath)
        {
            return new Texture(Engine.Device, filepath);
        }
    }
}
