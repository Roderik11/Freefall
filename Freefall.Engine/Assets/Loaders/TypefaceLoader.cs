using System.IO;
using Freefall.Assets.Packers;

namespace Freefall.Assets.Loaders
{
    /// <summary>Loads <see cref="Typeface"/> assets from cache (.font files).</summary>
    [AssetLoader(typeof(Typeface))]
    public class TypefaceLoader : IAssetLoader
    {
        private readonly TypefacePacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, nameof(TypefaceData));
            if (cachePath == null || !Artifacts.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for typeface '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            TypefaceData data;
            using (var stream = Artifacts.Open(cachePath))
                data = _packer.Read(stream);

            return new Typeface { Name = name, Data = data.Bytes };
        }
    }
}
