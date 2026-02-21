using System.IO;
using Freefall.Animation;
using Freefall.Assets;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads Skeleton assets from cache (.skel files).
    /// Unpacks via SkeletonPacker — the unpacked Skeleton
    /// is already the runtime type.
    /// </summary>
    [AssetLoader(typeof(Skeleton))]
    public class SkeletonLoader : IAssetLoader
    {
        private readonly SkeletonPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for skeleton '{name}'");

            Skeleton skeleton;
            using (var stream = File.OpenRead(cachePath))
                skeleton = _packer.Read(stream);

            skeleton.Name = name;
            return skeleton;
        }
    }
}
