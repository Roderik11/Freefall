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
            var cachePath = AssetDatabase.ResolveCachePath(name, "Skeleton");
            if (cachePath == null || !Artifacts.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for skeleton '{name}'");

            return LoadFromCache(cachePath, name, manager);
        }

        /// <summary>
        /// The skeleton in this cache file. Without it LoadByGuid fell back to Load(name), a lookup by the skeleton's
        /// name, which picks whichever model file's skeleton of that name comes first ("Armature" is common).
        /// </summary>
        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            Skeleton skeleton;
            using (var stream = Artifacts.Open(cachePath))
                skeleton = _packer.Read(stream);

            skeleton.Name = name;
            return skeleton;
        }
    }
}
