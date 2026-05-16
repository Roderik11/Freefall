using System.IO;
using Freefall.Animation;
using Freefall.Assets;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads AnimationClip assets from cache (.anim files).
    /// Unpacks via AnimationClipPacker — the unpacked AnimationClip
    /// is already the runtime type, so no further conversion needed.
    /// Resolves sibling Skeleton sub-asset for retargeting.
    /// </summary>
    [AssetLoader(typeof(AnimationClip))]
    public class AnimationClipLoader : IAssetLoader
    {
        private readonly AnimationClipPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name, "AnimationClip");
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for animation clip '{name}'");

            // Extract GUID from cache path (format: .../XX/{guid}.anim)
            var guid = System.IO.Path.GetFileNameWithoutExtension(cachePath);

            return LoadFromCache(cachePath, name, manager, guid);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            AnimationClip clip;
            using (var stream = File.OpenRead(cachePath))
                clip = _packer.Read(stream);

            clip.Name = name;

            // Resolve sibling Skeleton from the same source file
            if (!string.IsNullOrEmpty(sourceGuid))
            {
                var skelEntry = AssetDatabase.FindSiblingSubAsset(sourceGuid, nameof(Skeleton));
                if (skelEntry != null)
                    clip.Skeleton = manager.LoadByGuid<Skeleton>(skelEntry.Guid);
            }

            return clip;
        }
    }
}
