using System.IO;
using Freefall.Animation;
using Freefall.Assets;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads AnimationClip assets from cache (.anim files).
    /// Unpacks via AnimationClipPacker — the unpacked AnimationClip
    /// is already the runtime type, so no further conversion needed.
    /// </summary>
    [AssetLoader(typeof(AnimationClip))]
    public class AnimationClipLoader : IAssetLoader
    {
        private readonly AnimationClipPacker _packer = new();

        public Asset Load(string name, AssetManager manager)
        {
            var cachePath = AssetDatabase.ResolveCachePath(name);
            if (cachePath == null || !File.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for animation clip '{name}'");

            AnimationClip clip;
            using (var stream = File.OpenRead(cachePath))
                clip = _packer.Read(stream);

            clip.Name = name;
            return clip;
        }
    }
}
