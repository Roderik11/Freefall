using System.IO;
using Freefall.Animation;
using Freefall.Assets;

namespace Freefall.Assets.Loaders
{
    /// <summary>
    /// Loads AnimationClip assets from cache (.anim files).
    /// Unpacks via AnimationClipPacker â€” the unpacked AnimationClip
    /// is already the runtime type, so no further conversion needed.
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

            AnimationClip clip;
            using (var stream = File.OpenRead(cachePath))
                clip = _packer.Read(stream);

            clip.Name = name;
            if (clip.Channels?.Count > 0)
            {
                var ch = clip.Channels[0];
                if (ch.Scale?.Count > 0)
                    Debug.Log($"[AnimClipLoader] '{System.IO.Path.GetFileNameWithoutExtension(name)}' Ch0 Scale[0]={ch.Scale[0].Value}");
                if (ch.Position?.Count > 0)
                    Debug.Log($"[AnimClipLoader] '{System.IO.Path.GetFileNameWithoutExtension(name)}' Ch0 Pos[0]={ch.Position[0].Value}");
            }
            return clip;
        }
    }
}
