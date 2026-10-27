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
            if (cachePath == null || !Artifacts.Exists(cachePath))
                throw new FileNotFoundException($"Cache file not found for animation clip '{name}'");

            // Extract GUID from cache path (format: .../XX/{guid}.anim)
            var guid = System.IO.Path.GetFileNameWithoutExtension(cachePath);

            return LoadFromCache(cachePath, name, manager, guid);
        }

        public Asset LoadFromCache(string cachePath, string name, AssetManager manager, string sourceGuid = null)
        {
            AnimationClip clip;
            using (var stream = Artifacts.Open(cachePath))
                clip = _packer.Read(stream);

            clip.Name = name;

            // Resolve sibling Skeleton from the same source file
            if (FindSkeleton(sourceGuid) is { } skelEntry)
                clip.Skeleton = manager.LoadByGuid<Skeleton>(skelEntry.Guid);

            ApplyClipConfig(clip, sourceGuid);

            return Retargeted(clip, sourceGuid, manager);
        }

        /// <summary>
        /// The clip's skeleton: a sibling sub-asset of the same model file, as LoadFromCache finds it; and for a file
        /// whose clips are baked onto another rig (ModelImporter.RetargetTo), the two skeletons that takes.
        /// </summary>
        public IEnumerable<ImplicitArtifact> GetImplicitDependencies(string guid)
        {
            if (FindSkeleton(guid) is { } skelEntry)
                yield return new ImplicitArtifact(skelEntry.Guid);
            if (RetargetOf(guid) is { } importer)
            {
                yield return new ImplicitArtifact(importer.RetargetTo);
                if (!string.IsNullOrEmpty(importer.RetargetFrom))
                    yield return new ImplicitArtifact(importer.RetargetFrom);
            }
        }

        // The importer of a clip's file when its clips are to be baked onto another rig, else null. The settings are
        // read back only when they name one.
        private static Importers.ModelImporter RetargetOf(string clipGuid)
        {
            if (string.IsNullOrEmpty(clipGuid)) return null;
            var settings = AssetDatabase.GetMeta(clipGuid)?.ImporterSettings;
            if (settings == null || !settings.Contains("\"RetargetTo\"") || settings.Contains("\"RetargetTo\":\"\"")) return null;
            return AssetDatabase.GetImporter(clipGuid) is Importers.ModelImporter importer && !string.IsNullOrEmpty(importer.RetargetTo)
                ? importer : null;
        }

        /// <summary>
        /// A clip of a file whose clips are made on another rig (ModelImporter.RetargetTo) as a clip of the rig named
        /// there, with its events; the clip as it is otherwise, and when a skeleton is missing.
        /// </summary>
        private static AnimationClip Retargeted(AnimationClip clip, string clipGuid, AssetManager manager)
        {
            var importer = RetargetOf(clipGuid);
            if (importer == null) return clip;

            var source = string.IsNullOrEmpty(importer.RetargetFrom) ? clip.Skeleton : manager.LoadByGuid<Skeleton>(importer.RetargetFrom);
            var target = manager.LoadByGuid<Skeleton>(importer.RetargetTo);
            if (source == null || target == null)
            {
                Debug.LogWarning("AnimationClipLoader", $"'{clip.Name}': RetargetTo needs the skeleton it was made on and the one to bake it onto; it plays as made");
                return clip;
            }

            var baked = ClipRetarget.Bake(clip, source, target);
            if (baked == null)
            {
                Debug.LogWarning("AnimationClipLoader", $"'{clip.Name}': {source.Name} and {target.Name} share no bones; it plays as made");
                return clip;
            }
            return baked;
        }

        private static SubAssetEntry FindSkeleton(string clipGuid) =>
            string.IsNullOrEmpty(clipGuid) ? null : AssetDatabase.FindSiblingSubAsset(clipGuid, nameof(Skeleton));

        /// <summary>
        /// Put on the clip what is marked on it in the settings of the file it comes from
        /// (ModelImporter.Clips). Those are not in the cache, so marking an event needs no reimport.
        /// </summary>
        private static void ApplyClipConfig(AnimationClip clip, string clipGuid)
        {
            if (string.IsNullOrEmpty(clipGuid)) return;

            // Skip reading the importer back unless some clip of the file has settings
            var settings = AssetDatabase.GetMeta(clipGuid)?.ImporterSettings;
            if (settings == null || !settings.Contains("\"Events\"")) return;

            if (AssetDatabase.GetImporter(clipGuid) is not Importers.ModelImporter importer) return;

            var config = importer.FindClipConfig(AssetDatabase.ResolveFriendlyName(clipGuid));
            if (config?.Events == null || config.Events.Count == 0) return;

            var events = new System.Collections.Generic.List<AnimationEvent>(config.Events.Count);
            foreach (var e in config.Events)
                if (e != null) events.Add(e);
            clip.Events = events;
        }
    }
}
