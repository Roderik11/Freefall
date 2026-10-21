using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using Freefall.Assets.Packs;

namespace Freefall.Assets
{
    /// <summary>
    /// Where the loaders read artifacts from: the project's Cache folder (the editor), or pack files (a published game,
    /// PUBLISH_PLAN.md; the editor can mount packs over its folder to test them). A cache path is an artifact's key: its
    /// file name is the artifact's GUID, so mounted packs find it by the GUID alone, whatever folder the path names (the
    /// bucketed Cache/ab/{guid}.ext and the terrain's Cache/{guid}.dds alike). What the packs do not hold falls through
    /// to the folder. Loaders call Open/Exists instead of File.OpenRead/File.Exists; writing the Cache stays the
    /// editor's, with File.
    /// </summary>
    public static class Artifacts
    {
        private static PackSet _packs;
        private static long _fromPacks, _fromFolder;
        private static readonly ConcurrentDictionary<string, byte> _folderReads = new();

        /// <summary>The mounted packs, or null.</summary>
        public static PackSet Packs => _packs;

        /// <summary>Artifacts opened from the packs since they were mounted.</summary>
        public static long OpenedFromPacks => Interlocked.Read(ref _fromPacks);

        /// <summary>Artifacts opened from the folder while packs were mounted: what the packs lack.</summary>
        public static long OpenedFromFolder => Interlocked.Read(ref _fromFolder);

        /// <summary>The cache paths opened from the folder while packs were mounted (each once).</summary>
        public static System.Collections.Generic.ICollection<string> FolderReads => _folderReads.Keys;

        /// <summary>Read artifacts from <paramref name="packs"/> from now on, before the folder. Replaces packs mounted before.</summary>
        public static void Mount(PackSet packs)
        {
            var old = Interlocked.Exchange(ref _packs, packs);
            old?.Dispose();
            Interlocked.Exchange(ref _fromPacks, 0);
            Interlocked.Exchange(ref _fromFolder, 0);
            _folderReads.Clear();
        }

        /// <summary>Back to the folder alone.</summary>
        public static void Unmount() => Mount(null);

        public static string GuidOf(string cachePath) => Path.GetFileNameWithoutExtension(cachePath);

        /// <summary>The artifact's bytes: from the mounted packs if they hold it, else the file.</summary>
        public static Stream Open(string cachePath)
        {
            var packs = _packs;
            if (packs != null)
            {
                var stream = packs.Open(GuidOf(cachePath));
                if (stream != null)
                {
                    Interlocked.Increment(ref _fromPacks);
                    return stream;
                }
                Interlocked.Increment(ref _fromFolder);
                _folderReads.TryAdd(cachePath, 0);
            }
            return File.OpenRead(cachePath);
        }

        public static bool Exists(string cachePath)
        {
            if (string.IsNullOrEmpty(cachePath)) return false;
            var packs = _packs;
            return (packs != null && packs.Contains(GuidOf(cachePath))) || File.Exists(cachePath);
        }
    }
}
