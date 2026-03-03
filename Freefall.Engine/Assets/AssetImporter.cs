using System;
using System.Collections.Generic;
using System.IO;

namespace Freefall.Assets
{
    /// <summary>
    /// Marks a class as an asset importer for specific file extensions.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class AssetImporterAttribute : Attribute
    {
        public string[] Extensions { get; }

        public AssetImporterAttribute(params string[] extensions)
        {
            Extensions = extensions;
        }
    }

    /// <summary>
    /// Result of an import operation. Contains all artifacts produced
    /// from a single source file (subassets).
    /// </summary>
    public class ImportResult
    {
        public List<ImportArtifact> Artifacts { get; } = new();

        /// <summary>
        /// If true, always register artifacts as SubAssets (compound path).
        /// Used by ModelImporter so all subassets are independently browseable.
        /// </summary>
        public bool Compound { get; set; }
    }

    public class ImportArtifact
    {
        public string Name { get; set; }
        public string Type { get; set; }
        public object Data { get; set; }

        /// <summary>
        /// If true, this artifact is internal and won't be visible in the Asset Browser.
        /// </summary>
        public bool Hidden { get; set; }
    }

    /// <summary>
    /// Common interface for all importers (single-asset and multi-asset).
    /// Discovered via AssetImporterAttribute, invoked by AssetDatabase.
    /// </summary>
    public interface IImporter
    {
        ImportResult Import(string filepath);

        /// <summary>
        /// If non-null, indicates the concrete Asset type this importer produces.
        /// When set, GetInspectionTarget will load and return the actual asset
        /// instead of the importer itself.
        /// </summary>
        Type AssetType => null;

        /// <summary>
        /// Returns the object the editor inspector should display when
        /// this source file is selected.
        /// If AssetType is set, loads the actual asset for inspection.
        /// Otherwise returns the importer itself.
        /// </summary>
        object GetInspectionTarget(MetaFile meta)
        {
            if (AssetType != null && meta != null && !string.IsNullOrEmpty(meta.Guid) && Engine.Assets != null)
            {
                try
                {
                    var asset = Engine.Assets.LoadByGuid(meta.Guid, AssetType);
                    if (asset != null) return asset;
                }
                catch { }
            }
            return this;
        }
    }

    /// <summary>
    /// Base class for single-asset importers.
    /// - Load(filepath): runtime — load a single asset from file
    /// - Import(filepath): editor — produce all artifacts from source (subassets)
    /// </summary>
    public abstract class AssetImporter<T> : IImporter where T : Asset
    {
        /// <summary>
        /// Runtime: load a single asset from a file path.
        /// </summary>
        public abstract T Load(string filepath);

        /// <summary>
        /// Editor: import a source file and produce all artifacts.
        /// Default implementation wraps Load() for single-artifact importers.
        /// Override for importers that produce multiple subassets.
        /// </summary>
        public virtual ImportResult Import(string filepath)
        {
            var result = new ImportResult();
            var name = Path.GetFileNameWithoutExtension(filepath);
            var asset = Load(filepath);
            result.Artifacts.Add(new ImportArtifact
            {
                Name = name,
                Type = typeof(T).Name,
                Data = asset
            });
            return result;
        }
    }
}
