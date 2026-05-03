using System;
using System.Collections.Generic;

namespace Freefall.Assets
{
    /// <summary>
    /// Represents a .meta file stored in Library/{sourceGuid}.meta.
    /// Contains the serialized importer with its settings, and a list of
    /// subassets produced by the import.
    /// </summary>
    public class MetaFile
    {
        /// <summary>
        /// Unique identifier for the source asset.
        /// </summary>
        public string Guid { get; set; }

        /// <summary>
        /// Relative path to the source file within Assets/.
        /// </summary>
        public string SourcePath { get; set; }

        /// <summary>
        /// Fully-qualified type name of the importer (e.g. "Freefall.Assets.Importers.StaticMeshImporter").
        /// Used to deserialize the importer settings.
        /// </summary>
        public string ImporterType { get; set; }

        /// <summary>
        /// Serialized importer settings as JSON string.
        /// The importer instance is serialized/deserialized to preserve all import configuration.
        /// </summary>
        public string ImporterSettings { get; set; }

        /// <summary>
        /// Artifacts produced by the last import.
        /// </summary>
        public List<SubAssetEntry> SubAssets { get; set; } = new();

        /// <summary>
        /// UTC timestamp of the last successful import.
        /// </summary>
        public DateTime LastImported { get; set; }

        /// <summary>
        /// Size of the source file at last import. Used to detect changed files
        /// even when timestamps are older (e.g. copied from another project).
        /// </summary>
        public long FileSize { get; set; }

        /// <summary>
        /// For simple assets (1 artifact), the type name of the main asset cached under the source GUID.
        /// Null for compound assets that use SubAssets instead.
        /// </summary>
        public string MainAssetType { get; set; }

        /// <summary>
        /// For simple assets, the semantic asset type name (e.g. "PCGGraph", "Material")
        /// when it differs from MainAssetType (which stores the packer type like "AssetDefinitionData").
        /// Used by the asset browser for editor dispatch and display.
        /// </summary>
        public string MainSemanticType { get; set; }
    }

    /// <summary>
    /// Describes a single artifact produced by an import.
    /// </summary>
    public class SubAssetEntry
    {
        /// <summary>
        /// Unique identifier for this subasset.
        /// </summary>
        public string Guid { get; set; }

        /// <summary>
        /// Human-readable name (e.g. "idle", "walk", "character").
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Packer data type name (e.g. "MeshData", "AssetDefinitionData", "PrefabData").
        /// Used for cache path resolution.
        /// </summary>
        public string Type { get; set; }

        /// <summary>
        /// Semantic asset type when it differs from the data type
        /// (e.g. "Material" when Type is "AssetDefinitionData").
        /// Used for browser display and drag-drop type resolution.
        /// Null when Type already matches the asset type.
        /// </summary>
        public string AssetType { get; set; }

        /// <summary>
        /// If true, this subasset is internal and should not appear in the Asset Browser
        /// or be discoverable via user-facing name lookups. Used for pre-cooked physics data etc.
        /// </summary>
        public bool Hidden { get; set; }
    }
}
