using System.IO;

namespace Freefall.Assets.Packers
{
    /// <summary>The text of a project's .fx file, registered by type name ("EffectSourceData").</summary>
    public class EffectSourceData
    {
        public string Source { get; set; } = "";

        public EffectSourceData() { }
        public EffectSourceData(string source) => Source = source;
    }

    /// <summary>
    /// Packs a project's .fx to cache as its text. It is compiled when loaded (EffectLoader, Effect.FromAsset), with
    /// the includes it finds then, like the engine's shaders: a cached binary would go stale when common.fx changes.
    /// </summary>
    [AssetPacker(".fxsrc")]
    public class EffectSourcePacker : AssetPacker<EffectSourceData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter writer, EffectSourceData asset) => writer.Write(asset.Source ?? "");

        public override EffectSourceData Unpack(BinaryReader reader, int version) => new(reader.ReadString());
    }
}
