namespace Freefall.Assets
{
    /// <summary>
    /// A label a project defines for itself ("Forest Floor", "Road", ...). It carries nothing but its name:
    /// what a tag means is decided by whatever references it — a <see cref="Freefall.Components.SplatStamp"/>
    /// lists the tags that describe its ground, a PCG ExcludeStamps node lists the tags it does not mind.
    /// The engine knows no particular tag.
    /// </summary>
    [CreateAsset("Tag")]
    public class Tag : Asset
    {
    }
}
