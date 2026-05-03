using Freefall.Assets;
using Freefall.Graph;

namespace Freefall.PCG
{
    /// <summary>
    /// PCG-specific node graph asset.
    /// Extends NodeGraph with the asset type alias for the import pipeline.
    /// Referenced by PCGComponent.
    /// </summary>
    [AssetTypeAlias("PCGGraphData")]
    public class PCGGraph : NodeGraph
    {
    }
}
