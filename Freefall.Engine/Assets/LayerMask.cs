using System.Collections.Generic;

namespace Freefall.Assets
{
    /// <summary>
    /// A typed list of stable TextureLayer UIDs used for layer-driven decoration placement.
    /// Inherits from List&lt;ulong&gt; so it serializes identically, but provides a distinct
    /// type for the editor's property control system (GUIInspector.FindControlType matches
    /// LayerMask before the generic List&lt;&gt; fallback).
    /// </summary>
    public class LayerMask : List<ulong>
    {
        public LayerMask() { }
        public LayerMask(IEnumerable<ulong> collection) : base(collection) { }
    }
}
