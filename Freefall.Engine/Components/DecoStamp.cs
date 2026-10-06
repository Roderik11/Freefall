using Freefall.Assets;
using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    public enum DecoOp
    {
        /// <summary>Give the decorator at least Weight coverage here (0..1).</summary>
        Add,
        /// <summary>Scale existing coverage by Weight: 0 = clear, 1 = no change, above 1 = boost.</summary>
        Multiply,
    }

    /// <summary>
    /// Non-destructive decoration stamp: controls where a <see cref="TerrainDecorator"/> grows.
    ///
    /// Add places a decorator; a global Add stamp filtered by RequireLayers is "grass blades wherever
    /// the grass layer shows". Multiply thins or boosts what earlier stamps placed, for one decorator or,
    /// with no Decorator set, for all of them (the cleared ground around a house).
    ///
    /// The filter's layer terms read the finished splat result, so they do not depend on Priority.
    /// </summary>
    [Icon("icon_decostamp.png")]
    public class DecoStamp : CoverageStamp
    {
        /// <summary>The decorator this stamp affects. Empty = every decorator (Multiply only).</summary>
        [System.ComponentModel.Category("Decoration")]
        public TerrainDecorator Decorator;

        public DecoOp Op = DecoOp.Multiply;

        /// <summary>Add: coverage 0..1. Multiply: 0 = fully suppress, 1 = no change, above 1 = boost.</summary>
        [FormerlySerializedAs("Density")]
        [ValueRange(0f, 2f)]
        public float Weight = 0f;

        protected override Color4 GizmoColor => new Color4(0.5f, 0.7f, 1f, 1f); // blue
    }
}
