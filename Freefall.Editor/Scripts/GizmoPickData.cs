namespace Freefall.Editor
{
    /// <summary>
    /// Data payload for a gizmo axis pick.
    /// </summary>
    public class GizmoPickData
    {
        public int Slot;
        public MoveAxis Axis;
        public ToolBase Tool;
        /// <summary>
        /// MeshPart index from the GPU EntityIdBuffer pick (lower 8 bits of packed value).
        /// </summary>
        public uint MeshPart;
    }
}
