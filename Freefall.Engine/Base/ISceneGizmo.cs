using Freefall.Graphics;

namespace Freefall.Base
{
    /// <summary>
    /// Components implementing this interface draw scene gizmos
    /// when the entity is selected in the editor.
    /// Called during the editor's Draw phase.
    /// </summary>
    public interface ISceneGizmo
    {
        void DrawGizmos(GizmoContext ctx);
    }
}
