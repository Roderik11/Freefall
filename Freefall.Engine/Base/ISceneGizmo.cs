using Freefall.Graphics;

namespace Freefall.Base
{
    /// <summary>
    /// Components implementing this interface draw scene gizmos
    /// when the entity is selected in the editor.
    /// Called during the editor's Draw phase.
    ///
    /// An <see cref="EntitySystem"/> implementing it draws every frame in the editor, whatever is
    /// selected, in world space (<see cref="GizmoContext.Matrix"/> is the identity): markers over
    /// many things at once, which the system already keeps track of (<see cref="Systems.Gizmos"/>).
    /// </summary>
    public interface ISceneGizmo
    {
        void DrawGizmos(GizmoContext ctx);
    }
}
