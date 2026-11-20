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

    /// <summary>
    /// An <see cref="EntitySystem"/> whose gizmos are a debug view worth seeing while the game plays (forces on a ship, a
    /// path being followed): drawn while playing as well, in the editor's play mode and in a published game alike
    /// (<see cref="Graphics.PlayGizmoSystem"/>, which the editor's own gizmos do not reach: its camera is gone in play).
    /// It draws only what a switch of its own turns on (a Dev tab option); with nothing on it draws nothing. In edit mode
    /// it is an <see cref="ISceneGizmo"/> like any other.
    /// </summary>
    public interface IPlayGizmo : ISceneGizmo
    {
    }
}
