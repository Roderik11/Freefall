using Squid;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Spline inspector: standard fields plus "Center Pivot", which moves the entity to the centre of its spline
    /// without moving the spline (see <see cref="Spline.CenterPivot"/>).
    /// </summary>
    [GUIInspector(typeof(Spline))]
    public class SplineInspector : ComponentInspector
    {
        public SplineInspector(GUIObject target) : base(target)
        {
            if (target.Target is not Spline) return;

            var centerButton = new Button
            {
                Text = "Center Pivot",
                Style = "button",
                Size = new Point(26, 32),
                Margin = new Margin(8, 4, 8, 4),
                Dock = DockStyle.Top,
                Tooltip = "Move the entity to the centre of its spline points without moving the spline",
            };

            centerButton.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                int moved = 0;
                foreach (var t in target.Targets)
                    if (t is Spline spline && spline.CenterPivot() > 0f) moved++;
                Debug.Log($"[Spline] Centered pivot on {moved} spline(s)");
                MessageDispatcher.Send(Msg.RefreshInspector);
            };

            AddControl(centerButton);
        }
    }
}
