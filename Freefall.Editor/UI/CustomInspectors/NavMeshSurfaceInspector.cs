using Squid;
using Freefall.Base;
using Freefall.Components;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Custom inspector for NavMeshSurface. Extends the standard component inspector
    /// with a "Bake NavMesh" button and stats display.
    /// </summary>
    [GUIInspector(typeof(NavMeshSurface))]
    public class NavMeshSurfaceInspector : ComponentInspector
    {
        private readonly NavMeshSurface _surface;
        private Label _statsLabel;

        public NavMeshSurfaceInspector(GUIObject target) : base(target)
        {
            _surface = target.Target as NavMeshSurface;
            if (_surface == null) return;

            if (_surface.NavMesh != null)
            {
                var insp = new GenericInspector(new GUIObject(_surface.NavMesh), false, true);
                AddControl(insp);
            }   
                
            // Bake button
            var bakeButton = new Button
            {
                Text = "Bake NavMesh",
                Style = "button",
                Size = new Point(26, 32),
                Margin = new Margin(8, 4, 8, 4),
                Dock = DockStyle.Top,
            };

            bakeButton.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                BakeNavMesh();
            };

            AddControl(bakeButton);

            // Stats label
            _statsLabel = new Label
            {
                Text = GetStatsText(),
                Style = "tooltip",
                Dock = DockStyle.Top,
                Margin = new Margin(8, 4, 8, 4),
                Size = new Point(26, 20),
                AutoSize = AutoSize.Vertical,
            };

            AddControl(_statsLabel);
        }

        private void BakeNavMesh()
        {
            if (_surface == null) return;

            _statsLabel.Text = "Baking...";

            var result = _surface.Bake();

            if (result != null)
            {
                _statsLabel.Text = GetStatsText();
                Debug.Log($"[Editor] NavMesh baked successfully");
            }
            else
            {
                _statsLabel.Text = "Bake failed — check console for details.";
            }
        }

        private string GetStatsText()
        {
            var navMesh = _surface?.NavMesh;
            if (navMesh == null)
                return "No navmesh baked yet.";

            if (navMesh.PolyCount == 0 && navMesh.VertexCount == 0)
                return "No navmesh data.";

            return $"{navMesh.PolyCount} polys, {navMesh.VertexCount} verts";
        }
    }
}
