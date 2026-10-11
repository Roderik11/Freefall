using Squid;
using Freefall.Base;
using Freefall.Components;
using Freefall.Navigation;
using Freefall.Reflection;

namespace Freefall.Editor
{
    /// <summary>
    /// Custom inspector for NavMeshSurface. Extends the standard component inspector
    /// with bake buttons and a progress / stats display. The bake runs in the background.
    /// </summary>
    [GUIInspector(typeof(NavMeshSurface))]
    public class NavMeshSurfaceInspector : ComponentInspector
    {
        private readonly NavMeshSurface _surface;
        private Button _bakeButton;
        private Button _rebuildButton;
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

            // Bake: only the tiles whose surroundings changed since the last bake. Cancels while baking.
            _bakeButton = AddBakeButton("Bake NavMesh", () =>
            {
                if (_surface.IsBaking) _surface.CancelBake();
                else _surface.Bake();
            });

            _rebuildButton = AddBakeButton("Rebuild All Tiles", () =>
            {
                if (!_surface.IsBaking) _surface.Bake(rebuildAll: true);
            });

            // Stats label
            _statsLabel = new Label
            {
                Text = GetStatusText(),
                Style = "tooltip",
                Dock = DockStyle.Top,
                Margin = new Margin(8, 4, 8, 4),
                Size = new Point(26, 20),
                AutoSize = AutoSize.Vertical,
            };
            _statsLabel.Update += _ => Refresh();

            AddControl(_statsLabel);
        }

        private Button AddBakeButton(string text, System.Action onClick)
        {
            var button = new Button
            {
                Text = text,
                Style = "button",
                Size = new Point(26, 32),
                Margin = new Margin(8, 4, 8, 4),
                Dock = DockStyle.Top,
            };

            button.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                onClick();
            };

            AddControl(button);
            return button;
        }

        private void Refresh()
        {
            bool baking = _surface.IsBaking;

            string bakeText = baking ? "Cancel Bake" : "Bake NavMesh";
            if (_bakeButton.Text != bakeText) _bakeButton.Text = bakeText;
            if (_rebuildButton.Enabled == baking) _rebuildButton.Enabled = !baking;

            string status = GetStatusText();
            if (_statsLabel.Text != status) _statsLabel.Text = status;
        }

        private string GetStatusText()
        {
            var bake = _surface?.GetBake();
            if (bake != null)
            {
                switch (bake.Stage)
                {
                    case NavMeshBakeStage.PreparingMeshes:
                        return "Baking: preparing meshes...";
                    case NavMeshBakeStage.BuildingTiles:
                        return $"Baking: {bake.DoneCells} / {bake.TotalCells} tiles ({bake.Progress * 100f:0}%)";
                    case NavMeshBakeStage.Assembling:
                        return "Baking: assembling navmesh...";
                    case NavMeshBakeStage.Cancelled:
                        return "Bake cancelled.\n" + GetStatsText();
                    case NavMeshBakeStage.Failed:
                        return $"Bake failed: {bake.Error}";
                    case NavMeshBakeStage.Done when bake.Result != null:
                        return $"{GetStatsText()}\nLast bake: {bake.Result.Seconds:0.0} s, {bake.Result.RebuiltCells} tiles rebuilt, {bake.Result.ReusedCells} unchanged";
                }
            }

            return GetStatsText();
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
