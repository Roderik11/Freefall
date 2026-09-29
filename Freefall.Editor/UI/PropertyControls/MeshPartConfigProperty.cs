using Squid;
using Freefall.Reflection;
using Freefall.Assets.Importers;

namespace Freefall.Editor
{
    /// <summary>
    /// Compact property control for MeshPartConfig. Shows a single row:
    /// [Name label] [☑Render] [☑Collision] [LODMin] [LODMax]
    /// instead of the default expanded 5-row layout.
    /// </summary>
    [PropertyControl(typeof(MeshPartConfig))]
    public class MeshPartConfigProperty : PropertyControl
    {
        public MeshPartConfigProperty(GUIProperty property) : base(property)
        {
            RowHeight = 32;

            var config = property.GetValue() as MeshPartConfig;
            if (config == null) return;

            var obj = new GUIObject(config);

            var renderProp = obj.GetProperty("Render");
            var collisionProp = obj.GetProperty("Collision");
            var lodMinProp = obj.GetProperty("LODMin");
            var lodMaxProp = obj.GetProperty("LODMax");

            var renderCtrl = GUIInspector.GetPropertyControl(renderProp);
            var collisionCtrl = GUIInspector.GetPropertyControl(collisionProp);
            var lodMinCtrl = GUIInspector.GetPropertyControl(lodMinProp);
            var lodMaxCtrl = GUIInspector.GetPropertyControl(lodMaxProp);

            lodMinCtrl.Size = new Point(40, 32);
            lodMaxCtrl.Size = new Point(40, 32);

;           renderCtrl.Dock = DockStyle.Right;
            collisionCtrl.Dock = DockStyle.Right;
            lodMinCtrl.Dock = DockStyle.Right;
            lodMaxCtrl.Dock = DockStyle.Right;

            Controls.Add(lodMaxCtrl);
            Controls.Add(lodMinCtrl);
            Controls.Add(collisionCtrl);
            Controls.Add(renderCtrl);
        }
    }
}
