using Squid;
using Freefall.Reflection;
using Freefall.Assets;
using Freefall.Base;

namespace Freefall.Editor
{
   // [PropertyControl(typeof(Terrain.TextureLayer))]
    public class SplayLayerProperty : PropertyControl
    {
        private Terrain.TextureLayer layer;

        public SplayLayerProperty(GUIProperty property) : base(property)
        {
            RowHeight = 98;

            layer = property.GetValue() as Terrain.TextureLayer;
            var obj = new GUIObject(layer);
            var inspector = GUIInspector.GetInspector(obj);

            Controls.Add(inspector);

            //var prpDiffuse = obj.GetProperty("Diffuse");
            //var prpNormals = obj.GetProperty("Normals");
            //var prpTiling = obj.GetProperty("Tiling");

            //var diffuse = GUIInspector.GetPropertyControl(prpDiffuse);
            //var normals = GUIInspector.GetPropertyControl(prpNormals);
            //var tiling = GUIInspector.GetPropertyControl(prpTiling);

            //diffuse.Dock = DockStyle.Top;
            //normals.Dock = DockStyle.Top;
            //tiling.Dock = DockStyle.Top;
            
            //Controls.Add(diffuse);
            //Controls.Add(normals);
            //Controls.Add(tiling);

            //NoEvents = false;
        }

        public override void OnRowSelect()
        {
            MessageDispatcher.Send(Msg.SelectLayer, layer);
        }
    }
}
