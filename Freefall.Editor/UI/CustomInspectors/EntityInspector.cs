using Squid;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Editor
{
    [GUIInspector(typeof(Entity))]
    public class EntityInspector : GUIInspector
    {
        public EntityInspector(GUIObject target) : base(target)
        {
            //var entity = target.Target as Entity;

            AddCategory("Entity");

            foreach (var prop in target.GetProperties())
                AddProperty(prop);

            var entities = target.Targets;

            // foreach component type on all entities
            // add inspector for that set of compoonents shared across all selected entities
            // need to use GUIObject(params object[] targets) constructor to support multi-select with different component sets

            // 1. Collect all component types across all selected entities
            Dictionary<Type, List<Component?>> componentsByType = new();
            foreach (Entity entity in entities)
            {
                foreach(var comp in entity.Components)
                {
                    var type = comp.GetType();
                    if (!componentsByType.ContainsKey(type))
                        componentsByType[type] = new List<Component?>();

                    componentsByType[type].Add(comp);
                }
            }

            // 2. Create a GUIObject with those components as targets, and find an inspector for that type

            foreach(var kvp in componentsByType)
            {
                var obj = new GUIObject(kvp.Value.ToArray());
                var insp = GetInspector(obj);
                if (insp != null)
                {
                    Controls.Add(insp);
                    insp.PerformLayout();
                }
            }


            //foreach (var component in entity.Components)
            //{
            //    var obj = new GUIObject(component);
            //    var insp = GetInspector(obj);
            //    if (insp != null)
            //    {
            //        Controls.Add(insp);
            //        insp.PerformLayout();
            //    }
            //}
        }
    }
}
