using Freefall.Reflection;
using Squid;

namespace Freefall.Editor
{
    public class GenericInspector : GUIInspector
    {
        public GenericInspector(GUIObject target, bool header = true, bool expanded = true):base(target)
        {
            if (header)
            {
                 var cat = AddCategory(target.Name);
                 cat.Expanded = expanded;
            }

            var headers = new HashSet<string>();

            foreach (var prop in target.GetProperties())
            {
                var subcat = prop.GetAttribute<System.ComponentModel.CategoryAttribute>();
                if (subcat != null && !headers.Contains(subcat.Category))
                {
                    AddHeader(subcat.Category.ToUpperInvariant());
                    headers.Add(subcat.Category);
                }

                AddProperty(prop);
            }
        }
    }
}
