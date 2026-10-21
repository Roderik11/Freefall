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

            // [Button] methods (an asset's too), placed as ComponentInspector places them: each header's buttons
            // close its fields; those without a category come last
            var buttons = InspectorButtons.Of(target.Type);
            string? current = null;

            foreach (var prop in target.GetProperties())
            {
                var subcat = prop.GetAttribute<System.ComponentModel.CategoryAttribute>();
                if (subcat != null && !headers.Contains(subcat.Category))
                {
                    if (current != null) AddButtons(buttons, current, target);
                    AddHeader(subcat.Category.ToUpperInvariant());
                    headers.Add(subcat.Category);
                    current = subcat.Category;
                }

                AddProperty(prop);
            }

            if (current != null) AddButtons(buttons, current, target);
            while (buttons.Find(b => b.Category != null) is { } next)
            {
                AddHeader(next.Category.ToUpperInvariant());
                AddButtons(buttons, next.Category, target);
            }
            AddButtons(buttons, null, target);
        }

        /// <summary>Adds the buttons of one category (null = those without one) and takes them off the list.</summary>
        void AddButtons(List<InspectorButtons.Entry> buttons, string? category, GUIObject target)
        {
            var these = buttons.FindAll(b => b.Category == category);
            if (these.Count == 0) return;
            buttons.RemoveAll(b => b.Category == category);

            foreach (var control in InspectorButtons.Build(these, target))
                AddControl(control);
        }
    }
}
