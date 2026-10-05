using Squid;
using Freefall.Base;
using Freefall.Reflection;

namespace Freefall.Editor
{
    [GUIInspector(typeof(Component))]
    public class ComponentInspector : GUIInspector
    {
        public ComponentInspector(GUIObject target) : base(target)
        {
            var cat = AddCategory(target.Name);

            bool needsCheckbox = typeof(IUpdate).IsAssignableFrom(target.Type) || typeof(IDraw).IsAssignableFrom(target.Type)
                || typeof(Freefall.Graphics.IPersistentDrawSource).IsAssignableFrom(target.Type);

            if (needsCheckbox)
            {
                var sp = target.GetProperty(nameof(Component.Enabled));
                if (sp != null)
                {
                    var enabled = new BoolProperty(sp)
                    {
                        Size = new Point(18, 18),
                        Margin = new Margin(8, 9, 8, 9),
                        Dock = DockStyle.Right,
                    };
                    enabled.CheckButton.Dock = DockStyle.Fill;
                    enabled.CheckButton.Margin = new Margin(0);
                    enabled.Image.Dock = DockStyle.Fill;
                    enabled.Image.Margin = new Margin(3);

                    cat.lblName.GetElements().Add(enabled);
                }
            }

            CreateDropdownMenu(cat.iconFrame, target);

            var headers = new HashSet<string>();

            foreach (var prop in target.GetProperties())
            {
                var header = prop.GetAttribute<System.ComponentModel.CategoryAttribute>();
                if(header != null && !headers.Contains(header.Category))
                {
                    AddHeader(header.Category.ToUpperInvariant());
                    headers.Add(header.Category);
                }

                AddProperty(prop);
            }

            target.OnValueChanged += (property) => 
            {
                foreach(var comp in target.Targets)
                {      
                    if(comp is Component c)
                        c.OnMemberChanged();
                }
            };
        }


        void CreateDropdownMenu(Control parent, GUIObject target)
        {
            var btnCreate = new DropDownButton
            {
                Text = "...",
                Style = "catbutton",
                Size = new Point(30, 30),
                Margin = new Margin(2),
                Dock = DockStyle.Fill,
            };

            int itemHeight = 28;
            int dropWidth = 160;
            int dropHeight = 5 * (itemHeight + 1) + 6;

            btnCreate.Dropdown.Style = "popup";
            btnCreate.Dropdown.Padding = new Margin(4);
            btnCreate.Dropdown.Size = new Point(dropWidth, dropHeight);
            btnCreate.Dropdown.Resizable = false;
            btnCreate.Align = Alignment.TopLeft;
            btnCreate.Parent = parent;

            var dd = btnCreate.Dropdown.Controls;

            dd.Add(AddButton("Copy", (s, e) => { }));
            dd.Add(AddButton("Paste", (s, e) => { }));
            dd.Add(AddButton("Paste as New", (s, e) => { }));

            dd.Add(AddButton("Remove", (s, e) =>
            {
                Component comp = target.Target as Component;
                if (comp != null)
                {
                    var entity = comp.Entity;
                    entity.RemoveComponent(comp);
                    Desktop.CloseDropdowns();
                    MessageDispatcher.Send(Msg.RefreshInspector);
                }
            }));
        }

        Control AddButton(string text, MouseEvent onClick)
        {
            var btn = new Button
            {
                Text = text,
                Style = "button",
                Size = new Point(28, 28),
                Margin = new Margin(1),
                Dock = DockStyle.Top,
            };

            btn.MouseClick += Btn_MouseClick;
            btn.MouseClick += onClick;
            return btn;
        }


        private void Btn_MouseClick(Control sender, MouseEventArgs args)
        {
            Desktop.CloseDropdowns();
        }
    } 
}
