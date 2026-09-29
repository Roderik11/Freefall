using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Squid;
using System;
using System.Collections.Generic;
using System.Linq;
using Win32.Graphics.Direct3D;

namespace Freefall.Editor
{
    public class ExplorerControl : Frame
    {
        public class Header : Frame
        {
            private Button button1;
            private Button button2;
            private Label label1;
            private Label label2;

            public Header()
            {
                NoEvents = false;
                Size = new Point(32, 34);
                Padding = new Margin(0, 0, 0, 0);
                Margin = new Margin(0, 0, 0, 1);
                Dock = DockStyle.Top;

                button1 = new Button
                {
                    Margin = new Margin(0, 0, 1, 0),
                    Style = "header",
                    Dock = DockStyle.Left,
                    Size = new Point(26, 26),
                };

                button2 = new Button
                {
                    Margin = new Margin(0, 0, 1, 0),
                    Style = "header",
                    Dock = DockStyle.Left,
                    Size = new Point(26, 26),
                };

                label1 = new Label
                {
                    NoEvents = true,
                    Margin = new Margin(0, 0, 1, 0),
                    Size = new Point(200, 26),
                    Style = "header",
                    Dock = DockStyle.Left,
                    Text = "Name"
                };

                label2 = new Label
                {
                    NoEvents = true,
                    Margin = new Margin(0, 0, 0, 0),
                    Size = new Point(220, 26),
                    Style = "header",
                    Dock = DockStyle.Fill,
                    Text = "Type"
                };

                Controls.Add(button1);
                //Controls.Add(button2);
                Controls.Add(label1);
                Controls.Add(label2);
            }

            protected override void OnStateChanged()
            {
                label1.State = State;
                label2.State = State;
                button1.State = State;
                button2.State = State;
            }
        }

        public class ExplorerNode : Button
        {
            public ImageControl Button1 { get; private set; }
            public ImageControl Button2 { get; private set; }
            public ImageControl Foldout { get; private set; }
            public Label Label1 { get; private set; }

            public Control IndentFrame { get; private set; }

            public int Indent;

            public event Action<int, Entity> ExpandedChanged;

            public int NodeIndex { get; private set; }
            public Entity Entity { get; private set; }

            private Frame componentIcons;

            public ExplorerNode()
            {
                Size = new Point(100, 28);
                Dock = DockStyle.Top;
                Style = "";
                AllowFocus = false;
                AllowDrop = true;

                IndentFrame = new Control()
                {
                    Style = "indent18",
                    NoEvents = true,
                    Dock = DockStyle.Left,
                };

                Button1 = new ImageControl
                {
                    Style = "item",
                    Size = new Point(27, 26),
                    Dock = DockStyle.Left,
                    Tiling = TextureMode.Center,
                    Color = ColorInt.ARGB(1f, .5f, .5f, .5f)
                };

                Button2 = new ImageControl
                {
                    Style = "item",
                    Size = new Point(27, 26),
                    Dock = DockStyle.Left,
                    Tiling = TextureMode.Center,
                    Color = ColorInt.ARGB(1f, .5f, .5f, .5f)
                };

                Foldout = new ImageControl
                {
                    Style = "item",
                    NoEvents = true,
                    Size = new Point(26, 26),
                    Dock = DockStyle.Left,
                    Enabled = false,
                    Tiling = TextureMode.Center,
                    Color = ColorInt.ARGB(1f, .5f, .5f, .5f)
                };

                Label1 = new Button
                {
                    Style = "item",
                    Size = new Point(200 - 27, 20),
                    Dock = DockStyle.Left,
                    NoEvents = true
                };

                componentIcons = new Frame
                {
                    Style = "item",
                    Size = new Point(20, 20),
                    Dock = DockStyle.Fill,
                    Padding = new Margin(8, 0, 0, 0),
                };

                for (int i = 0; i < 6; i++)
                {
                    var icon = new ImageControl
                    {
                        Texture = "icon_mesh.png",
                        Size = new Point(16, 16),
                        Dock = DockStyle.Left,
                        Margin = new Margin(0, 6, 2, 6),
                        Visible = false
                    };

                    componentIcons.Controls.Add(icon);
                }

                Elements.Add(Button1);
                //Elements.Add(Button2);
                Elements.Add(IndentFrame);
                Elements.Add(Foldout);
                Elements.Add(Label1);
                Elements.Add(componentIcons);

                Foldout.MouseClick += Foldout_MouseClick;

                MouseDrag += (s, e) =>
                {
                    if (e.Button > 0) return;

                    var node = s as ExplorerNode;
                    if (node.Entity == null) return;

                    var proxy = new Label
                    {
                        Text = node.Entity.Name,
                        Size = s.Size,
                        Style = "tooltip",
                        Tag = node.Entity,
                        NoEvents = true
                    };

                    proxy.Position = s.Location;// Gui.MousePosition - proxy.Size / 2;
                    DoDragDrop(proxy);
                };

                DragResponse += Button_DragResponse;
                DragEnter += (s, e) =>
                {
                    savedStated = State;
                };

                DragLeave += (s, e) =>
                {
                    State = savedStated;
                };

                DragDrop += (s, e) =>
                {
                    if (!IsDropCompatible(e)) return;
                    var parent = Entity;
                    var child = e.DraggedControl.Tag as Entity;
                    
                    if (parent != null && child != null)
                    {
                        child.Transform.SetParent(parent.Transform, true);
                        MessageDispatcher.Send(Msg.RefreshExplorer);
                    }
                };
            }

            private ControlState savedStated;

            private bool IsDropCompatible(DragDropEventArgs e)
            {
                if (e.DraggedControl?.Tag is not Entity entity) return false;
            
                if(Entity.Transform.IsChildOf(entity.Transform) || entity == Entity)
                    return false;

                return true;
            }


            private void Button_DragResponse(Control sender, DragDropEventArgs e)
            {
                if (IsDropCompatible(e))
                    State = ControlState.Hot;
            }

            public void Bind(Entity entity, int index, bool flat)
            {
                Entity = entity;
                NodeIndex = index;
                Tag = entity;

                bool hasChildren = entity.Transform.GetChildCount() > 0;
                int depth = entity.Transform.Depth;

                if(flat)
                {
                    depth = 0;
                    hasChildren = false;
                }

                Foldout.Enabled = hasChildren;
                Foldout.NoEvents = !hasChildren;
                Foldout.Texture = hasChildren ? "nav_right.dds" : "";
                
                if(hasChildren)
                    Foldout.Texture = entity.Expanded ? "nav_down.dds" : Foldout.Texture;
                
                Selected = Selector.Selection.Contains(entity);

                Label1.Text = entity.Name;
                IndentFrame.Size = new Point(Indent * depth, IndentFrame.Size.y);
                Label1.Size = new Point((200 - 27) - Indent * depth, Label1.Size.y);
                if (Selected) Focus();

                foreach (var control in componentIcons.Controls)
                    control.Visible = false;

                int iconIndex = 0;

                Label1.Style = entity.Prefab != null ? "prefabItem" : "item";
                //if (entity.Prefab != null)
                //{
                //    var prefabIcon = componentIcons.Controls[0] as ImageControl;
                //    prefabIcon.Visible = true;
                //    prefabIcon.Texture = "icon_prefab_instance.png";
                //    iconIndex++;
                //}
                
                foreach (var comp in entity.Components)
                {
                    if (iconIndex >= componentIcons.Controls.Count)
                        break;

                    if (comp is Transform)
                        continue;

                    var att = Reflector.GetAttribute<IconAttribute>(comp.GetType());
                    if (att == null)
                        continue;

                    var icon = componentIcons.Controls[iconIndex] as ImageControl;
                    icon.Texture = att.Name;
                    icon.Visible = true;
                    iconIndex++;
                }
            }

            protected override void OnStateChanged()
            {
                IndentFrame.State = State;
                Foldout.State = State;
                Label1.State = State;
                Button1.State = State;
                Button2.State = State;
                componentIcons.State = State;
            }

            void Foldout_MouseClick(Control sender, MouseEventArgs args)
            {
                if (args.Button > 0) return;
                ExpandedChanged?.Invoke(NodeIndex, Entity);
            }
        }

        private VirtualList VirtualList;
        private Frame toolbar;
        private SearchBox searchbox;
        private Frame header;
        private bool selectionSender;

        private List<Entity> entities = new List<Entity>();
        private List<Entity> bindList;
        private Label footerLabel;

        bool IsSearching => !string.IsNullOrEmpty(searchbox.Text);

        public ExplorerControl()
        {
            Size = new Point(340, 200);

            toolbar = new Frame
            {
                Style = "frame",
                Size = new Point(16, 40),
                Dock = DockStyle.Top,
                Margin = new Margin(0, 0, 0, 1)
            };

            searchbox = new SearchBox
            {
                Size = new Point(200, 16),
                Dock = DockStyle.Fill,
                Margin = new Margin(28, 8, 8, 8)
            };

            header = new Header();

            footerLabel = new Label
            {
                Style = "header",
                Size = new Point(16, 28),
                Dock = DockStyle.Bottom,
                Margin = new Margin(0, 1, 0, 0),
                Text = "100 Entities"
            };

            searchbox.TextChanged += Searchbox_TextChanged;

            VirtualList = new VirtualList();
            VirtualList.Dock = DockStyle.Fill;
            VirtualList.Scrollbar.ButtonDown.Visible = false;
            VirtualList.Scrollbar.ButtonUp.Visible = false;
            VirtualList.Scrollbar.Slider.Ease = false;
            VirtualList.Scrollbar.Slider.MinHandleSize = 64;
            VirtualList.CreateItem = CreateNode;
            VirtualList.BindItem = BindNode;

            toolbar.Controls.Add(searchbox);
            Controls.Add(toolbar);
            Controls.Add(header);
            Controls.Add(footerLabel);
            Controls.Add(VirtualList);

            MessageDispatcher.AddListener(Msg.RefreshExplorer, (msg) => { Refresh(); });
            MessageDispatcher.AddListener(Msg.SelectionChanged, (msg) =>
            {
                if (selectionSender) return;

                var entity = msg.Data as Entity;
                if (entity != null)
                    ExpandTo(entity);

                VirtualList.Refresh();
            });

            bindList = entities;
        }

        private void ExpandTo(Entity entity)
        {
            try
            {
                var child = entity;
                var root = entity;

                while (root.Transform.Parent != null)
                    root = root.Transform.Parent.Entity;

                int index;

                if (root.Expanded)
                {
                    var remove = new List<Entity>();
                    index = entities.IndexOf(root);
                    FindExpandedChildren(root, remove);
                    entities.RemoveRange(index + 1, remove.Count);
                }

                while (entity.Transform.Parent != null)
                {
                    entity = entity.Transform.Parent.Entity;
                    entity.Expanded = true;
                }

                if (root.Expanded)
                {
                    var add = new List<Entity>();
                    FindExpandedChildren(root, add);
                    index = entities.IndexOf(root);
                    entities.InsertRange(index + 1, add);
                }

                int childindex = entities.IndexOf(child);

                VirtualList.Refresh();
                VirtualList.ScrollTo(childindex);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Explorer] Failed to expand to entity: {ex}");
            }
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();
         
            if (IsSearching)
                footerLabel.Text = $"{bindList.Count} Entities";
            else
                footerLabel.Text = $"{EntityManager.Entities.Count} Entities";
        }

        private void Searchbox_TextChanged(Control sender)
        {
            var str = searchbox.Text;
            bool isempty = string.IsNullOrEmpty(str);
            str = str.ToLower();

            bindList = entities;

            if (!isempty)
            {
                var list = new List<Entity>();

                foreach (Entity e in EntityManager.Entities)
                {
                    if (e.HideInHierarchy) continue;
                    //if (e.Transform.Parent != null) continue;
                    if (e.Name.ToLower().Contains(str))
                        list.Add(e);
                }
                bindList = list;
            }

            VirtualList.DataSource = bindList;
            VirtualList.Refresh();
        }

        public void Refresh()
        {
            entities.Clear();

            foreach (Entity e in EntityManager.Entities)
            {
                if (e.HideInHierarchy) continue;

                if (e.Transform.Parent != null) continue;
                entities.Add(e);

                if (e.Expanded)
                    FindExpandedChildren(e, entities);
            }

            VirtualList.DataSource = bindList;
            VirtualList.Refresh();
        }

        void FindExpandedChildren(Entity entity, List<Entity> result)
        {
            var childCount = entity.Transform.GetChildCount();
            for (int i = 0; i < childCount; i++)
            {
                var child = entity.Transform.GetChild(i);
                result.Add(child.Entity);

                if (child.Entity.Expanded)
                    FindExpandedChildren(child.Entity, result);
            }
        }

        private void BindNode(Control control, int index)
        {
            var node = control as ExplorerNode;
            var entity = bindList[index];
            node.Bind(entity, index, !string.IsNullOrEmpty(searchbox.Text));
        }

        private ExplorerNode CreateNode(int index)
        {
            var entity = bindList[index];

            ExplorerNode node = new ExplorerNode();
            node.Indent = 14;
            node.MouseClick += Node_MouseClick;
            node.ExpandedChanged += Node_ExpandedChanged;
            node.MouseDoubleClick += Node_MouseDoubleClick;
            node.AllowFocus = true;
            node.Bind(entity, index, !string.IsNullOrEmpty(searchbox.Text));

            node.KeyDown += (s, e) =>
            {
                var myEntity = (s as ExplorerNode).Entity;
                var myIndex = bindList.IndexOf(myEntity);

                if (e.Key == Squid.Keys.RIGHTARROW && !myEntity.Expanded)
                    Node_ExpandedChanged(myIndex, myEntity);

                if (e.Key == Squid.Keys.LEFTARROW && myEntity.Expanded)
                    Node_ExpandedChanged(myIndex, myEntity);

                if (e.Key == Squid.Keys.DOWNARROW)
                {
                    var nextIndex = Math.Min(myIndex + 1, bindList.Count - 1);
                    var nextDir = bindList[nextIndex];
                    Select(nextDir);
                    VirtualList.ScrollTo(nextIndex);
                }

                if (e.Key == Squid.Keys.UPARROW)
                {
                    var prev = Math.Max(myIndex - 1, 0);
                    var nextDir = bindList[prev];
                    Select(nextDir);
                    VirtualList.ScrollTo(prev);
                };
            };

            return node;
        }

        private void Node_MouseClick(Control sender, MouseEventArgs args)
        {
            var node = sender as ExplorerNode;

            if (args.Button > 0)
            {
                // Right-click: show context menu
                Select(node.Entity);
                ShowContextMenu(node.Entity);
                return;
            }

            // RANGE SELECTION
            if(Input.Shift)
            {
                var selected = Selector.SelectedEntity;
                if (selected != null && selected != node.Entity)
                {
                    selectionSender = true;
                    var selectedIndex = bindList.IndexOf(selected);
                    var clickedIndex = bindList.IndexOf(node.Entity);
                    int start = Math.Min(selectedIndex, clickedIndex);
                    int end = Math.Max(selectedIndex, clickedIndex);
                    for (int i = start; i <= end; i++)
                    {
                        var entity = bindList[i];
                     
                        Selector.AddToSelection(entity);
                    }
                    VirtualList.Refresh();
                    selectionSender = false;
                    return;
                }
            }
            else
            {
                Select(node.Entity);
            }
        }

        private void ShowContextMenu(Entity entity)
        {
            var desktop = Desktop;
            if (desktop == null) return;

            var menu = new Window
            {
                Style = "frame",
                AutoSize = AutoSize.Vertical,
                Size = new Point(180, 0),
                Position = new Point(Gui.MousePosition.x, Gui.MousePosition.y),
            };

            if (entity.IsPrefabInstance)
            {
                // Count how many instances of this prefab exist
                int count = EntityManager.Entities.Count(e => e.Prefab == entity.Prefab);

                AddMenuItem(menu, $"Apply to Prefab ({count})", () =>
                {
                    int updated = entity.Prefab.UpdateFromInstance(entity);
                    Debug.Log($"[Explorer] Applied to prefab, updated {updated} instances");
                    Refresh();
                });
            }

            AddMenuItem(menu, "Unpack", () =>
            {
                Unpack(entity);
                Refresh();
            });

            // Show "Merge Meshes" if entity has children with StaticMeshRenderers
            if (HasChildRenderers(entity))
            {
                AddMenuItem(menu, "Merge Meshes", () =>
                {
                    MergeMeshes(entity);
                });
            }

            AddMenuItem(menu, "Delete", () =>
            {
                entity.Destroy();
                Refresh();
            });

            desktop.ShowDropdown(menu, false);
        }

        private void Unpack(Entity entity)
        {
            entity.Prefab = null;
         
            foreach(Transform child in entity.Transform)
                Unpack(child.Entity);
        }

        private void AddMenuItem(Window menu, string text, Action action)
        {
            var item = new Button
            {
                Style = "button",
                Size = new Point(180, 28),
                Dock = DockStyle.Top,
                Text = text,
            };
            item.MouseClick += (s, e) =>
            {
                if (e.Button > 0) return;
                action();
                Desktop.CloseDropdowns();
            };
            menu.Controls.Add(item);
        }

        void Select(Entity entity)
        {
            selectionSender = true;
            Selector.SelectedEntity = entity;
            selectionSender = false;
            VirtualList.Refresh();
        }

        private void Node_ExpandedChanged(int index, Entity entity)
        {
            entity.Expanded = !entity.Expanded;

            var list = new List<Entity>();
            FindExpandedChildren(entity, list);

            if (entity.Expanded)
                entities.InsertRange(index + 1, list);
            else
                entities.RemoveRange(index + 1, list.Count);

            VirtualList.Refresh();
        }

        private void Node_MouseDoubleClick(Control sender, MouseEventArgs args)
        {
            var node = sender as ExplorerNode;
            var entity = node.Entity;
            MessageDispatcher.Send(Msg.FocusEntity, entity);
        }

        private bool HasChildRenderers(Entity entity)
        {
            int childCount = entity.Transform.GetChildCount();
            for (int i = 0; i < childCount; i++)
            {
                var child = entity.Transform.GetChild(i);
                if (child?.Entity?.GetComponent<MeshRenderer>() != null)
                    return true;
            }
            return false;
        }

        private void MergeMeshes(Entity entity)
        {
           // var sw = System.Diagnostics.Stopwatch.StartNew();

           // var result = MeshCombiner.Combine(entity);
           // if (result == null)
           // {
           //     Debug.LogWarning("Explorer", "Merge failed: no renderers found");
           //     return;
           // }

           // var staticMesh = MeshCombiner.CreateStaticMesh(Engine.Device, result);
           // staticMesh.Guid = Guid.NewGuid().ToString("N");
           // if (staticMesh == null)
           // {
           //     Debug.LogWarning("Explorer", "Merge failed: could not create mesh");
           //     return;
           // }

           // // Destroy all children
           // int childCount = entity.Transform.GetChildCount();
           // for (int i = childCount - 1; i >= 0; i--)
           // {
           //     var child = entity.Transform.GetChild(i);
           //     child?.Entity?.Destroy();
           // }

           // // Add merged renderer to root
           // var renderer = entity.AddComponent<StaticMeshRenderer>();
           // renderer.StaticMesh = staticMesh;

           // sw.Stop();
           //// Debug.Log($"[Explorer] Merged {result.SourceCount} renderers → {result.Materials.Count} materials, " +
           ////           $"{result.MeshDataPerLOD[0]?.Positions?.Length ?? 0} verts in {sw.ElapsedMilliseconds}ms");

           // Refresh();
        }
    }
}
