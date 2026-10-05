using System;
using System.Collections.Generic;
using Squid;
using System.Numerics;
using Freefall.Assets;
using Freefall.Graph;
using System.Runtime.CompilerServices;
using System.Data.Common;
using Vortice.Mathematics;
using Freefall.Reflection;
using Freefall.Base;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [CustomAssetEditor(typeof(NodeGraph))]
    public class GraphEditor : Frame, ICustomAssetEditor
    {
        private readonly GraphCanvas CanvasFrame;
        private readonly Frame Toolbar;

        private ScrollPanel scrollPanel;
        private GUIInspector nodeList;

        public NodeGraph Graph;

        private struct SortableNode
        {
            public Type Type;
            public string Category;
        }

        public GraphEditor()
        {
            Dock = DockStyle.Fill;
            Size = new Point(800, 600);
            Position = new Point(300, 300);
            MaxSize = Point.Zero;

            Toolbar = EditorToolbar.Create();
            Controls.Add(Toolbar);

            EditorToolbar.Add(Toolbar, EditorSkin.IconSave, "Save", BtnSave_MouseDown);
            EditorToolbar.Add(Toolbar, EditorSkin.IconOpen, "Load", BtnLoad_MouseDown);
            EditorToolbar.AddSeparator(Toolbar);
            EditorToolbar.Add(Toolbar, EditorSkin.IconPlay, "Execute", BtnExecute_MouseDown, primary: true);

            SplitContainer split = new SplitContainer();
            split.Dock = DockStyle.Fill;
            split.SplitFrame1.Size = new Point(200, 200);
            split.SplitButton.Margin = new Squid.Margin(1, 0, 1, 0);
            split.SplitButton.Size = new Squid.Point(2, 2);
            split.RetainAspect = false;
            Controls.Add(split);

            Graph = new NodeGraph();

            CanvasFrame = new GraphCanvas();
            CanvasFrame.Style = "canvas";
            CanvasFrame.Graph = Graph;

            split.SplitFrame2.Controls.Add(CanvasFrame);

            List<Type> types = Freefall.Reflection.Reflector.GetTypes<Node>();
            types.Sort((a, b) => a.Name.CompareTo(b.Name));


            var obj = new GUIObject(new object());
            nodeList = new GUIInspector(obj);
            scrollPanel = new ScrollPanel();
            scrollPanel.Content.Controls.Add(nodeList);
            split.SplitFrame1.Controls.Add(scrollPanel);

            var sortable = new List<SortableNode>();
            foreach (Type type in types)
            {
                var cat = Freefall.Reflection.Reflector.GetAttribute<System.ComponentModel.CategoryAttribute>(type);
                sortable.Add(new SortableNode { Type = type, Category = cat != null ? cat.Category : "Other" });
            }

            sortable.Sort((a, b) =>
            {
                int compare = a.Category.CompareTo(b.Category);
                if (compare != 0) return compare;
                return a.Type.Name.CompareTo(b.Type.Name);
            });

            string categoryName = string.Empty;

            foreach (var item in sortable)
            {
                if (item.Category != categoryName)
                {
                    categoryName = item.Category;
                    var c = nodeList.AddCategory(item.Category);
                    c.Size = new Point(c.Size.x, 24);
                }

                Button btn = new Button();
                btn.Text = item.Type.Name;
                btn.Size = new Point(100, 24);
                btn.Dock = DockStyle.Top;
                btn.MouseDrag += Btn_MouseDrag;
                btn.Tag = item.Type;
                btn.Style = "item";
                btn.Margin = new Squid.Margin(0, 0, 0, 1);
                nodeList.AddControl(btn);
            }
        }

        private void ResetCanvas(NodeGraph graph)
        {
            CanvasFrame.Clear();
            Graph = graph;
            CanvasFrame.SetGraph(graph);

            foreach (var node in Graph.Nodes)
                CanvasFrame.AddNode(node);
        }

        /// <summary>
        /// ICustomAssetEditor: open any NodeGraph-derived asset.
        /// </summary>
        public void OpenAsset(Asset asset)
        {
            if (asset is NodeGraph graph)
                OpenGraph(graph);
        }

        /// <summary>
        /// Open a NodeGraph for editing.
        /// Called from asset browser double-click or inspector.
        /// </summary>
        public void OpenGraph(NodeGraph graph)
        {
            ResetCanvas(graph);
        }

        void BtnSave_MouseDown(Control sender, MouseEventArgs args)
        {
            if (Graph == null) return;

            // If the graph has an asset path, save in place
            if (!string.IsNullOrEmpty(Graph.AssetPath))
            {
                Freefall.Assets.AssetFile.Save(Graph.AssetPath, Graph);
                Debug.Log($"[GraphEditor] Saved: {Graph.AssetPath}");
                return;
            }

            // Otherwise, Save As dialog
            using (System.Windows.Forms.SaveFileDialog dlg = new System.Windows.Forms.SaveFileDialog())
            {
                dlg.Filter = "Graph Asset|*.asset";
                if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    Graph.AssetPath = dlg.FileName;
                    Graph.Name = System.IO.Path.GetFileNameWithoutExtension(dlg.FileName);
                    Freefall.Assets.AssetFile.Save(dlg.FileName, Graph);
                    Debug.Log($"[GraphEditor] Saved: {dlg.FileName}");
                }
            }
        }

        void BtnLoad_MouseDown(Control sender, MouseEventArgs args)
        {
            using (System.Windows.Forms.OpenFileDialog dlg = new System.Windows.Forms.OpenFileDialog())
            {
                dlg.Filter = "Graph Asset|*.asset";
                if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    var loaded = Freefall.Assets.AssetFile.Load(dlg.FileName);
                    if (loaded is NodeGraph graph)
                    {
                        graph.AssetPath = dlg.FileName;
                        graph.RebuildAfterLoad();
                        ResetCanvas(graph);
                        Debug.Log($"[GraphEditor] Loaded: {dlg.FileName} ({graph.Nodes.Count} nodes)");
                    }
                    else
                    {
                        Debug.Log($"[GraphEditor] Failed to load graph from {dlg.FileName}");
                    }
                }
            }
        }

        void BtnExecute_MouseDown(Control sender, MouseEventArgs args)
        {
            if (Graph == null || Graph.Nodes.Count == 0)
                return;

            Graph.Execute();
            Debug.Log($"[GraphEditor] Executed graph: {Graph.Nodes.Count} nodes processed.");
        }

        void Btn_MouseDrag(Control sender, MouseEventArgs args)
        {
            Label label = new Label();
            label.Tag = sender.Tag;
            label.Text = ((Button)sender).Text;
            label.Size = sender.Size;
            label.Position = Gui.MousePosition;
            DoDragDrop(label);
        }
    }

    public class GraphCanvas : Window
    {
        private Control DownSource;
        private Point maxSize = new Point(1000000, 1000000);
        private bool isDragging;

        public NodeGraph Graph;

        public GraphCanvas()
        {
            DownSource = null;
            UIScale = 1f;
            Resizable = false;
            AllowDragOut = true;

            Dock = DockStyle.Center;
            Size = maxSize;
            Position = Size / -2;
            DragDrop += Canvas_DragDrop;
            AllowDrop = true;
            NoEvents = false;

            MouseDown += Canvas_MouseDown;
            MouseUp += Canvas_MouseUp;

            Gui.MouseDown += Gui_MouseDown;
            Gui.MouseUp += Gui_MouseUp;
        }

        public void Clear()
        {
            Controls.Clear();
        }

        public void SetGraph(NodeGraph graph)
        {
            Graph = graph;

            UIScale = 1;
            Size = maxSize;
            Position = Size / -2;
        }

        public void AddNode(Node node)
        {
            GraphFrame frame = new GraphFrame(node);
            Controls.Add(frame);
        }

        private void Gui_MouseUp(Control sender, MouseEventArgs args)
        {
            DownSource = null;
        }

        private void Gui_MouseDown(Control sender, MouseEventArgs args)
        {
            if (sender is Plug plug && plug.IsChildOf(this))
                DownSource = plug;
        }
        

        private void Canvas_MouseUp(Control sender, MouseEventArgs args)
        {
            if (args.Button == 2)
            {
                StopDrag();
                isDragging = false;
            }
        }

        private void Canvas_MouseDown(Control sender, MouseEventArgs args)
        {
            if (args.Button == 2)
            {
                StartDrag();
                isDragging = true;
            }
        }

        void Canvas_DragDrop(Control sender, DragDropEventArgs e)
        {
            if (e.DraggedControl == null) return;

            Type type = e.DraggedControl.Tag as Type;
            if (typeof(Node).IsAssignableFrom(type))
            {
                var node = Activator.CreateInstance(type) as Node;
                Graph.AddNode(node);
                
                GraphFrame frame = new GraphFrame(node);
                frame.Position = (Gui.MousePosition - Location) / UIScale;
                Controls.Add(frame);
            }
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();
            if (isDragging) return;

            var mouse = Gui.MousePosition;
            var mouseDelta = Math.Sign(Input.MouseWheelDelta);
            var oldScale = UIScale;

            if (Hit(mouse.x, mouse.y) && mouseDelta != 0)
                UIScale += .05f * mouseDelta;

            //if (Input.IsKey(Keys.Oemplus))
            //    UIScale += .020f;

            //if (Input.IsKey(Keys.OemMinus))
            //    UIScale -= .020f;

            UIScale = Math.Clamp(UIScale, 0.3f, 2);
            var batch = ((SquidRenderer)Gui.Renderer).SpriteBatch;

            if (UIScale == oldScale) return;
            var ratio = UIScale / oldScale;

            if (Dock != DockStyle.None)
            {
                var p = Position;
                var s = Size;
                Dock = DockStyle.None;
                Position = p;
                Size = s;
            }

            var pointInChild = (mouse - Location);

            Size = maxSize * UIScale;
            Position = (mouse - pointInChild * ratio) - Parent.Location;
        }

        private static readonly Color4 WireColor = new Color4(.50f, .66f, .92f, 1);
        private static readonly Color4 PendingWireColor = new Color4(1f, .47f, .37f, 1);

        private const int GridStep = 24;
        private const int GridMajorEvery = 5;

        /// <summary>
        /// Canvas background plus a grid that pans and zooms with the graph.
        /// </summary>
        protected override void DrawStyle(Style style, float opacity)
        {
            base.DrawStyle(style, opacity);
            if (Parent == null) return;

            float step = GridStep * UIScale;
            if (step < 6) return;

            // The canvas itself is huge; only the part inside the parent is visible
            Point origin = Location, view = Parent.Location, viewSize = Parent.Size;
            int minor = ColorInt.ARGB(.035f, 1f, 1f, 1f);
            int major = ColorInt.ARGB(.075f, 1f, 1f, 1f);

            int column = (int)MathF.Floor((view.x - origin.x) / step);
            for (float gx = origin.x + column * step; gx < view.x + viewSize.x; gx += step, column++)
                Gui.Renderer.DrawBox((int)gx, view.y, 1, viewSize.y, column % GridMajorEvery == 0 ? major : minor);

            int row = (int)MathF.Floor((view.y - origin.y) / step);
            for (float gy = origin.y + row * step; gy < view.y + viewSize.y; gy += step, row++)
                Gui.Renderer.DrawBox(view.x, (int)gy, viewSize.x, 1, row % GridMajorEvery == 0 ? major : minor);
        }

        protected override void DrawCustom()
        {
            if (DownSource is Plug plug)
            {
                Point a = plug.Location + plug.Size * UIScale / 2;
                int direction = (int)plug.Type * 2 - 1;
                DrawWire(a, Gui.MousePosition, direction, -direction, PendingWireColor);
            }
        }

        protected override void DrawBeforeChildren()
        {
            foreach (var connection in Graph.Connections)
            {
                if (connection.PortA == null || connection.PortB == null) continue;
                var plugA = connection.PortA.Tag as Plug;
                var plugB = connection.PortB.Tag as Plug;
                if (plugA == null || plugB == null) continue;

                Point start = plugA.Location + plugA.Size * UIScale / 2;
                Point end = plugB.Location + plugB.Size * UIScale / 2;

                DrawWire(start, end, (int)plugA.Port.Type * 2 - 1, (int)plugB.Port.Type * 2 - 1, WireColor);
            }
        }

        /// <summary>
        /// Cubic bezier between two plugs, leaving each one horizontally (direction: -1 = left, +1 = right).
        /// </summary>
        private void DrawWire(Point start, Point end, int startDirection, int endDirection, Color4 color)
        {
            var batch = ((SquidRenderer)Gui.Renderer).SpriteBatch;

            float reach = Math.Clamp(Math.Abs(end.x - start.x) * .5f, 40 * UIScale, 160 * UIScale);
            var p0 = new Vector2(start.x, start.y);
            var p3 = new Vector2(end.x, end.y);
            var p1 = p0 + new Vector2(reach * startDirection, 0);
            var p2 = p3 + new Vector2(reach * endDirection, 0);

            float savedWidth = batch.LineWidth;
            batch.LineWidth = MathF.Max(1.5f, 2 * UIScale);

            const int segments = 28;
            var last = p0;
            for (int i = 1; i <= segments; i++)
            {
                float t = i / (float)segments, u = 1 - t;
                var point = u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
                batch.DrawLine((int)last.X, (int)last.Y, (int)point.X, (int)point.Y, color);
                last = point;
            }

            batch.LineWidth = savedWidth;
        }
    }

    public enum PlugType
    {
        In, Out
    }

    public class Plug : Button
    {
        public Plug ConnectedTo;
        public PlugType Type;
        public object Owner;
        public GUIProperty Property;
        public Port Port;

        public Plug(Port port, GUIProperty property)
        {
            AllowDrop = true;

            Name = property.Name;

            Port = port;
            Property = property;
            Port.Tag = this;

            DragResponse += Plug_DragResponse;
            MouseDrag += Plug_MouseDrag;
            DragLeave += Plug_DragLeave;
            DragDrop += Plug_DragDrop;

            Selected = port.IsConnected;
        }

        /// <summary>
        /// Ring, plus a dot when connected or hovered. Drawn from art generated at the on-screen size
        /// instead of the skin texture, which blurs when the canvas is zoomed.
        /// </summary>
        protected override void DrawStyle(Style style, float opacity)
        {
            if (opacity == 0) return;

            float scale = GetScale();
            int diameter = Math.Max(6, (int)MathF.Round(14 * scale));
            int cx = Location.x + (int)(Size.x * scale / 2);
            int cy = Location.y + (int)(Size.y * scale / 2);

            int color = Type == PlugType.In ? ColorInt.ARGB(1f, .42f, .66f, .95f) : ColorInt.ARGB(1f, 1f, .52f, .40f);
            bool filled = State != ControlState.Default;   // connected (Selected) or under the mouse / a dragged wire

            LandingArt.Centered(EditorSkin.PlugArt(diameter, dot: false), cx, cy, color);
            if (filled)
                LandingArt.Centered(EditorSkin.PlugArt(diameter, dot: true), cx, cy, color);
        }

        private void Plug_DragDrop(Control sender, DragDropEventArgs e)
        {
            var plug = e.Source as Plug;
            var graph = Port.Graph;
           
            var portA = Port;
            var portB = plug.Port;

            if (!graph.CanConnect(portA, portB))
                return;

            if (portA.ConnectionType == ConnectionType.Single)
            {
                Selected = false;
                graph.ClearConnections(portA);
            }

            if (portB.ConnectionType == ConnectionType.Single)
            {
                plug.Selected = false;
                graph.ClearConnections(portB);
            }

            graph.AddConnection(portA, portB);
            plug.Selected = Selected = true;
        }

        private void Plug_MouseDrag(Control sender, MouseEventArgs args)
        {
            bool inport = Type == PlugType.In;

            Control drag = new Control()
            {
                Style = inport ? "inport" : "outport",
                Tag = this,
                Size = Size,
                NoEvents = true,
                Position = Gui.MousePosition - ClipRect.Size / 2
            };

            DoDragDrop(drag);
        }

        private void Plug_DragLeave(Control sender, DragDropEventArgs e)
        {
            State = ControlState.Default;
        }

        private void Plug_DragResponse(Control sender, DragDropEventArgs e)
        {
            State = ControlState.Selected;
        }
    }

    public class PortControl : Control
    {
        private readonly Label Label;
        private readonly GUIProperty Property;
        private readonly Node Node;
        
        public Plug Plug { get; private set; }

        public PortControl(GUIProperty property, Node node, PlugType type)
        {
            Node = node;
            Property = property;

            Dock = DockStyle.Top;
            Size = new Point(100, 20);
            Dock = DockStyle.Top;
            NoEvents = true;

            bool inport = type == PlugType.In;
            var port = node.GetPort(property.Name);

            Plug = new Plug(port, property)
            {
                Style = inport ? "inport" : "outport",
                Type = type,
                Size = new Point(20, 20),
                Dock = inport ? DockStyle.Left : DockStyle.Right,
                Owner = Node,
                Property = Property
            };

            Label = new Label();
            Label.Size = new Point(20, 20);
            Label.Dock = DockStyle.Fill;
            Label.Text = Property.Name;
            Label.TextAlign = inport ? Alignment.MiddleLeft : Alignment.MiddleRight;
            Label.NoEvents = true;

            Elements.Add(Plug);
            Elements.Add(Label);
        }
    }

    public class GuiWindow : Window
    {
        public TitleBar Titlebar { get; private set; }

        private Point FloatSize;
        private bool Docked;
        private bool dragging;

        public GuiWindow()
        {
            Style = "window";
            MaxSize = Point.Zero;
            SnapDistance = 0;
            Padding = new Margin(1);
            FloatSize = new Point(200, 400);

            Titlebar = new TitleBar();
            Titlebar.Dock = DockStyle.Top;
            Titlebar.Size = new Squid.Point(122, 28);
            Titlebar.Margin = new Squid.Margin(0, 0, 0, 1);
            Titlebar.Cursor = Cursors.Move;
            Titlebar.Style = "frame";

            Titlebar.Button.MouseClick += Button_MouseClick;

            //Titlebar.MouseDown += delegate(Control sender, MouseEventArgs args)
            //{
            //    //Size = FloatSize; 
            //    //Dock = DockStyle.None; 
            //    StartDrag();
            //};

            Titlebar.MouseDrag += delegate (Control sender, MouseEventArgs args)
            {
                //Size = FloatSize; 
                //Dock = DockStyle.None; 
                StartDrag();

                if (Dock != DockStyle.None)
                {
                    Point p = Position;
                    Dock = DockStyle.None;
                    Position = p;
                }

                //BringToFront();
            };

            Titlebar.MouseUp += delegate (Control sender, MouseEventArgs args)
            {
                StopDrag();
            };

            Controls.Add(Titlebar);
        }

        void Button_MouseClick(Control sender, MouseEventArgs args)
        {
            Close();
        }
    }

    public class TitleBar : Label
    {
        public Button Button { get; private set; }

        public TitleBar()
        {
            Button = new Button();
            Button.Size = new Point(24, 16);
            Button.Style = "close";
            Button.Tooltip = "Close Window";
            Button.Dock = DockStyle.Right;
            Button.Margin = new Margin(0, 2, 2, 2);
            Button.Tint = ColorInt.ARGB(1, 0, 1, 1);
            Elements.Add(Button);
        }
    }

    public class GraphFrame : GuiWindow
    {
        public Node Node { get; private set; }
        private PropertyFrame propertyGrid;

        public GraphFrame(Node node)
        {
            Node = node;
            AllowDragOut = true;
            Resizable = true;
            Cursor = Cursors.Move;
            SnapGrid = 8;
          //  Style = "GraphNode";

            MinSize = new Point(168, 56);
            MaxSize = new Point(216, 600);

            // The card (body, header band, outline) is painted in DrawStyle; children only add text and plugs
            Style = "";
            Padding = new Margin(1, 1, 1, 6);

            Titlebar.Cursor = Cursors.Move;
            Titlebar.Text = Node.GetType().Name;
            Titlebar.Style = "graphTitle";
            Titlebar.Button.Visible = false;

            Position = new Point((int)node.Position.X, (int)node.Position.Y);

            propertyGrid = new PropertyFrame
            {
                Dock = DockStyle.Fill
            };
            Controls.Add(propertyGrid);

            var obj = new GUIObject(Node);

            // Title bar (28 + 1) and padding, plus a little air under the last port; each port row is 20 high
            int h = 40;
            foreach (var property in obj.GetProperties())
            {
                var input = property.GetAttribute<InputAttribute>();
                var output = property.GetAttribute<OutputAttribute>();

                if (input != null)
                {
                    PortControl conn = new PortControl(property, Node, PlugType.In);
                    propertyGrid.Controls.Add(conn);
                    h += 20;
                }
                else if (output != null)
                {
                    PortControl conn = new PortControl(property, Node, PlugType.Out);
                    propertyGrid.Controls.Add(conn);
                    h += 20;
                }
                //else
                //{
                //    propertyGrid.AddProperty(info);
                //    h += 26;
                //}
            }

            Size = new Point(136, h);

            MouseDown += (s, e) =>
            {
                Selector.SelectedObject = Node;
                BringToFront();
                StartDrag();
            };

            MouseDrag += (s, e) => StartDrag();

            MouseUp += (s, e) => StopDrag();

            Titlebar.MouseDown += (s, e) =>
            {
                Selector.SelectedObject = Node;
                BringToFront();
            };

            PositionChanged += GraphFrame_PositionChanged;
        }

        protected override void DrawStyle(Style style, float opacity)
        {
            if (opacity == 0) return;

            // Sizes are in canvas units; the canvas zoom scales them on screen
            float scale = GetScale();
            int x = Location.x, y = Location.y;
            int w = (int)(Size.x * scale), h = (int)(Size.y * scale);
            int header = (int)((Titlebar.Size.y + 1) * scale);
            int edge = Math.Max(1, (int)MathF.Round(LandingArt.RadiusSmall * scale));

            LandingArt.Slice(LandingArt.RoundSmall, x, y, w, h, LandingArt.RadiusSmall, EditorSkin.NodeBodyColor, scale);

            // Header band: rounded on top, squared off where it meets the body
            LandingArt.Slice(LandingArt.RoundSmall, x, y, w, header + edge, LandingArt.RadiusSmall, EditorSkin.NodeHeaderColor, scale);
            Gui.Renderer.DrawBox(x, y + header, w, edge, EditorSkin.NodeBodyColor);

            bool selected = ReferenceEquals(Selector.SelectedObject, Node);
            LandingArt.Slice(LandingArt.OutlineSmall, x, y, w, h, LandingArt.RadiusSmall,
                selected ? LandingArt.Coral : ColorInt.ARGB(.14f, 1f, 1f, 1f), scale);
        }

        private void GraphFrame_PositionChanged(Control sender)
        {
            Node.Position = new Vector2(sender.Position.x, sender.Position.y);
        }
    }
}

