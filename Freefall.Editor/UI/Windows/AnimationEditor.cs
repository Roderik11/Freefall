using Freefall.Animation;
using Freefall.Assets;
using Freefall.Base;
using Squid;
using System;
using System.Collections.Generic;
using System.Numerics;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    using Point = Squid.Point;

    [CustomAssetEditor(typeof(Animation.Animation))]
    public class AnimationEditor : Frame, ICustomAssetEditor
    {
        private readonly AnimationCanvas Canvas;
        private readonly Frame Toolbar;
        private readonly Frame ParamPanel;
        private readonly ScrollPanel ParamScroll;

        private Animation.Animation _animation;

        public AnimationEditor()
        {
            Dock = DockStyle.Fill;
            Size = new Point(800, 600);

            // ── Toolbar ──
            Toolbar = new Frame { Style = "frame", Size = new Point(16, 24), Dock = DockStyle.Top };
            Toolbar.Margin = new Margin(0, 0, 0, 1);
            Controls.Add(Toolbar);

            AddToolbarButton("Save", BtnSave_Click);
            AddToolbarButton("Add State", BtnAddState_Click);
            AddToolbarButton("Add Tree", BtnAddTree_Click);
            AddToolbarButton("Add Param", BtnAddParam_Click);
            AddToolbarButton("Add Test", CreateBlendTreeAnimations);

            // ── Split: Params (left) | Canvas (right) ──
            var split = new SplitContainer();
            split.Dock = DockStyle.Fill;
            split.SplitFrame1.Size = new Point(200, 200);
            split.SplitButton.Margin = new Margin(1, 0, 1, 0);
            split.SplitButton.Size = new Point(2, 2);
            split.RetainAspect = false;
            Controls.Add(split);

            // Parameter panel
            ParamScroll = new ScrollPanel();
            ParamPanel = new Frame { Dock = DockStyle.Fill };
            ParamScroll.Content.Controls.Add(ParamPanel);
            split.SplitFrame1.Controls.Add(ParamScroll);

            // Canvas
            Canvas = new AnimationCanvas(this);
            Canvas.Style = "canvas";
            split.SplitFrame2.Controls.Add(Canvas);
        }

        private void AddToolbarButton(string text, MouseEvent handler)
        {
            var btn = new Button
            {
                Text = text,
                Size = new Point(100, 20),
                Dock = DockStyle.Left,
                Style = "button",
                Margin = new Margin(1)
            };
            btn.MouseClick += handler;
            Toolbar.Controls.Add(btn);
        }

        // ═══════════════════════════════════════════════════════
        //  ICustomAssetEditor
        // ═══════════════════════════════════════════════════════

        public void OpenAsset(Asset asset)
        {
            if (asset is Animation.Animation anim)
                Open(anim);
        }

        public void Open(Animation.Animation animation)
        {
            _animation = animation;
            Canvas.SetAnimation(animation);
            RebuildParamPanel();
        }

        // ═══════════════════════════════════════════════════════
        //  Toolbar handlers
        // ═══════════════════════════════════════════════════════

        private void BtnSave_Click(Control sender, MouseEventArgs args)
        {
            if (_animation == null) return;

            if (!string.IsNullOrEmpty(_animation.AssetPath))
            {
                AssetFile.Save(_animation.AssetPath, _animation);
                Debug.Log($"[AnimEditor] Saved: {_animation.AssetPath}");
                return;
            }

            using var dlg = new System.Windows.Forms.SaveFileDialog();
            dlg.Filter = "Animation Asset|*.asset";
            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                _animation.AssetPath = dlg.FileName;
                _animation.Name = System.IO.Path.GetFileNameWithoutExtension(dlg.FileName);
                AssetFile.Save(dlg.FileName, _animation);
                Debug.Log($"[AnimEditor] Saved: {dlg.FileName}");
            }
        }

        void CreateBlendTreeAnimations(Control sender, MouseEventArgs args)
        {
            _animation ??= new Animation.Animation { Name = "New Animation" };

            var animation = _animation;

            // Load animation clips (by relative path — resolved through asset database)
            string animDir = "Characters/Knight/";
            var walkAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Walking.dae");
            var idleAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Idle.dae");
            var runAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Running.dae");
            var walkBack = Engine.Assets.Load<AnimationClip>($"{animDir}Walking Backward.dae");
            var strafeRightAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Right Strafe Walking.dae");
            var strafeLeftAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Left Strafe Walking.dae");
            var strafeRightRunAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Right Strafe.dae");
            var strafeLeftRunAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Left Strafe.dae");
            var jogBack = Engine.Assets.Load<AnimationClip>($"{animDir}Jog Backward.dae");
            var jumpAnim = Engine.Assets.Load<AnimationClip>($"{animDir}Jumping Up Quick.dae");
            var falling = Engine.Assets.Load<AnimationClip>($"{animDir}Falling Idle.dae");
            var landing = Engine.Assets.Load<AnimationClip>($"{animDir}Jumping Down Quick.dae");

            // Create animation states
            var idleState = new AnimationState { Name = "Idle", Clip = idleAnim, Loop = true };
            var walkState = new AnimationState { Name = "Walk", Clip = walkAnim, Loop = true };
            var walkBackState = new AnimationState { Name = "WalkBack", Clip = walkBack, Loop = true };
            var runState = new AnimationState { Name = "Run", Clip = runAnim, Loop = true };
            var strafeRightState = new AnimationState { Name = "StrafeRight", Clip = strafeRightAnim, Loop = true };
            var strafeLeftState = new AnimationState { Name = "StrafeLeft", Clip = strafeLeftAnim, Loop = true };
            var strafeRightRunState = new AnimationState { Name = "StrafeRightRun", Clip = strafeRightRunAnim, Loop = true };
            var strafeLeftRunState = new AnimationState { Name = "StrafeLeftRun", Clip = strafeLeftRunAnim, Loop = true };
            var jogBackState = new AnimationState { Name = "JogBack", Clip = jogBack, Loop = true };
            var jumpState = new AnimationState { Name = "Jump", Clip = jumpAnim, Loop = false };
            var fallingState = new AnimationState { Name = "Falling", Clip = falling, Loop = true };
            var landingState = new AnimationState { Name = "Landing", Clip = landing, Loop = false };

            // Add animation events
            walkAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.3f });
            walkAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.75f });
            runAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.3f });
            runAnim.Events.Add(new AnimationEvent { Name = "footstep", Time = 0.75f });
            jumpAnim.Events.Add(new AnimationEvent { Name = "jump", Time = 0.001f });
            landing.Events.Add(new AnimationEvent { Name = "land", Time = 0.001f });

            // Create blend tree for locomotion
            var blendTree = new AnimationBlendTree
            {
                Name = "Locomotion",
                ParameterA = "axisX",
                ParameterB = "axisY",
            };

            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = idleState, Values = new Vector2(0, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = walkState, Values = new Vector2(0, 0.5f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = runState, Values = new Vector2(0, 1f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = walkBackState, Values = new Vector2(0, -.5f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = jogBackState, Values = new Vector2(0, -1f) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeRightState, Values = new Vector2(.5f, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeLeftState, Values = new Vector2(-.5f, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeRightRunState, Values = new Vector2(1f, 0) });
            blendTree.Layers.Add(new AnimationBlendTree.BlendLayer { Animation = strafeLeftRunState, Values = new Vector2(-1f, 0) });

            animation.Name = "PlayerAnimation";

            // Create animation layer
            var layer = new AnimationLayer { Name = "Default" };
            animation.Layers.Add(layer);

            // Add states (assigns IDs automatically)
            animation.AddState(layer, blendTree);
            animation.AddState(layer, jumpState);
            animation.AddState(layer, fallingState);
            animation.AddState(layer, landingState);

            // Create transitions
            var toJump = new AnimationTransition
            {
                Source = blendTree,
                Target = jumpState,
                Conditions = {
                    new AnimationCondition {
                        Parameter = "jump",
                        Comparison = ComparisonType.Greater,
                        Value = 0
                    }
                }
            };

            var toFalling = new AnimationTransition
            {
                Source = jumpState,
                Target = fallingState,
            };

            var toLanding = new AnimationTransition
            {
                Source = fallingState,
                Target = landingState,
                Conditions = {
                    new AnimationCondition {
                        Parameter = "landing",
                        Comparison = ComparisonType.Greater,
                        Value = 0
                    }
                }
            };

            var toBlendTree = new AnimationTransition
            {
                Source = landingState,
                Target = blendTree,
            };

            var fromBlendToFalling = new AnimationTransition
            {
                Source = blendTree,
                Target = fallingState,
                Conditions = {
                    new AnimationCondition {
                        Parameter = "falling",
                        Comparison = ComparisonType.Greater,
                        Value = 0
                    }
                }
            };

            // Add transitions (syncs SourceID/TargetID automatically)
            animation.AddTransition(layer, toJump);
            animation.AddTransition(layer, toFalling);
            animation.AddTransition(layer, toLanding);
            animation.AddTransition(layer, toBlendTree);
            animation.AddTransition(layer, fromBlendToFalling);

            animation.Parameters.Add(new AnimationParameter("axisX"));
            animation.Parameters.Add(new AnimationParameter("axisY"));
            animation.Parameters.Add(new AnimationParameter("jump", isTrigger: true));
            animation.Parameters.Add(new AnimationParameter("landing", isTrigger: true));
            animation.Parameters.Add(new AnimationParameter("falling", isTrigger: true));

            Canvas.SetAnimation(animation);
            RebuildParamPanel();
        }

        private void BtnAddTree_Click(Control sender, MouseEventArgs args)
        {
            var layer = GetOrCreateLayer();
            var state = new AnimationBlendTree { Name = "New Blend" };
            state.Position = Canvas.GetVisibleCenter();
            _animation.AddState(layer, state);

            Canvas.AddStateNode(state);
        }

        private void BtnAddState_Click(Control sender, MouseEventArgs args)
        {
            var layer = GetOrCreateLayer();
            var state = new AnimationState { Name = "New State" };
            state.Position = Canvas.GetVisibleCenter();
            _animation.AddState(layer, state);

            Canvas.AddStateNode(state);
        }

        private void BtnAddParam_Click(Control sender, MouseEventArgs args)
        {
            GetOrCreateLayer(); // ensure animation exists

            _animation.Parameters.Add(new AnimationParameter($"param{_animation.Parameters.Count}"));
            _animation.InvalidateParamCache();
            RebuildParamPanel();
        }

        private AnimationLayer GetOrCreateLayer()
        {
            if (_animation == null)
            {
                _animation = new Animation.Animation { Name = "New Animation" };
                Canvas.SetAnimation(_animation);
                RebuildParamPanel();
            }

            if (_animation.Layers.Count == 0)
                _animation.Layers.Add(new AnimationLayer { Name = "Default" });
            return _animation.Layers[0];
        }

        // ═══════════════════════════════════════════════════════
        //  Parameter Panel
        // ═══════════════════════════════════════════════════════

        private void RebuildParamPanel()
        {
            ParamPanel.Controls.Clear();

            if (_animation == null) return;

            var header = new Label
            {
                Text = "Parameters",
                Style = "label",
                Size = new Point(100, 22),
                Dock = DockStyle.Top,
                Margin = new Margin(4, 4, 4, 2)
            };
            ParamPanel.Controls.Add(header);

            for (int i = 0; i < _animation.Parameters.Count; i++)
            {
                var param = _animation.Parameters[i];
                var row = CreateParamRow(param, i);
                ParamPanel.Controls.Add(row);
            }
        }

        private Frame CreateParamRow(AnimationParameter param, int index)
        {
            var row = new Frame
            {
                Size = new Point(100, 24),
                Dock = DockStyle.Top,
                Margin = new Margin(2, 0, 2, 1),
                Style = "frame"
            };

            // Delete button (rightmost)
            var btnDelete = new Button
            {
                Text = "X",
                Size = new Point(20, 20),
                Dock = DockStyle.Right,
                Style = "button",
                Margin = new Margin(1),
                Tag = index
            };
            btnDelete.MouseClick += (s, a) =>
            {
                int idx = (int)((Control)s).Tag;
                if (idx < _animation.Parameters.Count)
                {
                    _animation.Parameters.RemoveAt(idx);
                    _animation.InvalidateParamCache();
                    RebuildParamPanel();
                }
            };
            row.Controls.Add(btnDelete);

            // Trigger checkbox
            var chkTrigger = new Button
            {
                Text = param.IsTrigger ? "T" : "F",
                Size = new Point(20, 20),
                Dock = DockStyle.Right,
                Style = "button",
                Margin = new Margin(1),
                Tooltip = "IsTrigger",
                Tag = param
            };
            chkTrigger.MouseClick += (s, a) =>
            {
                var p = (AnimationParameter)((Control)s).Tag;
                p.IsTrigger = !p.IsTrigger;
                ((Button)s).Text = p.IsTrigger ? "T" : "F";
            };
            row.Controls.Add(chkTrigger);

            // Name field
            var txtName = new TextBox
            {
                Text = param.Name ?? "",
                Size = new Point(80, 20),
                Dock = DockStyle.Fill,
                Style = "textbox",
                Margin = new Margin(1),
                Tag = param
            };
            txtName.TextCommit += (s, a) =>
            {
                var p = (AnimationParameter)((Control)s).Tag;
                p.Name = ((TextBox)s).Text;
                _animation.InvalidateParamCache();
            };
            row.Controls.Add(txtName);

            return row;
        }

        // ═══════════════════════════════════════════════════════
        //  Public API for canvas → editor communication
        // ═══════════════════════════════════════════════════════

        internal Animation.Animation Animation => _animation;

        internal void CreateTransition(AnimStateNode from, AnimStateNode to)
        {
            if (_animation == null) return;

            var layer = GetOrCreateLayer();
            var transition = new AnimationTransition
            {
                Source = from.State,
                Target = to.State
            };
            _animation.AddTransition(layer, transition);
        }

        internal void RemoveState(AnimationState state)
        {
            if (_animation == null) return;

            foreach (var layer in _animation.Layers)
            {
                layer.States.Remove(state);
                layer.Transitions.RemoveAll(t =>
                    t.Source == state || t.Target == state ||
                    t.SourceID == state.ID || t.TargetID == state.ID);
            }
        }

        internal void RemoveTransition(AnimationTransition transition)
        {
            if (_animation == null) return;

            foreach (var layer in _animation.Layers)
                layer.Transitions.Remove(transition);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  Canvas — pannable/zoomable area for state nodes
    // ═══════════════════════════════════════════════════════════

    public class AnimationCanvas : Window
    {
        internal readonly AnimationEditor _editor;
        private readonly Point maxSize = new Point(1000000, 1000000);
        private bool isDragging;

        // Wiring state: drag from an output port to create a transition
        internal AnimStateNode WiringSource;

        private Animation.Animation _animation;

        public AnimationCanvas(AnimationEditor editor)
        {
            _editor = editor;
            UIScale = 1f;
            Resizable = false;
            AllowDragOut = true;

            Dock = DockStyle.Center;
            Size = maxSize;
            Position = Size / -2;
            NoEvents = false;

            MouseDown += Canvas_MouseDown;
            MouseUp += Canvas_MouseUp;

            Gui.MouseDown += Gui_MouseDown;
            Gui.MouseUp += Gui_MouseUp;
        }

        public void SetAnimation(Animation.Animation animation)
        {
            _animation = animation;
            Controls.Clear();

            UIScale = 1;
            Size = maxSize;
            Position = Size / -2;

            if (_animation == null) return;

            var center = GetVisibleCenter();

            // Create nodes for all states across all layers
            foreach (var layer in _animation.Layers)
            {
                for (int i = 0; i < layer.States.Count; i++)
                {
                    var state = layer.States[i];

                    // Auto-layout if no editor position set
                    if (state.Position == Vector2.Zero)
                        state.Position = center + new Vector2(i * 200, 300);

                    AddStateNode(state);
                }
            }
        }

        public void AddStateNode(AnimationState state)
        {
            var node = new AnimStateNode(state, this);
            Controls.Add(node);
        }

        /// <summary>
        /// Returns the center of the currently visible canvas area in canvas-local coordinates.
        /// </summary>
        public Vector2 GetVisibleCenter()
        {
            // Canvas Location is its screen position; Parent gives the viewport
            var parentSize = Parent != null ? Parent.Size : new Point(800, 600);
            var center = (parentSize / 2 - Position) / Math.Max(UIScale, 0.01f);
            return new Vector2(center.x, center.y);
        }

        // ── Pan & Zoom ──

        private void Canvas_MouseDown(Control sender, MouseEventArgs args)
        {
            if (args.Button == 2) { StartDrag(); isDragging = true; }
        }

        private void Canvas_MouseUp(Control sender, MouseEventArgs args)
        {
            if (args.Button == 2) { StopDrag(); isDragging = false; }

            // Cancel wiring if clicking on empty canvas
            if (args.Button == 0 && WiringSource != null)
            {
                WiringSource = null;
                return;
            }

            // Hit-test transition arrows
            if (_animation != null && (args.Button == 0 || args.Button == 1))
            {
                var hit = HitTestTransition(Gui.MousePosition);
                if (hit != null)
                {
                    if (args.Button == 0)
                    {
                        // Left-click: select transition in inspector
                        Selector.SelectedObject = hit;
                    }
                    else if (args.Button == 1)
                    {
                        // Right-click: delete transition
                        _editor.RemoveTransition(hit);
                    }
                }
            }
        }

        private void Gui_MouseDown(Control sender, MouseEventArgs args) { }

        private void Gui_MouseUp(Control sender, MouseEventArgs args)
        {
            // Complete wiring: if mouse released on a state node
            if (WiringSource != null && sender is AnimStateNode target && target != WiringSource)
            {
                _editor.CreateTransition(WiringSource, target);
                WiringSource = null;
            }
        }

        /// <summary>
        /// Returns the transition whose drawn arrow is closest to the given screen point,
        /// or null if nothing is within click distance.
        /// </summary>
        private AnimationTransition HitTestTransition(Point mouse)
        {
            const float hitThreshold = 8f;

            var nodeMap = new Dictionary<int, AnimStateNode>();
            foreach (Control ctrl in Controls)
            {
                if (ctrl is AnimStateNode node)
                    nodeMap[node.State.ID] = node;
            }

            AnimationTransition closest = null;
            float closestDist = hitThreshold;

            foreach (var layer in _animation.Layers)
            {
                foreach (var transition in layer.Transitions)
                {
                    if (!nodeMap.TryGetValue(transition.SourceID, out var fromNode)) continue;
                    if (!nodeMap.TryGetValue(transition.TargetID, out var toNode)) continue;

                    Point start = fromNode.Location + new Point(fromNode.Size.x, fromNode.Size.y / 2) * UIScale;
                    Point end = toNode.Location + new Point(0, toNode.Size.y / 2) * UIScale;

                    float dist = PointToSegmentDist(mouse, start, end);
                    if (dist < closestDist)
                    {
                        closestDist = dist;
                        closest = transition;
                    }
                }
            }

            return closest;
        }

        private static float PointToSegmentDist(Point p, Point a, Point b)
        {
            float dx = b.x - a.x, dy = b.y - a.y;
            float lenSq = dx * dx + dy * dy;
            if (lenSq < 0.001f) return MathF.Sqrt((p.x - a.x) * (p.x - a.x) + (p.y - a.y) * (p.y - a.y));

            float t = Math.Clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq, 0f, 1f);
            float projX = a.x + t * dx, projY = a.y + t * dy;
            float ex = p.x - projX, ey = p.y - projY;
            return MathF.Sqrt(ex * ex + ey * ey);
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

            UIScale = Math.Clamp(UIScale, 0.3f, 2);

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

        // ── Draw transition arrows ──

        protected override void DrawCustom()
        {
            var batch = ((SquidRenderer)Gui.Renderer).SpriteBatch;

            // Draw wiring preview line
            if (WiringSource != null)
            {
                var from = WiringSource.Location + WiringSource.Size * UIScale / 2;
                batch.DrawLine(from.x, from.y, Gui.MousePosition.x, Gui.MousePosition.y, new Color4(1, 0.8f, 0.3f, 1));
            }
        }

        protected override void DrawBeforeChildren()
        {
            if (_animation == null) return;

            var batch = ((SquidRenderer)Gui.Renderer).SpriteBatch;

            // Build state ID → node lookup
            var nodeMap = new Dictionary<int, AnimStateNode>();
            foreach (Control ctrl in Controls)
            {
                if (ctrl is AnimStateNode node)
                    nodeMap[node.State.ID] = node;
            }

            // Draw transition arrows
            foreach (var layer in _animation.Layers)
            {
                foreach (var transition in layer.Transitions)
                {
                    if (!nodeMap.TryGetValue(transition.SourceID, out var fromNode)) continue;
                    if (!nodeMap.TryGetValue(transition.TargetID, out var toNode)) continue;

                    // Arrow from right edge of source to left edge of target
                    Point start = fromNode.Location + new Point(fromNode.Size.x, fromNode.Size.y / 2) * UIScale;
                    Point end = toNode.Location + new Point(0, toNode.Size.y / 2) * UIScale;

                    // Color: orange for conditional, white for unconditional
                    var color = transition.Conditions.Count > 0
                        ? new Color4(1f, 0.7f, 0.2f, 1f)
                        : new Color4(0.7f, 0.8f, 1f, 1f);

                    // Simple 3-segment line (out, across, in)
                    int nudge = (int)(15 * UIScale);
                    var midStart = new Point(start.x + nudge, start.y);
                    var midEnd = new Point(end.x - nudge, end.y);

                    batch.DrawLine(start.x, start.y, midStart.x, midStart.y, color);
                    batch.DrawLine(midStart.x, midStart.y, midEnd.x, midEnd.y, color);
                    batch.DrawLine(midEnd.x, midEnd.y, end.x, end.y, color);

                    // Arrowhead
                    int ah = (int)(6 * UIScale);
                    batch.DrawLine(end.x, end.y, end.x - ah, end.y - ah, color);
                    batch.DrawLine(end.x, end.y, end.x - ah, end.y + ah, color);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  State Node — draggable rectangle representing an AnimationState
    // ═══════════════════════════════════════════════════════════

    public class AnimStateNode : Window
    {
        public AnimationState State;
        private readonly AnimationCanvas _canvas;
        private readonly Label _titleBar;

        public AnimStateNode(AnimationState state, AnimationCanvas canvas)
        {
            State = state;
            _canvas = canvas;

            Style = "window";
            Size = new Point(150, 44);
            Position = new Point((int)state.Position.X, (int)state.Position.Y);
            Resizable = false;
            AllowDragOut = true;
            MaxSize = Point.Zero;
            SnapDistance = 0;
            Padding = new Margin(1);
            Tooltip = state is AnimationBlendTree ? "Blend Tree" : (state.Clip?.Name ?? "No Clip");

            _titleBar = new Label
            {
                Text = state.Name ?? "State",
                Style = "header",
                Dock = DockStyle.Fill,
                Size = new Point(100, 28),
                Cursor = Cursors.Move,
                TextAlign = Alignment.MiddleCenter
            };
            Controls.Add(_titleBar);

            _titleBar.MouseDrag += (s, a) =>
            {
                StartDrag();
            };

            _titleBar.MouseUp += (s, a) =>
            {
                StopDrag();
            };

            // Left-click: select state in inspector
            _titleBar.MouseClick += (s, a) =>
            {
                if (a.Button == 0)
                {
                    if(_canvas.WiringSource != null && _canvas.WiringSource != this)
                        _canvas._editor.CreateTransition(_canvas.WiringSource, this);
                    else
                        Selector.SelectedObject = State;

                    _canvas.WiringSource = null;
                }
                else if (a.Button == 1)
                    _canvas.WiringSource = this;
            };

            PositionChanged += (s) =>
            {
                State.Position = new Vector2(Position.x, Position.y);
            };
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();
            _titleBar.Text = State.Name ?? "State";
        }
    }
}
