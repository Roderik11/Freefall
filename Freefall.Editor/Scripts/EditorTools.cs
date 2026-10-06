using System.Collections.Generic;
using System.Numerics;
using System.Text;
using Squid;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Freefall.Serialization;

namespace Freefall.Editor
{
    [UpdateInEditor]
    public class EditorTools : Component, IDraw, IUpdate
    {
        private TransformTranslate translate;
        private TransformRotate rotate;
        private TransformScale scale;
        private GizmoManager gizmoManager;

        private ToolBase activeTool;

        // Gizmo slot → axis mapping for pick routing
        private Dictionary<int, (ToolBase tool, MoveAxis axis)> gizmoSlotMap = new Dictionary<int, (ToolBase, MoveAxis)>();

        // Pick state tracking
        private bool _pendingClickPick;   // True when pick was from a mouse click
        private bool _pendingHoverPick;   // True when pick was from hover (don't affect selection)

        public static EditorTools Instance { get; private set; }
        public static ToolBase ActiveTool => Instance?.activeTool;

        protected override void Awake()
        {
            Instance = this;

            translate = new TransformTranslate();
            translate.Initialize();

            rotate = new TransformRotate();
            rotate.Initialize();

            scale = new TransformScale();
            scale.Initialize();

            gizmoManager = new GizmoManager();

            activeTool = translate;

            // Listen for pick results
            MessageDispatcher.AddListener(Msg.PickResult, OnPickResult);
            MessageDispatcher.AddListener(Msg.GizmoPicked, OnGizmoPicked);
            MessageDispatcher.AddListener(Msg.HandleClick, OnHandleClick);
        }

        public override void Destroy()
        {
            MessageDispatcher.RemoveListener(Msg.PickResult, OnPickResult);
            MessageDispatcher.RemoveListener(Msg.GizmoPicked, OnGizmoPicked);
            MessageDispatcher.RemoveListener(Msg.HandleClick, OnHandleClick);
        }

        public void Update()
        {
            if (Input.IsKeyPressed(Keys.P))
                Toast.Show("Toast is getting tested now.", 4);

            if (Input.IsKeyPressed(Keys.F1))
                activeTool = translate;
            if (Input.IsKeyPressed(Keys.F2))
                activeTool = rotate;
            if (Input.IsKeyPressed(Keys.F3))
                activeTool = scale;

            if (!EditorUI.KeyboardCaptured)
            {
                if (Input.IsKeyPressed(Keys.Delete))
                {
                    if (Selector.Selection.Count > 0)
                    {
                        var entities = new List<Entity>(Selector.Selection);

                        Selector.SelectedEntity = null;

                        foreach (var entity in entities)
                        {
                            if (entity.GetComponent<TerrainRenderer>() != null)
                                continue;

                            entity.Destroy();
                        }

                        MessageDispatcher.Send(Msg.RefreshExplorer);
                    }
                }

                if (Input.Control && Input.IsKeyPressed(Keys.D))
                {
                    if (Selector.Selection.Count > 0)
                    {
                        var serializer = new EntitySerializer { DuplicateMode = true };
                        var yaml = serializer.SaveToString(Selector.Selection);
                        var bytes = Encoding.UTF8.GetBytes(yaml);

                        var duplicates = serializer.LoadFromBytes(bytes);

                        if (duplicates.Count > 0)
                        {
                            Selector.SelectedEntity = duplicates[0];
                            for (int i = 1; i < duplicates.Count; i++)
                                Selector.SelectOrDeselect(duplicates[i]);

                            MessageDispatcher.Send(Msg.RefreshExplorer);
                        }
                    }
                }
            }

            // Rebuild gizmo slot map each frame
            gizmoSlotMap.Clear();
            foreach (var (slot, axis) in translate.GetSlotMappings())
                gizmoSlotMap[slot] = (translate, axis);
            foreach (var (slot, axis) in rotate.GetSlotMappings())
                gizmoSlotMap[slot] = (rotate, axis);
            foreach (var (slot, axis) in scale.GetSlotMappings())
                gizmoSlotMap[slot] = (scale, axis);
            // Register component gizmo slot (handle picks routed via MeshPart)
            gizmoSlotMap[gizmoManager.GizmoSlot] = (null, MoveAxis.Free);

            activeTool.Update(Camera.Main);

            // Release mouse capture on mouse up (drag end)
            if (activeTool.IsClicked && !Input.IsMouseDown(0))
            {
                activeTool.Disable();
            }

            if (activeTool.MouseCaptured) return;

            // Left-click to pick (only when not right-click orbiting and mouse is over viewport)
            if (!EditorUI.MouseCaptured && Input.IsMousePressed(0) && !Input.IsMouseDown(1))
            {
                MessageDispatcher.Send(Msg.RequestPick, Input.MousePosition);
                _pendingClickPick = true;
                _pendingHoverPick = false;
            }
            // Hover pick for gizmo highlighting (no mouse buttons down, something selected, no pick in flight)
            else if (!EditorUI.MouseCaptured && !Input.IsMouseDown(0) && !Input.IsMouseDown(1)
                     && Selector.Selection.Count > 0 && !_pendingClickPick && !_pendingHoverPick)
            {
                MessageDispatcher.Send(Msg.RequestPick, Input.MousePosition);
                _pendingHoverPick = true;
            }
        }

        public void Draw()
        {
            if(gizmoManager == null)
                return;

            // Always run gizmo lifecycle so stale geometry gets cleared
            // when selection changes or becomes empty.
            gizmoManager.DrawGizmos(Selector.Selection);

            if (Selector.Selection.Count < 1) return;
            activeTool.Render();
        }

        private void OnHandleClick(Message message)
        {
            if(!Input.Control)
                return;

            var serializer = new EntitySerializer { DuplicateMode = true };
            var yaml = serializer.SaveToString(Selector.Selection);
            var bytes = Encoding.UTF8.GetBytes(yaml);

            var parents = new List<Transform>();
            foreach (Entity entity in Selector.Selection)
                parents.Add(entity.Transform.Parent);

            var duplicates = serializer.LoadFromBytes(bytes);

            if (duplicates.Count > 0)
            {
                for (int i = 0; i < duplicates.Count; i++)
                    duplicates[i].Transform?.SetParent(parents[i], keepWorld: true);

                Selector.SelectedEntity = duplicates[0];
                for (int i = 1; i < duplicates.Count; i++)
                    Selector.SelectOrDeselect(duplicates[i]);
         
               MessageDispatcher.Send(Msg.RefreshExplorer);
            }
        }

        private void OnPickResult(Message message)
        {
            // Hover picks: ignore entirely (don't change selection)
            if (_pendingHoverPick)
            {
                _pendingHoverPick = false;
                // Clear all highlights since we're not hovering a gizmo
                translate.SetHighlight(-1);
                scale.SetHighlight(-1);
                rotate.SetHighlight(-1);
                gizmoManager.ClearHover();

                return;
            }

            // Only process click-initiated picks
            if (!_pendingClickPick) return;
            _pendingClickPick = false;

            var entity = message.Data as Entity;

            // first check if entity is child of a prefab
            // and if so, select the prefab root instead for better UX
            // unless the root is already selected
            if (entity != null && !Input.Shift)
            {
                var root = FindPrefabRoot(entity);

                if (Selector.SelectedEntity != root && Selector.SelectedEntity != entity)
                    entity = root;
            }

            if (Input.Shift && entity != null)
                Selector.SelectOrDeselect(entity);
            else
                Selector.SelectedEntity = entity;
        }

        private Entity FindPrefabRoot(Entity entity)
        {
            if (entity.Prefab != null)
                return entity;

            // Traverse up the parent hierarchy to find the prefab root
            var current = entity.Transform;
            while (current.Parent != null)
            {
                current = current.Parent;
                if (current.Entity.Prefab != null)
                    return current.Entity;
            }

            return entity;
        }

        private void OnGizmoPicked(Message message)
        {
            if (message.Data is GizmoPickData data)
            {
                // Route to component gizmo handles
                if (data.Slot == gizmoManager.GizmoSlot)
                {
                    if (_pendingHoverPick)
                    {
                        _pendingHoverPick = false;
                        gizmoManager.OnGizmoPicked(data.MeshPart, isClick: false);
                    }
                    else if (_pendingClickPick)
                    {
                        _pendingClickPick = false;
                        gizmoManager.OnGizmoPicked(data.MeshPart, isClick: true);
                    }
                    return;
                }

                // Hover pick: just highlight, don't start drag
                if (_pendingHoverPick)
                {
                    _pendingHoverPick = false;
                    translate.SetHighlight(-1);
                    scale.SetHighlight(-1);
                    rotate.SetHighlight(-1);
                    gizmoManager.ClearHover();

                    if (activeTool == translate)
                        translate.SetHighlight((int)data.MeshPart);
                    else if (activeTool == scale)
                        scale.SetHighlight((int)data.MeshPart);
                    else if (activeTool == rotate)
                        rotate.SetHighlight((int)data.MeshPart);

                    return;
                }

                // Click pick: start drag
                _pendingClickPick = false;

                if (activeTool.IsClicked) return; // Already dragging

                if (Input.IsMouseDown(0))
                {
                    var camera = Camera.Main;
                    if (camera != null)
                    {
                        activeTool.StartDrag(camera, (int)data.MeshPart);
                    }
                }
            }
        }

        /// <summary>
        /// Check if a TransformSlot belongs to a gizmo axis.
        /// </summary>
        public bool TryGetGizmoAxis(int slot, out ToolBase tool, out MoveAxis axis)
        {
            if (gizmoSlotMap.TryGetValue(slot, out var mapping))
            {
                tool = mapping.tool;
                axis = mapping.axis;
                return true;
            }
            tool = null;
            axis = MoveAxis.Free;
            return false;
        }
    }
}
