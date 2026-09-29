using System;
using System.Collections.Generic;
using Squid;
using Freefall.Base;
using Freefall.Graphics;
using Freefall.Components;
using Freefall.Assets;

namespace Freefall.Editor
{
    public class InnerViewport : ImageControl { }

    public class ViewportControl : Frame
    {
        public RenderView View { get; private set; }
        
        private readonly InnerViewport image;
        private readonly string textureId;

        public ViewportControl()
        {
            textureId = System.IO.Path.GetRandomFileName();

            image = new InnerViewport
            {
                NoEvents = false,
                Texture = textureId,
                Dock = DockStyle.Fill,
                AllowDrop = true,
                AllowFocus = false,
            };

            Controls.Add(image);

            // Viewport border overlay
            image.GetElements().Add(new Frame
            {
                Size = new Point(100, 100),
                Style = "viewport",
                Dock = DockStyle.Fill
            });
            image.DragResponse += Scene_DragResponse;
            image.DragDrop += Scene_DragDrop;

            // Create headless RenderView (Apex pattern: creates own BackBuffer + Pipeline)
            int w = Math.Max(64, image.Size.x);
            int h = Math.Max(64, image.Size.y);

            View = new RenderView(w, h, Engine.Device);
            View.Pipeline = new DeferredRenderer();
            View.Pipeline.Initialize(w, h);

            // Register the BackBuffer texture in Squid (Apex: rend.InsertTexture(textureId, View.BackBufferTexture))
            var renderer = Gui.Renderer as SquidRenderer;
            renderer.InsertTexture(textureId, View.BackBufferTexture.BindlessIndex, w, h);

            // When resize completes, swap the texture in Squid (Apex: View_OnResized)
            View.OnResized += View_OnResized;

            // Size change triggers deferred resize (Apex: View.Resize sets ResizePending)
            SizeChanged += ViewportControl_SizeChanged;

            // Mouse picking
            MessageDispatcher.AddListener(Msg.RequestPick, OnRequestPick);
        }

        // Placement state
        private static Material? _ghostMaterial;
        private Entity _placingEntity;       // Entity being dragged into position

        /// <summary>
        /// Fires every frame while an asset is being dragged over the viewport.
        /// First call: loads prefab and requests pick. Subsequent calls: requests pick at current mouse pos.
        /// </summary>
        private void Scene_DragResponse(Control sender, DragDropEventArgs e)
        {
            if (View.Pipeline is not DeferredRenderer deferred) return;
            if (e.DraggedControl?.Tag is not AssetDragData data) return;

            // Convert mouse position to viewport-local coordinates
            var loc = image.Location;
            int localX = Math.Clamp(Input.MousePosition.X - loc.x, 0, View.Width - 1);
            int localY = Math.Clamp(Input.MousePosition.Y - loc.y, 0, View.Height - 1);


            // First DragResponse: load the prefab and start placement
            if (_placingEntity == null)
            {
                var asset = Engine.Assets.LoadByGuid(data.Guid, data.AssetType);
                //if (asset is not Prefab prefab) return;

                _ghostMaterial ??= new Material(new Effect("ghost"));
                
                if(asset is Prefab prefab)
                    _placingEntity = prefab.Instantiate();

                if(asset is Mesh mesh)
                {
                    _placingEntity = new Entity();
                    var mr = _placingEntity.AddComponent<MeshRenderer>();
                    mr.Mesh = mesh;
                }

                // Set ghost material on all MeshRenderers (renders in Forward pass, not GBuffer)
                if (_placingEntity != null)
                    SetGhostMode(_placingEntity, _ghostMaterial);
            }

            // Request pick at current mouse position (result comes next frame)
            deferred.RequestPick(localX, localY);
        }

        /// <summary>
        /// Fires on mouse release — finalize placement.
        /// </summary>
        private void Scene_DragDrop(Control sender, DragDropEventArgs e)
        {
            if (_placingEntity != null)
            {
                // Restore normal materials (back to GBuffer rendering)
                SetGhostMode(_placingEntity, null);

                MessageDispatcher.Send(Msg.RefreshExplorer);
                Selector.SelectedEntity = _placingEntity;
                _placingEntity = null;
                _smoothedNormal = System.Numerics.Vector3.UnitY;
            }
        }

        /// <summary>
        /// Set or clear ghost material on the entity's MeshRenderer.
        /// </summary>
        private static void SetGhostMode(Entity entity, Material? ghostMat)
        {
            var mr = entity.GetComponent<MeshRenderer>();
            if (mr != null) mr.ReplacementMaterial = ghostMat;

            foreach (Transform child in entity.Transform)
                SetGhostMode(child.Entity, ghostMat);
        }

        // Smoothed normal for stable rotation during placement
        private System.Numerics.Vector3 _smoothedNormal = System.Numerics.Vector3.UnitY;

        /// <summary>
        /// Place or update an entity at the picked world position with surface alignment.
        /// </summary>
        private void ApplyPlacement(Entity entity, Camera cam, DeferredRenderer deferred)
        {
            System.Numerics.Vector3 worldPos;
            System.Numerics.Vector3 normal;

            if (deferred.PickedDepth > 0)
            {
                worldPos = cam.UnprojectPixel(
                    deferred.PickedPixelX, deferred.PickedPixelY,
                    deferred.PickedDepth,
                    View.Width, View.Height);
                normal = deferred.PickedNormal;
            }
            else
            {
                // Sky/empty — fallback to Y=0 plane
                var ray = cam.MouseRay();
                if (MathF.Abs(ray.Direction.Y) > 0.0001f)
                {
                    float t = -ray.Position.Y / ray.Direction.Y;
                    worldPos = t > 0 ? ray.Position + ray.Direction * t : cam.Position + cam.Forward * 10f;
                }
                else
                {
                    worldPos = cam.Position + cam.Forward * 10f;
                }
                normal = System.Numerics.Vector3.UnitY;
            }

            // Smooth the normal to avoid jittery rotation
            if (normal.LengthSquared() > 0.5f)
            {
                float lerpT = MathF.Min(1f, 8f * Time.Delta);
                _smoothedNormal = System.Numerics.Vector3.Normalize(
                    System.Numerics.Vector3.Lerp(_smoothedNormal, System.Numerics.Vector3.Normalize(normal), lerpT));
            }

            // Offset by mesh bounds so object sits ON surface (pivot → bottom)
            var mr = entity.GetComponent<MeshRenderer>();
            if (mr?.Mesh != null)
            {
                float bottomOffset = -mr.Mesh.BoundingBox.Min.Y;
                worldPos += _smoothedNormal * bottomOffset;
            }

            // Snap to grid if enabled
            var snap = EditorPreferences.Instance.Snapping;
            if (snap.SnapToGrid)
            {
                var g = snap.GridSnap;
                if (g.X > 0) worldPos.X = MathF.Round(worldPos.X / g.X) * g.X;
                if (g.Y > 0) worldPos.Y = MathF.Round(worldPos.Y / g.Y) * g.Y;
                if (g.Z > 0) worldPos.Z = MathF.Round(worldPos.Z / g.Z) * g.Z;
            }

            entity.Transform.Position = worldPos;

            // Surface alignment from smoothed normal
            var up = _smoothedNormal;
            float dot = System.Numerics.Vector3.Dot(up, System.Numerics.Vector3.UnitY);

            if (dot < 0.999f)
            {
                // Choose reference forward that isn't parallel to normal
                var refFwd = MathF.Abs(System.Numerics.Vector3.Dot(up, System.Numerics.Vector3.UnitZ)) < 0.99f
                    ? System.Numerics.Vector3.UnitZ
                    : System.Numerics.Vector3.UnitX;

                var right = System.Numerics.Vector3.Normalize(
                    System.Numerics.Vector3.Cross(up, refFwd));
                var forward = System.Numerics.Vector3.Cross(right, up);

                entity.Transform.Rotation = System.Numerics.Quaternion.CreateFromRotationMatrix(
                    new System.Numerics.Matrix4x4(
                        right.X, right.Y, right.Z, 0,
                        up.X, up.Y, up.Z, 0,
                        forward.X, forward.Y, forward.Z, 0,
                        0, 0, 0, 1));
            }
            else
            {
                entity.Transform.Rotation = System.Numerics.Quaternion.Identity;
            }
        }

        private void ViewportControl_SizeChanged(Control sender)
        {
            // Deferred resize (Apex pattern: just sets ResizePending flag)
            View.Resize(image.Size.x, image.Size.y);
        }

        private void View_OnResized()
        {
            // Swap texture in Squid after resize completes (Apex: rend.Replace(textureId, View.BackBufferTexture))
            var renderer = Gui.Renderer as SquidRenderer;
            renderer.UpdateTexture(textureId, View.BackBufferTexture.BindlessIndex, View.Width, View.Height);
            image.TextureRect = new Squid.Rectangle();
        }

        private void OnRequestPick(Message message)
        {
            if (message.Data is not Vortice.Mathematics.Int2 screenPos) return;
            if (View.Pipeline is not DeferredRenderer deferred) return;

            // Convert screen coords to viewport-local coords
            var loc = image.Location;
            int localX = screenPos.X - loc.x;
            int localY = screenPos.Y - loc.y;

            // Only handle if click is within this viewport
            if (localX < 0 || localY < 0 || localX >= image.Size.x || localY >= image.Size.y)
                return;

            deferred.RequestPick(localX, localY);
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();

            if (View.Pipeline is not DeferredRenderer deferred) return;

            // Promote pick result early to cut 1-frame delay
            deferred.TryPromotePickResult();

            // Check for completed pick readback
            if (deferred.HasPickResult)
            {
                var cam = Camera.Main;

                // Continuous drag: update placement position
                if (_placingEntity != null && cam != null)
                    ApplyPlacement(_placingEntity, cam, deferred);

                uint slot = deferred.GetPickedEntityId();

                // Check gizmo slots first
                if (slot != 0 && EditorTools.Instance != null
                    && EditorTools.Instance.TryGetGizmoAxis((int)slot, out var tool, out var axis))
                {
                    MessageDispatcher.Send(Msg.GizmoPicked, new GizmoPickData
                    {
                        Slot = (int)slot,
                        Axis = axis,
                        Tool = tool,
                        MeshPart = deferred.PickedMeshPartId
                    });
                    return;
                }

                // Regular entity pick
                Entity picked = null;
                if (slot != 0)
                    picked = TransformBuffer.Instance?.GetEntityBySlot((int)slot);

                MessageDispatcher.Send(Msg.PickResult, picked);
            }
        }
    }
}
