using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Freefall.Assets;
using Freefall.Base;
using Freefall.Components;
using Freefall.Graphics;
using Vortice.DXGI;

namespace Freefall.Editor
{
    public class TransformRotate : ToolBase
    {
        private Mesh mesh;
        private List<Material> materials = new();
        private int gizmoSlot;
        private bool initialized;

        // Pre-created textures
        private Texture[] baseTextures;
        private Texture highlightTexture;

        // State
        private RotateAxis? activeAxis;
        private Plane pickPlane;
        private Vector3 pickPoint;    // Initial point on plane
        private Quaternion pickQuat;  // Entity rotation at drag start
        private Vector3 gizmoCenter;  // World center of gizmo at drag start
        private int highlightPart = -1;

        // Part index → RotateAxis mapping (matches OBJ group order)
        private static readonly RotateAxis[] PartToAxis = {
            RotateAxis.Free, // 0: Free ring = Yellow
            RotateAxis.X,    // 1: X ring = Red
            RotateAxis.Y,    // 2: Y ring = Green
            RotateAxis.Z,    // 3: Z ring = Blue
        };

        public override void Initialize()
        {
            EnsureInitialized();
        }

        private void EnsureInitialized()
        {
            if (initialized) return;
            initialized = true;

            var device = Engine.Device;

            string meshPath = Path.Combine(AppContext.BaseDirectory, "Resources", "Meshes", "transform_rotate.obj");
            mesh = Mesh.LoadOBJ(device, meshPath);

            // Apex color order: Yellow(free), Red(X), Blue(Z), Green(Y)
            var colors = new (byte r, byte g, byte b)[] {
                (220, 220, 50),   // Part 0: Free = Yellow
                (230, 50, 50),    // Part 1: X axis = Red
                (50, 120, 230),   // Part 2: Z axis = Blue
                (80, 210, 50),    // Part 3: Y axis = Green
            };

            baseTextures = new Texture[colors.Length];
            highlightTexture = CreateColorTexture(device, 255, 255, 255);

            var effect = new Effect("gbuffer_gizmo");
            for (int i = 0; i < Math.Min(mesh.MeshParts.Count, colors.Length); i++)
            {
                var (r, g, b) = colors[i];
                baseTextures[i] = CreateColorTexture(device, r, g, b);
                var mat = new Material(effect);
                mat.SetTexture("AlbedoTex", baseTextures[i]);
                materials.Add(mat);
            }

            gizmoSlot = TransformBuffer.Instance.AllocateSlot();
        }

        private static Texture CreateColorTexture(GraphicsDevice device, byte r, byte g, byte b)
        {
            byte[] data = new byte[4 * 4 * 4];
            for (int i = 0; i < data.Length; i += 4)
            {
                data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
            }
            return Texture.CreateFromData(device, 4, 4, data, Format.R8G8B8A8_UNorm);
        }

        public IEnumerable<(int slot, MoveAxis axis)> GetSlotMappings()
        {
            if (!initialized) yield break;
            yield return (gizmoSlot, MoveAxis.Free);
        }

        public override void StartDrag(Camera camera, int meshPart)
        {
            if (Selector.Selection.Count < 1 || Selector.SelectedEntity == null) return;
            if (meshPart < 0 || meshPart >= PartToAxis.Length) return;

            activeAxis = PartToAxis[meshPart];
            IsClicked = true;
            MouseCaptured = true;

            // Compute center of selection
            Vector3 center = Vector3.Zero;
            foreach (Entity e in Selector.Selection)
                center += e.Transform.WorldPosition;
            center /= Selector.Selection.Count;

            // Build gizmo matrix for plane normals
            Quaternion rotation = Quaternion.Identity;
            if (Selector.Selection.Count == 1)
                Matrix4x4.Decompose(Selector.Selection[0].Transform.Matrix, out _, out rotation, out _);

            var gizmoMatrix = Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(center);
            gizmoCenter = center;

            // Choose the pick plane based on which axis ring was clicked
            Vector3 planeNormal;
            switch (activeAxis.Value)
            {
                case RotateAxis.Y:
                    planeNormal = new Vector3(gizmoMatrix.M21, gizmoMatrix.M22, gizmoMatrix.M23); // Up
                    break;
                case RotateAxis.X:
                    planeNormal = new Vector3(gizmoMatrix.M11, gizmoMatrix.M12, gizmoMatrix.M13); // Right
                    break;
                case RotateAxis.Z:
                    planeNormal = new Vector3(gizmoMatrix.M31, gizmoMatrix.M32, gizmoMatrix.M33); // Forward
                    break;
                default: // Free
                    planeNormal = camera.Forward;
                    break;
            }

            pickPlane = new Plane(planeNormal, -Vector3.Dot(planeNormal, center));

            Ray ray = camera.MouseRay();
            Collision.RayIntersectsPlane(ref ray, ref pickPlane, out pickPoint);
            pickQuat = Selector.SelectedEntity.Transform.Rotation;
        }

        public override void Update(Camera camera)
        {
            if (!IsClicked || !activeAxis.HasValue) return;
            if (camera == null || Selector.SelectedEntity == null) return;

            Ray ray = camera.MouseRay();
            Collision.RayIntersectsPlane(ref ray, ref pickPlane, out Vector3 dragPoint);

            // Compute signed angle between initial pick direction and current drag direction
            var a = Vector3.Normalize(dragPoint - gizmoCenter);
            var b = Vector3.Normalize(pickPoint - gizmoCenter);
            float angle = Collision.ClockwiseAngle(a, b, pickPlane.Normal);

            // Snap to angle increments if enabled
            var snap = EditorPreferences.Instance.Snapping;
            if (snap.SnapToGrid && snap.AngleSnap > 0)
            {
                float step = snap.AngleSnap * (MathF.PI / 180f);
                angle = MathF.Round(angle / step) * step;
            }

            // Apply rotation around the plane normal
            var dragQuat = Quaternion.CreateFromAxisAngle(pickPlane.Normal, -angle);
            Selector.SelectedEntity.Transform.Rotation = Quaternion.Normalize(dragQuat * pickQuat);
        }

        public override void Render()
        {
            EnsureInitialized();
            if (Selector.Selection.Count < 1) return;

            var camera = Camera.Main;
            if (camera == null) return;

            Vector3 center = Vector3.Zero;
            foreach (Entity e in Selector.Selection)
                center += e.Transform.WorldPosition;
            center /= Selector.Selection.Count;

            Vector3 cam = camera.Position;
            Vector3 cen = cam + Vector3.Normalize(center - cam) * 3f;

            Quaternion rotation = Quaternion.Identity;
            if (Selector.Selection.Count == 1)
                Matrix4x4.Decompose(Selector.Selection[0].Transform.Matrix, out _, out rotation, out _);

            var matrix = Matrix4x4.CreateScale(0.5f) * Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(cen);
            TransformBuffer.Instance.SetTransform(gizmoSlot, matrix);

            // Apply highlight by swapping pre-created textures
            for (int i = 0; i < materials.Count && i < baseTextures.Length; i++)
            {
                var tex = (i == highlightPart) ? highlightTexture : baseTextures[i];
                materials[i].SetTexture("AlbedoTex", tex);
            }

            var block = new MaterialBlock();
            for (int i = 0; i < mesh.MeshParts.Count && i < materials.Count; i++)
            {
                CommandBuffer.Enqueue(mesh, i, materials[i], block, gizmoSlot);
            }
        }


        public void SetHighlight(int meshPart)
        {
            highlightPart = meshPart;
        }

        public override void Disable()
        {
            activeAxis = null;
            IsClicked = false;
            MouseCaptured = false;
            highlightPart = -1;
        }
    }
}
