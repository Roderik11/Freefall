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
    public class TransformScale : ToolBase
    {
        private Mesh mesh;
        private List<Material> materials = new();
        private int gizmoSlot;
        private bool initialized;

        // Pre-created textures
        private Texture[] baseTextures;
        private Texture highlightTexture;

        // State
        private MoveAxis? activeAxis;
        private Plane pickPlane;
        private Vector3 pickOffset; // Initial ray-plane hit point
        private Dictionary<Entity, Vector3> Cache = new();
        private int highlightPart = -1;

        // Axis data
        private Dictionary<MoveAxis, AxisData> Data = new();

        // Part index → MoveAxis mapping (matches OBJ group order for scale gizmo)
        // Scale gizmo has 4 parts: Center(uniform), X, Z, Y
        private static readonly MoveAxis[] PartToAxis = {
            MoveAxis.Free, // 0: Center = uniform scale
            MoveAxis.X,    // 1: X axis
            MoveAxis.Z,    // 2: Z axis
            MoveAxis.Y,    // 3: Y axis
        };

        public override void Initialize()
        {
            Data.Add(MoveAxis.X, new AxisData { Axis1 = TransformAxis.LocalForward, Axis2 = TransformAxis.LocalUp });
            Data.Add(MoveAxis.Y, new AxisData { Axis1 = TransformAxis.LocalRight, Axis2 = TransformAxis.LocalForward });
            Data.Add(MoveAxis.Z, new AxisData { Axis1 = TransformAxis.LocalRight, Axis2 = TransformAxis.LocalUp });
            Data.Add(MoveAxis.XY, new AxisData { Axis1 = TransformAxis.LocalForward, Axis2 = TransformAxis.None });
            Data.Add(MoveAxis.ZY, new AxisData { Axis1 = TransformAxis.LocalRight, Axis2 = TransformAxis.None });
            Data.Add(MoveAxis.XZ, new AxisData { Axis1 = TransformAxis.LocalUp, Axis2 = TransformAxis.None });
            Data.Add(MoveAxis.Free, new AxisData { Axis1 = TransformAxis.CameraForward, Axis2 = TransformAxis.None });
            EnsureInitialized();
        }

        private void EnsureInitialized()
        {
            if (initialized) return;
            initialized = true;

            var device = Engine.Device;

            string meshPath = Path.Combine(AppContext.BaseDirectory, "Resources", "Meshes", "transform_scale.obj");
            mesh = Mesh.LoadOBJ(device, meshPath);

            // Apex color order: Yellow(center), Red(X), Blue(Z), Green(Y)
            var colors = new (byte r, byte g, byte b)[] {
                (220, 220, 50),   // Part 0: Center = Yellow
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
            if (Selector.Selection.Count < 1) return;
            if (meshPart < 0 || meshPart >= PartToAxis.Length) return;

            activeAxis = PartToAxis[meshPart];
            if (!Data.ContainsKey(activeAxis.Value)) return;

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

            Vector3[] normals = GetPlaneNormals(camera, gizmoMatrix);
            pickPlane = new Plane(normals[0], -Vector3.Dot(normals[0], center));

            Ray ray = camera.MouseRay();
            Collision.RayIntersectsPlane(ref ray, ref pickPlane, out pickOffset);

            // Cache each entity's current scale
            Cache.Clear();
            foreach (Entity e in Selector.Selection)
                Cache[e] = e.Transform.Scale;
        }

        public override void Update(Camera camera)
        {
            if (!IsClicked || !activeAxis.HasValue) return;
            if (!Data.ContainsKey(activeAxis.Value)) return;
            if (camera == null) return;

            Ray ray = camera.MouseRay();
            Collision.RayIntersectsPlane(ref ray, ref pickPlane, out Vector3 intersect);

            // Build scale mask based on axis (Apex pattern)
            Vector3 scaleMask = Vector3.Zero;
            switch (activeAxis.Value)
            {
                case MoveAxis.Free: scaleMask = Vector3.One; break;
                case MoveAxis.X: scaleMask = Vector3.UnitX; break;
                case MoveAxis.Y: scaleMask = Vector3.UnitY; break;
                case MoveAxis.Z: scaleMask = Vector3.UnitZ; break;
            }

            foreach (Entity e in Selector.Selection)
            {
                Vector3 dir = (intersect - pickOffset) * 0.1f;
                float delta = dir.Y; // Use Y component of drag delta for scale magnitude
                if (Cache.ContainsKey(e))
                    e.Transform.Scale = scaleMask * delta + Cache[e];
            }
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
            Cache.Clear();
            highlightPart = -1;
        }

        private Dictionary<TransformAxis, Vector3> vectors = new();

        private Vector3[] GetPlaneNormals(Camera camera, Matrix4x4 matrix)
        {
            vectors[TransformAxis.LocalForward] = new Vector3(matrix.M31, matrix.M32, matrix.M33);
            vectors[TransformAxis.LocalRight] = new Vector3(matrix.M11, matrix.M12, matrix.M13);
            vectors[TransformAxis.LocalUp] = new Vector3(matrix.M21, matrix.M22, matrix.M23);
            vectors[TransformAxis.CameraForward] = camera.Forward;
            vectors[TransformAxis.None] = Vector3.Zero;

            if (!Data.ContainsKey(activeAxis ?? MoveAxis.Free))
                return new Vector3[] { Vector3.UnitY, Vector3.Zero };

            AxisData data = Data[activeAxis ?? MoveAxis.Free];
            return new Vector3[] { vectors[data.Axis1], vectors[data.Axis2] };
        }
    }
}
