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
    public class TransformTranslate : ToolBase
    {
        // Loaded OBJ mesh with MeshParts (7 parts: 3 axis cone+shaft, 1 center, 3 plane handles)
        private Mesh mesh;

        // Per-MeshPart materials (matching Apex color order)
        private List<Material> materials = new();

        // Single TransformSlot for the entire gizmo
        private int gizmoSlot;
        private bool initialized;

        // Pre-created textures (allocated once)
        private Texture[] baseTextures;
        private Texture highlightTexture;

        // Pick / drag state
        private MoveAxis? activeAxis;
        private Plane pickPlane;
        private Plane constrainPlane;
        private Vector3 pickOffset;
        private Vector3 selectionCenter; // world-space center at drag start
        private Dictionary<Entity, Vector3> positionCache = new();
        private int highlightPart = -1;

        // Part index → MoveAxis mapping (matches OBJ group order)
        private static readonly MoveAxis[] PartToAxis = {
            MoveAxis.Z,    // 0: Z cone+shaft
            MoveAxis.X,    // 1: X cone+shaft
            MoveAxis.Y,    // 2: Y cone+shaft
            MoveAxis.Free, // 3: Center cube
            MoveAxis.XZ,   // 4: XZ plane handle
            MoveAxis.XY,   // 5: XY plane handle
            MoveAxis.ZY,   // 6: ZY plane handle
        };

        // Axis data for plane computation
        private Dictionary<MoveAxis, AxisData> Data = new();

        public override void Initialize()
        {
            Data.Add(MoveAxis.X, new AxisData { Axis1 = TransformAxis.LocalForward, Axis2 = TransformAxis.LocalUp });
            Data.Add(MoveAxis.XY, new AxisData { Axis1 = TransformAxis.LocalForward, Axis2 = TransformAxis.None });
            Data.Add(MoveAxis.Y, new AxisData { Axis1 = TransformAxis.LocalRight, Axis2 = TransformAxis.LocalForward });
            Data.Add(MoveAxis.ZY, new AxisData { Axis1 = TransformAxis.LocalRight, Axis2 = TransformAxis.None });
            Data.Add(MoveAxis.Z, new AxisData { Axis1 = TransformAxis.LocalRight, Axis2 = TransformAxis.LocalUp });
            Data.Add(MoveAxis.XZ, new AxisData { Axis1 = TransformAxis.LocalUp, Axis2 = TransformAxis.None });
            Data.Add(MoveAxis.Free, new AxisData { Axis1 = TransformAxis.CameraForward, Axis2 = TransformAxis.None });
            EnsureInitialized();
        }

        private void EnsureInitialized()
        {
            if (initialized) return;
            initialized = true;

            var device = Engine.Device;

            // Load OBJ mesh from resources
            string meshPath = Path.Combine(AppContext.BaseDirectory, "Resources", "Meshes", "transform_translate.obj");
            mesh = Mesh.LoadOBJ(device, meshPath);

            // Per-part colors: Z=Blue, X=Red, Y=Green, Center=White, XZ=Red, XY=Blue, ZY=Green
            var colors = new (byte r, byte g, byte b)[] {
                (50, 120, 230),   // Part 0: Z axis = Blue
                (230, 50, 50),    // Part 1: X axis = Red
                (80, 210, 50),    // Part 2: Y axis = Green
                (220, 220, 220),  // Part 3: Center = White/Yellow
                (230, 50, 50),    // Part 4: XZ plane = Red
                (50, 120, 230),   // Part 5: XY plane = Blue
                (80, 210, 50),    // Part 6: ZY plane = Green
            };

            // Pre-create all textures ONCE
            baseTextures = new Texture[colors.Length];
            highlightTexture = CreateColorTexture(device, 255, 255, 255);

            var effect = new Effect("gbuffer_gizmo");
            for (int i = 0; i < Math.Min(mesh.MeshParts.Count, colors.Length); i++)
            {
                var (r, g, b) = colors[i];
                baseTextures[i] = CreateColorTexture(device, r, g, b);
                materials.Add(CreateColorMaterial(effect, baseTextures[i]));
            }

            // Allocate single TransformSlot for the whole gizmo
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

        private static Material CreateColorMaterial(Effect effect, Texture tex)
        {
            var mat = new Material(effect);
            mat.SetTexture("AlbedoTex", tex);
            return mat;
        }

        /// <summary>
        /// Returns the TransformSlot used by this gizmo for pick detection.
        /// </summary>
        public IEnumerable<(int slot, MoveAxis axis)> GetSlotMappings()
        {
            if (!initialized) yield break;
            yield return (gizmoSlot, MoveAxis.Free); // Single slot for whole gizmo
        }

        /// <summary>
        /// Called when GPU picking detects a click on this gizmo's mesh part.
        /// Sets up planes and caches for the drag operation.
        /// </summary>
        public override void StartDrag(Camera camera, int meshPart)
        {
            if (Selector.Selection.Count < 1) return;
            if (meshPart < 0 || meshPart >= PartToAxis.Length) return;

            activeAxis = PartToAxis[meshPart];
            if (!Data.ContainsKey(activeAxis.Value)) return;

            IsClicked = true;
            MouseCaptured = true;

            // Compute center of selection (world-space)
            selectionCenter = Vector3.Zero;
            foreach (Entity e in Selector.Selection)
                selectionCenter += e.Transform.WorldPosition;
            selectionCenter /= Selector.Selection.Count;

            // Build the gizmo matrix (same as Render) to get correct plane normals
            Quaternion rotation = Quaternion.Identity;
            if (Selector.Selection.Count == 1)
                Matrix4x4.Decompose(Selector.Selection[0].Transform.Matrix, out _, out rotation, out _);

            var gizmoMatrix = Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(selectionCenter);

            // Compute pick/constrain planes
            Vector3[] normals = GetPlaneNormals(camera, gizmoMatrix);

            pickPlane = new Plane(normals[0], -Vector3.Dot(normals[0], selectionCenter));
            constrainPlane = new Plane(normals[1], -Vector3.Dot(normals[1], selectionCenter));

            Ray ray = camera.MouseRay();
            Collision.RayIntersectsPlane(ref ray, ref pickPlane, out Vector3 hitPoint);
            pickOffset = selectionCenter - hitPoint;

            // Cache each entity's position relative to center
            positionCache.Clear();
            foreach (Entity e in Selector.Selection)
                positionCache[e] = e.Transform.WorldPosition - selectionCenter;

            MessageDispatcher.Send(Msg.HandleClick);
        }

        public override void Update(Camera camera)
        {
            if (!IsClicked || !activeAxis.HasValue) return;
            if (!Data.ContainsKey(activeAxis.Value)) return;
            if (camera == null) return;

            Ray ray = camera.MouseRay();

            // Raycast against the pick plane
            Collision.RayIntersectsPlane(ref ray, ref pickPlane, out Vector3 intersect);
            intersect += pickOffset;

            // If single-axis constrained, project onto the axis line
            if (Data[activeAxis.Value].Axis2 != TransformAxis.None)
            {
                float dot = Plane.DotCoordinate(constrainPlane, intersect);
                float dir = dot < 0 ? 1 : -1;

                // Re-derive constrain normal from current gizmo matrix
                Quaternion rotation = Quaternion.Identity;
                if (Selector.Selection.Count == 1)
                    Matrix4x4.Decompose(Selector.Selection[0].Transform.Matrix, out _, out rotation, out _);
                var gizmoMatrix = Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(selectionCenter);
                Vector3[] normals = GetPlaneNormals(camera, gizmoMatrix);
                Vector3 constrain = normals[1];

                ray = new Ray(intersect, Vector3.Normalize(constrain * dir));
                Collision.RayIntersectsPlane(ref ray, ref constrainPlane, out intersect);
            }

            // Snap to grid if enabled
            var snap = EditorPreferences.Instance.Snapping;
            if (snap.SnapToGrid)
            {
                var g = snap.GridSnap;
                if (g.X > 0) intersect.X = MathF.Round(intersect.X / g.X) * g.X;
                if (g.Y > 0) intersect.Y = MathF.Round(intersect.Y / g.Y) * g.Y;
                if (g.Z > 0) intersect.Z = MathF.Round(intersect.Z / g.Z) * g.Z;
            }

            // Apply to each selected entity
            foreach (Entity e in Selector.Selection)
            {
                var inv = Matrix4x4.Identity;
                if (e.Transform.Parent != null)
                {
                    Matrix4x4.Invert(e.Transform.Parent.Matrix, out inv);
                }

                var cachepos = positionCache.ContainsKey(e) ? positionCache[e] : Vector3.Zero;
                e.Transform.Position = Vector3.Transform(intersect + cachepos, inv);
            }
        }

        public override void Render()
        {
            EnsureInitialized();
            if (Selector.Selection.Count < 1) return;

            var camera = Camera.Main;
            if (camera == null) return;

            // Compute center of selection
            Vector3 center = Vector3.Zero;
            foreach (Entity e in Selector.Selection)
                center += e.Transform.WorldPosition;
            center /= Selector.Selection.Count;

            // Position gizmo at fixed distance from camera (constant screen size, Apex pattern)
            Vector3 cam = camera.Position;
            Vector3 cen = cam + Vector3.Normalize(center - cam) * 3f;

            // Extract entity rotation (for local-axis alignment) but discard entity scale
            Quaternion rotation = Quaternion.Identity;
            if (Selector.Selection.Count == 1)
            {
                Matrix4x4.Decompose(Selector.Selection[0].Transform.Matrix, out _, out rotation, out _);
            }

            // Build world matrix: fixed scale + entity rotation + screen-constant position
            var matrix = Matrix4x4.CreateScale(0.5f) * Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(cen);

            // Set the single gizmo transform
            TransformBuffer.Instance.SetTransform(gizmoSlot, matrix);

            // Apply highlight by swapping pre-created textures (no per-frame allocation)
            for (int i = 0; i < materials.Count && i < baseTextures.Length; i++)
            {
                var tex = (i == highlightPart) ? highlightTexture : baseTextures[i];
                materials[i].SetTexture("AlbedoTex", tex);
            }

            // Draw all MeshParts with per-part materials
            var block = new MaterialBlock();
            for (int i = 0; i < mesh.MeshParts.Count && i < materials.Count; i++)
            {
                CommandBuffer.Enqueue(mesh, i, materials[i], block, gizmoSlot);
            }
        }

        /// <summary>
        /// Set highlight from external source (EditorTools passes GPU pick mesh part).
        /// </summary>
        public void SetHighlight(int meshPart)
        {
            highlightPart = meshPart;
        }

        public override void Disable()
        {
            activeAxis = null;
            IsClicked = false;
            MouseCaptured = false;
            positionCache.Clear();
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
