using System;
using System.Numerics;
using Freefall.Components;
using Freefall.Base;

namespace Freefall.Editor
{
    /// <summary>
    /// Fly-through editor camera — right-click to orbit, WASD to move, scroll to dolly.
    /// Ported from Apex's EditorCamera component.
    /// </summary>
    [UpdateInEditor]
    public class EditorCamera : Component, IUpdate
    {
        private Vector2 angle;
        private Vector2 newAngle;
        private float smoothness = 10;

        private static Vector3 Position = new Vector3(0, 100, -20);
        private static Quaternion Rotation = Quaternion.Identity;
        
        public static Camera Camera { get; private set; }

        public static void CreateCamera()
        {
            // --- Editor Camera ---
            var cameraEntity = new Entity("EditorCamera");
            cameraEntity.Flags = EntityFlags.DontDestroy | EntityFlags.HideAndDontSave;
            var camera = cameraEntity.AddComponent<Camera>();
            camera.FarPlane = 16384;
            cameraEntity.AddComponent<EditorCamera>();
            cameraEntity.AddComponent<EditorTools>();
            cameraEntity.AddComponent<AudioListener>();
            cameraEntity.AddComponent<Hologrid>();
            camera.Activate();
            Camera = camera;
        }

        public static void DestroyCamera()
        {
            Camera?.Entity?.Destroy();
            Camera = null;
        }

        protected override void Awake()
        {
            Transform.Position = Position;
            Transform.Rotation = Rotation;

            MessageDispatcher.AddListener(Msg.FocusEntity, OnFocusEntity);
        }

        public override void Destroy()
        {
            Position = Transform.Position;
            Rotation = Transform.Rotation;
            MessageDispatcher.RemoveListener(Msg.FocusEntity, OnFocusEntity);
        }

        /// <summary>Freezes the camera (input and smoothing drift) while an agent screenshot is being taken.</summary>
        public static bool InputLocked;

        public void Update()
        {
            if (InputLocked)
            {
                newAngle = angle;
                return;
            }

            // Right-mouse drag to orbit (allow when mouse isn't captured by UI)
            if (!EditorUI.MouseCaptured && Input.IsMouseDown(1))
                newAngle += new Vector2(Input.MouseDelta.X, Input.MouseDelta.Y) * (float)Time.Delta;

            angle += (newAngle - angle) * (float)Time.Delta * smoothness;
            Entity.Transform.Rotation = Quaternion.CreateFromYawPitchRoll(angle.X, angle.Y, 0);

            // Scroll wheel to dolly forward/back
            if (!EditorUI.MouseCaptured)
                Entity.Transform.Position += Entity.Transform.Forward * Input.MouseWheelDelta * (float)Time.Delta * 4;

            // WASD movement (only when keyboard isn't captured by UI text fields)
            if (!EditorUI.KeyboardCaptured)
            {
                float speed = 3;
                if (Input.Shift)
                    speed = Math.Max(20, Entity.Transform.Position.Y);

                speed *= (float)Time.Delta;

                if (Input.IsKeyDown(Keys.W))
                    Entity.Transform.Position += Entity.Transform.Forward * speed;
                if (Input.IsKeyDown(Keys.A))
                    Entity.Transform.Position -= Entity.Transform.Right * speed;
                if (Input.IsKeyDown(Keys.S))
                    Entity.Transform.Position -= Entity.Transform.Forward * speed;
                if (Input.IsKeyDown(Keys.D))
                    Entity.Transform.Position += Entity.Transform.Right * speed;
            }
        }

        /// <summary>
        /// Programmatic camera control — sets position and derives internal yaw/pitch
        /// from a look direction so Update() preserves the new orientation.
        /// </summary>
        public void SetView(Vector3 position, Vector3? lookAtTarget = null)
        {
            Entity.Transform.Position = position;

            if (lookAtTarget.HasValue)
            {
                var dir = Vector3.Normalize(lookAtTarget.Value - position);
                float yaw = MathF.Atan2(dir.X, dir.Z);
                float pitch = MathF.Asin(-dir.Y);
                angle = new Vector2(yaw, pitch);
                newAngle = angle;
                Entity.Transform.Rotation = Quaternion.CreateFromYawPitchRoll(yaw, pitch, 0);
            }
        }

        /// <summary>
        /// Set yaw/pitch directly (radians). Updates internal state so Update() preserves it.
        /// </summary>
        public void SetAngles(float yaw, float pitch)
        {
            angle = new Vector2(yaw, pitch);
            newAngle = angle;
            Entity.Transform.Rotation = Quaternion.CreateFromYawPitchRoll(yaw, pitch, 0);
        }

        void OnFocusEntity(Message message)
        {
            if (message.Data is not Entity entity)
                return;

            float distance = 3;
            var center = entity.Transform.WorldPosition;

            // Move camera to face the entity, update internal angles
            var dir = Vector3.Normalize(center - Entity.Transform.Position);
            float yaw = MathF.Atan2(dir.X, dir.Z);
            float pitch = MathF.Asin(-dir.Y);
            angle = new Vector2(yaw, pitch);
            newAngle = angle;

            var mesh = entity.GetComponent<MeshRenderer>()?.Mesh;
            if(mesh != null)
            {
                distance = Math.Max(distance, mesh.BoundingBox.Extent.Length() * 2);
            }

            Entity.Transform.Rotation = Quaternion.CreateFromYawPitchRoll(yaw, pitch, 0);
            Entity.Transform.Position = center - Entity.Transform.Forward * distance;
        }
    }
}
