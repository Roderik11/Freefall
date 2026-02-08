using System;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Scripts
{
    /// <summary>
    /// Third-person camera that orbits around a target entity.
    /// Based on Apex's WowCamera but simplified without PhysX collision sweeps.
    /// </summary>
    public class ThirdPersonCamera : Component, IUpdate
    {
        public Entity Target;
        public Vector3 Offset = Vector3.UnitY * 1.85f;

        private float zoom = 5f;
        private float newZoom = 5f;
        private Vector2 angle;      // Radians (X = yaw, Y = pitch)
        private Vector2 newAngle;
        private float smoothness = 20f;

        // Terrain reference for simple collision
        public Terrain Terrain;

        public void Update()
        {
            if (Target == null || Transform == null) return;

            float delta = Time.SmoothDelta;
            float lim = MathF.PI / 2f - 0.1f; // ~85 degrees limit

            // Mouse input for rotation (only when mouse locked) - like Apex line 39
            if (Input.IsMouseLocked)
            {
                newAngle += new Vector2(Input.MouseDelta.X, Input.MouseDelta.Y) * delta * 1f;
                newAngle.Y = Math.Clamp(newAngle.Y, -lim, lim);
            }

            // Zoom with Z/X keys (Apex uses mouse wheel)
            if (Input.IsKeyDown(Keys.Z)) newZoom += 10f * delta;
            if (Input.IsKeyDown(Keys.X)) newZoom -= 10f * delta;
            newZoom = Math.Clamp(newZoom, 3f, 50f);

            // Smooth interpolation - like Apex lines 48-49
            angle += (newAngle - angle) * Math.Min(1f, delta * smoothness);
            zoom += (newZoom - zoom) * Math.Min(1f, delta * smoothness);

            // Update CharacterController yaw - like Apex line 51
            CharacterController.CameraYaw = angle.X;

            // Set rotation first - like Apex line 53
            Transform.Rotation = Quaternion.CreateFromYawPitchRoll(angle.X, angle.Y, 0);

            // Get direction from rotation - like Apex line 79
            Vector3 direction = Transform.Forward;
            
            // Target position with offset - like Apex line 80
            Vector3 targetPos = Target.Transform.Position + Offset;
            
            // Position = target MINUS direction * zoom (camera behind target) - like Apex line 84
            Vector3 desiredPos = targetPos - direction * zoom;

            // Simple terrain collision
            if (Terrain != null)
            {
                float terrainHeight = Terrain.GetHeight(desiredPos) + 1f;
                if (desiredPos.Y < terrainHeight)
                    desiredPos.Y = terrainHeight;
            }

            // Update position - like Apex line 92
            Transform.Position = desiredPos;
        }
    }
}



