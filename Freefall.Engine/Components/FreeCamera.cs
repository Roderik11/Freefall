using System;
using System.Numerics;
using Freefall.Components;
using Freefall.Base;
using Vortice.Mathematics;

namespace Freefall.Components
{
    public class FreeCamera : Component, IUpdate
    {
        public float MoveSpeed { get; set; } = 200.0f;
        public float LookSpeed { get; set; } = 0.15f;

        private float yaw = 0;
        private float pitch = 0;

        public void Update()
        {
            if (Transform == null) return;

            float dt = Time.Delta;
            float move = MoveSpeed * dt;

            // Get current forward/right from Transform
            Vector3 forward = Transform.Forward;
            Vector3 right = Transform.Right;

            // Movement
            if (Input.IsKeyDown(Keys.W)) Transform.Position += forward * move;
            if (Input.IsKeyDown(Keys.S)) Transform.Position -= forward * move;
            if (Input.IsKeyDown(Keys.A)) Transform.Position -= right * move;
            if (Input.IsKeyDown(Keys.D)) Transform.Position += right * move;
            if (Input.IsKeyDown(Keys.Q)) Transform.Position -= Vector3.UnitY * move;
            if (Input.IsKeyDown(Keys.E)) Transform.Position += Vector3.UnitY * move;

            // Look (only when mouse locked)
            if (Input.IsMouseLocked)
            {
                var delta = Input.MouseDelta;
                yaw += delta.X * LookSpeed;
                pitch -= delta.Y * LookSpeed;

                if (pitch > 89.0f) pitch = 89.0f;
                if (pitch < -89.0f) pitch = -89.0f;

                // Update Transform rotation from yaw/pitch
                float yawRad = yaw * (MathF.PI / 180f);
                float pitchRad = pitch * (MathF.PI / 180f);
                Transform.Rotation = Quaternion.CreateFromYawPitchRoll(yawRad, pitchRad, 0);
            }
        }
    }
}


