using System;
using System.Numerics;
using Freefall.Base;
using Freefall.Components;

namespace Freefall.Scripts
{
    /// <summary>
    /// Third-person character controller with WASD movement, terrain following, and jumping.
    /// Based on Apex's CharacterController but simplified without PhysX physics.
    /// </summary>
    public class CharacterController : Component, IUpdate
    {
        // Configuration
        public float RunSpeed = 4.7f;
        public float WalkSpeed = 1.7f;
        public float BackwardSpeed = 1.5f;
        public float JumpVelocity = 5f;
        public float Height = 1.8f;

        // Air control
        public float AirControlMultiplier = 0.3f;

        // Smoothing
        private const float InputSmoothSpeed = 6f;
        private const float VerticalSmoothTime = 0.05f;

        // Ground sticking
        private const float SlopeStickForce = -2f;
        private const float FallingVelocityThreshold = -2.5f;
        private const float FallingTimeThreshold = 0.5f;

        // Terrain reference for height sampling
        public IHeightProvider Terrain;
        private float groundOffset = 0.05f;
        
        // Movement state
        private Vector2 smoothInputVector;
        private Vector3 velocity;
        private readonly Vector3 gravityAcceleration = new Vector3(0, -9.81f, 0);

        // Jump momentum
        private Vector3 jumpMomentum;
        private bool didJump;

        // Ground state
        private float smoothedYPosition;
        private float smoothedYVelocity;
        private bool wasGroundedLastFrame;
        private bool isGrounded;
        private float timeSinceGrounded;
        private bool inAir;

        // Camera angle (set by ThirdPersonCamera)
        public static float CameraYaw;

        // Animation
        private Animator animator;

        protected override void Awake()
        {
            smoothedYPosition = Transform.Position.Y;
            animator = Entity.GetComponent<Animator>();
        }

        public void Update()
        {
            float delta = Time.SmoothDelta;

            // Capture input
            bool forward = Input.IsKeyDown(Keys.W);
            bool backward = Input.IsKeyDown(Keys.S);
            bool left = Input.IsKeyDown(Keys.A);
            bool right = Input.IsKeyDown(Keys.D);
            bool run = Input.IsKeyDown(Keys.ShiftKey);
            bool jump = Input.IsKeyDown(Keys.Space);

            // Update rotation based on camera
            Transform.Rotation = Quaternion.CreateFromYawPitchRoll(CameraYaw, 0, 0);

            // Calculate movement direction
            Vector3 inputVelocity = Vector3.Zero;
            if (forward) inputVelocity += Transform.Forward;
            else if (backward) inputVelocity -= Transform.Forward;
            if (left) inputVelocity -= Transform.Right;
            else if (right) inputVelocity += Transform.Right;

            // Update animator parameters
            UpdateAnimator(delta);

            // Calculate speed
            float speed = 0f;
            if (left || right)
                speed = run ? RunSpeed : WalkSpeed;
            else if (forward)
                speed = run ? RunSpeed : WalkSpeed;
            else if (backward)
                speed = run ? BackwardSpeed * 3 : BackwardSpeed;

            // Normalize and apply speed
            if (inputVelocity.LengthSquared() > 0.0f)
                inputVelocity = Vector3.Normalize(inputVelocity);
            inputVelocity *= speed;

            // Apply air control if jumping
            Vector3 targetVelocity;
            if (!isGrounded && didJump)
            {
                Vector3 airControl = inputVelocity * AirControlMultiplier;
                targetVelocity = jumpMomentum + airControl;
            }
            else
            {
                targetVelocity = inputVelocity;
            }

            // Apply gravity
            velocity += gravityAcceleration * delta;
            UpdateJumpFallLand(run, jump, targetVelocity, delta);

            // Apply movement
            targetVelocity += velocity;
            Vector3 newPos = Transform.Position + targetVelocity * delta;

            // Sample terrain height (+ offset so feet don't clip through)
            float terrainHeight = 0;
            if (Terrain != null)
            {
                terrainHeight = Terrain.GetHeight(newPos) + groundOffset;
            }

            // Ground detection
            bool wasGrounded = isGrounded;
            isGrounded = newPos.Y <= terrainHeight + 0.1f;

            if (isGrounded)
            {
                timeSinceGrounded = 0;
                velocity.Y = Math.Max(velocity.Y, SlopeStickForce);

                // Smooth vertical position
                if (wasGroundedLastFrame)
                {
                    smoothedYPosition = SmoothDamp(smoothedYPosition, terrainHeight, ref smoothedYVelocity, 
                        VerticalSmoothTime, 100f, delta);
                    newPos.Y = smoothedYPosition;
                }
                else
                {
                    smoothedYPosition = terrainHeight;
                    smoothedYVelocity = 0;
                    newPos.Y = terrainHeight;
                }
            }
            else
            {
                timeSinceGrounded += delta;
                smoothedYPosition = newPos.Y;
                smoothedYVelocity = 0;
            }

            wasGroundedLastFrame = isGrounded;
            Transform.Position = newPos;
        }

        private void UpdateAnimator(float delta)
        {
            if (animator == null) return;

            // Capture input
            bool forward = Input.IsKeyDown(Keys.W);
            bool backward = Input.IsKeyDown(Keys.S);
            bool left = Input.IsKeyDown(Keys.A);
            bool right = Input.IsKeyDown(Keys.D);
            bool run = Input.IsKeyDown(Keys.ShiftKey);

            if (!didJump) // Using didJump as proxy for Input.Jump for now, or just send 0
                animator.SetParam("jump", 0);

            Vector2 axis = Vector2.Zero;

            if (forward)
                axis.Y = run ? 1f : .5f;
            else if (backward)
                axis.Y = run ? -1f : -.5f;

            if (left)
                axis.X = run ? -1 : -.5f;
            else if (right)
                axis.X = run ? 1 : .5f;

            smoothInputVector = Vector2.Lerp(smoothInputVector, axis, Math.Min(1, delta * InputSmoothSpeed));

            animator.SetParam("axisX", smoothInputVector.X);
            animator.SetParam("axisY", smoothInputVector.Y);
        }

        private void UpdateJumpFallLand(bool isRunning, bool tryJump, Vector3 currentHorizontalVelocity, float delta)
        {
            if (isGrounded)
            {
                timeSinceGrounded = 0;
            }
            else
            {
                timeSinceGrounded += delta;
            }

            // Apex Logic: "isFallingFast" threshold
            bool isFallingFast = velocity.Y < FallingVelocityThreshold;

            bool shouldLand = inAir && isGrounded;
            // Apex Logic: Time threshold (+ falling fast) prevents instant falling state
            bool shouldFall = timeSinceGrounded > FallingTimeThreshold && isFallingFast;
            bool canJump = !shouldFall && !shouldLand && !inAir;
            bool shouldJump = canJump && tryJump;

            if (animator != null)
            {
                if (shouldLand)
                {
                    SetAnimationState(jump: 0, falling: 0, landing: 1);
                    jumpMomentum = Vector3.Zero;
                    didJump = false;
                }
                else if (shouldFall)
                {
                    SetAnimationState(jump: 0, falling: 1, landing: 0);
                }
                else if (shouldJump)
                {
                    velocity.Y = isRunning ? JumpVelocity * 1.5f : JumpVelocity;
                    SetAnimationState(jump: 1, falling: 0, landing: 0);

                    jumpMomentum = new Vector3(currentHorizontalVelocity.X, 0, currentHorizontalVelocity.Z);
                    didJump = true;
                }
            }
            else
            {
                 // Fallback if no animator
                 if (shouldJump)
                 {
                    velocity.Y = isRunning ? JumpVelocity * 1.5f : JumpVelocity;
                    jumpMomentum = new Vector3(currentHorizontalVelocity.X, 0, currentHorizontalVelocity.Z);
                    didJump = true;
                 }
                 if (shouldLand)
                 {
                    jumpMomentum = Vector3.Zero;
                    didJump = false;
                 }
            }

            inAir = !isGrounded;
        }

        private void SetAnimationState(int jump, int falling, int landing)
        {
            animator.SetParam("jump", jump);
            animator.SetParam("falling", falling);
            animator.SetParam("landing", landing);
        }

        private float SmoothDamp(float current, float target, ref float currentVelocity,
            float smoothTime, float maxSpeed, float deltaTime)
        {
            smoothTime = Math.Max(0.0001f, smoothTime);
            float omega = 2f / smoothTime;
            float x = omega * deltaTime;
            float exp = 1f / (1f + x + 0.48f * x * x + 0.235f * x * x * x);

            float change = current - target;
            float originalTo = target;
            float maxChange = maxSpeed * smoothTime;
            change = Math.Clamp(change, -maxChange, maxChange);
            target = current - change;

            float temp = (currentVelocity + omega * change) * deltaTime;
            currentVelocity = (currentVelocity - omega * temp) * exp;

            float output = target + (change + temp) * exp;
            if (originalTo - current > 0.0f == output > originalTo)
            {
                output = originalTo;
                currentVelocity = (output - originalTo) / deltaTime;
            }

            return output;
        }
    }
}
