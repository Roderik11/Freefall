using Freefall.Graphics;
using Vortice.Mathematics;
using System.Numerics;
using System;
using Freefall.Base;

namespace Freefall.Components
{
    public class Camera : Component, IUpdate
    {
        public RenderView Target;
        public static Camera Main { get; set; }

        public Matrix4x4 View { get; private set; }
        public Matrix4x4 Projection { get; private set; }

        // Position/Forward/Right convenience properties that map to Transform
        public Vector3 Position
        {
            get => Transform?.Position ?? Vector3.Zero;
            set { if (Transform != null) Transform.Position = value; }
        }

        public Vector3 Forward => Transform?.Forward ?? Vector3.UnitZ;
        public Vector3 Right => Transform?.Right ?? Vector3.UnitX;
        public Vector3 Up => Transform?.Up ?? Vector3.UnitY;

        public float FieldOfView { get; set; } = 45f; // In degrees, like Apex
        public float AspectRatio { get; set; } = 16.0f / 9.0f;
        public float NearPlane { get; set; } = 0.1f;
        public float FarPlane { get; set; } = 2048f;

        public Camera() 
        {
            if (Main == null) Main = this;
            if (Engine.RenderView != null) Target = Engine.RenderView;
        }

        public Camera(RenderView target) : this()
        {
            Target = target;
            if (Target != null)
            {
                Target.OnResize += OnResize;
                OnResize(Target.Width, Target.Height);
            }
        }

        protected override void Awake()
        {
            if (Target == null && Engine.RenderView != null) Target = Engine.RenderView;
            if (Main == null) Main = this;
        }

        private void OnResize(int width, int height)
        {
            if (height == 0) height = 1;
            AspectRatio = (float)width / height;
        }

        public void Update()
        {
            // Update Aspect Ratio if Target changes
            if (Target != null && Target.Height != 0)
            {
                float ar = (float)Target.Width / Target.Height;
                if (Math.Abs(ar - AspectRatio) > 0.001f) AspectRatio = ar;
            }
            
            UpdateMatrices();
        }

        private void UpdateMatrices()
        {
            // Use Transform for position and orientation like Apex
            Vector3 position = Position;
            Vector3 forward = Forward;
            Vector3 up = Up;

            // Create view matrix from Transform (like Apex line 53)
            View = Matrix4x4.CreateLookAtLeftHanded(position, position + forward, up);
            
            // Create projection matrix
            float vFOV = FieldOfView * (MathF.PI / 180f);
            Projection = Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(vFOV, AspectRatio, NearPlane, FarPlane);
        }
        
        /// <summary>
        /// Set shader parameters globally on all effects (Apex pattern).
        /// Called by DeferredRenderer before rendering.
        /// </summary>
        public void SetShaderParams()
        {
            UpdateMatrices();
            var view = View;
            Matrix4x4.Invert(view, out var viewInv);
            var proj = Projection;
            Matrix4x4.Invert(proj, out var projInv);
            var cameraPos = Transform?.Position ?? Vector3.Zero;
            var time = Freefall.Base.Time.TotalTime;
            
            foreach (var pair in Graphics.Effect.MasterEffects)
            {
                pair.Value.SetParameter("CameraPosition", cameraPos);
                pair.Value.SetParameter("View", view);
                pair.Value.SetParameter("ViewInverse", viewInv);
                pair.Value.SetParameter("Projection", proj);
                pair.Value.SetParameter("ProjectionInverse", projInv);
                pair.Value.SetParameter("ViewProjection", view * proj);
                pair.Value.SetParameter("Time", time);
                pair.Value.SetParameter("AmbientScale", SkyboxRenderer.AmbientScale);
            }
        }
        
        /// <summary>
        /// Get a projection matrix with custom near/far planes.
        /// Used for shadow cascade computation.
        /// </summary>
        public Matrix4x4 GetProjectionMatrix(float nearPlane, float farPlane)
        {
            float vFOV = FieldOfView * (MathF.PI / 180f);
            return Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(vFOV, AspectRatio, nearPlane, farPlane);
        }

        /// <summary>
        /// Get View-Projection matrix for GPU culling.
        /// </summary>
        public Matrix4x4 ViewProjection => View * Projection;

        /// <summary>
        /// Get frustum planes for GPU culling.
        /// Returns 6 planes as Vector4 (xyz=normal, w=distance).
        /// </summary>
        public Vector4[] GetFrustumPlanes()
        {
            var frustum = new Graphics.Frustum(ViewProjection);
            return frustum.GetPlanesAsVector4();
        }

        public void Render(Vortice.Direct3D12.ID3D12GraphicsCommandList list)
        {
             if (Target != null && Target.Pipeline != null)
             {
                 Target.Pipeline.Clear(this);
                 Target.Pipeline.Render(this, list);
             }
        }
    }
}


