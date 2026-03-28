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
        public float FoVFactor { get; private set; }

        public Camera() 
        {
            if (Main == null) Main = this;
        }

        public Camera(RenderView target) : this()
        {
            Target = target;
            if (Target != null)
            {
                Target.OnResized += HandleTargetResize;
                HandleTargetResize();
            }
        }

        protected override void Awake()
        {
            if (Main == null) Main = this;
        }

        private void HandleTargetResize()
        {
            int width = Target.Width;
            int height = Target.Height;
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
            
            
            // Create reverse-Z projection matrix (near→1.0, far→0.0)
            // Can't use CreatePerspectiveFieldOfViewLeftHanded — it validates near < far
            float vFOV = FieldOfView * (MathF.PI / 180f);
            Projection = CreateReverseZPerspectiveLH(vFOV, AspectRatio, NearPlane, FarPlane);
        
            FoVFactor = (180f / MathF.PI) / FieldOfView;
            FoVFactor *= FoVFactor; // squared to match squared size comparison
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
                float shadowRes = Graphics.DeferredRenderer.Current?.ShadowTextureArray?.Width ?? 2048f;
                pair.Value.SetParameter("ShadowTexelSize", 1.0f / shadowRes);
                pair.Value.SetParameter("FogEnabled", Engine.Settings.Fog ? 1.0f : 0.0f);
                pair.Value.SetParameter("FogDensity", Engine.Settings.FogDensity);
                pair.Value.SetParameter("FogSunDirection", SkyboxRenderer.CurrentSunDirection);
                pair.Value.SetParameter("CamPos", cameraPos);
            }
        }
        
        /// <summary>
        /// Get a standard-Z projection matrix with custom near/far planes.
        /// Used for shadow cascade frustum computation — standard Z is correct here
        /// since we only unproject NDC corners to get world-space frustum geometry.
        /// </summary>
        public Matrix4x4 GetProjectionMatrix(float nearPlane, float farPlane)
        {
            float vFOV = FieldOfView * (MathF.PI / 180f);
            return Matrix4x4.CreatePerspectiveFieldOfViewLeftHanded(vFOV, AspectRatio, nearPlane, farPlane);
        }

        /// <summary>
        /// Build a left-handed perspective projection matrix with reverse-Z (near→1, far→0).
        /// The .NET CreatePerspectiveFieldOfViewLeftHanded validates near &lt; far and rejects swapped values,
        /// so we build the matrix manually.
        /// </summary>
        private static Matrix4x4 CreateReverseZPerspectiveLH(float fovY, float aspect, float near, float far)
        {
            float h = 1.0f / MathF.Tan(fovY * 0.5f);
            float w = h / aspect;

            // Standard LH perspective maps [near,far] → [0,1]:
            //   M33 = far / (far - near),  M43 = -near * far / (far - near)
            // Reverse-Z swaps near↔far in the standard formula:
            //   M33 = near / (near - far)   [negative]
            //   M43 = far * near / (far - near)
            float range = far - near;
            return new Matrix4x4(
                w,    0,    0,    0,
                0,    h,    0,    0,
                0,    0,    near / (near - far),  1,
                0,    0,    far * near / range, 0
            );
        }

        /// <summary>
        /// Get View-Projection matrix for GPU culling.
        /// </summary>
        public Matrix4x4 ViewProjection => View * Projection;

        /// <summary>
        /// Build a world-space ray from the current mouse position through the camera.
        /// Uses Input.MousePosition and Target viewport dimensions.
        /// </summary>
        public Ray MouseRay()
        {
            int vpW = Target?.Width ?? 1;
            int vpH = Target?.Height ?? 1;

            // Mouse position (screen-space, relative to window)
            var mp = Input.MousePosition;

            // Convert to NDC: x ∈ [-1,1], y ∈ [-1,1] (Y flipped: top=-1 in clip space)
            float ndcX = (2f * mp.X / vpW) - 1f;
            float ndcY = 1f - (2f * mp.Y / vpH);

            // Unproject near and far points through inverse ViewProjection
            var vp = View * Projection;
            Matrix4x4.Invert(vp, out var vpInv);

            var nearNdc = new Vector3(ndcX, ndcY, 1f);   // Reverse-Z: near = 1
            var farNdc  = new Vector3(ndcX, ndcY, 0f);   // Reverse-Z: far  = 0

            var nearW = Vector4.Transform(new Vector4(nearNdc, 1f), vpInv);
            var farW  = Vector4.Transform(new Vector4(farNdc, 1f), vpInv);

            var near = new Vector3(nearW.X, nearW.Y, nearW.Z) / nearW.W;
            var far  = new Vector3(farW.X, farW.Y, farW.Z) / farW.W;

            return new Ray(near, Vector3.Normalize(far - near));
        }

        /// <summary>
        /// Reconstruct world position from a viewport pixel and its linear view-space depth.
        /// Uses the same zero-translation inverse VP pattern as the light pass.
        /// </summary>
        public Vector3 UnprojectPixel(int pixelX, int pixelY, float linearDepth, int vpWidth, int vpHeight)
        {
            // Pixel -> NDC (same as MouseRay)
            float ndcX = (2f * pixelX / vpWidth) - 1f;
            float ndcY = 1f - (2f * pixelY / vpHeight);

            // Zero-translation camera-relative VP (matches FillLightBuffer pattern)
            var cvp = Matrix4x4.CreateLookAtLeftHanded(Vector3.Zero, Forward, Up) * Projection;
            Matrix4x4.Invert(cvp, out var cvpInv);

            // Unproject NDC point at z=1 (reverse-Z near) to get camera-relative ray direction
            var clipPoint = new Vector4(ndcX, ndcY, 1f, 1f);
            var viewDir = Vector4.Transform(clipPoint, cvpInv);
            var dir = new Vector3(viewDir.X, viewDir.Y, viewDir.Z) / viewDir.W;

            // Scale by linear depth / length to get camera-relative position, then add camera world pos
            float scale = linearDepth / dir.Length();
            return Position + dir * scale;
        }

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


