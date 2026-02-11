using System.Numerics;
using Vortice.Mathematics;
using Freefall.Base;
using Freefall.Graphics;

namespace Freefall.Components
{
    public class Transform : Component
    {
        public Vector3 Position { get; set; } = Vector3.Zero;
        public Quaternion Rotation { get; set; } = Quaternion.Identity;
        public Vector3 Scale { get; set; } = Vector3.One;

        /// <summary>
        /// GPU persistent transform slot in global TransformBuffer.
        /// Allocated on first access, reused across frames.
        /// </summary>
        public int TransformSlot
        {
            get
            {
                if (_transformSlot < 0 && TransformBuffer.Instance != null)
                    _transformSlot = TransformBuffer.Instance.AllocateSlot();
                return _transformSlot;
            }
        }
        private int _transformSlot = -1;

        public Matrix4x4 WorldMatrix
        {
            get
            {
                // TRS Matrix
                return Matrix4x4.CreateScale(Scale) * 
                       Matrix4x4.CreateFromQuaternion(Rotation) * 
                       Matrix4x4.CreateTranslation(Position);
            }
        }
        
        public Vector3 Forward => Vector3.Transform(Vector3.UnitZ, Rotation);
        public Vector3 Right => Vector3.Transform(Vector3.UnitX, Rotation);
        public Vector3 Up => Vector3.Transform(Vector3.UnitY, Rotation);

        public void SetRotation(float pitch, float yaw, float roll)
        {
            Rotation = Quaternion.CreateFromYawPitchRoll(yaw, pitch, roll);
        }

        public void LookAt(Vector3 target, Vector3 up)
        {
             // Calculate rotation to look at target
             Matrix4x4 lookAt = Matrix4x4.CreateLookAtLeftHanded(Position, target, up);
             // Invert because LookAt creates a View Matrix (World to Camera), we want World Matrix (Model to World) rotation?
             // Actually, LookAt returns View Matrix. Inverse of View Matrix (rotation part) is Camera World Rotation.
             Matrix4x4 inv;
             if (Matrix4x4.Invert(lookAt, out inv))
             {
                 Rotation = Quaternion.CreateFromRotationMatrix(inv);
             }
        }
        
        public void LookAt(Vector3 target) => LookAt(target, Vector3.UnitY);
    }
}
