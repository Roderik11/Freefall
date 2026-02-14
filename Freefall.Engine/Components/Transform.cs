using System;
using System.Collections.Generic;
using System.Numerics;
using Freefall.Base;
using Freefall.Graphics;
using Vortice.Mathematics;

namespace Freefall.Components
{
    public sealed class Transform : Component
    {
        private int _depth;
        private Transform _parent;
        private bool _isDirty = true;
        private bool _isFiringChanged;
        private Vector3 _position;
        private Quaternion _rotation = Quaternion.Identity;
        private Vector3 _scale = Vector3.One;
        private Matrix4x4 _matrix = Matrix4x4.Identity;
        private Matrix4x4 _localTransform = Matrix4x4.Identity;
        private Vector3 _worldPosition;
        private readonly List<Transform> Children = new List<Transform>();

        /// <summary>
        /// Fired when any local property changes (position, rotation, scale, parent).
        /// Renderers subscribe to push dirty data to GPU buffers.
        /// </summary>
        public event Action OnChanged;

        public int Depth => _depth;
        public int GetChildCount() => Children.Count;
        public Transform GetChild(int i) => Children.Count > i ? Children[i] : null;
        public bool IsDirty => _isDirty ? true : (_parent != null ? _parent.IsDirty : false);
        public Vector3 WorldPosition => IsDirty ? Matrix.Translation : _worldPosition;
        public Vector3 WorldScale => _parent != null ? _scale * _parent.WorldScale : _scale;
        public Vector3 Up => Vector3.Normalize(new Vector3(Matrix.M21, Matrix.M22, Matrix.M23));
        public Vector3 Forward => Vector3.Normalize(new Vector3(Matrix.M31, Matrix.M32, Matrix.M33));
        public Vector3 Right => Vector3.Normalize(new Vector3(Matrix.M11, Matrix.M12, Matrix.M13));
        public int Count => Children.Count;
        public System.Collections.IEnumerator GetEnumerator() => Children.GetEnumerator();

        public Matrix4x4 RootRotation = Matrix4x4.Identity;

        internal void SetDirty()
        {
            _isDirty = true;

            for (int i = 0; i < Children.Count; i++)
                Children[i].SetDirty();

            TransformBuffer.Instance!.SetTransform(TransformSlot, RootRotation * Matrix);

            // Re-entrancy guard: if a subscriber modifies the transform
            // inside OnChanged, the recursive SetDirty still propagates
            // dirty flags and updates the GPU buffer, but we skip firing
            // the event again to prevent stack overflow.
            if (!_isFiringChanged)
            {
                _isFiringChanged = true;
                try { OnChanged?.Invoke(); }
                finally { _isFiringChanged = false; }
            }
        }

        [System.ComponentModel.Browsable(false)]
        public Transform Parent
        {
            get => _parent;
            set
            {
                if (_parent == value) return;
                _parent?.Children.Remove(this);
                _parent = value;
                _parent?.Children.Add(this);
                _depth = _parent != null ? _parent.Depth + 1 : 0;
                SetDirty();
            }
        }

        public Vector3 Position
        {
            get => _position;
            set
            {
                if (_position == value) return;
                _position = value; SetDirty();
            }
        }

        public Quaternion Rotation
        {
            get => _rotation;
            set
            {
                if (_rotation == value) return;
                _rotation = value; SetDirty();
            }
        }

        public Vector3 Scale
        {
            get => _scale;
            set
            {
                if (_scale == value) return;
                _scale = value; SetDirty();
            }
        }

        /// <summary>
        /// Cached world matrix — lazy recompute only when dirty.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public Matrix4x4 Matrix
        {
            get
            {
                if (IsDirty)
                {
                    _isDirty = false;
                    _localTransform = Matrix4x4.CreateScale(_scale) * Matrix4x4.CreateFromQuaternion(_rotation) * Matrix4x4.CreateTranslation(_position);
                    _matrix = _parent != null ? _localTransform * _parent.Matrix : _localTransform;
                    _worldPosition = _matrix.Translation;
                }

                return _matrix;
            }
            set
            {
                Matrix4x4 m = value;

                if (_parent != null && Matrix4x4.Invert(_parent.Matrix, out var inverse))
                    m *= inverse;

                Matrix4x4.Decompose(m, out _, out _rotation, out _position);

                SetDirty();
            }
        }

        /// <summary>
        /// Backward-compatible alias for Matrix property.
        /// </summary>
        [System.ComponentModel.Browsable(false)]
        public Matrix4x4 WorldMatrix
        {
            get => Matrix;
            set => Matrix = value;
        }

        /// <summary>
        /// GPU persistent transform slot in global TransformBuffer.
        /// Allocated on first access, reused across frames.
        /// Will be replaced by RenderSlot in SceneBuffers (Phase 2).
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

        public void Set(Vector3 position, Quaternion rotation)
        {
            Position = position;
            Rotation = rotation;
        }

        public void RotateLocal(Vector3 axis, float angle)
        {
            Rotation = Quaternion.CreateFromAxisAngle(axis, MathHelper.ToRadians(angle)) * _rotation;
        }

        public void Rotate(Vector3 axis, float angle)
        {
            Rotation = _rotation * Quaternion.CreateFromAxisAngle(axis, MathHelper.ToRadians(angle));
        }

        public void Move(float forward, float up, float right)
        {
            Vector3 f = Forward;
            Vector3 u = Up;
            Vector3 s = Right;

            if (_parent != null)
            {
                Quaternion m = Quaternion.Inverse(_parent.Rotation);

                if (forward != 0) f = Vector3.Transform(f, m);
                if (up != 0) u = Vector3.Transform(u, m);
                if (right != 0) s = Vector3.Transform(s, m);
            }

            if (forward != 0) _position += f * forward;
            if (up != 0) _position += u * up;
            if (right != 0) _position += s * right;

            SetDirty();
        }

        public void SetRotation(float pitch, float yaw, float roll)
        {
            Rotation = Quaternion.CreateFromYawPitchRoll(yaw, pitch, roll);
        }

        public void LookAt(Vector3 target, Vector3 up)
        {
            Matrix4x4 lookAt = Matrix4x4.CreateLookAtLeftHanded(Position, target, up);
            if (Matrix4x4.Invert(lookAt, out var inv))
            {
                Rotation = Quaternion.CreateFromRotationMatrix(inv);
            }
        }

        public void LookAt(Vector3 target) => LookAt(target, Vector3.UnitY);

        public bool IsChildOf(Transform other)
        {
            if (other.Children.Contains(this))
                return true;

            foreach (Transform child in other.Children)
            {
                if (IsChildOf(child))
                    return true;
            }

            return false;
        }
    }
}
