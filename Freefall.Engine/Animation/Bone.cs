using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace Freefall.Animation
{
    /// <summary>
    /// Represents a bone in a skeleton hierarchy.
    /// </summary>
    public class Bone
    {
        private string _name = string.Empty;

        public int Hash { get; private set; }

        public string Name
        {
            get => _name;
            set
            {
                _name = value;
                Hash = string.IsNullOrEmpty(value) ? 0 : _name.GetHashCode();
            }
        }

        /// <summary>The local bind pose (position, rotation, scale).</summary>
        public BonePose BindPose;
        
        /// <summary>Bind pose as a matrix.</summary>
        public Matrix4x4 BindPoseMatrix;
        
        /// <summary>Inverse bind pose matrix (transforms from mesh space to bone space).</summary>
        public Matrix4x4 OffsetMatrix;
        
        /// <summary>Optional correction matrix for retargeting.</summary>
        public Matrix4x4 Correction = Matrix4x4.Identity;
        
        /// <summary>Scale factor for bone translations.</summary>
        public float ScaleFactor = 1;

        /// <summary>Parent bone index (-1 for root).</summary>
        public int Parent;

        public override string ToString() => Name;

        public void Write(BinaryWriter writer)
        {
            writer.Write(_name);
            writer.Write(BindPose.Position.X);
            writer.Write(BindPose.Position.Y);
            writer.Write(BindPose.Position.Z);
            writer.Write(BindPose.Rotation.X);
            writer.Write(BindPose.Rotation.Y);
            writer.Write(BindPose.Rotation.Z);
            writer.Write(BindPose.Rotation.W);
            writer.Write(BindPose.Scale.X);
            writer.Write(BindPose.Scale.Y);
            writer.Write(BindPose.Scale.Z);
            writer.Write(Parent);
        }

        public void Read(BinaryReader reader)
        {
            Name = reader.ReadString();
            BindPose.Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            BindPose.Rotation = new Quaternion(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            BindPose.Scale = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            Parent = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Represents a bone's transform as position, rotation, and scale.
    /// </summary>
    public struct BonePose
    {
        public Vector3 Position;
        public Quaternion Rotation;
        public Vector3 Scale;

        public static BonePose Identity => new BonePose
        {
            Position = Vector3.Zero,
            Rotation = Quaternion.Identity,
            Scale = Vector3.One
        };
    }

    /// <summary>
    /// Per-vertex bone weights for skinning.
    /// Supports up to 4 bones per vertex.
    /// </summary>
    public struct BoneWeight
    {
        public Vector4 BoneIDs;
        public Vector4 Weights;

        public static BoneWeight Default => new BoneWeight
        {
            BoneIDs = Vector4.Zero,
            Weights = new Vector4(1, 0, 0, 0) // All weight on bone 0
        };
    }
}
