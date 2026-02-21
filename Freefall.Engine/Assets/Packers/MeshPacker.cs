using System.IO;
using System.Numerics;
using Freefall.Assets;
using Freefall.Animation;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Binary packer for MeshData (CPU-only mesh representation).
    /// Stores positions, normals, UVs, indices, mesh parts, bounding box,
    /// bones, and bone weights. Uses length-prefixed arrays.
    /// 
    /// This packs MeshData (not Mesh) because Mesh has GPU resources.
    /// At load time, the caller creates GPU buffers from the unpacked MeshData.
    /// </summary>
    [AssetPacker(".mesh")]
    public class MeshPacker : AssetPacker<MeshData>
    {
        public override int Version => 1;

        public override void Pack(BinaryWriter w, MeshData data)
        {
            // Vertex arrays
            w.WriteArray(data.Positions);
            w.WriteArray(data.Normals);
            w.WriteArray(data.UVs);

            // Indices
            w.WriteArray(data.Indices);

            // Bounding box
            w.Write(data.BoundingBox.Min);
            w.Write(data.BoundingBox.Max);

            // Mesh parts
            w.Write(data.Parts?.Count ?? 0);
            if (data.Parts != null)
            {
                foreach (var part in data.Parts)
                {
                    w.Write(part.Name ?? string.Empty);
                    w.Write(part.BaseVertex);
                    w.Write(part.BaseIndex);
                    w.Write(part.NumIndices);
                    w.Write(part.BoundingBox.Min);
                    w.Write(part.BoundingBox.Max);
                    w.Write(part.BoundingSphere);
                }
            }

            // Skeleton
            bool hasBones = data.Bones != null && data.Bones.Length > 0;
            w.Write(hasBones);
            if (hasBones)
            {
                w.Write(data.Bones.Length);
                foreach (var bone in data.Bones)
                    bone.Write(w);

                // Bone weights (same count as vertices)
                w.Write(data.BoneWeights?.Length ?? 0);
                if (data.BoneWeights != null)
                {
                    foreach (var bw in data.BoneWeights)
                    {
                        w.Write(bw.BoneIDs);
                        w.Write(bw.Weights);
                    }
                }
            }
        }

        public override MeshData Unpack(BinaryReader r, int version)
        {
            var data = new MeshData();

            // Vertex arrays
            data.Positions = r.ReadVector3Array();
            data.Normals = r.ReadVector3Array();
            data.UVs = r.ReadVector2Array();

            // Indices
            data.Indices = r.ReadUInt32Array();

            // Bounding box
            data.BoundingBox = new BoundingBox(r.ReadVector3(), r.ReadVector3());

            // Mesh parts
            int partCount = r.ReadInt32();
            for (int i = 0; i < partCount; i++)
            {
                var part = new MeshPart
                {
                    Name = r.ReadString(),
                    BaseVertex = r.ReadInt32(),
                    BaseIndex = r.ReadInt32(),
                    NumIndices = r.ReadInt32(),
                    BoundingBox = new BoundingBox(r.ReadVector3(), r.ReadVector3()),
                    BoundingSphere = r.ReadVector4()
                };
                data.Parts.Add(part);
            }

            // Skeleton
            bool hasBones = r.ReadBoolean();
            if (hasBones)
            {
                int boneCount = r.ReadInt32();
                data.Bones = new Bone[boneCount];
                for (int i = 0; i < boneCount; i++)
                {
                    data.Bones[i] = new Bone();
                    data.Bones[i].Read(r);
                }

                int weightCount = r.ReadInt32();
                if (weightCount > 0)
                {
                    data.BoneWeights = new BoneWeight[weightCount];
                    for (int i = 0; i < weightCount; i++)
                    {
                        data.BoneWeights[i] = new BoneWeight
                        {
                            BoneIDs = r.ReadVector4(),
                            Weights = r.ReadVector4()
                        };
                    }
                }
            }

            return data;
        }
    }
}
