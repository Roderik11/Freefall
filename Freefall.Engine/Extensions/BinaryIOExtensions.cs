using System.IO;
using System.Numerics;

namespace Freefall
{
    /// <summary>
    /// Extension methods for BinaryWriter and BinaryReader.
    /// Shared by all AssetPackers for reading/writing common math types.
    /// </summary>
    public static class BinaryIOExtensions
    {
        // ── Write ──

        public static void Write(this BinaryWriter w, Vector2 v)
        { w.Write(v.X); w.Write(v.Y); }

        public static void Write(this BinaryWriter w, Vector3 v)
        { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); }

        public static void Write(this BinaryWriter w, Vector4 v)
        { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); w.Write(v.W); }

        public static void Write(this BinaryWriter w, Quaternion q)
        { w.Write(q.X); w.Write(q.Y); w.Write(q.Z); w.Write(q.W); }

        public static void Write(this BinaryWriter w, Matrix4x4 m)
        {
            w.Write(m.M11); w.Write(m.M12); w.Write(m.M13); w.Write(m.M14);
            w.Write(m.M21); w.Write(m.M22); w.Write(m.M23); w.Write(m.M24);
            w.Write(m.M31); w.Write(m.M32); w.Write(m.M33); w.Write(m.M34);
            w.Write(m.M41); w.Write(m.M42); w.Write(m.M43); w.Write(m.M44);
        }

        public static void WriteArray(this BinaryWriter w, Vector3[] arr)
        {
            w.Write(arr?.Length ?? 0);
            if (arr != null)
                foreach (var v in arr) w.Write(v);
        }

        public static void WriteArray(this BinaryWriter w, Vector2[] arr)
        {
            w.Write(arr?.Length ?? 0);
            if (arr != null)
                foreach (var v in arr) w.Write(v);
        }

        public static void WriteArray(this BinaryWriter w, uint[] arr)
        {
            w.Write(arr?.Length ?? 0);
            if (arr != null)
                foreach (var v in arr) w.Write(v);
        }

        // ── Read ──

        public static Vector2 ReadVector2(this BinaryReader r)
            => new(r.ReadSingle(), r.ReadSingle());

        public static Vector3 ReadVector3(this BinaryReader r)
            => new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());

        public static Vector4 ReadVector4(this BinaryReader r)
            => new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle());

        public static Quaternion ReadQuaternion(this BinaryReader r)
            => new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle());

        public static Matrix4x4 ReadMatrix4x4(this BinaryReader r)
        {
            return new Matrix4x4(
                r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle(),
                r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle(),
                r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle(),
                r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle()
            );
        }

        public static Vector3[] ReadVector3Array(this BinaryReader r)
        {
            int count = r.ReadInt32();
            var arr = new Vector3[count];
            for (int i = 0; i < count; i++)
                arr[i] = r.ReadVector3();
            return arr;
        }

        public static Vector2[] ReadVector2Array(this BinaryReader r)
        {
            int count = r.ReadInt32();
            var arr = new Vector2[count];
            for (int i = 0; i < count; i++)
                arr[i] = r.ReadVector2();
            return arr;
        }

        public static uint[] ReadUInt32Array(this BinaryReader r)
        {
            int count = r.ReadInt32();
            var arr = new uint[count];
            for (int i = 0; i < count; i++)
                arr[i] = r.ReadUInt32();
            return arr;
        }
    }
}
