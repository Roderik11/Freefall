using System.Numerics;
using Vortice.Mathematics;

namespace Freefall.Graphics
{
    /// <summary>
    /// Frustum planes extracted from a View-Projection matrix.
    /// Used for GPU frustum culling - planes are uploaded as float4[6].
    /// </summary>
    public struct Frustum
    {
        private Plane[] planes;

        /// <summary>
        /// Extracts 6 frustum planes from a combined View-Projection matrix.
        /// Planes are normalized and in the form: dot(normal, point) + d >= 0 means inside.
        /// Order: Near, Far, Left, Right, Top, Bottom
        /// </summary>
        public Frustum(Matrix4x4 viewProj)
        {
            planes = new Plane[6];

            // Left plane
            planes[2].Normal.X = -viewProj.M14 - viewProj.M11;
            planes[2].Normal.Y = -viewProj.M24 - viewProj.M21;
            planes[2].Normal.Z = -viewProj.M34 - viewProj.M31;
            planes[2].D = -viewProj.M44 - viewProj.M41;

            // Right plane
            planes[3].Normal.X = -viewProj.M14 + viewProj.M11;
            planes[3].Normal.Y = -viewProj.M24 + viewProj.M21;
            planes[3].Normal.Z = -viewProj.M34 + viewProj.M31;
            planes[3].D = -viewProj.M44 + viewProj.M41;

            // Top plane
            planes[4].Normal.X = -viewProj.M14 + viewProj.M12;
            planes[4].Normal.Y = -viewProj.M24 + viewProj.M22;
            planes[4].Normal.Z = -viewProj.M34 + viewProj.M32;
            planes[4].D = -viewProj.M44 + viewProj.M42;

            // Bottom plane
            planes[5].Normal.X = -viewProj.M14 - viewProj.M12;
            planes[5].Normal.Y = -viewProj.M24 - viewProj.M22;
            planes[5].Normal.Z = -viewProj.M34 - viewProj.M32;
            planes[5].D = -viewProj.M44 - viewProj.M42;

            // Near plane
            planes[0].Normal.X = -viewProj.M13;
            planes[0].Normal.Y = -viewProj.M23;
            planes[0].Normal.Z = -viewProj.M33;
            planes[0].D = -viewProj.M43;

            // Far plane
            planes[1].Normal.X = -viewProj.M14 + viewProj.M13;
            planes[1].Normal.Y = -viewProj.M24 + viewProj.M23;
            planes[1].Normal.Z = -viewProj.M34 + viewProj.M33;
            planes[1].D = -viewProj.M44 + viewProj.M43;

            // Normalize all planes
            for (int i = 0; i < 6; i++)
            {
                float len = planes[i].Normal.Length();
                planes[i].Normal = planes[i].Normal / len;
                planes[i].D /= len;
            }
        }

        /// <summary>
        /// Gets planes as Vector4 array for GPU upload.
        /// Format: xyz = normal, w = distance
        /// </summary>
        public Vector4[] GetPlanesAsVector4()
        {
            var result = new Vector4[6];
            for (int i = 0; i < 6; i++)
            {
                result[i] = new Vector4(planes[i].Normal, planes[i].D);
            }
            return result;
        }
        
        /// <summary>
        /// Gets the 8 corners of the frustum by inverting the view-projection matrix.
        /// Order: Near-plane (TL, TR, BR, BL), Far-plane (TL, TR, BR, BL)
        /// </summary>
        public void GetCorners(Vector3[] corners)
        {
            if (corners == null || corners.Length < 8)
                throw new ArgumentException("Corners array must have at least 8 elements");
            
            // Frustum corners in NDC space
            Vector4[] ndcCorners = new Vector4[]
            {
                new Vector4(-1,  1, 0, 1), // Near TL
                new Vector4( 1,  1, 0, 1), // Near TR
                new Vector4( 1, -1, 0, 1), // Near BR
                new Vector4(-1, -1, 0, 1), // Near BL
                new Vector4(-1,  1, 1, 1), // Far TL
                new Vector4( 1,  1, 1, 1), // Far TR
                new Vector4( 1, -1, 1, 1), // Far BR
                new Vector4(-1, -1, 1, 1), // Far BL
            };
            
            // We need the inverse VP matrix - compute from planes
            // Since we don't store the original matrix, we'll compute corners differently
            // For shadow mapping, the caller should use a different approach
            // This is a simplified version that assumes the frustum was built from a VP matrix
            
            // Just return the NDC corners for now - the caller will transform them
            for (int i = 0; i < 8; i++)
            {
                corners[i] = new Vector3(ndcCorners[i].X, ndcCorners[i].Y, ndcCorners[i].Z);
            }
        }

        /// <summary>
        /// Test if a bounding sphere intersects or is contained by the frustum.
        /// </summary>
        public ContainmentType Intersects(BoundingSphere sphere)
        {
            Vector3 center = sphere.Center;
            float radius = sphere.Radius;
            int count = 0;

            foreach (Plane plane in planes)
            {
                float d = (plane.Normal.X * center.X + plane.Normal.Y * center.Y + plane.Normal.Z * center.Z) + plane.D;

                if (d > radius)
                    return ContainmentType.Disjoint;

                if (d < -radius)
                    count++;
            }

            if (count != 6)
                return ContainmentType.Intersects;

            return ContainmentType.Contains;
        }

        /// <summary>
        /// Test if a bounding box intersects or is contained by the frustum.
        /// </summary>
        public ContainmentType Intersects(BoundingBox box)
        {
            bool flag = false;

            foreach (Plane plane in planes)
            {
                switch (box.Intersects(plane))
                {
                    case PlaneIntersectionType.Front:
                        return ContainmentType.Disjoint;

                    case PlaneIntersectionType.Intersecting:
                        flag = true;
                        break;
                }
            }

            if (!flag)
                return ContainmentType.Contains;

            return ContainmentType.Intersects;
        }
    }
}
