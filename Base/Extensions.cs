using System;
using System.Collections.Generic;
using System.Numerics;
using Vortice.Mathematics;

namespace Freefall
{
    public static class Extensions
    {
        public static Vector3[] Transform(this Vector3[] points, Matrix4x4 transform)
        {
            for (int i = 0; i < points.Length; i++)
                points[i] = Vector3.Transform(points[i], transform);

            return points;
        }

        public static void GetCorners(this BoundingBox box, Vector3[] points, Matrix4x4 transform)
        {
            box.GetCorners(points);
            points.Transform(transform);
        }

        public static void For<T>(this IList<T> list, Action<T> action, bool parallel = false)
        {
            if (parallel)
            {
                System.Threading.Tasks.Parallel.For(0, list.Count, (i) => action(list[i]));
            }
            else
            {
                for (int i = 0; i < list.Count; i++)
                    action(list[i]);
            }
        }
    }
}
