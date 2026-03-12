using System.Numerics;

namespace Freefall
{
    /// <summary>
    /// A ray defined by a position and direction.
    /// </summary>
    public struct Ray
    {
        public Vector3 Position;
        public Vector3 Direction;

        public Ray(Vector3 position, Vector3 direction)
        {
            Position = position;
            Direction = direction;
        }
    }
}
