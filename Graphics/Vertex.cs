using System.Numerics;
using System.Runtime.InteropServices;
using Vortice.DXGI;
using Vortice.Direct3D12;

namespace Freefall.Graphics
{
    [StructLayout(LayoutKind.Sequential)]
    public struct VertexPositionNormalTexture
    {
        public Vector3 Position;
        public Vector3 Normal;
        public Vector2 TextureCoordinate;

        public static readonly int SizeInBytes = Marshal.SizeOf<VertexPositionNormalTexture>();

        public VertexPositionNormalTexture(Vector3 position, Vector3 normal, Vector2 textureCoordinate)
        {
            Position = position;
            Normal = normal;
            TextureCoordinate = textureCoordinate;
        }
        
        public static InputElementDescription[] InputElements = new[]
        {
            new InputElementDescription("POSITION", 0, Format.R32G32B32_Float, 0, 0),
            new InputElementDescription("NORMAL", 0, Format.R32G32B32_Float, 12, 0),
            new InputElementDescription("TEXCOORD", 0, Format.R32G32_Float, 24, 0)
        };
    }
}
