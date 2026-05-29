using System;
using System.Collections.Generic;
using System.ComponentModel;
using Freefall.Graph;
using System.Numerics;

namespace Freefall.ShaderGraph
{
    public enum SampleType
    {
        Point,
        Bilinear,
    }

    public enum WrapType
    {
        Clamp,
        Wrap
    }

    public enum CoordinateSpace
    {
        Object,
        World,
        View,
        Tangent
    }

    [Category("Sampler")]
    public class Texture2DSample : Node
    {
        [Input]
        public Vector2 UV;

        [Output]
        public Vector4 RGBA;
    
        public SampleType SampleType = SampleType.Bilinear;
        public WrapType WrapType = WrapType.Wrap;
    }

    [Category("Sampler")]
    public class Texture2DArraySample : Node
    {
        [Input]
        public Vector3 UV;

        [Output]
        public Vector4 RGBA;

        public SampleType SampleType = SampleType.Bilinear;
        public WrapType WrapType = WrapType.Wrap;
    }

    [Category("Sampler")]
    public class Texture3DSample : Node
    {
        [Input]
        public Vector3 UV;

        [Output]
        public Vector4 RGBA;

        public SampleType SampleType = SampleType.Bilinear;
        public WrapType WrapType = WrapType.Wrap;
    }

    [Category("Math")]
    public class Add : Node
    {
        [Input]
        public Vector3 A;

        [Input]
        public Vector3 B;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class Subtract : Node
    {
        [Input]
        public Vector3 A;

        [Input]
        public Vector3 B;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class Multiply : Node
    {
        [Input]
        public Vector3 A;

        [Input]
        public Vector3 B;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class Clamp : Node
    {
        [Input]
        public Vector3 Value;

        [Input]
        public Vector2 Range;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class Saturate : Node
    {
        [Input]
        public Vector3 Value;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class Power : Node
    {
        [Input]
        public Vector3 Value;

        [Input]
        public float Exponent;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class Min : Node
    {
        [Input]
        public Vector3 A;

        [Input]
        public Vector3 B;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class Max : Node
    {
        [Input]
        public Vector3 A;

        [Input]
        public Vector3 B;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class TransformCoord : Node
    {
        [Input]
        public Vector3 A;

        [Input]
        public Matrix4x4 B;

        [Output]
        public Vector3 Result;
    }

    [Category("Math")]
    public class TransformMatrix : Node
    {
        [Input]
        public Matrix4x4 A;

        [Input]
        public Matrix4x4 B;

        [Output]
        public Matrix4x4 Result;
    }

    [Category("Math")]
    public class InvertMatrix : Node
    {
        [Input]
        public Matrix4x4 Value;

        [Output]
        public Matrix4x4 Result;
    }


    [Category("Inputs")]
    public class Position : Node
    {
        [Output]
        public Vector4 Result;

        public CoordinateSpace Space = CoordinateSpace.Object;
    }

    [Category("Inputs")]
    public class Normal : Node
    {
        [Output]
        public Vector3 Result;

        public CoordinateSpace Space = CoordinateSpace.Object;
    }

    [Category("Master")]
    public class MasterNodeDeferred : Node
    {
        [Input]
        public Vector3 Diffuse;

        [Input]
        public Vector3 Normal;

        [Input]
        public Vector3 Emissive;

        [Input]
        public float Roughness;

        [Input]
        public float Metallic;

        [Input]
        public float Height;

        [Input]
        public float Alpha;
    }
}
