using System;
using System.Collections.Generic;
using System.Numerics;
using LiteYaml.Emitter;
using LiteYaml.Parser;

namespace Freefall.Serialization
{
    /// <summary>
    /// Type-specific YAML read/write handler.
    /// Register converters with YAMLSerializer for custom type support.
    /// </summary>
    public interface IYAMLConverter
    {
        object Read(ref YamlParser parser);
        void Write(object value, ref Utf8YamlEmitter emitter);
    }

    // ── Primitive converters ──────────────────────────────────────

    public class IntYAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser) => parser.ReadScalarAsInt32();
        public void Write(object value, ref Utf8YamlEmitter emitter) => emitter.WriteInt32((int)value);
    }

    public class FloatYAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser) => parser.ReadScalarAsFloat();
        public void Write(object value, ref Utf8YamlEmitter emitter) => emitter.WriteFloat((float)value);
    }

    public class DoubleYAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser) => parser.ReadScalarAsDouble();
        public void Write(object value, ref Utf8YamlEmitter emitter) => emitter.WriteDouble((double)value);
    }

    public class BoolYAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser) => parser.ReadScalarAsBool();
        public void Write(object value, ref Utf8YamlEmitter emitter) => emitter.WriteBool((bool)value);
    }

    public class StringYAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser) => parser.ReadScalarAsString();
        public void Write(object value, ref Utf8YamlEmitter emitter) => emitter.WriteString((string)value);
    }

    // ── System.Numerics converters ────────────────────────────────

    public class Vector2YAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser)
        {
            parser.Read(); // SequenceStart
            var x = parser.ReadScalarAsFloat();
            var y = parser.ReadScalarAsFloat();
            parser.Read(); // SequenceEnd
            return new Vector2(x, y);
        }

        public void Write(object value, ref Utf8YamlEmitter emitter)
        {
            var v = (Vector2)value;
            emitter.BeginSequence(SequenceStyle.Flow);
            emitter.WriteFloat(v.X);
            emitter.WriteFloat(v.Y);
            emitter.EndSequence();
        }
    }

    public class Vector3YAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser)
        {
            parser.Read(); // SequenceStart
            var x = parser.ReadScalarAsFloat();
            var y = parser.ReadScalarAsFloat();
            var z = parser.ReadScalarAsFloat();
            parser.Read(); // SequenceEnd
            return new Vector3(x, y, z);
        }

        public void Write(object value, ref Utf8YamlEmitter emitter)
        {
            var v = (Vector3)value;
            emitter.BeginSequence(SequenceStyle.Flow);
            emitter.WriteFloat(v.X);
            emitter.WriteFloat(v.Y);
            emitter.WriteFloat(v.Z);
            emitter.EndSequence();
        }
    }

    public class Vector4YAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser)
        {
            parser.Read(); // SequenceStart
            var x = parser.ReadScalarAsFloat();
            var y = parser.ReadScalarAsFloat();
            var z = parser.ReadScalarAsFloat();
            var w = parser.ReadScalarAsFloat();
            parser.Read(); // SequenceEnd
            return new Vector4(x, y, z, w);
        }

        public void Write(object value, ref Utf8YamlEmitter emitter)
        {
            var v = (Vector4)value;
            emitter.BeginSequence(SequenceStyle.Flow);
            emitter.WriteFloat(v.X);
            emitter.WriteFloat(v.Y);
            emitter.WriteFloat(v.Z);
            emitter.WriteFloat(v.W);
            emitter.EndSequence();
        }
    }

    public class QuaternionYAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser)
        {
            parser.Read(); // SequenceStart
            var x = parser.ReadScalarAsFloat();
            var y = parser.ReadScalarAsFloat();
            var z = parser.ReadScalarAsFloat();
            var w = parser.ReadScalarAsFloat();
            parser.Read(); // SequenceEnd
            return new Quaternion(x, y, z, w);
        }

        public void Write(object value, ref Utf8YamlEmitter emitter)
        {
            var q = (Quaternion)value;
            emitter.BeginSequence(SequenceStyle.Flow);
            emitter.WriteFloat(q.X);
            emitter.WriteFloat(q.Y);
            emitter.WriteFloat(q.Z);
            emitter.WriteFloat(q.W);
            emitter.EndSequence();
        }
    }

    // ── Vortice.Mathematics converters ─────────────────────────────

    public class Color3YAMLConverter : IYAMLConverter
    {
        public object Read(ref YamlParser parser)
        {
            parser.Read(); // SequenceStart
            var r = parser.ReadScalarAsFloat();
            var g = parser.ReadScalarAsFloat();
            var b = parser.ReadScalarAsFloat();
            parser.Read(); // SequenceEnd
            return new Vortice.Mathematics.Color3(r, g, b);
        }

        public void Write(object value, ref Utf8YamlEmitter emitter)
        {
            var c = (Vortice.Mathematics.Color3)value;
            emitter.BeginSequence(SequenceStyle.Flow);
            emitter.WriteFloat(c.R);
            emitter.WriteFloat(c.G);
            emitter.WriteFloat(c.B);
            emitter.EndSequence();
        }
    }
}
