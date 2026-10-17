using System.Collections.Generic;
using System.Numerics;
using Squid;
using Freefall.Components;
using Freefall.Graphics;
using Freefall.Reflection;
using Vortice.Mathematics;

namespace Freefall.Editor
{
    /// <summary>
    /// MeshRenderer and SkinnedMeshRenderer: the standard component inspector, then this object's values for the
    /// per-object data its effects declare (their struct Instance, "//=" members) under INSTANCE, with the effect's
    /// descriptions, ranges (sliders) and colour pickers. A value not set on the object shows the effect's default;
    /// setting it saves it with the entity (PersistentRenderer.Instance).
    /// </summary>
    [GUIInspector(typeof(PersistentRenderer))]
    public class PersistentRendererInspector : ComponentInspector
    {
        public PersistentRendererInspector(GUIObject target) : base(target)
        {
            if (target.Target is not PersistentRenderer renderer) return;

            var seen = new HashSet<string>();
            bool header = false;
            foreach (var effect in renderer.GetInstanceEffects())
            {
                foreach (var member in effect.InstanceParams)
                {
                    if (!seen.Add(member.Name)) continue;
                    var handle = InstanceValueHandle.For(renderer, member);
                    if (handle == null) continue;

                    if (!header)
                    {
                        AddHeader("INSTANCE");
                        header = true;
                    }

                    var property = new GUIProperty(target, new Field(handle.GetType().GetProperty("Value")), handle)
                    {
                        DescriptionOverride = member.Description,
                    };
                    if (member.Min is float lo && member.Max is float hi)
                        property.RangeOverride = new ValueRangeAttribute(lo, hi, (hi - lo) / 100f);
                    AddProperty(property, member.Name);
                }
            }
        }
    }

    /// <summary>
    /// The inspector's handle on one instance member of a renderer: a Value of the member's type that reads the
    /// renderer's own value (or the effect's default) and sets it through PersistentRenderer.SetInstance.
    /// </summary>
    public abstract class InstanceValueHandle
    {
        protected readonly PersistentRenderer Renderer;
        protected readonly MaterialParamDesc Member;

        protected InstanceValueHandle(PersistentRenderer renderer, MaterialParamDesc member)
        {
            Renderer = renderer;
            Member = member;
        }

        protected Vector4 Current => Renderer.TryGetInstance(Member.Name, out var value) ? value : InstanceData.DefaultOf(Member);

        protected void Set(Vector4 value) => Renderer.SetInstance(Member.Name, value);

        public static InstanceValueHandle For(PersistentRenderer renderer, MaterialParamDesc member) => member.Type switch
        {
            "float" => new FloatHandle(renderer, member),
            "float2" => new Float2Handle(renderer, member),
            "float3" when member.IsColor => new ColorHandle(renderer, member),
            "float3" => new Float3Handle(renderer, member),
            "float4" => new Float4Handle(renderer, member),
            "int" or "uint" => new IntHandle(renderer, member),
            "bool" => new BoolHandle(renderer, member),
            _ => null,
        };

        public sealed class FloatHandle(PersistentRenderer r, MaterialParamDesc m) : InstanceValueHandle(r, m)
        {
            public float Value { get => Current.X; set => Set(new Vector4(value, 0f, 0f, 0f)); }
        }

        public sealed class Float2Handle(PersistentRenderer r, MaterialParamDesc m) : InstanceValueHandle(r, m)
        {
            public Vector2 Value { get => new(Current.X, Current.Y); set => Set(new Vector4(value, 0f, 0f)); }
        }

        public sealed class Float3Handle(PersistentRenderer r, MaterialParamDesc m) : InstanceValueHandle(r, m)
        {
            public Vector3 Value { get => new(Current.X, Current.Y, Current.Z); set => Set(new Vector4(value, 0f)); }
        }

        public sealed class ColorHandle(PersistentRenderer r, MaterialParamDesc m) : InstanceValueHandle(r, m)
        {
            public Color3 Value
            {
                get { var c = Current; return new Color3(c.X, c.Y, c.Z); }
                set => Set(new Vector4(value.R, value.G, value.B, 0f));
            }
        }

        public sealed class Float4Handle(PersistentRenderer r, MaterialParamDesc m) : InstanceValueHandle(r, m)
        {
            public Vector4 Value { get => Current; set => Set(value); }
        }

        public sealed class IntHandle(PersistentRenderer r, MaterialParamDesc m) : InstanceValueHandle(r, m)
        {
            public int Value { get => (int)Current.X; set => Set(new Vector4(value, 0f, 0f, 0f)); }
        }

        public sealed class BoolHandle(PersistentRenderer r, MaterialParamDesc m) : InstanceValueHandle(r, m)
        {
            public bool Value { get => Current.X != 0f; set => Set(new Vector4(value ? 1f : 0f, 0f, 0f, 0f)); }
        }
    }
}
