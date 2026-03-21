using Freefall.Components;
using Freefall.Graphics;
using System;
using System.Collections.Generic;

namespace Freefall.Reflection
{
    public interface IValuesProvider
    {
        IList<ProviderValue> GetValues(object context);
    }

    public struct ProviderValue
    {
        public string Name;
        public object Value;

        public override string ToString() => Name;
    }

    public class ValueSelectAttribute : Attribute
    {
        public IValuesProvider Provider { get; private set; }

        public ValueSelectAttribute(Type type)
        {
            Provider = Activator.CreateInstance(type) as IValuesProvider;
        }
    }

    public class MeshPartProvider : IValuesProvider
    {
        public IList<ProviderValue> GetValues(object context)
        {
            var result = new List<ProviderValue>();
            Mesh mesh = null;

            if (context is Assets.MeshElement target)
                mesh = target.Mesh;

            if (mesh == null) return result;

            for (int i = 0; i < mesh.MeshParts.Count; i++)
            {
                result.Add(new ProviderValue
                {
                    Name = mesh.MeshParts[i].Name,
                    Value = i
                });
            }

            return result;
        }
    }
}
