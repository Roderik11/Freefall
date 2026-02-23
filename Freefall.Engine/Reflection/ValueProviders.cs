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
}
