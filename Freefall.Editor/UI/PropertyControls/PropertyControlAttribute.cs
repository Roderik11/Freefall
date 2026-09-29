
namespace Freefall.Editor
{
    public class PropertyControlAttribute : Attribute
    {
        public Type Type { get; private set; }

        public PropertyControlAttribute(Type type)
        {
            Type = type;
        }
    }
}
