
namespace Freefall.Editor
{
    public class GUIInspectorAttribute : Attribute
    {
        public Type Type { get; private set; }

        public GUIInspectorAttribute(Type type)
        {
            Type = type;
        }
    }
}
